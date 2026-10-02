import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3" / "scripts"))
import recap_assembly as A


def fixture(n=1):
    goals = [{"id": f"home:{i}", "chosen": {"primary_cam": "cam2", "detection_s": 30. + i * 40,
              "start_s": 20. + i * 40, "end_s": 40. + i * 40}} for i in range(n)]
    layout = {"cam2": [{"start": 0., "end": 100000., "seek": 20., "files": [("a.mp4", 100020.)]}]}
    return {"goals": goals}, layout


def replays(plan, kind=None):
    return [r for r in plan["replays"] if kind is None or r["kind"] == kind]


def output_seconds(plan, kind=None):
    return sum(r["frames"] for r in replays(plan, kind)) / A.FPS


def test_tier_1_gets_wide_and_tight_replays():
    plan = A.plan_recap(*fixture(1))
    assert [r["kind"] for r in plan["replays"]] == ["wide", "tight"]
    assert output_seconds(plan) == pytest.approx(8.0, abs=.1)  # 2 shots, ~8 s
    assert all(r["camera"] == "cam2" and r["speed"] == A.SLOW_MO for r in plan["replays"])


def test_tier_2_gets_one_long_tight_shot():
    plan = A.plan_recap(*fixture(7))
    assert [r["kind"] for r in plan["replays"]] == ["tight"] * 7
    assert output_seconds(plan, "tight") / 7 == pytest.approx(5.0, abs=.1)
    assert not replays(plan, "wide")


def test_tier_3_gets_one_short_tight_shot():
    plan = A.plan_recap(*fixture(13))
    assert len(replays(plan, "tight")) == 13
    assert output_seconds(plan, "tight") / 13 == pytest.approx(4.0, abs=.1)
    assert not replays(plan, "wide")


def test_replays_follow_their_goal_in_sequence_order():
    plan = A.plan_recap(*fixture(2))
    order = [(r["goal_id"], r["kind"]) for r in plan["replays"]]
    assert order == [("home:0", "wide"), ("home:0", "tight"), ("home:1", "wide"), ("home:1", "tight")]
    assert all(r["order"] == int(r["goal_id"].split(":")[1]) for r in plan["replays"])


def test_replay_window_lies_inside_recording_coverage():
    plan = A.plan_recap(*fixture(1))
    for r in plan["replays"]:
        assert r["start_s"] >= 0 and r["end_s"] <= 100000
        assert r["start_s"] < 30 + A.REPLAY_WINDOW_BIAS < r["end_s"]
    # Whistle detection leads the crossing: the window centers on detection + bias.
    r = plan["replays"][0]
    assert r["start_s"] == pytest.approx(30 + A.REPLAY_WINDOW_BIAS - 1.0)
    assert r["end_s"] == pytest.approx(30 + A.REPLAY_WINDOW_BIAS + 1.0)


def test_replay_shot_without_room_is_omitted_not_fatal():
    selection, layouts = fixture(1)
    layouts["cam2"] = [{"start": 29.9, "end": 30.6, "seek": 0, "files": [("a", 10.5)]}]
    plan = A.plan_recap(selection, layouts)
    assert len(plan["clips"]) == 1  # the live clip still fits
    assert not plan["replays"]
    assert any("replay wide shot has no room" in f for f in plan["flags"])
    assert any("replay tight shot has no room" in f for f in plan["flags"])


def test_live_clip_gets_slow_push_in_and_wide_stays_wide():
    plan = A.plan_recap(*fixture(1))
    assert plan["clips"][0]["zoom"] == {"type": "push", "z0": A.ZOOM_MIN, "z1": A.ZOOM_MAX}
    assert replays(plan, "wide")[0]["zoom"] == {"type": "static", "z": A.ZOOM_MIN}


def test_tight_replay_punches_in_on_the_goal_frame_box():
    rois = {"cam2": {"net": [640, 360, 200, 100]}}
    plan = A.plan_recap(*fixture(1), rois=rois)
    zoom = replays(plan, "tight")[0]["zoom"]
    keep = 1 - 2 * A.FISHEYE_CROP
    assert zoom["type"] == "roi"
    # Box plus context, but never tighter than 1/ZOOM_TIGHT_MAX of the cropped frame.
    raw = min(1.0, max(1.5 * 200 / 1280, 1.5 * 100 / 720)) / keep
    assert zoom["wf"] == pytest.approx(max(raw, 1 / A.ZOOM_TIGHT_MAX))
    assert zoom["cx"] == pytest.approx(((640 + 100) / 1280 - A.FISHEYE_CROP) / keep)
    assert zoom["cy"] == pytest.approx(((360 + 50) / 720 - A.FISHEYE_CROP) / keep)


def test_tight_replay_zoom_is_clamped_to_zoom_tight_max():
    # A tiny box (150 px wide) would ask for a ~5x punch-in; the clamp holds 2.5x.
    rois = {"cam2": {"net": [600, 340, 150, 90]}}
    plan = A.plan_recap(*fixture(1), rois=rois)
    zoom = replays(plan, "tight")[0]["zoom"]
    assert zoom["wf"] == pytest.approx(1 / A.ZOOM_TIGHT_MAX)
    assert zoom["hf"] == pytest.approx(1 / A.ZOOM_TIGHT_MAX)


def test_tight_replay_without_goal_box_falls_back_and_flags():
    plan = A.plan_recap(*fixture(1))
    assert replays(plan, "tight")[0]["zoom"] == {"type": "static", "z": A.ZOOM_MAX}
    assert any("no goal-frame box" in f for f in plan["flags"])
    plan = A.plan_recap(*fixture(1), rois={"cam2": {"slot": [0, 0, 10, 10]}})
    assert replays(plan, "tight")[0]["zoom"]["type"] == "static"
    assert any("no goal-frame box" in f for f in plan["flags"])


def test_tight_replay_clamps_the_box_inside_the_frame():
    rois = {"cam2": {"net": [1200, 600, 300, 300]}}
    plan = A.plan_recap(*fixture(1), rois=rois)
    zoom = replays(plan, "tight")[0]["zoom"]
    assert zoom["wf"] <= 1 and zoom["cx"] - zoom["wf"] / 2 >= 0
    assert zoom["cx"] + zoom["wf"] / 2 <= 1 and zoom["cy"] + zoom["hf"] / 2 <= 1


def test_native_slow_motion_when_footage_is_120fps():
    plan = A.plan_recap(*fixture(1), source_fps={"cam2": 120.0})
    assert all(r["slowmo"] == "native" for r in plan["replays"])
    plan = A.plan_recap(*fixture(1), source_fps={"cam2": 119.88})
    assert all(r["slowmo"] == "native" for r in plan["replays"])


def test_rife_only_when_footage_is_below_60fps(monkeypatch):
    plan = A.plan_recap(*fixture(1), source_fps={"cam2": 30.0})
    assert all(r["slowmo"] == "duplicated" for r in plan["replays"])
    assert any("RIFE unavailable" in f for f in plan["flags"])
    monkeypatch.setattr(A, "rife_ready", lambda: True)
    plan = A.plan_recap(*fixture(1), source_fps={"cam2": 30.0})
    assert all(r["slowmo"] == "rife" for r in plan["replays"])
    plan = A.plan_recap(*fixture(1), source_fps={"cam2": 120.0})
    assert all(r["slowmo"] == "native" for r in plan["replays"])
    # 0.5x at 30 fps advances 1/60 s per frame: 60 fps is already native slow motion.
    plan = A.plan_recap(*fixture(1), source_fps={"cam2": 60.0})
    assert all(r["slowmo"] == "native" for r in plan["replays"])
    assert not any("RIFE unavailable" in f for f in plan["flags"])


def test_unknown_frame_rate_never_claims_native_slow_motion(monkeypatch):
    # An unprobed rate is unknown, not native: duplicate and say so.
    plan = A.plan_recap(*fixture(1))
    assert all(r["slowmo"] == "duplicated" for r in plan["replays"])
    assert any("frame rate unknown; replay slow motion without interpolation" in f
               for f in plan["flags"])
    monkeypatch.setattr(A, "rife_ready", lambda: True)
    plan = A.plan_recap(*fixture(1))
    assert all(r["slowmo"] == "rife" for r in plan["replays"])


def test_horizon_angle_levels_the_live_and_replay_shots():
    plan = A.plan_recap(*fixture(1), horizon={"cam2": 3.5})
    assert plan["clips"][0]["angle_deg"] == pytest.approx(3.5)
    assert all(r["angle_deg"] == pytest.approx(3.5) for r in plan["replays"])


def test_frame_horizon_detector_demands_three_agreeing_board_lines():
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    def draw(angles):
        image = np.zeros((180, 320, 3), dtype=np.uint8)
        ys = [40, 90, 140][:len(angles)]
        for y, deg in zip(ys, angles):
            x0, x1 = 10, 310
            y1 = y + int(300 * math.tan(math.radians(deg)))
            cv2.line(image, (x0, y), (x1, y1), (255, 255, 255), 2)
        return image

    # Three gentle parallel lines read as the boards line.
    angle = A._frame_horizon_deg(draw([5, 5, 5]))
    assert angle == pytest.approx(5.0, abs=1.5)
    # One line is not evidence; a steep line is not a boards line; disagreement is not either.
    assert A._frame_horizon_deg(draw([5])) is None
    assert A._frame_horizon_deg(draw([20, 20, 20])) is None
    assert A._frame_horizon_deg(draw([2, 5, 8])) is None
    flat = np.zeros((180, 320, 3), dtype=np.uint8)
    assert A._frame_horizon_deg(flat) is None


def test_horizon_angle_is_clamped_and_zero_without_lines(tmp_path):
    source = tmp_path / "flat.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=black:size=320x180:rate=30:duration=1", "-c:v", "libx264",
                    str(source)], check=True)
    assert A.horizon_angle(str(source)) == 0.0
    assert A.horizon_angle(str(tmp_path / "missing.mp4")) == 0.0


def test_replays_dropped_first_when_over_the_cap():
    plan = A.plan_recap(*fixture(40))
    assert len(plan["clips"]) == 40
    assert not plan["replays"]
    assert any("replay shot(s) to fit the four-minute cap" in f for f in plan["flags"])
    assert plan["duration_s"] <= 240
    assert all(c["end_s"] - c["moment_s"] >= 3.0 - 1e-9 for c in plan["clips"])


def test_partial_replay_drops_keep_the_cap():
    plan = A.plan_recap(*fixture(24), speed=1.0)
    assert plan["duration_s"] <= 240
    assert len(plan["clips"]) == 24
    live = sum(c["frames"] for c in plan["clips"])
    replay = sum(r["frames"] for r in plan["replays"])
    assert live + replay <= A.MAX_FRAMES
    assert live + replay + 4 * A.FPS > A.MAX_FRAMES  # dropping one more shot would overshoot


def test_duration_includes_replays():
    plan = A.plan_recap(*fixture(2))
    expected = (sum(c["frames"] for c in plan["clips"])
                + sum(r["frames"] for r in plan["replays"])) / A.FPS
    assert plan["duration_s"] == pytest.approx(expected)


def test_real_render_sequence_with_slow_motion_and_framing(tmp_path):
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=1280x720:rate=120:duration=3", "-c:v", "libx264",
                    str(source)], check=True)
    selection = {"goals": [{"id": "g1", "chosen": {"primary_cam": "cam1", "detection_s": 1,
                             "start_s": 0, "end_s": 3}}]}
    layouts = {"cam1": [{"start": 0, "end": 3, "seek": 0, "files": [(str(source), 3)]}]}
    plan = A.plan_recap(selection, layouts, rois={"cam1": {"net": [640, 360, 200, 100]}},
                        source_fps={"cam1": 120.0}, horizon={"cam1": 0.0})
    assert [e["kind"] if "kind" in e else "live" for e in _sequence(plan)] == \
        ["live", "wide", "tight"]
    output = tmp_path / "Recap.mp4"
    A.render(plan, output)
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_format",
                                                "-show_streams", "-of", "json", str(output)]))
    stream = probe["streams"][0]
    assert stream["width"] == 1920 and stream["height"] == 1080
    assert int(stream["nb_frames"]) == sum(e["frames"] for e in _sequence(plan))
    assert abs(float(probe["format"]["duration"]) - plan["duration_s"]) <= .1
    # slow motion: the replay shots stretch 2 s of source to 4 s at 30 fps
    assert all(r["frames"] == 120 for r in plan["replays"])


def _sequence(plan):
    sequence = []
    for clip in plan["clips"]:
        sequence.append(clip)
        sequence.extend(r for r in plan.get("replays", []) if r["goal_id"] == clip["goal_id"])
    return sequence


def test_rife_slowmo_fails_loudly_without_the_binary(tmp_path):
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x180:rate=30:duration=1", "-c:v", "libx264",
                    str(source)], check=True)
    entry = {"goal_id": "g", "kind": "tight", "camera": "cam1", "speed": A.SLOW_MO,
             "slowmo": "rife", "source_fps": 30.0, "angle_deg": 0.0,
             "parts": [{"file": str(source), "seek_s": 0, "duration_s": 1}],
             "frames": 60, "zoom": {"type": "static", "z": 1.3}}
    dest = tmp_path / "out.mp4"
    with pytest.raises((OSError, subprocess.CalledProcessError)):
        A._rife_slowmo(source, dest, tmp_path, entry)  # rife-ncnn-vulkan is missing
    assert len(list((tmp_path / "rife-in").glob("*.png"))) == 30  # extraction did run at 30 fps


def test_rife_extraction_keeps_native_frames_of_sub_60fps_footage(tmp_path):
    """24 fps footage: no duplication before the interpolator, every native frame kept."""
    source = tmp_path / "source24.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x180:rate=24:duration=2", "-c:v", "libx264",
                    str(source)], check=True)
    entry = {"goal_id": "g", "kind": "tight", "camera": "cam1", "speed": A.SLOW_MO,
             "slowmo": "rife", "source_fps": 24.0, "angle_deg": 0.0,
             "parts": [{"file": str(source), "seek_s": 0, "duration_s": 2}],
             "frames": 120, "zoom": {"type": "static", "z": 1.3}}
    dest = tmp_path / "out.mp4"
    with pytest.raises((OSError, subprocess.CalledProcessError)):
        A._rife_slowmo(source, dest, tmp_path, entry, 24)  # fails at the first rife call
    assert len(list((tmp_path / "rife-in").glob("*.png"))) == 48  # native 24 fps, not padded to 30


def test_encode_thins_only_when_not_feeding_rife(tmp_path):
    source = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x180:rate=60:duration=2", "-c:v", "libx264",
                    str(source)], check=True)
    entry = {"goal_id": "g", "kind": "tight", "camera": "cam1", "speed": A.SLOW_MO,
             "angle_deg": 0.0, "frames": 120, "start_s": 0, "end_s": 2,
             "parts": [{"file": str(source), "seek_s": 0, "duration_s": 2}],
             "zoom": {"type": "static", "z": 1.3}}

    def probe(path):
        return int(json.loads(subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)]
        ))["streams"][0]["nb_read_frames"])

    plain = tmp_path / "plain-rife.mp4"
    A._encode(entry, plain, speed=1.0, frames=None, fps_floor=60)
    assert probe(plain) >= 120  # RIFE intermediate: no 60 fps native frame thinned away
    duplicated = tmp_path / "duplicated.mp4"
    A._encode(entry, duplicated, speed=A.SLOW_MO, frames=120)
    assert probe(duplicated) == 120  # normal slow-motion path still caps at 30 fps output


def test_render_rife_branch_runs_the_two_pass_pipeline(tmp_path, monkeypatch):
    """With a stub interpolator on PATH the rife entries render via plain-XXXX + _rife_slowmo."""
    stub = tmp_path / "bin"
    stub.mkdir()
    rife_stub = stub / "rife-ncnn-vulkan"
    rife_stub.write_text(
        "#!/bin/sh\n"
        'in="$2"; out="$4"; mkdir -p "$out"\n'
        'i=0; for f in "$in"/*.png; do\n'
        '  i=$((i+1)); cp "$f" "$out/$(printf %08d.png "$i")"\n'
        '  i=$((i+1)); cp "$f" "$out/$(printf %08d.png "$i")"\n'
        "done\n"
    )
    rife_stub.chmod(0o755)
    monkeypatch.setenv("PATH", str(stub) + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr(A, "rife_ready", lambda: True)

    source = tmp_path / "source30.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=1280x720:rate=30:duration=2", "-c:v", "libx264",
                    str(source)], check=True)
    selection = {"goals": [{"id": "g1", "chosen": {"primary_cam": "cam1", "detection_s": 1,
                             "start_s": 0, "end_s": 2}}]}
    layouts = {"cam1": [{"start": 0, "end": 2, "seek": 0, "files": [(str(source), 2)]}]}
    plan = A.plan_recap(selection, layouts, rois={"cam1": {"net": [640, 360, 200, 100]}},
                        source_fps={"cam1": 30.0}, horizon={"cam1": 0.0})
    assert all(r["slowmo"] == "rife" for r in plan["replays"])

    output = tmp_path / "Recap.mp4"
    A.render(plan, output)
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_format",
                                                "-show_streams", "-of", "json", str(output)]))
    assert int(probe["streams"][0]["nb_frames"]) == sum(e["frames"] for e in _sequence(plan))
    assert abs(float(probe["format"]["duration"]) - plan["duration_s"]) <= .1


def test_encode_filter_chain_carries_framing(tmp_path):
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=1280x720:rate=30:duration=1", "-c:v", "libx264",
                    str(source)], check=True)
    entry = {"goal_id": "g", "kind": "wide", "camera": "cam1", "speed": A.SLOW_MO,
             "slowmo": "native", "source_fps": 120.0, "angle_deg": 2.0,
             "start_s": 0, "end_s": 1,
             "parts": [{"file": str(source), "seek_s": 0, "duration_s": 1}],
             "frames": 60, "zoom": {"type": "static", "z": A.ZOOM_MIN}}
    dest = tmp_path / "shot.mp4"
    A._encode(entry, dest, speed=A.SLOW_MO, frames=60)
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams",
                                                "v:0", "-show_streams", "-of", "json", str(dest)]))
    assert probe["streams"][0]["width"] == 1920
    assert int(probe["streams"][0]["nb_frames"]) == 60


def test_verify_sources_drops_short_replays(tmp_path):
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x180:rate=30:duration=2", "-c:v", "libx264",
                    str(source)], check=True)
    selection = {"goals": [{"id": "g1", "chosen": {"primary_cam": "cam1", "detection_s": 1.9,
                             "start_s": 0, "end_s": 2.9}}]}
    layouts = {"cam1": [{"start": 0, "end": 2.9, "seek": 0, "files": [(str(source), 2.9)]}]}
    plan = A.plan_recap(selection, layouts)
    assert len(plan["clips"]) == 1
    assert plan["replays"]  # the plan cannot see the stream length yet
    with pytest.raises(ValueError, match="no selected goals"):
        A.verify_sources(plan)
    assert any("replay" in f and "shorter than its manifest duration" in f
               for f in plan["flags"])


def test_verify_sources_keeps_replays_only_for_kept_goals(tmp_path):
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x180:rate=30:duration=3", "-c:v", "libx264",
                    str(source)], check=True)
    selection = {"goals": [{"id": "g1", "chosen": {"primary_cam": "cam1", "detection_s": 1.5,
                             "start_s": 0, "end_s": 3}}]}
    layouts = {"cam1": [{"start": 0, "end": 3, "seek": 0, "files": [(str(source), 3)]}]}
    plan = A.plan_recap(selection, layouts)
    A.verify_sources(plan)
    assert len(plan["clips"]) == 1 and len(plan["replays"]) == 2
    assert plan["duration_s"] == pytest.approx(
        sum(e["frames"] for e in plan["clips"] + plan["replays"]) / A.FPS)


def test_load_rois_reads_both_formats(tmp_path):
    (tmp_path / "rois.json").write_text(json.dumps(
        {"camera_1": {"net": [1, 2, 3, 4], "slot": [0, 0, 1, 1]}}))
    assert A.load_rois(tmp_path) == {"cam1": {"net": [1, 2, 3, 4], "slot": [0, 0, 1, 1]}}
    (tmp_path / "rois.json").unlink()
    (tmp_path / "rois_auto.json").write_text(json.dumps(
        {"camera_2": {"goal_box": [10, 20, 110, 120], "confidence": .9}}))
    assert A.load_rois(tmp_path) == {"cam2": {"net": [10, 20, 100, 100]}}
    assert A.load_rois(tmp_path / "nowhere") == {}


def test_cli_main_probes_and_renders(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x180:rate=30:duration=2", "-c:v", "libx264",
                    str(source)], check=True)
    (tmp_path / "cam1_concat.txt").write_text(f"ffconcat version 1.0\nfile '{source}'\n")
    (tmp_path / "selection.json").write_text(json.dumps({"goals": [{"id": "home:1", "chosen": {
        "primary_cam": "cam1", "detection_s": 1, "start_s": 0, "end_s": 2}}]}))
    (tmp_path / "game_sheet.json").write_text(json.dumps({
        "teams": {"home": "Opponent", "away": "Ice Pak"}, "date": "2026-01-01"}))
    (tmp_path / "rois.json").write_text(json.dumps({"camera_1": {"net": [600, 300, 200, 100]}}))
    report = {}
    def fake_render(plan, output):
        report.update(plan)
        output.write_bytes(b"new")
    monkeypatch.setattr(A, "render", fake_render)
    assert A.main([str(tmp_path)]) == 0
    assert report["schema_version"] == 2
    assert len(report["replays"]) == 2
    assert all(r["slowmo"] == "duplicated" for r in report["replays"])  # 30 fps source, no RIFE
    assert any("RIFE unavailable" in f for f in report["flags"])
    assert json.loads((tmp_path / "recap_assembly.json").read_text())["replays"]


def test_load_rois_merges_per_camera_with_rois_json_precedence(tmp_path):
    (tmp_path / "rois.json").write_text(json.dumps(
        {"camera_1": {"net": [10, 10, 100, 60]}}))
    (tmp_path / "rois_auto.json").write_text(json.dumps(
        {"camera_1": {"goal_box": [20, 20, 120, 80]},
         "camera_2": {"goal_box": [30, 30, 130, 90]}}))
    rois = A.load_rois(tmp_path)
    assert rois["cam1"]["net"] == [10, 10, 100, 60]  # rois.json wins per camera
    assert rois["cam2"]["net"] == [30, 30, 100, 60]  # auto fills the missing camera


def test_replay_punch_in_centers_the_goal_box_pixel(tmp_path):
    """Pixel test: the box center must land at frame center, with and without levelling."""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    def marker_centroid(marker, angle):
        mx, my = marker
        source = tmp_path / f"src-{mx}-{my}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"color=c=gray:size=1280x720:rate=60:duration=1",
                        "-vf", f"drawbox=x={mx - 10}:y={my - 10}:w=20:h=20:color=white:t=fill",
                        "-c:v", "libx264", str(source)], check=True)
        rois = {"cam1": {"net": [mx - 100, my - 50, 200, 100]}}  # box center = marker
        zoom, _found = A._tight_zoom("cam1", rois, angle)
        entry = {"goal_id": "g", "kind": "tight", "camera": "cam1", "speed": A.SLOW_MO,
                 "angle_deg": angle, "start_s": 0, "end_s": 1, "frames": 60,
                 "parts": [{"file": str(source), "seek_s": 0, "duration_s": 1}],
                 "zoom": zoom, "slowmo": "duplicated", "source_fps": 60.0}
        shot = tmp_path / f"shot-{mx}-{my}-{angle}.mp4"
        A._encode(entry, shot, speed=A.SLOW_MO, frames=60)
        ok, frame = cv2.VideoCapture(str(shot)).read()
        assert ok
        ys, xs = np.where(frame[:, :, 2] > 200)
        assert len(xs) > 50  # the marker is present
        return xs.mean(), ys.mean()

    for marker in ((800, 300), (900, 250), (400, 450)):
        for angle in (0.0, 6.0):
            cx, cy = marker_centroid(marker, angle)
            assert cx == pytest.approx(960, abs=12), f"{marker} angle {angle}: x {cx:.0f}"
            assert cy == pytest.approx(540, abs=12), f"{marker} angle {angle}: y {cy:.0f}"


def test_tight_zoom_rotation_is_aspect_corrected():
    """The ROI center rotates in pixel space, not fraction space (16:9 frame)."""
    rois = {"cam1": {"net": [700, 250, 200, 100]}}  # raw center fraction (0.64205, 0.40530)
    zoom, found = A._tight_zoom("cam1", rois, 6.0)
    assert found and zoom["type"] == "roi"
    assert zoom["wf"] == pytest.approx(0.4)  # clamped by ZOOM_TIGHT_MAX
    dx, dy = zoom["cx"] - .5, zoom["cy"] - .5
    raw_cx, raw_cy = (800 / 1280 - A.FISHEYE_CROP) / (1 - 2 * A.FISHEYE_CROP), \
        (300 / 720 - A.FISHEYE_CROP) / (1 - 2 * A.FISHEYE_CROP)
    angle = math.radians(6.0)
    expected_cx = .5 + (raw_cx - .5) * math.cos(angle) + (raw_cy - .5) * math.sin(angle) * (720 / 1280)
    expected_cy = .5 - (raw_cx - .5) * math.sin(angle) * (1280 / 720) + (raw_cy - .5) * math.cos(angle)
    assert dx == pytest.approx(expected_cx - .5, abs=1e-4)
    assert dy == pytest.approx(expected_cy - .5, abs=1e-4)


def test_verify_sources_keeps_a_live_clip_whose_replay_overruns(tmp_path):
    source = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x180:rate=30:duration=3", "-c:v", "libx264",
                    str(source)], check=True)
    plan = {"clips": [{"goal_id": "g1", "frames": 60,
                       "parts": [{"file": str(source), "seek_s": 0, "duration_s": 2.9}]}],
            "replays": [{"goal_id": "g1", "kind": "tight", "frames": 30,
                         "parts": [{"file": str(source), "seek_s": 2.5, "duration_s": 1.0}]}],
            "flags": []}
    A.verify_sources(plan)
    assert len(plan["clips"]) == 1  # the live clip survives
    assert not plan["replays"]  # the overruning replay is dropped
    assert any("replay tight shot" in f and "shorter than its manifest duration" in f
               for f in plan["flags"])
