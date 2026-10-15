import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3" / "scripts"))


def test_short_selects_highest_interest_covered_plays_in_game_order():
    from short_clips import plan_short
    selection = {"goals": [{"id": f"away:{i}", "interest": score,
                 "chosen": {"primary_cam": "cam1", "detection_s": t,
                            "start_s": t - 6, "end_s": t + 6}}
                for i, t, score in [(1, 20, .5), (2, 40, .9), (3, 60, .8), (4, 80, .7),
                                    (5, 100, .6), (6, 120, .1)]],
                 "plays": [{"id": "play:icing", "kind": "net_play", "score": 1.,
                            "primary_cam": "cam1", "detection_s": 10.}]}
    selection["goals"].append({"id": "away:7", "interest": 1., "chosen": None})
    layouts = {"cam1": [{"start": 0., "end": 200., "seek": 0., "files": [("synthetic.mp4", 200.)]}]}
    before = copy.deepcopy(selection)
    plan = plan_short(selection, layouts, rois={"cam1": {"net": [500, 250, 100, 100]}})
    assert [p["goal_id"] for p in plan["clips"]] == ["away:1", "away:2", "away:3", "away:4", "away:5"]
    assert plan["opening"][0]["goal_id"] == "away:2"
    assert 30 <= plan["duration_s"] <= 60
    assert 30 <= plan["opening"][0]["frames"] <= 60
    assert plan["final"]["frames"] == 150
    assert len(plan["single_goals"]) == 3
    assert all(r["layout"] == "9:16" for r in plan["replays"])
    assert selection == before


def test_focus_short_preserves_sheet_score_for_uncovered_prior_goal():
    from short_clips import plan_short
    goals = [{"id": "home:1", "period": 1, "elapsed_s": 2., "interest": 1., "chosen": None}]
    goals += [{"id": f"away:{i}", "period": 1, "elapsed_s": i*10., "interest": .8,
               "chosen": {"primary_cam": "cam1", "detection_s": i*20.}} for i in range(1, 4)]
    goals.append({"id": "home:2", "period": 1, "elapsed_s": 5., "interest": 1.,
                  "chosen": {"primary_cam": "cam1", "detection_s": 10.}})
    selection = {"goals": goals, "plays": []}
    layouts = {"cam1": [{"start": 0., "end": 100., "seek": 0., "files": [("synthetic.mp4", 100.)]}]}
    plan = plan_short(selection, layouts, focus_side="away")
    assert [c["goal_id"] for c in plan["clips"]] == ["away:1", "away:2", "away:3"]
    assert plan["clips"][0]["score_before"] == [2, 0]
    assert plan["clips"][0]["score"] == [2, 1]


def test_motion_spread_selects_window_and_smooths_offline_tracking():
    from short_clips import frame_play
    # Coordinates are fractions of a 16:9 source. Full-height 9:16 covers 0.3164 of its width.
    narrow = [[i / 10, .3 + (i % 2) * .04, .2, .5 + (i % 2) * .04, .8] for i in range(20)]
    assert frame_play(narrow)["layout"] == "9:16"
    boundary = narrow[:17] + [[1.7 + i / 10, .1, .2, .8, .8] for i in range(3)]
    assert frame_play(boundary)["layout"] == "9:16"  # exactly 85 percent
    assert frame_play(narrow[:16] + [[2, .2, .2, .6, .8]] * 4)["layout"] == "4:5"
    wide = [[i / 10, .1, .2, .9, .8] for i in range(20)]
    assert frame_play(wide)["layout"] == "1:1"
    tracked = frame_play(narrow)
    assert tracked["tracking"][0][1] != .4  # future samples affect the first center
    assert max(abs(a[1] - b[1]) for a, b in zip(tracked["tracking"], tracked["tracking"][1:])) < .01


def test_focus_short_excludes_opponent_penalties_without_side_field():
    from short_clips import plan_short
    selection = {"goals": [{"id": f"away:{i}", "interest": .5, "chosen": {
        "primary_cam": "cam1", "detection_s": i*20.}} for i in range(1, 4)],
        "plays": [{"id": "penalty:home:1", "kind": "penalty", "score": 1.,
                   "primary_cam": "cam1", "detection_s": 10.}]}
    layouts = {"cam1": [{"start": 0., "end": 100., "seek": 0., "files": [("synthetic.mp4", 100.)]}]}
    plan = plan_short(selection, layouts, focus_side="away")
    assert [c["goal_id"] for c in plan["clips"]] == ["away:1", "away:2", "away:3"]


def test_focus_short_infers_net_plays_from_period_net_mapping():
    from short_clips import plan_short
    selection = {"goals": [], "mapping": {"defender_teams": {"A": "home", "B": "away"}, "source": "explicit"},
                 "periods": [{"n": 1, "start": 0., "end": 100.}],
                 "plays": [{"id": f"play:{i}", "kind": "net_play", "primary_cam": "cam1",
                             "score": .8, "detection_s": i*20., "review_status": "keep"} for i in range(1, 4)]}
    layouts = {"cam1": [{"start": 0., "end": 100., "seek": 0., "files": [("synthetic.mp4", 100.)]}]}
    coverage = {"cam1": {"periods": [{"n": 1, "defender": "A"}]}}
    plan = plan_short(selection, layouts, focus_side="away", coverage=coverage)
    assert len(plan["clips"]) == 3
    assert all(c["kind"] == "net_play" for c in plan["clips"])
    assert any("inferred" in f for f in plan["flags"])


def test_short_score_changes_at_goal_moment_and_replay_keeps_new_score():
    from short_clips import score_at
    entry = {"kind": "goal", "start_s": 10., "moment_s": 13., "speed": 1.,
             "score_before": [2, 0], "score": [2, 1]}
    assert score_at(entry, 2.9) == [2, 0]
    assert score_at(entry, 3.) == [2, 1]
    assert score_at(dict(entry, kind="replay", speed=.5), 0.) == [2, 1]


def test_short_replay_uses_native_gate_and_optional_interpolator():
    from short_clips import replay_mode
    assert replay_mode(60., True) == "native"
    assert replay_mode(120., False) == "native"
    assert replay_mode(30., True) == "rife"
    assert replay_mode(59.94, False) == "duplicated"
    assert replay_mode(None, False) == "duplicated"


def test_short_plan_places_bar_cuts_captions_and_both_team_final():
    from short_clips import plan_short, short_sequence
    selection = {"goals": [{"id": "away:1", "interest": .9, "chosen": {
        "primary_cam": "cam1", "detection_s": 20., "start_s": 14., "end_s": 26.}}],
        "plays": [{"id": "p1", "kind": "penalty", "primary_cam": "cam1", "score": .8,
                   "detection_s": 40., "start_s": 34., "end_s": 46., "caption": "#12 Tripping"},
                  {"id": "p2", "kind": "save", "primary_cam": "cam1", "score": .7,
                   "detection_s": 60., "start_s": 54., "end_s": 66.}]}
    layouts = {"cam1": [{"start": 0., "end": 100., "seek": 0., "files": [("synthetic.mp4", 100.)]}]}
    sheet = {"teams": {"home": "Salty Boys", "away": "Ice Pak"},
             "goals": {"home": [], "away": [{"scorer": "12", "per": "1"}]}}
    plan = plan_short(selection, layouts, sheet=sheet, bar_s=2.4)
    sequence = short_sequence(plan)
    cursor = sequence[0]["frames"]
    assert cursor == 60
    for entry in sequence[1:-1]:
        assert entry["frames"] % 72 == 0
        cursor += entry["frames"]
    assert 30 <= plan["duration_s"] <= 60
    assert "#12" in plan["clips"][0]["caption"]
    assert plan["clips"][1]["caption"] == "#12 Tripping"
    assert plan["clips"][2]["caption"] is None
    assert plan["clips"][2]["score"] == [0, 1]
    assert plan["final"]["teams"] == sheet["teams"]
    assert plan["final"]["score"] == [0, 1]
    assert plan["bed_offset_s"] == .4


def test_short_budget_handles_long_bars_and_skips_short_recording():
    from short_clips import plan_short
    selection = {"goals": [{"id": f"away:{i}", "interest": 1/i, "chosen": {
        "primary_cam": "cam1", "detection_s": i*20.}} for i in range(1, 7)],
        "plays": [{"id": "short", "kind": "save", "score": 2.,
                   "primary_cam": "cam2", "detection_s": 2.}]}
    layouts = {"cam1": [{"start": 0., "end": 150., "seek": 0., "files": [("synthetic.mp4", 150.)]}],
               "cam2": [{"start": 0., "end": 1., "seek": 0., "files": [("short.mp4", 1.)]}]}
    for bar in (2., 6., 8., 1.875):
        plan = plan_short(selection, layouts, bar_s=bar)
        assert 30 <= plan["duration_s"] <= 60
        assert 3 <= len(plan["clips"]) <= 5
        assert "short" not in [c["goal_id"] for c in plan["clips"]]
        cursor = 2.
        from short_clips import short_sequence
        for entry in short_sequence(plan)[1:-1]:
            cursor += entry["frames"] / 30
            assert abs((cursor - 2.) / bar - round((cursor - 2.) / bar)) * bar <= 1/60 + 1e-6
    saves = {"goals": [], "plays": [{"id": f"p{i}", "kind": "save", "score": 1.,
             "primary_cam": "cam1", "detection_s": i*20.} for i in range(1, 6)]}
    saves["plays"].insert(0, {"id": "limited", "kind": "save", "score": 2.,
                            "primary_cam": "cam3", "detection_s": 3.})
    layouts["cam3"] = [{"start": 0., "end": 7., "seek": 0., "files": [("limited.mp4", 7.)]}]
    plan = plan_short(saves, layouts, bar_s=2.4)
    assert len(plan["clips"]) == 5
    assert "limited" in [c["goal_id"] for c in plan["clips"]]
    assert plan["opening"][0]["goal_id"] == "limited"


def test_synthetic_render_uses_vertical_window_and_final(tmp_path, monkeypatch):
    import subprocess
    import json
    from short_clips import render_short
    import recap_assembly
    monkeypatch.setattr(recap_assembly, "rife_ready", lambda: False)
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=s=320x180:r=60:d=2", "-c:v", "libx264", "-y", str(source)], check=True)
    entry = {"kind": "goal", "camera": "cam1", "goal_id": "away:1", "frames": 15,
             "speed": 1., "start_s": 0., "end_s": .5, "caption": "GOAL #12", "score": [0, 1],
             "parts": [{"file": str(source), "seek_s": 0., "duration_s": .5}],
             "framing": {"layout": "4:5", "tracking": [[0., .5, .5]], "fit": 1., "flags": []}}
    plan = {"opening": [], "clips": [entry], "replays": [], "teams": {"home": "Red", "away": "Blue"},
            "final": {"kind": "final", "frames": 15, "score": [0, 1], "table": {}, "goalies": {}},
            "duration_s": 1., "flags": []}
    entry["slowmo"] = "rife"
    target = tmp_path / "synthetic_Recap_short.mp4"
    render_short(plan, target)
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams",
                                               "-of", "json", str(target)]))["streams"][0]
    assert (probe["width"], probe["height"], probe["r_frame_rate"]) == (1080, 1920, "30/1")
    assert int(probe["nb_frames"]) == 30
    assert any("RIFE unavailable" in f for f in plan["flags"])


def test_motion_measurement_reuses_cache_and_single_goals_are_separate(tmp_path):
    import subprocess
    from short_clips import measure_motion, plan_short, single_goal_plan
    source = tmp_path / "moving.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=30:d=3",
                    "-c:v", "libx264", "-y", str(source)], check=True)
    entry = {"parts": [{"file": str(source), "seek_s": 0., "duration_s": 2.}], "speed": 1.}
    samples = measure_motion(entry, tmp_path / "cache")
    assert len(samples) >= 8
    cache = list((tmp_path / "cache").glob("*.json"))
    assert len(cache) == 1
    stamp = cache[0].stat().st_mtime_ns
    assert measure_motion(entry, tmp_path / "cache") == samples
    assert cache[0].stat().st_mtime_ns == stamp
    selection = {"goals": [{"id": f"away:{i}", "interest": 1/i, "chosen": {
        "primary_cam": "cam1", "detection_s": i * 10}} for i in range(1, 4)], "plays": []}
    layouts = {"cam1": [{"start": 0., "end": 40., "seek": 0., "files": [(str(source), 40.)]}]}
    plan = plan_short(selection, layouts)
    one = single_goal_plan(plan, "away:2")
    assert [c["goal_id"] for c in one["clips"]] == ["away:2"]
    assert [c["goal_id"] for c in one["replays"]] == ["away:2"]
    assert one["opening"] == []
    assert one["final"]["frames"] == 150
    assert plan["single_goals"] == ["away:1", "away:2", "away:3"]


def test_motion_measurement_covers_static_intervals_and_repairs_bad_cache(tmp_path):
    import subprocess
    from short_clips import measure_motion, frame_play
    source = tmp_path / "static.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=320x180:r=30:d=2",
                    "-c:v", "libx264", "-y", str(source)], check=True)
    entry = {"parts": [{"file": str(source), "seek_s": 0., "duration_s": 2.}]}
    samples = measure_motion(entry, tmp_path / "cache")
    assert len(samples) == 10
    assert frame_play(samples)["layout"] == "1:1"
    next((tmp_path / "cache").glob("*.json")).write_text("corrupt")
    assert measure_motion(entry, tmp_path / "cache") == samples


def test_short_audio_mutes_pa_and_measures_encoded_loudness(tmp_path):
    import subprocess
    import numpy as np
    from short_clips import prepare_audio, mix_short
    source = tmp_path / "rink.wav"
    bed = tmp_path / "bed.wav"
    for path, hz in [(source, 440), (bed, 110)]:
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        f"sine=frequency={hz}:sample_rate=48000:duration=8", "-y", str(path)], check=True)
    video = tmp_path / "video.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=1080x1920:r=30:d=8",
                    "-c:v", "libx264", "-preset", "ultrafast", "-y", str(video)], check=True)
    entry = {"kind": "save", "camera": "cam1", "goal_id": "p1", "frames": 90, "speed": 1.,
             "start_s": 0., "parts": [{"file": str(source), "seek_s": 0., "duration_s": 3.}]}
    plan = {"opening": [], "clips": [entry], "replays": [], "duration_s": 8., "bed_offset_s": 0.,
            "final": {"kind": "final", "frames": 150}, "flags": []}
    track = {"id": "synthetic", "file": str(bed), "duration_s": 8., "bar_s": 2.}
    spans = [{"camera": "cam1", "start": 1., "end": 2., "state": "mute"}]
    master, report = prepare_audio(plan, track, spans)
    # Only the 110 Hz Bed remains inside the muted window.
    t = np.arange(48000) / 48000
    def amplitude(start, hz):
        return abs(np.mean(master[int(start*48000):int((start+1)*48000), 0] * np.exp(-2j*np.pi*hz*t)))
    assert amplitude(1, 440) < amplitude(0, 440) * .02
    plain, _ = prepare_audio(plan, track, [])
    plain_rink = abs(np.mean(plain[:48000, 0] * np.exp(-2j*np.pi*440*t)))
    assert abs(amplitude(0, 440) / plain_rink - 1.) < .02
    assert report["muted_windows"] == 1
    assert report["rink_under_bed_db"] == 12.
    output = tmp_path / "synthetic_Recap_short.mp4"
    report = mix_short(plan, video, output, track, spans)
    assert -15.5 <= report["loudnorm"]["output_i"] <= -13.5
    assert report["loudnorm"]["output_tp"] <= -1.
    assert output.exists()


def test_short_bed_starts_at_measured_first_beat_and_ducks_for_goal(tmp_path):
    import subprocess
    import numpy as np
    from short_clips import prepare_audio
    source = tmp_path / "offset.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "sine=frequency=110:duration=6", "-af", "adelay=500", "-y", str(source)], check=True)
    plan = {"opening": [], "clips": [{"kind": "goal", "goal_id": "away:1", "frames": 90,
            "start_s": 10., "moment_s": 11., "speed": 1.}], "replays": [],
            "final": {"kind": "final", "frames": 150}, "duration_s": 8., "bed_offset_s": 0.}
    bed = {"id": "synthetic", "file": str(source), "duration_s": 6.5,
           "bar_s": 2., "beats": [.5, 1., 1.5, 2., 2.5, 3.]}
    master, report = prepare_audio(plan, bed, [])
    rms = lambda t: float(np.sqrt(np.mean(master[int(t*48000):int((t+.1)*48000)]**2)))
    assert rms(.1) > .1  # The half-second intro must not precede the first beat.
    assert rms(1.1) < rms(.1) * .6
    assert report["bed_source_start_s"] == .5
    assert report["duck_events"][0]["t"] == 1.
    plan["opening"] = [dict(plan["clips"][0], kind="hook", frames=60)]
    plan["duration_s"] += 2.
    _, hooked = prepare_audio(plan, bed, [])
    assert [e["t"] for e in hooked["duck_events"]] == [1., 3.]


def test_short_runner_registers_outputs_and_invalidates_changed_inputs(tmp_path):
    import json
    from recap_runner import STAGES, Context, Reporter
    from hockeyrecap import build_parser
    stage = next(s for s in STAGES if s.name == "short")
    assert stage.needs == ("selection",)
    assert build_parser().parse_args(["rerun", str(tmp_path), "--from", "short"]).from_stage == "short"
    (tmp_path / "short_assembly.json").write_text(json.dumps({"outputs": ["test_Recap_short.mp4", "test_Recap_goal_1.mp4"]}))
    ctx = Context(tmp_path, {"date": "2026-02-01", "focus_team": "Ice Pak"}, Reporter())
    assert stage.outputs(ctx) == ["short_assembly.json", "short_audio.json", "test_Recap_short.mp4", "test_Recap_goal_1.mp4"]
    first = stage.fingerprint(ctx)
    (tmp_path / "music_spans.json").write_text('{"spans": []}')
    assert stage.fingerprint(ctx) != first
    second = stage.fingerprint(ctx)
    (tmp_path / "rois.json").write_text('{"camera_1": {"net": [10, 20, 30, 40]}}')
    assert stage.fingerprint(ctx) != second


def test_short_cli_writes_compilation_and_three_goal_outputs(tmp_path, freeze_sheet):
    import subprocess
    import json
    source = tmp_path / "synthetic.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=60:d=40",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=40", "-c:v", "libx264",
                    "-preset", "ultrafast", "-c:a", "aac", "-y", str(source)], check=True)
    (tmp_path / "cam1_concat.txt").write_text(f"ffconcat version 1.0\nfile '{source}'\n")
    selection = {"goals": [{"id": f"away:{i}", "interest": 1/i, "chosen": {
        "primary_cam": "cam1", "detection_s": i*10.}} for i in range(1, 4)], "plays": []}
    sheet = {"teams": {"home": "Red", "away": "Blue"}, "goals": {"home": [], "away": [
        {"scorer": "12", "assists": [], "per": "1"} for _ in range(3)]}}
    sheet["rosters"] = {"home": {}, "away": {"12": "Blue Skater"}}
    freeze_sheet(sheet, {"focus_team": "Blue"})
    for name, data in [("selection.json", selection), ("game_sheet.json", sheet),
                       ("rois.json", {"camera_1": {"net": [500, 250, 100, 100]}})]:
        (tmp_path / name).write_text(json.dumps(data))
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "v3/scripts/short_clips.py"),
               str(tmp_path), "--options", json.dumps({"date": "2026-02-01", "focus_team": "Blue",
                   "audio": {"manifest": str(tmp_path / "missing-manifest.json")}})]
    subprocess.run(command, check=True)
    plan = json.loads((tmp_path / "short_assembly.json").read_text())
    report = json.loads((tmp_path / "short_audio.json").read_text())
    assert len(plan["outputs"]) == 4
    assert report["placeholder"] is True
    assert any("approved" in f for f in report["flags"])
    for name in plan["outputs"]:
        assert (tmp_path / name).exists()
        assert "Recap" in name  # discovery already excludes this naming family
        streams = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams",
                                                      "-of", "json", str(tmp_path / name)]))["streams"]
        assert (streams[0]["width"], streams[0]["height"]) == (1080, 1920)
        assert streams[1]["codec_name"] == "aac"
    assert 30 <= plan["duration_s"] <= 60


def test_non_goal_play_keeps_uncovered_goal_counted_before_it():
    from short_clips import plan_short
    goals = [{"id": "home:1", "period": 1, "elapsed_s": 2., "interest": .1, "chosen": None}]
    goals += [{"id": f"away:{i}", "period": 1, "elapsed_s": i*10., "interest": .8,
               "chosen": {"primary_cam": "cam1", "detection_s": i*20.}} for i in range(1, 4)]
    selection = {"goals": goals, "plays": [{"id": "p_save", "kind": "save", "primary_cam": "cam1",
                                            "score": 1., "detection_s": 70.}]}
    layouts = {"cam1": [{"start": 0., "end": 100., "seek": 0., "files": [("synthetic.mp4", 100.)]}]}
    plan = plan_short(selection, layouts)
    scores = {c["goal_id"]: c["score"] for c in plan["clips"]}
    assert scores["away:3"] == [1, 3]
    assert scores["p_save"] == [1, 3]   # was [0, 3]: the score went down


def test_fractional_seek_still_fills_every_frame(tmp_path, monkeypatch):
    # ffmpeg -ss 0.011 -t 6 with fps=30 gives 179 frames, not 180.
    import subprocess
    import json
    from short_clips import render_short
    import recap_assembly
    monkeypatch.setattr(recap_assembly, "rife_ready", lambda: False)
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=s=320x180:r=30:d=8", "-c:v", "libx264", "-g", "30", "-y", str(source)], check=True)
    entry = {"kind": "goal", "camera": "cam1", "goal_id": "away:1", "frames": 180,
             "speed": 1., "start_s": .011, "end_s": 6.011, "caption": None, "score": [0, 1],
             "parts": [{"file": str(source), "seek_s": .011, "duration_s": 6.}],
             "framing": {"layout": "4:5", "tracking": [[0., .5, .5]], "fit": 1., "flags": []}}
    plan = {"opening": [], "clips": [entry], "replays": [], "teams": {"home": "Red", "away": "Blue"},
            "final": {"kind": "final", "frames": 15, "score": [0, 1], "table": {}, "goalies": {}},
            "duration_s": 6.5, "flags": []}
    target = tmp_path / "fractional_Recap_short.mp4"
    render_short(plan, target)
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams",
                                               "-of", "json", str(target)]))["streams"][0]
    assert int(probe["nb_frames"]) == 195


def test_goal_with_unknown_sheet_clock_is_ordered_by_camera_time():
    from short_clips import plan_short
    goals = [{"id": "home:1", "period": None, "elapsed_s": None, "interest": .8,
              "chosen": {"primary_cam": "cam1", "detection_s": 20.}},
             {"id": "away:1", "period": 1, "elapsed_s": 50., "interest": .8,
              "chosen": {"primary_cam": "cam1", "detection_s": 100.}}]
    selection = {"goals": goals, "plays": [{"id": "p_save", "kind": "save", "primary_cam": "cam1",
                                            "score": 1., "detection_s": 50.}]}
    layouts = {"cam1": [{"start": 0., "end": 200., "seek": 0., "files": [("synthetic.mp4", 200.)]}]}
    scores = {c["goal_id"]: c["score"] for c in plan_short(selection, layouts)["clips"]}
    assert scores == {"home:1": [1, 0], "p_save": [1, 0], "away:1": [1, 1]}   # was 1-1, 1-1, 0-1
