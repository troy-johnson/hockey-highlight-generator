import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3" / "scripts"))
import recap_assembly as A
import recap_runner as R


def fixture(n=1):
    goals = [{"id": f"home:{i}", "chosen": {"primary_cam": "cam2", "detection_s": 20. + i * 20,
             "start_s": 14. + i * 20, "end_s": 24. + i * 20}} for i in range(n)]
    layout = {"cam2": [{"start": 0., "end": 10000., "seek": 20., "files": [("a.mp4", 10020.)]}]}
    return {"goals": goals}, layout


@pytest.mark.parametrize("count,before,after", [(6, 6, 6), (7, 5, 5), (12, 5, 5), (13, 4, 3)])
def test_budget_tiers(count, before, after):
    plan = A.plan_recap(*fixture(count))
    clip = plan["clips"][0]
    assert clip["start_s"] == 20 - before
    assert clip["end_s"] == pytest.approx(20 + after)
    assert clip["parts"][0]["seek_s"] == 40 - before
    assert plan["duration_s"] <= 240


def test_tier_counts_selected_goals_not_unmatched_claims():
    selection, layouts = fixture(6)
    selection["goals"].append({"id": "away:9", "chosen": None})
    plan = A.plan_recap(selection, layouts)
    assert plan["clips"][0]["start_s"] == 14  # 6-goal tier, not the 7-goal tier


def test_keeps_goals_in_order_and_reports_unmatched():
    selection, layouts = fixture(2)
    selection["goals"].reverse()
    selection["goals"].append({"id": "away:1", "chosen": None})
    original = copy.deepcopy(selection)
    plan = A.plan_recap(selection, layouts)
    assert [c["goal_id"] for c in plan["clips"]] == ["home:0", "home:1"]
    assert "away:1" in plan["flags"][0]
    assert selection == original


def test_cap_trims_build_up_but_keeps_goal_action():
    plan = A.plan_recap(*fixture(40))
    assert len(plan["clips"]) == 40
    assert plan["duration_s"] <= 240
    assert all(c["end_s"] - c["moment_s"] == pytest.approx(3.0) for c in plan["clips"])
    assert all(2.0 <= c["moment_s"] - c["start_s"] <= 4.0 for c in plan["clips"])
    assert any("four-minute cap" in f for f in plan["flags"])


def test_cap_raises_when_goal_action_would_be_cut():
    with pytest.raises(ValueError, match="four-minute cap"):
        A.plan_recap(*fixture(100), speed=.5)


@pytest.mark.parametrize("count,speed", [(38, 1.1), (49, 1.1), (35, 1.0), (47, 1.0)])
def test_cap_fits_games_near_the_limit(count, speed):
    plan = A.plan_recap(*fixture(count), speed=speed)
    assert len(plan["clips"]) == count
    assert plan["duration_s"] <= 240
    assert all(c["end_s"] - c["moment_s"] >= 3.0 - 1e-9 for c in plan["clips"])
    assert all(c["moment_s"] - c["start_s"] >= 2.0 - 1e-9 for c in plan["clips"])


def test_chapter_boundary_and_recording_gap():
    layout = [{"start": 0, "end": 20, "seek": 2, "files": [("a", 12), ("b", 10)]}]
    assert A.source_parts(layout, 9, 13) == [
        {"file": "a", "seek_s": 11, "duration_s": 1}, {"file": "b", "seek_s": 0, "duration_s": 3}]
    layout.append({"start": 25, "end": 35, "seek": 0, "files": [("c", 10)]})
    with pytest.raises(ValueError, match="gap"):
        A.source_parts(layout, 19, 26)


@pytest.mark.parametrize("speed", [0, -1, float("nan"), float("inf")])
def test_invalid_speed(speed):
    with pytest.raises(ValueError):
        A.plan_recap(*fixture(), speed=speed)


@pytest.mark.parametrize("field,value", [("detection_s", "20"), ("start_s", True), ("end_s", None),
                                         ("detection_s", float("nan"))])
def test_invalid_selected_times(field, value):
    selection, layouts = fixture()
    selection["goals"][0]["chosen"][field] = value
    with pytest.raises(ValueError, match="invalid|outside"):
        A.plan_recap(selection, layouts)


def test_clip_is_trimmed_to_recording_coverage_not_failed():
    selection, layouts = fixture(2)
    layouts["cam2"] = [{"start": 100.04, "end": 149.967, "seek": 0, "files": [("a", 50)]}]
    for g, shift in zip(selection["goals"], (85, 105)):  # moments at 105 and 145, near the block edges
        g["chosen"]["detection_s"] += shift
        g["chosen"]["start_s"] = g["chosen"]["detection_s"] - 6
        g["chosen"]["end_s"] = g["chosen"]["detection_s"] + 4
    plan = A.plan_recap(selection, layouts)
    assert [c["goal_id"] for c in plan["clips"]] == ["home:0", "home:1"]
    assert plan["clips"][0]["start_s"] == 100.04
    assert plan["clips"][1]["end_s"] == 149.967
    assert any("trimmed to Recording coverage" in f for f in plan["flags"])


def test_goal_in_recording_gap_is_omitted_not_fatal():
    selection, layouts = fixture(2)
    layouts["cam2"] = [{"start": 0, "end": 22, "seek": 0, "files": [("a", 22)]},
                       {"start": 30, "end": 100, "seek": 0, "files": [("b", 70)]}]
    selection["goals"][0]["chosen"]["detection_s"] = 25  # in the gap between Recordings
    selection["goals"][0]["chosen"]["start_s"] = 19
    selection["goals"][0]["chosen"]["end_s"] = 29
    plan = A.plan_recap(selection, layouts)
    assert [c["goal_id"] for c in plan["clips"]] == ["home:1"]
    assert any("home:0" in f and "omitted" in f for f in plan["flags"])


def test_filename_is_safe_and_focus_first(tmp_path):
    name = A.output_name(tmp_path, {"date": "2026-01-01", "focus_team": "Ice Pak"},
                         {"teams": {"home": "Bad/../Name", "away": "Ice Pak"}})
    assert name.startswith("2026-01-01_Ice-Pak-vs-")
    assert "/" not in name and name.endswith("_Recap.mp4")


def test_real_render_and_failure_preserves_output(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x180:rate=30:duration=2", "-c:v", "libx264", str(source)], check=True)
    selection = {"goals": [{"id": "g1", "chosen": {"primary_cam": "cam1", "detection_s": 1,
                            "start_s": 0, "end_s": 2}}]}
    layouts = {"cam1": [{"start": 0, "end": 2, "seek": 0, "files": [(str(source), 2)]}]}
    plan = A.plan_recap(selection, layouts)
    output = tmp_path / "Recap.mp4"
    A.render(plan, output)
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)]))
    assert probe["streams"][0]["width"] == 1920
    seq = plan["clips"] + plan.get("replays", [])
    assert int(probe["streams"][0]["nb_frames"]) == sum(e["frames"] for e in seq)
    old = output.read_bytes()
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "ffmpeg")
    monkeypatch.setattr(A.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        A.render(plan, output)
    assert output.read_bytes() == old


def test_short_chapter_is_flagged_and_omitted(tmp_path):
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x180:rate=30:duration=2", "-c:v", "libx264", str(source)], check=True)
    selection = {"goals": [{"id": "g1", "chosen": {"primary_cam": "cam1", "detection_s": 1,
                            "start_s": 0, "end_s": 2.9}}]}
    layouts = {"cam1": [{"start": 0, "end": 2.9, "seek": 0, "files": [(str(source), 2.9)]}]}
    plan = A.plan_recap(selection, layouts)
    assert len(plan["clips"]) == 1  # the plan cannot see the stream length
    with pytest.raises(ValueError, match="no selected goals"):
        A.verify_sources(plan)
    assert any("shorter than its manifest duration" in f for f in plan["flags"])


def test_older_recap_name_is_reported(tmp_path, monkeypatch):
    (tmp_path / "2001-01-01_Old-vs-Name_Recap.mp4").write_bytes(b"old")
    selection = {"goals": [{"id": "g1", "chosen": {"primary_cam": "cam1", "detection_s": 1,
                            "start_s": 0, "end_s": 2}}]}
    layouts = {"cam1": [{"start": 0, "end": 2, "seek": 0, "files": [("a.mp4", 2)]}]}
    plan = A.plan_recap(selection, layouts)
    monkeypatch.setattr(A, "verify_sources", lambda p: None)
    monkeypatch.setattr(A, "render", lambda p, o: o.write_bytes(b"new"))
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    (tmp_path / "game_sheet.json").write_text(json.dumps({
        "date": "2026-01-01", "teams": {"home": "Opponent", "away": "Ice Pak"}}))
    monkeypatch.setattr(A, "load_layouts", lambda root, cameras: layouts)
    assert A.main([str(tmp_path)]) == 0
    report = json.loads((tmp_path / "recap_assembly.json").read_text())
    assert (tmp_path / "2026-01-01_Opponent-vs-Ice-Pak_Recap.mp4").read_bytes() == b"new"
    assert (tmp_path / "2001-01-01_Old-vs-Name_Recap.mp4").read_bytes() == b"old"
    assert any("older Recap kept" in f for f in report["flags"])
    assert not (tmp_path / "2026-01-01_Opponent-vs-Ice-Pak_Recap.mp4.tmp").exists()


def test_runner_wires_assembly_and_fingerprints_options(tmp_path):
    stage = next(s for s in R.STAGES if s.name == "assembly")
    assert stage.needs == ("selection",)
    ctx = R.Context(tmp_path, {"live_play_speed": 1.1}, R.Reporter())
    before = R._fp_assembly(ctx)
    ctx.options["live_play_speed"] = 1.2
    assert R._fp_assembly(ctx) != before


def test_configured_focus_id_resolves_to_display_name(tmp_path):
    sheet = {"teams": {"home": "Opponent", "away": "Ice Pak"}}
    (tmp_path / "game_sheet.json").write_text(json.dumps(sheet))
    ctx = R.Context(tmp_path, {"date": "2026-01-01", "focus_team": "ice-pak",
                              "teams": {"ice-pak": {"name": "Ice Pak"}}}, R.Reporter())
    assert R.assembly_outputs(ctx)[1] == "2026-01-01_Ice-Pak-vs-Opponent_Recap.mp4"


def test_cli_run_renders_assembly_and_reuses_it(tmp_path, monkeypatch):
    import hockeyrecap as cli
    source = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x180:rate=30:duration=2", "-c:v", "libx264", str(source)], check=True)
    (tmp_path / "cam1_concat.txt").write_text(f"ffconcat version 1.0\nfile '{source}'\n")
    (tmp_path / "selection.json").write_text(json.dumps({"goals": [{"id": "home:1", "chosen": {
        "primary_cam": "cam1", "detection_s": 1, "start_s": 0, "end_s": 2}}]}))
    (tmp_path / "game_sheet.json").write_text(json.dumps({
        "teams": {"home": "Opponent", "away": "Ice Pak"}, "date": "2026-01-01"}))
    R.save_status(tmp_path, {"stages": {"selection": {"state": "done"}}})
    monkeypatch.setattr(R, "STAGES", [s for s in R.STAGES if s.name == "assembly"])
    args = ["run", str(tmp_path), "--config-dir", str(tmp_path / "config"), "--focus-team", "Ice Pak"]
    assert cli.main(args) == 0
    output = tmp_path / "2026-01-01_Ice-Pak-vs-Opponent_Recap.mp4"
    assert output.exists()
    assert R.load_status(tmp_path)["stages"]["assembly"]["state"] == "flagged"
    before = output.stat().st_mtime_ns
    assert cli.main(args) == 0
    assert R.load_status(tmp_path)["stages"]["assembly"]["reused"]
    assert output.stat().st_mtime_ns == before
