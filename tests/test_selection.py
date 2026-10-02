import json
from pathlib import Path

import numpy as np
import pytest

import selection as S
import selection_inputs as I
import recap_runner as rr


@pytest.mark.parametrize("direction,clock,expected", [("remaining", 570, 210), ("elapsed", 210, 210)])
def test_elapsed_time(direction, clock, expected):
    assert S.elapsed_time(clock, 780, direction) == expected


@pytest.mark.parametrize("clock,length,direction", [(None, 780, "remaining"), (800, 780, "remaining"),
                                                    (-1, 780, "elapsed"), (5, 0, "elapsed"),
                                                    (float("nan"), 780, "elapsed"), (5, 780, None)])
def test_invalid_clock(clock, length, direction):
    assert S.elapsed_time(clock, length, direction) is None


def test_clock_windows_and_uncertain_boundaries():
    period = {"start": 100, "end": 1100}
    running = S.clock_window(period, 200, 780, "running")
    assert running["expected"] == 300 and (running["start"], running["end"]) == (285, 315)
    stop = S.clock_window(period, 200, 780, "stop")
    assert (stop["start"], stop["end"]) == (285, 535)
    period["break_before"] = {"uncertain": True}
    uncertain = S.clock_window(period, 200, 780, "stop")
    assert uncertain["start"] < stop["start"] and uncertain["end"] > stop["end"]


def test_missing_center_faceoff_does_not_inflate_score():
    features = {k: 1 for k in S.WEIGHTS}
    features["center_faceoff"] = None
    assert S.weighted_score(features) == pytest.approx(0.9)
    assert S.weighted_score({"center_faceoff": 1}) == pytest.approx(0.1)
    assert S.weighted_score({"flow": 100}) == pytest.approx(0.15)


def test_monotonic_matching_preserves_order_and_allows_unmatched():
    goals = [{"period": 1, "elapsed_s": t} for t in (10, 50, 60)]
    candidates = [{"t": t} for t in (100, 120, 160)]
    edges = {(0, 0): {"score": .9}, (1, 1): {"score": .95}, (1, 2): {"score": .85},
             (2, 2): {"score": .5}}
    _, path = S.ordered_match(goals, candidates, edges, "stop")
    assert path == {0: 0, 1: 2}  # 120 is too early for 40 clock seconds after 100.
    _, path = S.ordered_match(goals, candidates, edges, "stop", forbidden=(1, 2))
    assert path == {1: 1}


def test_candidate_cannot_be_used_twice_and_clock_resets_between_periods():
    goals = [{"period": 1, "elapsed_s": 700}, {"period": 2, "elapsed_s": 10}]
    candidates = [{"t": 100}, {"t": 110}]
    edges = {(i, j): {"score": .9} for i in range(2) for j in range(2)}
    _, path = S.ordered_match(goals, candidates, edges, "stop")
    assert path == {0: 0, 1: 1}


@pytest.mark.parametrize("second_moment", [100, 99, 104])
def test_matching_checks_camera_moments_not_candidate_seeds(second_moment):
    goals = [{"period": 1, "elapsed_s": t} for t in (10, 20)]
    candidates = [{"t": 100}, {"t": 115}]
    edges = {(0, 0): {"score": .9, "t": 100}, (1, 1): {"score": .9, "t": second_moment}}
    _, path = S.ordered_match(goals, candidates, edges, "stop")
    assert len(path) == 1


def test_audio_stoppage_is_a_soft_order_penalty_not_goal_proof():
    before = {"period": 1, "elapsed_s": 10}
    goal = {"period": 1, "elapsed_s": 40}
    candidate = {"t": 150}
    previous = {"t": 100, "cameras": {"cam1": {"stoppage_s": 40}}}
    prior_edge = {"t": 102, "camera": "cam1"}
    edge = {"t": 148, "features": {"clock": 1}}
    penalty = S.order_penalty(before, goal, previous, candidate, prior_edge, edge, "stop")
    assert 0 < penalty < S.WEIGHTS["clock"]
    assert S.order_penalty(before, goal, previous, candidate, prior_edge, edge, "running") == 0
    assert S.order_penalty(before, dict(goal, period=2), previous, candidate, prior_edge, edge, "stop") == 0


def test_matching_allows_reversed_seed_order_when_camera_moments_increase():
    goals = [{"period": 1, "elapsed_s": t} for t in (10, 12)]
    candidates = [{"t": 100}, {"t": 100.5}]
    edges = {(0, 1): {"score": .9, "t": 98.75}, (1, 0): {"score": .9, "t": 100.9}}
    _, path = S.ordered_match(goals, candidates, edges, "stop")
    assert path == {0: 1, 1: 0}


def test_matching_remembers_used_candidate_across_other_matches():
    goals = [{"period": 1, "elapsed_s": t} for t in (10, 12, 16)]
    candidates = [{"t": 100}, {"t": 102}]
    edges = {(0, 0): {"score": .9, "t": 100}, (1, 1): {"score": .9, "t": 102},
             (2, 0): {"score": .9, "t": 106}}
    _, path = S.ordered_match(goals, candidates, edges, "stop")
    assert len(path) == 2 and len(set(path.values())) == 2


def test_uncertain_game_end_and_next_break_widen_both_periods():
    structure = {"game": {"end_uncertain": True}, "periods": [
        {"n": 1, "start": 0, "end": 100},
        {"n": 2, "start": 200, "end": 300, "break_before": {"uncertain": True}},
        {"n": 3, "start": 400, "end": 500}]}
    periods = S.selection_periods(structure)
    assert all(S.boundary_uncertain(p) for p in periods)
    assert "end_uncertain" not in structure["periods"][0]
    assert S.clock_window(periods[-1], 50, 60, "running")["end"] == 570


def test_flow_peak_outside_whistle_search_window_is_not_discarded():
    net = np.zeros(2000)
    net[1000] = 10
    candidates = S.build_candidates({"whistles": [{"t": 106.2}]},
        [{"primary_cam": "1", "start_s": 99, "end_s": 101}],
        {"cam1": {"net": net, "slot": net}}, {}, {"cam1": {"spans": [[0, 200]]}}, 10)
    peak = next(c for c in candidates if c["id"] == "flow:0")
    assert peak["cameras"]["cam1"]["t"] == 100


def stoppage_audio():
    return {"whistles": [
        {"t": 100, "role": "stoppage", "stoppage": 0},
        {"t": 110, "role": "in_stoppage", "stoppage": 0},
        {"t": 140, "role": "no_stoppage", "stoppage": None},
        {"t": 160, "role": "stoppage", "stoppage": 1}],
        "stoppages": [{"start": 100, "end": 130, "whistles": [100, 110]},
                      {"start": 160, "end": 190, "whistles": [160]}]}


def test_repeated_whistles_share_one_candidate_without_requiring_a_stoppage():
    audio = stoppage_audio()
    candidates = S.build_candidates(audio, [], {}, {}, {"cam1": {"spans": [[0, 200]]}}, 10)
    assert [c["id"] for c in candidates] == ["whistle:0", "whistle:2", "whistle:3"]
    assert audio["whistles"][1]["t"] == 110
    assert candidates[1]["cameras"]["cam1"]["whistle_t"] == 140


@pytest.mark.parametrize("missing", ["role", "index", "stoppage", "member", "leader", "bounds"])
def test_uncorroborated_stoppage_membership_keeps_the_whistle_candidate(missing):
    audio = stoppage_audio()
    if missing == "role":
        audio["whistles"][1].pop("role")
    elif missing == "index":
        audio["whistles"][1]["stoppage"] = 7
    elif missing == "stoppage":
        audio["stoppages"] = []
    elif missing == "member":
        audio["stoppages"][0]["whistles"] = [100]
    elif missing == "leader":
        audio["whistles"][0]["role"] = "no_stoppage"
    else:
        audio["stoppages"][0]["end"] = 109
    candidates = S.build_candidates(audio, [], {}, {}, {"cam1": {"spans": [[0, 200]]}}, 10)
    assert "whistle:1" in {c["id"] for c in candidates}


def test_grouping_repeated_whistles_preserves_an_independent_flow_peak():
    audio = stoppage_audio()
    net = np.zeros(2000)
    net[1100] = 10
    candidates = S.build_candidates(audio,
        [{"primary_cam": "1", "start_s": 109, "end_s": 111}],
        {"cam1": {"net": net, "slot": net}}, {}, {"cam1": {"spans": [[0, 200]]}}, 10)
    assert "whistle:1" not in {c["id"] for c in candidates}
    peak = next(c for c in candidates if c["id"] == "flow:0")
    assert peak["cameras"]["cam1"]["t"] == 110


def inputs():
    sheet = {"teams": {"home": "H", "away": "A"}, "goals": {
        "home": [{"per": "1", "time": "0:50", "time_s": 50, "scorer": "20", "status": "ok"}],
        "away": []}}
    coverage = {"cam1": {"spans": [[0, 200]], "periods": [{"n": 1, "defender": "A"}]},
                "cam2": {"spans": [[0, 200]], "periods": [{"n": 1, "defender": "B"}]}}
    structure = {"coverage": coverage, "periods": [{"n": 1, "start": 0, "end": 60}]}
    layouts = {cam: [{"start": 0, "end": 200, "seek": 20 if cam == "cam2" else 0,
                     "files": [(f"/g/{cam}.MP4", 220)]}] for cam in coverage}
    flow = {cam: {"net": np.ones(2400), "slot": np.ones(2400)} for cam in coverage}
    activity = {cam: np.full(20000, .1) for cam in coverage}
    audio = {"whistles": [{"t": 11.2}], "stoppages": [{"start": 11.2, "end": 36}]}
    rules = {"period_minutes": 1, "clock": "running", "time_direction": "remaining"}
    return sheet, structure, audio, [], flow, activity, layouts, rules


def test_explicit_mapping_selects_opposite_net_and_chapter_seek():
    result = S.select_goals(*inputs(), options={"defender_teams": {"A": "home", "B": "away"}})
    goal = result["goals"][0]
    assert goal["status"] == "selected" and goal["chosen"]["primary_cam"] == "cam2"
    t = goal["chosen"]["detection_s"]
    assert goal["chosen"]["chapters"]["cam2"][1] == pytest.approx(t + 20, abs=.01)
    assert goal["sheet"]["scorer"] == "20"
    assert any("center faceoff" in f for f in result["flags"])


def test_unknown_mapping_tie_is_flagged_not_forced():
    result = S.select_goals(*inputs())
    assert result["mapping"]["defender_teams"] is None
    assert result["goals"][0]["chosen"] is None
    assert any("mapping" in f for f in result["flags"])


def test_gap_does_not_invent_a_clip_or_drop_a_sheet_goal():
    args = list(inputs())
    args[1]["coverage"]["cam2"]["spans"] = [[40, 100]]
    result = S.select_goals(*args, options={"defender_teams": {"A": "home", "B": "away"}})
    assert len(result["goals"]) == 1
    assert result["goals"][0]["status"] == "no clip found"


def test_missing_rules_preserve_goals_with_flags():
    args = list(inputs())
    args[-1] = {}
    result = S.select_goals(*args)
    assert len(result["goals"]) == 1 and result["goals"][0]["chosen"] is None
    assert any("clock" in f for f in result["goals"][0]["flags"])


def test_stop_clock_tail_penalty_respects_uncertain_end():
    args = list(inputs())
    args[-1] = {"period_minutes": 1, "clock": "stop", "time_direction": "remaining"}
    args[1]["periods"][0]["end"] = 85
    args[2]["whistles"][0]["t"] = 30
    args[2]["stoppages"] = [{"start": 30, "end": 60}]
    mapping = {"defender_teams": {"A": "home", "B": "away"}, "minimum_margin": 0}
    certain = S.select_goals(*args, options=mapping)["goals"][0]
    args[1]["game"] = {"end_uncertain": True}
    uncertain = S.select_goals(*args, options=mapping)["goals"][0]
    assert certain["features"]["clock"] < uncertain["features"]["clock"]
    assert any("boundary uncertain" in f for f in uncertain["flags"])


def test_close_alternative_is_flagged():
    args = list(inputs())
    args[2]["whistles"].append({"t": 14.2})
    result = S.select_goals(*args, options={"defender_teams": {"A": "home", "B": "away"}})
    assert result["goals"][0]["chosen"] is None
    assert "goal moment ambiguous" in result["goals"][0]["flags"]


def test_recording_placement_leaves_nan_gaps():
    out = I.place([(0, np.ones(2)), (4, np.full(2, 2))], 1)
    assert np.isnan(out[2:4]).all() and out.tolist()[:2] == [1, 1]


def test_missing_caches_never_extract_signals(tmp_path, monkeypatch):
    import audio_signals as A
    import coverage as C
    import signals
    video = tmp_path / "A.MP4"
    video.write_bytes(b"placeholder")
    (tmp_path / "cam1_concat.txt").write_text(f"file '{video}'\n")
    (tmp_path / "rois.json").write_text(json.dumps({
        f"camera_{i}": {"net": [0, 0, 1, 1], "slot": [0, 0, 1, 1]} for i in (1, 2)}))
    monkeypatch.setattr(C, "_duration", lambda p: 100)
    def forbidden(*a, **kw):
        pytest.fail("Selection must not decode signals")
    monkeypatch.setattr(A, "camera_features", forbidden)
    monkeypatch.setattr(signals, "extract_signals", forbidden)
    flags = []
    flows, activity, layout = I.cached_camera(tmp_path, "cam1", 12, 1280, False, flags)
    assert len(flows["net"]) == len(activity) == 0 and layout
    assert any("flow cache unavailable" in f for f in flags)
    assert any("audio cache unavailable" in f for f in flags)


def test_selection_stage_wiring_and_fingerprints(tmp_path):
    stage = next(s for s in rr.STAGES if s.name == "selection")
    assert set(stage.needs) == {"coverage", "scoresheet", "audio", "detection"}
    assert stage.outputs(None) == ["selection.json"]
    ctx = rr.Context(tmp_path, {"league_rules": {"period_minutes": 13, "clock": "stop", "time_direction": "remaining"},
                               "selection": {"defender_teams": {"A": "home", "B": "away"}}}, rr.Reporter())
    argv = rr.selection_argv(ctx)
    assert argv[1].endswith("selection.py")
    assert json.loads(argv[argv.index("--league") + 1])["time_direction"] == "remaining"
    before = stage.fingerprint(ctx)
    (tmp_path / "game_sheet.json").write_text('{"goals": {}}')
    assert stage.fingerprint(ctx) != before
    before = stage.fingerprint(ctx)
    ctx.options["league_rules"]["time_direction"] = "elapsed"
    assert stage.fingerprint(ctx) != before


def test_selection_stage_reports_flags_without_stopping(tmp_path, monkeypatch):
    ctx = rr.Context(tmp_path, {}, rr.Reporter())
    lines = ["[selection] 0/2 goals selected, 2 no clip found", "[selection] flag: cache unavailable"]
    def run(argv, on_line):
        for line in lines:
            on_line(line)
        return 0, lines
    monkeypatch.setattr(ctx, "run_cmd", run)
    result = rr._run_selection(ctx)
    assert result.state == "flagged" and result.flags == ["cache unavailable"] and not result.fatal


def test_failed_output_write_preserves_previous_selection(tmp_path, monkeypatch):
    output = tmp_path / "selection.json"
    previous = '{"previous": true}\n'
    output.write_text(previous)
    monkeypatch.setattr(S, "analyse", lambda *a: {"goals": [], "flags": []})
    monkeypatch.setattr(S.sys, "argv", ["selection.py", str(tmp_path)])
    write_text = Path.write_text
    def interrupted(path, text, *a, **kw):
        write_text(path, "partial output", *a, **kw)
        raise OSError(12, "Cannot allocate memory")
    monkeypatch.setattr(Path, "write_text", interrupted)
    with pytest.raises(OSError, match="Cannot allocate memory"):
        S.main()
    assert output.read_text() == previous


def test_selection_cli_replaces_output_with_complete_json(tmp_path, monkeypatch):
    result = {"schema_version": 1, "goals": [], "flags": []}
    (tmp_path / "selection.json").write_text('{"previous": true}\n')
    monkeypatch.setattr(S, "analyse", lambda *a: result)
    monkeypatch.setattr(S.sys, "argv", ["selection.py", str(tmp_path)])
    assert S.main() == 0
    assert json.loads((tmp_path / "selection.json").read_text()) == result
