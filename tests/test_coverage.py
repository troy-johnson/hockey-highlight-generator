"""Tests for v3/scripts/coverage.py pure logic and the coverage stage wiring."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v3", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v2", "scripts"))

import coverage as C  # noqa: E402

PLAY, QUIET = 0.8, 0.05


def synthetic_game(start=300, period=1100, brk=70, n=3, tail=200, pre_quiet=True, weak=None):
    """
    Activity per second and per-camera flow for a game: periods of play,
    quiet breaks, and a quiet tail after the game. `weak` = list of
    (start, length) in-play lulls (audio quiet, flow high).
    """
    total = start + n * period + (n - 1) * brk + tail
    act = np.full(total, PLAY, np.float32)
    flow = np.full(total, 1.0, np.float32)
    if pre_quiet:
        act[:start] = QUIET
        flow[:start] = 0.3
    breaks, starts = [], [start]
    t = start
    for k in range(n):
        t += period
        if k < n - 1:
            act[t:t + brk] = QUIET
            flow[t:t + brk] = 0.2
            breaks.append((t, t + brk))
            t += brk
            starts.append(t)
    act[t:] = QUIET
    flow[t:] = 0.1
    for s, L in weak or []:
        act[s:s + L] = QUIET
    whistles = [{"t": float(x)} for x in range(start - 1, t, 60)]
    flows = {"cam1": flow.copy(), "cam2": flow.copy()}
    return act, flows, whistles, breaks, starts, t


def test_quiet_runs_merge_and_min_length():
    s = np.array([1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0], np.float32)
    assert C.quiet_runs(s, 0.5, 2, 1) == [(1, 7), (10, 12)]
    assert C.quiet_runs(s, 0.5, 3, 0) == [(4, 7)]


def test_quiet_runs_nan_is_not_quiet():
    s = np.array([0, 0, np.nan, 0, 0], np.float32)
    assert C.quiet_runs(s, 0.5, 1, 0) == [(0, 2), (3, 5)]


def test_max_over_keeps_nan_only_where_no_camera():
    a = np.array([0.1, np.nan, np.nan], np.float32)
    b = np.array([0.3, 0.2], np.float32)
    out = C.max_over([a, b])
    assert out[0] == pytest.approx(0.3) and out[1] == pytest.approx(0.2) and np.isnan(out[2])


def test_per_second_means():
    x = np.concatenate([np.full(100, 1.0), np.full(100, 3.0), np.full(100, np.nan)])
    out = C.per_second(x, 100)
    assert list(out[:2]) == [1.0, 3.0] and np.isnan(out[2])


def test_candidate_score_prefers_long_quiet_and_low_flow():
    assert C.candidate_score(60, 0.2) == 1.0
    assert C.candidate_score(30, 0.2) < C.candidate_score(60, 0.2)
    assert C.candidate_score(60, 1.0) == 0.5
    assert C.candidate_score(60, None) == 0.75


def test_break_candidates_score_breaks_over_in_play_lulls():
    act, flows, _, breaks, _, end = synthetic_game(weak=[(800, 35)])
    cands = C.break_candidates(act, flows, 400, end)
    by_start = {c["start"]: c for c in cands}
    assert by_start[800.0]["score"] < 0.5
    for s, e in breaks:
        assert by_start[float(s)]["score"] >= 0.9


def test_first_game_whistle_skips_lone_warmup_whistles():
    ws = [{"t": 50.0}] + [{"t": float(t)} for t in (400, 450, 500, 560)]
    assert C.first_game_whistle(ws) == 400.0


def test_game_start_is_first_faceoff_after_whistle():
    act = np.full(600, QUIET, np.float32)
    act[405:] = PLAY
    ws = [{"t": float(t)} for t in (400, 450, 500, 560)]
    g = C.game_start(ws, act)
    assert g["t"] == 405.0 and g["whistle"] == 400.0


def test_game_start_falls_back_to_whistle():
    act = np.full(600, QUIET, np.float32)
    ws = [{"t": float(t)} for t in (400, 450, 500, 560)]
    g = C.game_start(ws, act)
    assert g["t"] == 400.0 and "no faceoff" in g["rule"]


def test_game_end_first_long_quiet_and_later_candidates():
    act = np.full(1000, PLAY, np.float32)
    act[500:520] = QUIET     # short lull: not the end
    act[700:780] = QUIET     # end of game
    act[850:950] = QUIET     # post-game
    end = C.game_end(act, 400, 1000, [{"t": 690.0}, {"t": 860.0}])
    assert end["t"] == 700.0 and not end["uncertain"]
    ts = [c["t"] for c in end["candidates"]]
    assert 500.0 in ts and 690.0 in ts and 850.0 in ts and 1000.0 in ts
    assert 860.0 not in ts  # a post-game whistle is not an end candidate


def test_game_end_quiet_to_end_of_recording_is_accepted():
    act = np.full(1000, PLAY, np.float32)
    act[950:] = QUIET
    end = C.game_end(act, 400, 1000)
    assert end["t"] == 950.0 and "end of the Recording" in end["rule"] and not end["uncertain"]


def test_game_end_without_quiet_is_uncertain_at_signal_end():
    act = np.full(1000, PLAY, np.float32)
    end = C.game_end(act, 400, 1000)
    assert end["t"] == 1000.0 and end["uncertain"]


def test_league_timing_defaults_and_rules():
    assert C.league_timing(None)["periods"] == 3
    assert C.league_timing(None)["source"].startswith("default")
    t = C.league_timing({"periods": 2, "period_minutes": 20, "clock": "running", "break_minutes": 2})
    assert t == {"periods": 2, "period_s": 1200.0, "clock": "running", "break_s": 120.0, "source": "League"}


def test_plan_periods_finds_both_breaks_and_period_starts():
    act, flows, ws, breaks, starts, end = synthetic_game(weak=[(800, 35), (2900, 30)])
    cands = C.break_candidates(act, flows, starts[0] + 120, len(act))
    plan = C.plan_periods(cands, float(starts[0]), act, float(len(act)), ws, C.league_timing(None))
    got = [p["start"] for p in plan["periods"]]
    assert got == [float(s) for s in starts]
    assert [p["break_before"]["start"] for p in plan["periods"][1:]] == [float(s) for s, _ in breaks]
    assert all(not p["break_before"]["uncertain"] for p in plan["periods"][1:])
    assert plan["end"]["t"] == float(end)
    assert not plan["flags"]


def test_plan_candidates_exclude_the_other_chosen_breaks():
    act, flows, ws, breaks, starts, _ = synthetic_game(weak=[(800, 35)])
    cands = C.break_candidates(act, flows, starts[0] + 120, len(act))
    plan = C.plan_periods(cands, float(starts[0]), act, float(len(act)), ws, C.league_timing(None))
    chosen = {float(s) for s, _ in breaks}
    for p in plan["periods"][1:]:
        assert all(c["start"] not in chosen for c in p["break_before"]["candidates"])


def test_uncertain_break_is_flagged_with_candidates():
    # Break 2 is a short, weak quiet; an in-play lull nearby is a close rival.
    act, flows, ws, breaks, starts, _ = synthetic_game(brk=70)
    s2, e2 = breaks[1]
    act[s2 + 30:e2] = PLAY           # break 2 only 30 s quiet
    flows["cam1"][s2:e2] = 0.7
    flows["cam2"][s2:e2] = 0.7
    act[s2 - 200:s2 - 170] = QUIET   # rival lull, 30 s
    cands = C.break_candidates(act, flows, starts[0] + 120, len(act))
    plan = C.plan_periods(cands, float(starts[0]), act, float(len(act)), ws, C.league_timing(None))
    brk = plan["periods"][2]["break_before"]
    assert brk["uncertain"]
    assert brk["reasons"]
    assert brk["candidates"], "an uncertain break lists other candidates"
    assert all({"start", "period_start", "plan_score"} <= set(c) for c in brk["candidates"])


def test_plan_falls_back_to_fewer_breaks_with_flag():
    act, flows, ws, _, starts, _ = synthetic_game(n=2)
    cands = C.break_candidates(act, flows, starts[0] + 120, len(act))
    plan = C.plan_periods(cands, float(starts[0]), act, float(len(act)), ws, C.league_timing(None))
    assert len(plan["periods"]) == 2
    assert any("found 1 of 2 breaks" in f for f in plan["flags"])


def test_league_periods_change_the_number_of_breaks():
    act, flows, ws, breaks, starts, _ = synthetic_game(n=2)
    cands = C.break_candidates(act, flows, starts[0] + 120, len(act))
    plan = C.plan_periods(cands, float(starts[0]), act, float(len(act)), ws, C.league_timing({"periods": 2}))
    assert [p["start"] for p in plan["periods"]] == [float(s) for s in starts]
    assert not plan["flags"]


def test_stop_clock_rejects_periods_shorter_than_the_period_length():
    act, flows, ws, breaks, starts, _ = synthetic_game(weak=[(800, 60)])
    flows["cam1"][800:860] = 0.2
    flows["cam2"][800:860] = 0.2
    cands = C.break_candidates(act, flows, starts[0] + 120, len(act))
    timing = C.league_timing({"periods": 3, "period_minutes": 15, "clock": "stop"})
    plan = C.plan_periods(cands, float(starts[0]), act, float(len(act)), ws, timing)
    assert [p["start"] for p in plan["periods"]] == [float(s) for s in starts]


def test_manifest_layout_and_chapter_time_with_seek():
    lines = ["ffconcat version 1.0", "# seek 20.0", "file '/g/A.MP4'", "file '/g/B.MP4'"]
    lay = C.manifest_layout(lines, {"/g/A.MP4": 100.0, "/g/B.MP4": 50.0})
    assert lay[0]["start"] == 0.0 and lay[0]["end"] == 130.0
    assert C.chapter_at(lay, 10.0) == ("A.MP4", 30.0)
    assert C.chapter_at(lay, 90.0) == ("B.MP4", 10.0)
    assert C.chapter_at(lay, 131.0) is None


def test_manifest_layout_recordings_with_gap():
    lines = ["ffconcat version 1.0", "# recording 0.0", "file '/g/A.MP4'", "# recording 200.0", "file '/g/C.MP4'"]
    lay = C.manifest_layout(lines, {"/g/A.MP4": 100.0, "/g/C.MP4": 60.0})
    assert [(b["start"], b["end"]) for b in lay] == [(0.0, 100.0), (200.0, 260.0)]
    assert C.chapter_at(lay, 150.0) is None
    assert C.chapter_at(lay, 210.0) == ("C.MP4", 10.0)


def test_coverage_gaps_flags_and_defenders():
    periods = [{"n": 1, "start": 0.0, "end": 100.0}, {"n": 2, "start": 110.0, "end": 200.0},
               {"n": 3, "start": 210.0, "end": 300.0}, {"n": 4, "start": 310.0, "end": 350.0}]
    spans = {"cam1": [(0.0, 400.0)], "cam2": [(0.0, 250.0)]}
    cov, flags = C.coverage(spans, periods, 3)
    assert [r["defender"] for r in cov["cam1"]["periods"]] == ["A", "B", "A", None]
    assert [r["defender"] for r in cov["cam2"]["periods"]] == ["B", "A", "B", None]
    p3 = cov["cam2"]["periods"][2]
    assert p3["covered_s"] == 40.0 and p3["gaps"] == [[250.0, 300.0]]
    assert any("cam2" in f and "period 3" in f for f in flags)
    assert cov["cam2"]["periods"][3]["fraction"] == 0.0


def test_coverage_short_gap_is_not_flagged():
    periods = [{"n": 1, "start": 0.0, "end": 100.0}]
    _, flags = C.coverage({"cam1": [(0.0, 80.0)]}, periods, 3)
    assert flags == []


def test_swap_verdict():
    red, green = (155.0, 140.0), (127.0, 132.0)
    assert C.swap_verdict([red] * 5, [green] * 5)["result"] == "swapped"
    assert C.swap_verdict([red] * 5, [red] * 5)["result"] == "same"
    assert C.swap_verdict([red] * 3, [red] * 3)["result"] == "unclear"  # too few for "same"
    assert C.swap_verdict([red] * 2, [green] * 5)["result"] == "unclear"


def test_break_swap_flag_only_on_confident_same():
    assert C.break_swap_flag({"cam1": {"result": "same"}, "cam2": {"result": "unclear"}})
    assert C.break_swap_flag({"cam1": {"result": "same"}, "cam2": {"result": "swapped"}}) is None
    assert C.break_swap_flag({"cam1": {"result": "unclear"}}) is None


def test_jersey_color_ignores_white_and_grey():
    crop = np.full((100, 60, 3), 230, np.uint8)  # white pads and ice
    crop[20:60, 10:50] = (40, 40, 200)            # red jersey (BGR)
    a, b = C.jersey_color(crop)
    assert a > 150
    assert C.jersey_color(np.full((100, 60, 3), 200, np.uint8)) is None


def test_net_figure_prefers_goalie_in_crease():
    goal = ("goal", 0.9, [500.0, 300.0, 860.0, 540.0])
    far = ("goalie", 0.9, [100.0, 100.0, 150.0, 200.0])
    skater = ("player", 0.8, [600.0, 250.0, 680.0, 500.0])
    goalie = ("goalie", 0.4, [650.0, 260.0, 740.0, 520.0])
    assert C.net_figure([goal, far, skater, goalie]) == [650, 260, 740, 520]
    assert C.net_figure([goal, far, skater]) == [600, 250, 680, 500]
    assert C.net_figure([far, skater]) is None


def test_mmss():
    assert C.mmss(375) == "6:15" and C.mmss(None) == "?"
