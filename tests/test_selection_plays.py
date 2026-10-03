"""Interest score, goal cutting, non-goal plays and penalties (hhg-3r5.36)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v3", "scripts"))

import selection as S


def goal(gid, side, period, elapsed, team=None, per=None, gtype=None, score=None):
    return {"id": gid, "side": side, "team": team or side.upper(), "period": period,
            "elapsed_s": elapsed, "period_s": 780.0,
            "sheet": {"per": per if per is not None else str(period or ""), "type": gtype},
            "score": score, "flags": []}


def rules():
    return {"period_minutes": 13, "clock": "stop", "time_direction": "remaining", "periods": 3}


class TestGameContext:
    def test_first_tying_lead_change_and_winner(self):
        goals = [goal("h:1", "home", 1, 60, score=.9),
                 goal("a:1", "away", 1, 200, score=.8),   # 1-1 tie
                 goal("a:2", "away", 2, 100, score=.7),   # 1-2 lead away
                 goal("h:2", "home", 3, 100, score=.8),   # 2-2 tie
                 goal("h:3", "home", 3, 300, score=.9),   # 3-2 lead home
                 goal("h:4", "home", 3, 400, score=.6)]   # 4-2 final
        out = S.game_context(goals, rules())
        reasons = {g["id"]: out[i]["reasons"] for i, g in enumerate(goals)}
        assert "first goal" in reasons["h:1"]
        assert "tying goal" in reasons["a:1"]
        assert "lead change" in reasons["a:2"]
        assert "tying goal" in reasons["h:2"]
        assert "lead change" in reasons["h:3"]
        assert "game-winner" in reasons["h:3"]
        assert "game-winner" not in reasons["h:4"]

    def test_tied_game_has_no_winner(self):
        goals = [goal("h:1", "home", 1, 60), goal("a:1", "away", 2, 60)]
        out = S.game_context(goals, rules())
        assert not any("game-winner" in c["reasons"] for c in out)

    def test_overtime_and_shootout_reasons(self):
        goals = [goal("h:1", "home", 1, 60), goal("a:1", "away", 3, 60),
                 goal("h:2", "home", None, None, per="OT", gtype="EV", score=.8)]
        out = S.game_context(goals, rules())
        assert "overtime" in out[2]["reasons"]
        so = [goal("h:1", "home", 1, 60), goal("a:1", "away", 3, 60),
              goal("h:2", "home", None, None, per="SO", gtype="SO", score=.8)]
        assert "shootout" in S.game_context(so, rules())[2]["reasons"]

    def test_last_two_minutes_and_focus_team(self):
        goals = [goal("h:1", "home", 3, 770, score=.7),        # 10 s left in period 3
                 goal("a:1", "away", 1, 100, team="Ice Pak", score=.7)]
        out = S.game_context(goals, rules(), focus_team="Ice Pak")
        assert "last two minutes" in out[0]["reasons"]
        assert "focus team" in out[1]["reasons"]

    def test_interest_blends_context_and_match_score(self):
        # 1-0 away, then 1-1, 2-1, 3-1. h:2 is the game-winner (home's 2nd);
        # h:3 is plain but has the highest match score.
        goals = [goal("a:1", "away", 1, 60, score=.5),
                 goal("h:1", "home", 1, 200, score=.5),
                 goal("h:2", "home", 2, 60, score=.5),
                 goal("h:3", "home", 3, 400, score=.9)]
        out = S.game_context(goals, rules())
        i = {g["id"]: out[k]["interest"] for k, g in enumerate(goals)}
        assert i["h:2"] > i["a:1"] > i["h:3"]   # context outweighs the match score
        assert 0 <= min(i.values()) <= max(i.values()) <= 1

    def test_protected_covers_reasons_and_the_three_best(self):
        # Home wins 3-2. h:3 is the game-winner (home's (2+1)th goal); a:2 has
        # no protected reason but belongs to the three best by interest.
        goals = [goal("h:1", "home", 1, 60, score=.2),   # first goal
                 goal("a:1", "away", 1, 200, score=.5),  # tying goal
                 goal("h:2", "home", 2, 100, score=.5),
                 goal("h:3", "home", 2, 300, score=1.0),  # 3-1 = game-winner
                 goal("a:2", "away", 3, 100, score=1.0)]
        out = S.game_context(goals, rules())
        protected = {g["id"] for g, c in zip(goals, out) if c["protected"]}
        # h:2 takes the lead back after a tie (go-ahead goal), so it is protected
        # and takes a best slot; a:2 (plain, interest .58) does not.
        assert protected == {"h:1", "a:1", "h:2", "h:3"}
        assert "lead change" in out[2]["reasons"]
        assert "a:2" not in protected
        # the three best by interest are marked for the full treatment (§5.8)
        best = {g["id"] for g, c in zip(goals, out) if c.get("best_goal")}
        # h:2 is a best goal now: the go-ahead rule keeps its interest (.68)
        # above plain a:2 (.58).
        assert best == {"h:2", "a:1", "h:3"}

    def test_a_goal_without_a_clip_takes_no_best_slot(self):
        goals = [goal("h:1", "home", 1, 60, score=.9),
                 goal("a:1", "away", 1, 200, score=.8),
                 goal("h:2", "home", 2, 100, score=.7),
                 goal("h:3", "home", 3, 100, score=.9)]
        goals[0]["chosen"] = None  # no clip found: never a best slot
        out = S.game_context(goals, rules())
        best = {g["id"] for g, c in zip(goals, out) if c.get("best_goal")}
        assert "h:1" not in best
        assert len(best) == 3


class TestPlanCuts:
    def big_game(self, n=12, low_from=3):
        goals, i = [], 0
        # home:1 is protected (first goal); eleven home goals, one away goal.
        while i < n - 1:
            goals.append(goal(f"h:{i + 1}", "home", 1, 60 + 60 * i, score=.5 - (i >= low_from) * .2))
            i += 1
        goals.append(goal(f"a:{n}", "away", 3, 700, score=.9))
        return goals

    def test_cuts_only_low_interest_goals_in_high_scoring_games(self):
        goals = self.big_game()
        out = S.game_context(goals, rules())
        cut = S.plan_cuts(goals, out)
        assert cut == ["h:3", "h:4", "h:5", "h:6", "h:7", "h:8", "h:9", "h:10", "h:11"]
        assert all(c["cut"] for g, c in zip(goals, out) if g["id"] in cut)
        assert not any(c["cut"] for g, c in zip(goals, out) if g["id"] not in cut)

    def test_never_cuts_protected_goals(self):
        goals = self.big_game()
        out = S.game_context(goals, rules())
        cut = set(S.plan_cuts(goals, out))
        assert "h:1" not in cut and "a:12" not in cut  # first goal and the three best

    def test_no_cuts_in_a_low_scoring_game(self):
        goals = self.big_game(n=6)
        out = S.game_context(goals, rules())
        assert S.plan_cuts(goals, out) == []


def fixture_plays():
    """Two cameras, three 13-minute stop-clock periods; sheet goal at 5:00 remaining (elapsed 480)."""
    coverage = {"cam1": {"spans": [[0, 2340]], "periods": [{"n": 1, "defender": "A"}, {"n": 2, "defender": "A"}, {"n": 3, "defender": "A"}]},
                "cam2": {"spans": [[0, 2340]], "periods": [{"n": 1, "defender": "B"}, {"n": 2, "defender": "B"}, {"n": 3, "defender": "B"}]}}
    structure = {"coverage": coverage,
                 "periods": [{"n": 1, "start": 0, "end": 780}, {"n": 2, "start": 780, "end": 1560}, {"n": 3, "start": 1560, "end": 2340}]}
    layouts = {cam: [{"start": 0, "end": 1000, "seek": 20 if cam == "cam2" else 0,
                      "files": [(f"/g/{cam}.MP4", 1020)]}] for cam in coverage}
    flow = {cam: {"net": np.ones(24000), "slot": np.ones(24000)} for cam in coverage}
    activity = {cam: np.full(200000, .1) for cam in coverage}

    def audio(whistles):
        return {"whistles": [{"t": t} for t in whistles], "stoppages": []}

    sheet = {"teams": {"home": "H", "away": "A"},
             "goals": {"home": [{"per": "1", "time": "5:00", "time_s": 300, "status": "ok"}], "away": []},
             "penalties": {"home": [], "away": []}}
    rules_ = {"period_minutes": 13, "clock": "stop", "time_direction": "remaining", "periods": 3}
    return sheet, structure, audio, flow, activity, layouts, rules_


class TestPlays:
    def run(self, whistles=(120.0, 400.0), extra_events=(), penalties=None, options=None):
        sheet, structure, audio_fn, flow, activity, layouts, rules_ = fixture_plays()
        audio = audio_fn(whistles)
        if penalties is not None:
            sheet["penalties"] = penalties
        events = [dict(zip(("start_s", "end_s", "score", "primary_cam", "confidence"), e))
                  for e in extra_events]
        opts = {"defender_teams": {"A": "home", "B": "away"}}
        if options:
            opts.update(options)
        return S.select_goals(sheet, structure, audio, events, flow, activity, layouts, rules_,
                              options=opts)

    def penalty(self, off="3:00", infraction="TRIPPING", status="ok"):
        return {"per": "1", "player": "9", "infraction": infraction,
                "minutes": "2", "off": off, "on": None, "status": status}

    def test_unmatched_net_candidates_become_plays_and_next_best(self):
        # The sheet goal (5:00 remaining -> expected 480) stays unmatched: both
        # whistles sit far from 480 under the stop prior, so all four seeds are free.
        result = self.run(extra_events=[(200, 230, 2.0, 2, .5), (600, 640, 1.5, 1, .5)])
        plays = result["plays"]
        assert len(plays) == 4
        assert {p["kind"] for p in plays} == {"net_play"}
        assert all(p["chosen"] is False for p in plays)
        detections = [p["detection_s"] for p in plays]
        assert detections == sorted(detections)
        assert any(199 <= t <= 201 for t in detections)       # flow seed at 200
        assert any(599 <= t <= 601 for t in detections)       # flow seed at 600
        best = result["next_best"]
        assert best and {n["id"] for n in best} <= {p["id"] for p in plays}
        scores = [n["score"] for n in best]
        assert scores == sorted(scores, reverse=True)

    def test_plays_do_not_overlap_chosen_goal_windows(self):
        # Whistle at 482 puts the goal moment at ~480.8 inside the clock window,
        # so the goal is chosen; the flow event right at it must not become a play.
        result = self.run(whistles=(120.0, 482.0),
                          extra_events=[(476, 482, 2.0, 2, .5), (600, 640, 1.5, 1, .5)])
        chosen = result["goals"][0]["chosen"]
        assert chosen is not None and chosen["detection_s"] > 460
        plays = result["plays"]
        assert len(plays) == 2
        for p in plays:
            assert not chosen["start_s"] - 12 <= p["detection_s"] <= chosen["end_s"] + 12

    def test_penalty_row_matches_a_later_whistle(self):
        result = self.run(whistles=(600.0,), penalties={"home": [], "away": [self.penalty()]})
        pens = [p for p in result["plays"] if p["kind"] == "penalty"]
        assert len(pens) == 1
        assert pens[0]["chosen"] is True
        assert "TRIPPING" in pens[0]["label"]
        assert pens[0]["whistle_t"] == pytest.approx(600.0)
        assert pens[0]["end_s"] > pens[0]["whistle_t"]        # clip ends just after the whistle
        # the matched candidate was claimed, so no duplicate net play exists
        assert len(result["plays"]) == 1

    def test_penalty_without_confident_match_is_flagged(self):
        result = self.run(whistles=(100.0,), penalties={"home": [], "away": [self.penalty()]})
        assert not [p for p in result["plays"] if p["kind"] == "penalty"]
        assert any("no confident time match" in f for f in result["flags"])

    def test_penalty_over_a_selected_goal_clip_is_flagged(self):
        # Goal moment ~479 (whistle 482 chosen by the goal); a second whistle at
        # 470 (t~467) matches the penalty clock (5:00 remaining -> expected 480)
        # but sits inside the chosen goal clip +-gap. minimum_margin 0 keeps the
        # goal chosen despite the near alternative, isolating the overlap path.
        result = self.run(whistles=(482.0, 470.0),
                          penalties={"home": [], "away": [self.penalty(off="5:00")]},
                          options={"minimum_margin": 0})
        assert not [p for p in result["plays"] if p["kind"] == "penalty"]
        assert any("overlap a selected goal clip" in f for f in result["flags"])

    def test_penalty_row_with_review_status_is_skipped(self):
        result = self.run(whistles=(600.0,),
                          penalties={"home": [], "away": [self.penalty(status="review")]})
        assert not [p for p in result["plays"] if p["kind"] == "penalty"]
        assert any("row needs review; skipped" in f for f in result["flags"])

    def test_penalty_with_unreadable_clock_is_flagged(self):
        result = self.run(whistles=(600.0,),
                          penalties={"home": [], "away": [self.penalty(off="bogus")]})
        assert not [p for p in result["plays"] if p["kind"] == "penalty"]
        assert any("no confident time match" in f for f in result["flags"])

    def test_ot_penalty_matches_the_period_after_regulation(self):
        coverage = {"cam1": {"spans": [[0, 3120]],
                             "periods": [{"n": n, "defender": "A"} for n in (1, 2, 3, 4)]},
                    "cam2": {"spans": [[0, 3120]],
                             "periods": [{"n": n, "defender": "B"} for n in (1, 2, 3, 4)]}}
        structure = {"coverage": coverage,
                     "periods": [{"n": 1, "start": 0, "end": 780},
                                 {"n": 2, "start": 780, "end": 1560},
                                 {"n": 3, "start": 1560, "end": 2340},
                                 {"n": 4, "start": 2340, "end": 3120}]}
        layouts = {cam: [{"start": 0, "end": 1000, "seek": 20 if cam == "cam2" else 0,
                          "files": [(f"/g/{cam}.MP4", 1020)]}] for cam in coverage}
        flow = {cam: {"net": np.ones(40000), "slot": np.ones(40000)} for cam in coverage}
        activity = {cam: np.full(400000, .1) for cam in coverage}
        # 8:00 remaining in OT = 5:00 played = elapsed 300 -> expected 2340 + 300.
        sheet = {"teams": {"home": "H", "away": "A"},
                 "goals": {"home": [{"per": "1", "time": "5:00", "time_s": 300, "status": "ok"}],
                           "away": []},
                 "penalties": {"home": [], "away": [
                     {"per": "OT", "player": "9", "infraction": "HOOKING", "minutes": "2",
                      "off": "8:00", "on": None, "status": "ok"}]}}
        rules_ = {"period_minutes": 13, "clock": "stop", "time_direction": "remaining", "periods": 3}
        result = S.select_goals(sheet, structure, {"whistles": [{"t": 2640.0}], "stoppages": []},
                                [], flow, activity, layouts, rules_,
                                options={"defender_teams": {"A": "home", "B": "away"}})
        pens = [p for p in result["plays"] if p["kind"] == "penalty"]
        assert len(pens) == 1 and "HOOKING" in pens[0]["label"]
        assert not any("unreadable period" in f for f in result["flags"])

    def test_two_nearby_candidates_make_one_play(self):
        # Candidates 5 s apart produce overlapping windows; the guard keeps the
        # first in game order and silently skips the second.
        result = self.run(whistles=(120.0, 400.0),
                          extra_events=[(200, 230, 2.0, 2, .5), (205, 235, 2.0, 2, .5)])
        near = [p for p in result["plays"] if 195 <= p["detection_s"] <= 210]
        assert len(near) == 1

    def test_cut_goal_is_marked_in_output(self):
        sheet = {"teams": {"home": "H", "away": "A"},
                 "goals": {"home": [{"per": "1", "time": "5:00", "time_s": 220, "status": "ok"}],
                           "away": [{"per": "1", "time": "9:00", "time_s": 120 + 60 * i, "status": "ok"}
                                    for i in range(11)]},
                 "penalties": {"home": [], "away": []}}
        structure = {"coverage": {"cam1": {"spans": [[0, 1000]], "periods": [{"n": 1, "defender": "A"}]},
                                  "cam2": {"spans": [[0, 1000]], "periods": [{"n": 1, "defender": "B"}]}},
                     "periods": [{"n": 1, "start": 0, "end": 780}]}
        layouts = {cam: [{"start": 0, "end": 1000, "seek": 0, "files": [(f"/g/{cam}.MP4", 1000)]}]
                   for cam in ("cam1", "cam2")}
        flow = {cam: {"net": np.ones(24000), "slot": np.ones(24000)} for cam in layouts}
        activity = {cam: np.full(200000, .1) for cam in layouts}
        audio = {"whistles": [{"t": 60 + 60 * i} for i in range(12)], "stoppages": []}
        rules_ = {"period_minutes": 13, "clock": "stop", "time_direction": "remaining", "periods": 3}
        result = S.select_goals(sheet, structure, audio, [], flow, activity, layouts, rules_)
        cut = [g for g in result["goals"] if g.get("cut")]
        assert cut, "a 12-goal game must cut low-interest goals"
        assert all("interest" in g for g in result["goals"])
        assert result["schema_version"] == 2
        assert any("cut" in f for f in result["flags"])
