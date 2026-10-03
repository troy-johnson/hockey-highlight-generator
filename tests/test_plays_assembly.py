"""Non-goal plays, cut goals and the ~3-minute fill in assembly (hhg-3r5.36)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v3", "scripts"))

import pytest

import recap_assembly as A

FPS = A.FPS
SPEED = 1.1
FILL_FRAMES = A.FILL_TARGET_S * FPS


def goal(gid, i, moment=None):
    moment = moment if moment is not None else 30.0 + i * 40.0
    return {
        "id": gid,
        "chosen": {
            "candidate_id": f"c-{gid}",
            "detection_s": moment,
            "primary_cam": "cam2",
            "start_s": moment - 10.0,
            "end_s": moment + 10.0,
        },
    }


def play(pid, moment, score=0.5, chosen=False):
    return {
        "id": pid,
        "kind": "net_play",
        "label": "Net play",
        "detection_s": moment,
        "primary_cam": "cam2",
        "score": score,
        "candidate_id": pid,
        "start_s": moment - 6.0,
        "end_s": moment + 4.0,
        "whistle_t": None,
        "chosen": chosen,
        "flags": [],
    }


def selection(goals, plays=(), next_best=()):
    return {
        "schema_version": 2,
        "goals": goals,
        "plays": list(plays),
        "next_best": list(next_best),
        "flags": [],
    }


def layouts():
    return {
        "cam2": [{"start": 0.0, "end": 100000.0, "seek": 20.0, "files": [("a.mp4", 100020.0)]}],
    }


def sequence(plan):
    out = []
    for clip in plan["clips"]:
        out.append(clip)
        out.extend(r for r in plan["replays"] if r["goal_id"] == clip["goal_id"])
    return out


def total_frames(plan):
    return sum(e["frames"] for e in sequence(plan))


class TestCutGoals:
    def test_cut_goals_are_omitted_and_flags_explain_it(self):
        goals = [goal(f"home:{i}", i) for i in range(1, 13)]
        goals[3]["cut"] = True
        result = A.plan_recap(selection(goals), layouts())
        goal_ids = [c["goal_id"] for c in result["clips"] if c["kind"] == "goal"]
        assert "home:4" not in goal_ids
        assert len(goal_ids) == 11
        # 11 surviving goals still sit in tier 2: one tight replay each, 5 s output.
        tight = [r for r in result["replays"] if r["kind"] == "tight"]
        assert len(tight) == 11
        assert any("home:4: cut by interest policy; omitted from Recap" in f
                   for f in result["flags"])

    def test_lower_tier_when_cuts_take_the_count_across_a_boundary(self):
        goals = [goal(f"home:{i}", i) for i in range(1, 13)]
        for g in goals[2:9]:
            g["cut"] = True
        result = A.plan_recap(selection(goals), layouts())
        kinds = [r["kind"] for r in result["replays"]]
        # 5 surviving goals -> tier 1: one wide and one tight shot each.
        assert kinds.count("wide") == 5 and kinds.count("tight") == 5


class TestChosenPlays:
    def play_selection(self):
        plays = [play("p-early", 25.0, score=0.9, chosen=True),
                 play("p-late", 75.0, score=0.8, chosen=True)]
        return selection([goal("home:1", 0)], plays, [])

    def test_chosen_plays_render_as_clips_at_their_moment(self):
        result = A.plan_recap(self.play_selection(), layouts())
        play_clips = [c for c in result["clips"] if c["kind"] == "play"]
        assert [c["goal_id"] for c in play_clips] == ["p-early", "p-late"]
        # chronological order: the 25 s play lands before the 30 s goal
        assert result["clips"][0]["goal_id"] == "p-early"
        # a 10 s play at 110 % speed
        assert play_clips[0]["frames"] == pytest.approx(10 / SPEED * FPS, abs=1)
        assert play_clips[0]["play_kind"] == "net_play"
        # goals get the animated push-in; plays get the fixed punch-in crop
        assert play_clips[0]["zoom"] == {"type": "static", "z": A.ZOOM_MIN}

    def test_the_best_play_gets_one_tight_replay(self):
        result = A.plan_recap(self.play_selection(), layouts())
        play_replays = [r for r in result["replays"] if r["goal_id"] == "p-early"]
        goal_replays = [r for r in result["replays"] if r["goal_id"] == "home:1"]
        assert len(play_replays) == 1
        assert play_replays[0]["kind"] == "tight"
        assert play_replays[0]["speed"] == A.SLOW_MO
        assert not any(r["goal_id"] == "p-late" for r in result["replays"])
        # goal replays remain per-tier (single goal -> tier 1: wide + tight)
        assert {r["kind"] for r in goal_replays} == {"wide", "tight"}
        assert any("best-play replay on p-early" in f for f in result["flags"])

    def test_play_window_validation_and_coverage_flags(self):
        bad_window = play("p-bad", 25.0)
        bad_window["start_s"] = 30.0
        bad_window["end_s"] = 40.0
        outside = play("p-outside", 250000.0)
        next_best = [{"id": "p-bad"}, {"id": "p-outside"}]
        result = A.plan_recap(selection([goal("home:1", 0)], [bad_window, outside], next_best),
                              layouts())
        flags = "\n".join(result["flags"])
        assert "p-bad: invalid play window; omitted" in flags
        assert "p-outside: play outside Recording coverage; omitted" in flags


class TestFill:
    def filler_selection(self, n):
        goals = [goal("home:1", 0)]
        plays = [play(f"p-{i}", 60.0 + i * 30.0, score=0.5 - i * 0.001, chosen=False)
                 for i in range(n)]
        next_best = [{"id": p["id"], "kind": "net_play", "detection_s": p["detection_s"],
                      "camera": "cam2", "score": p["score"], "why": "net play"} for p in plays]
        return selection(goals, plays, next_best)

    def test_fill_adds_unchosen_plays_toward_three_minutes(self):
        result = A.plan_recap(self.filler_selection(40), layouts())
        filler = [c for c in result["clips"] if c["kind"] == "play"]
        assert filler, "the fill must convert next_best plays into clips"
        total = total_frames(result)
        assert total >= FILL_FRAMES  # the fill reaches the soft target
        assert total <= A.MAX_FRAMES
        # stops within one play of the target: the last added filler crossed it
        assert total - filler[-1]["frames"] < FILL_FRAMES

    def test_fill_skips_plays_already_chosen_and_missing_ids(self):
        plays = [play("p-chosen", 60.0, chosen=True), play("p-filler", 90.0)]
        sel = selection([goal("home:1", 0)], plays,
                        [{"id": "p-chosen"}, {"id": "p-missing"}, {"id": "p-filler"}])
        result = A.plan_recap(sel, layouts())
        play_clips = [c for c in result["clips"] if c["kind"] == "play"]
        assert [c["goal_id"] for c in play_clips] == ["p-chosen", "p-filler"]

    def test_fill_leaves_room_when_the_cap_is_close(self):
        # 12 goals in tier 2 leave little room: the filler adds only what fits.
        goals = [goal(f"home:{i}", i) for i in range(1, 13)]
        plays = [play(f"p-{i}", 70.0 + i * 30.0) for i in range(30)]
        next_best = [{"id": p["id"]} for p in plays]
        result = A.plan_recap(selection(goals, plays, next_best), layouts())
        assert total_frames(result) <= A.MAX_FRAMES


class TestCapOrder:
    def twelve_goals(self):
        return [goal(f"home:{i}", i) for i in range(1, 13)]

    def test_cap_trims_replays_then_goal_context(self):
        # 39 goals land in tier 3 (7 s windows = 191 frames each = 7449 frames,
        # just over the cap): the replay shots drop wholesale, then goal context
        # is trimmed. No play is present to be touched.
        goals = [goal(f"home:{i}", i) for i in range(1, 40)]
        result = A.plan_recap(selection(goals, [], []), layouts())
        flags = "\n".join(result["flags"])
        assert "goal context shortened to fit the four-minute cap" in flags
        assert "dropped" in flags
        assert total_frames(result) <= A.MAX_FRAMES
        # every surviving goal keeps at least the MIN_BUILD_UP_S floor
        for clip in result["clips"]:
            assert clip["frames"] >= A.MIN_BUILD_UP_S / 1.1 * A.FPS - 1

    def test_fill_never_exceeds_the_cap_even_with_a_chosen_play(self):
        # 12 goals (273 each) + one chosen play + 12 replays + the best-play
        # replay = 5469 frames, already past the 180 s fill target, so the fill
        # must not add anything and no cap stage may fire.
        chosen = [play("p-keep", 40.0, score=0.99, chosen=True)]
        filler = [play(f"p-f{i}", 46.0 + i * 20.0, score=0.05) for i in range(9)]
        next_best = [{"id": p["id"]} for p in filler]
        result = A.plan_recap(selection(self.twelve_goals(), chosen, next_best), layouts())
        assert total_frames(result) <= A.MAX_FRAMES
        assert not [f for f in result["flags"] if f.startswith("dropped")]
        assert "p-keep" in [c["goal_id"] for c in result["clips"] if c["kind"] == "play"]

    def test_chosen_plays_drop_before_replay_shots(self):
        # 12 goals (tier 2, 10 s windows = 273 frames each = 3276) + 12 tight
        # replays (150 each = 1800) + 8 chosen plays (273 each = 2184) = 7260,
        # 60 over the cap. Plays absorb it; every replay shot survives.
        goals = self.twelve_goals()
        plays = [play(f"p-{chr(97 + i)}", 40.0 + i * 20.0,
                      score=0.9 - i * 0.05, chosen=True) for i in range(8)]
        result = A.plan_recap(selection(goals, plays, []), layouts())
        dropped = [f for f in result["flags"] if f.startswith("dropped")]
        assert dropped == ["dropped 1 chosen play(s) to fit the four-minute cap (p-h)"]
        # 12 goal tight replays + 1 best-play tight replay (p-a has the top score)
        assert len([r for r in result["replays"] if r["kind"] == "tight"]) == 13
        assert sum(1 for c in result["clips"] if c["kind"] == "play") == 7
        assert total_frames(result) <= A.MAX_FRAMES

    def test_best_goal_gets_the_full_treatment_regardless_of_tier(self):
        # 12 goals land in tier 2 (one tight shot each); the top-3 by interest
        # still get the tier-1 block: one wide and one tight shot.
        goals = self.twelve_goals()
        goals[0]["best_goal"] = True
        result = A.plan_recap(selection(goals), layouts())
        best = [r for r in result["replays"] if r["goal_id"] == "home:1"]
        others = [r for r in result["replays"] if r["goal_id"] != "home:1"]
        assert {r["kind"] for r in best} == {"wide", "tight"}
        assert all(r.get("best_goal") for r in best)
        assert all(r["kind"] == "tight" for r in others)

    def test_best_goal_replays_drop_last_at_the_cap(self):
        # 24 goals in tier 3: 4584 goal frames + 21 plain tight shots (2520) and
        # wide+tight for the three best goals (720) = 7824. Six plain tight
        # shots go first (latest goals first); every best-goal shot survives.
        goals = [goal(f"home:{i}", i) for i in range(1, 25)]
        for g in goals[:3]:
            g["best_goal"] = True
        result = A.plan_recap(selection(goals, [], []), layouts())
        drop_flag = next((f for f in result["flags"]
                          if f.startswith("dropped") and "replay shot" in f), None)
        assert drop_flag is not None
        ids = drop_flag.split("(")[-1].rstrip(")").split(", ")
        assert ids == ["home:24", "home:23", "home:22", "home:21", "home:20", "home:19"]
        best_shots = [r for r in result["replays"] if r.get("best_goal")]
        assert len(best_shots) == 6  # wide + tight for each of the three
        assert total_frames(result) <= A.MAX_FRAMES


class TestSchema:
    def test_schema_version_three(self):
        plays = [play("p-early", 25.0, score=0.9, chosen=True)]
        result = A.plan_recap(selection([goal("home:1", 0)], plays, []), layouts())
        assert result["schema_version"] == 3
