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
        "review_status": "keep",
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
    def test_high_score_filler_never_preempts_available_goals(self):
        goals = [goal(f"home:{i}", i) for i in range(1, 19)]
        for g in goals[5:]:
            g["cut"] = True
            g["interest"] = 0.1
        plays = [play("p-high", 900, score=1)]
        result = A.plan_recap(selection(goals, plays, [{"id": "p-high"}]), layouts())
        assert not [c for c in result["clips"] if c["kind"] == "play"]

    def test_cut_goal_is_restored_by_the_fill(self):
        goals = [goal(f"home:{i}", i) for i in range(1, 13)]
        goals[3]["cut"] = True
        result = A.plan_recap(selection(goals), layouts())
        goal_ids = [c["goal_id"] for c in result["clips"] if c["kind"] == "goal"]
        # The fill restores the cut goal: 11 goals leave the Recap short of the
        # three-minute target.
        assert "home:4" in goal_ids
        assert len(goal_ids) == 12
        assert any("home:4: restored from the cut list to fill the Recap" in f
                   for f in result["flags"])
        assert not any("home:4: cut by interest policy" in f for f in result["flags"])

    def test_lower_tier_when_cuts_take_the_count_across_a_boundary(self):
        goals = [goal(f"home:{i}", i) for i in range(1, 13)]
        for g in goals[2:9]:
            g["cut"] = True
        result = A.plan_recap(selection(goals), layouts())
        kinds = [r["kind"] for r in result["replays"]]
        # Shorter goal clips leave room to restore all seven cut goals.
        assert kinds == ["tight"] * 12

    def test_fill_restores_cut_goals_before_plays(self):
        goals = [goal(f"home:{i}", i) for i in range(1, 10)]
        for g, interest in ((goals[3], 0.45), (goals[5], 0.4), (goals[7], 0.35)):
            g["cut"] = True
            g["interest"] = interest
        plays = [play("p-1", 500.0, score=0.5), play("p-2", 900.0, score=0.5)]
        result = A.plan_recap(selection(goals, plays, [{"id": "p-1"}, {"id": "p-2"}]),
                              layouts())
        goal_ids = [c["goal_id"] for c in result["clips"] if c["kind"] == "goal"]
        assert all(f"home:{i}" in goal_ids for i in (4, 6, 8))
        # Cut goals come back in descending interest order, ahead of any play.
        restored = [f.split(":")[0] + ":" + f.split(":")[1]
                    for f in result["flags"] if "restored from the cut list" in f]
        assert restored == ["home:4", "home:6", "home:8"]
        assert len([c for c in result["clips"] if c["kind"] == "play"]) == 2

    def test_high_score_reviewed_play_follows_restored_goals(self):
        goals = [goal(f"home:{i}", i) for i in range(1, 11)]
        for g in goals[2:9]:
            g["cut"] = True
            g["interest"] = 0.45
        plays = [play("p-1", 500.0, score=0.9)]
        result = A.plan_recap(selection(goals, plays, [{"id": "p-1"}]), layouts())
        # All goals return before the reviewed play fills the remaining budget.
        assert [c["goal_id"] for c in result["clips"] if c["kind"] == "play"] == ["p-1"]
        assert any("home:3: restored from the cut list to fill the Recap" in f
                    for f in result["flags"])
        assert sum("restored from the cut list" in f for f in result["flags"]) == 7

    def test_a_close_play_never_preempts_a_cut_goal(self):
        goals = [goal(f"home:{i}", i) for i in range(1, 11)]
        for g in goals[2:9]:
            g["cut"] = True
            g["interest"] = 0.45
        plays = [play("p-1", 500.0, score=0.6)]
        result = A.plan_recap(selection(goals, plays, [{"id": "p-1"}]), layouts())
        # The shorter goals leave room for the reviewed play after restoration.
        assert [c["goal_id"] for c in result["clips"] if c["kind"] == "play"] == ["p-1"]
        restored = [f for f in result["flags"] if "restored from the cut list" in f]
        assert len(restored) == 7

    def test_cut_goals_stay_omitted_when_there_is_no_fill_room(self):
        goals = [goal(f"home:{i}", i) for i in range(1, 25)]
        goals[3]["cut"] = True
        goals[5]["cut"] = True
        result = A.plan_recap(selection(goals), layouts())
        goal_ids = [c["goal_id"] for c in result["clips"] if c["kind"] == "goal"]
        assert "home:4" not in goal_ids and "home:6" not in goal_ids
        assert sum("cut by interest policy" in f for f in result["flags"]) == 2


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
        assert len(goal_replays) == 1 and goal_replays[0]["kind"] == "tight"
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
    def test_unreviewed_and_rejected_net_plays_do_not_pad_the_recap(self):
        plays = [play("icing", 60, score=1), play("offside", 90, score=.9)]
        del plays[0]["review_status"]
        plays[1]["review_status"] = "reject"
        sel = selection([goal("home:1", 0)], plays, [{"id": p["id"]} for p in plays])
        result = A.plan_recap(sel, layouts())
        assert [c["goal_id"] for c in result["clips"]] == ["home:1"]
        assert result["duration_s"] < 30
        assert any("review" in f for f in result["flags"])

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
        # The best-play replay is added after the fill reaches its target.
        best_replay_frames = sum(r["frames"] for r in result["replays"] if r["goal_id"].startswith("p-"))
        assert total - best_replay_frames - filler[-1]["frames"] < FILL_FRAMES

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
        # Drop lower-ranked plays before any goal replay at the cap.
        goals = self.twelve_goals()
        plays = [play(f"p-{chr(97 + i)}", 40.0 + i * 20.0,
                      score=0.9 - i * 0.05, chosen=True) for i in range(10)]
        result = A.plan_recap(selection(goals, plays, []), layouts())
        dropped = [f for f in result["flags"] if f.startswith("dropped")]
        assert dropped == ["dropped 5 chosen play(s) to fit the four-minute cap (p-j, p-i, p-h, p-g, p-f)"]
        # 12 goal tight replays + 1 best-play tight replay (p-a has the top score)
        assert len([r for r in result["replays"] if r["kind"] == "tight"]) == 13
        assert sum(1 for c in result["clips"] if c["kind"] == "play") == 5
        assert total_frames(result) <= A.MAX_FRAMES

    def test_best_goal_gets_the_full_treatment_regardless_of_tier(self):
        # Best goals retain priority but do not receive a second replay.
        goals = self.twelve_goals()
        goals[0]["best_goal"] = True
        result = A.plan_recap(selection(goals), layouts())
        best = [r for r in result["replays"] if r["goal_id"] == "home:1"]
        others = [r for r in result["replays"] if r["goal_id"] != "home:1"]
        assert len(best) == 1 and best[0]["kind"] == "tight"
        assert all(r.get("best_goal") for r in best)
        assert all(r["kind"] == "tight" for r in others)

    def test_best_goal_replays_drop_last_at_the_cap(self):
        # Four-second source replays exceed the cap on a 24-goal game.
        # Plain replays drop first; all three best-goal replays survive.
        goals = [goal(f"home:{i}", i) for i in range(1, 25)]
        for g in goals[:3]:
            g["best_goal"] = True
        result = A.plan_recap(selection(goals, [], []), layouts())
        drop_flag = next((f for f in result["flags"]
                          if f.startswith("dropped") and "replay shot" in f), None)
        assert drop_flag is not None
        ids = drop_flag.split("(")[-1].rstrip(")").split(", ")
        assert ids == ["home:24", "home:23", "home:22", "home:21", "home:20", "home:19", "home:18",
                       "home:17", "home:16", "home:15", "home:14", "home:13", "home:12", "home:11"]
        best_shots = [r for r in result["replays"] if r.get("best_goal")]
        assert len(best_shots) == 3
        assert total_frames(result) <= A.MAX_FRAMES


class TestSchema:
    def test_schema_version_three(self):
        plays = [play("p-early", 25.0, score=0.9, chosen=True)]
        result = A.plan_recap(selection([goal("home:1", 0)], plays, []), layouts())
        assert result["schema_version"] == 3
