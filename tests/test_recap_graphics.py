import json
import subprocess
from types import SimpleNamespace

import pytest


def game():
    sheet = {
        "teams": {"home": "Ice Pak", "away": "Visitors"},
        "rosters": {"home": {"17": "A. Skater", "8": "B. Skater"}, "away": {}},
        "goals": {"home": [
            {"per": "1", "time_s": 600, "scorer": "17", "assists": ["8"], "type": "PP"},
            {"per": "2", "time_s": 400, "scorer": "8", "assists": [], "type": "ES"}],
            "away": []},
        "penalties": {"home": [{"per": "1", "player": "8", "minutes": "2", "infraction": "TRIP"}], "away": []},
        "goalkeeping": {"home": [{"player": "30", "name": "C. Goalie", "saves": 0}], "away": []},
    }
    selection = {"schema_version": 2, "periods": [
        {"n": 1, "start": 0, "end": 100}, {"n": 2, "start": 100, "end": 200}],
        "goals": [{"id": "home:1", "period": 1, "elapsed_s": 20},
                  {"id": "home:2", "period": 2, "elapsed_s": 40}]}
    def clip(gid, start, moment, kind="goal", **extra):
        return {"goal_id": gid, "kind": kind, "start_s": start, "moment_s": moment,
                "end_s": start + 10, "speed": 1, "frames": 300, **extra}
    plan = {"schema_version": 3, "fps": 30, "output": "A_Recap.mp4",
            "clips": [clip("home:1", 10, 15),
                      clip("penalty:home:1", 30, 35, "play", play_kind="penalty"),
                      clip("home:2", 120, 125)],
            "replays": [clip("home:1", 14, 15, "tight")], "duration_s": 40}
    audio = {"output": "A_Recap_audio.mp4", "video": "A_Recap.mp4", "duration_s": 40,
             "cues": [{"cue": "sfx_goal_card", "t": 5, "goal_id": "home:1"},
                      {"cue": "sfx_penalty_card", "t": 20, "goal_id": "penalty:home:1"},
                      {"cue": "sfx_period_wipe", "t": 30},
                      {"cue": "sfx_goal_card", "t": 35, "goal_id": "home:2"},
                      {"cue": "sfx_final_card", "t": 38}]}
    return sheet, selection, plan, audio


def test_props_writer_credits_sheet_players_and_cue_times(tmp_path):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    audio["cues"].insert(0, audio["cues"].pop())
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {})
    saved = json.loads((tmp_path / "recap_graphics_props.json").read_text())
    assert saved == props
    goal = next(e for e in props["events"] if e["kind"] == "goal")
    assert goal["startFrame"] == 150
    assert goal["data"] == {"id": "home:1", "team": "home", "num": "17", "scorer": "A. Skater",
                            "assists": ["8 B. Skater"], "type": "PP", "period": 1}
    final = next(e for e in props["events"] if e["kind"] == "final")
    assert final["startFrame"] == 1050 and final["durationFrames"] == 150
    assert final["data"]["score"] == [2, 0]
    assert final["data"]["table"]["home"]["skaters"] == [
        {"num": "8", "name": "B. Skater", "g": 1, "a": 1, "pts": 2, "pim": 2},
        {"num": "17", "name": "A. Skater", "g": 1, "a": 0, "pts": 1, "pim": 0}]
    assert final["data"]["table"]["home"]["goalies"] == [
        {"num": "30", "name": "C. Goalie", "saves": 0}]
    assert props["events"][-1]["kind"] == "final"
    assert "FINAL overlaps a late card; review the closing timeline" in props["flags"]


def test_props_score_changes_once_and_replay_keeps_post_goal_score(tmp_path):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {})
    bugs = [e for e in props["events"] if e["kind"] == "scorebug"]
    assert [(b["startFrame"], b["durationFrames"], b["data"]["score"]) for b in bugs] == [
        (0, 150, [0, 0]), (150, 150, [1, 0]), (300, 300, [1, 0]),
        (600, 300, [1, 0]), (900, 150, [1, 0]), (1050, 150, [2, 0])]
    penalty = next(e for e in props["events"] if e["kind"] == "penalty")
    assert penalty["startFrame"] == 600
    assert penalty["data"] == {"id": "penalty:home:1", "team": "home", "num": "8",
                               "name": "B. Skater", "minutes": "2", "infraction": "TRIP"}
    period = next(e for e in props["events"] if e["kind"] == "period")
    assert period["startFrame"] == 900 and period["data"] == {"period": 2, "score": [1, 0]}
    assert props["durationFrames"] == 1200


def test_cut_goal_still_counts_in_later_score_and_final(tmp_path):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    sheet["goals"]["away"] = [{"per": "1", "time_s": 500, "scorer": "9", "assists": []}]
    selection["goals"].insert(1, {"id": "away:1", "period": 1, "elapsed_s": 30,
                                  "cut": True, "chosen": {"detection_s": 25}})
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {})
    bugs = [e for e in props["events"] if e["kind"] == "scorebug"]
    assert bugs[-2]["data"]["score"] == [1, 1]
    assert bugs[-1]["data"]["score"] == [2, 1]
    assert next(e for e in props["events"] if e["kind"] == "final")["data"]["score"] == [2, 1]


def test_team_config_theme_and_missing_goalie_stats_are_explicit(tmp_path):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    sheet.pop("goalkeeping")
    options = {"focus_team": "ice-pak", "perspective": "focus",
               "teams": {"ice-pak": {"name": "Ice Pak", "short_name": "IPK",
                                      "colors": {"primary": "#5AA4D0", "secondary": "#222222"},
                                      "logo": "missing.png"}},
               "_layers": {"config_dir": str(tmp_path)}}
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, options)
    assert props["teams"]["home"] == {"name": "Ice Pak", "acronym": "IPK", "primary": "#5AA4D0",
                                        "secondary": "#222222", "logo": None}
    assert props["focusSide"] == "home" and props["perspective"] == "focus"
    assert props["teams"]["away"]["name"] == "Visitors"
    assert "home: goalie saves unavailable in game_sheet.json" in props["flags"]
    assert "home: logo missing; using wordmark" in props["flags"]
    logo = '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><rect width="32" height="32" fill="#5AA4D0"/></svg>'
    (tmp_path / "team.svg").write_text(logo)
    options["teams"]["ice-pak"]["logo"] = "team.svg"
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, options)
    assert props["teams"]["home"]["logo"] == "home.svg"
    assert (tmp_path / ".recap_cache/graphics/public/home.svg").read_text() == logo
    assert "home: logo missing; using wordmark" not in props["flags"]
    options["teams"]["ice-pak"]["name"] = "IcePak"
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, options)
    assert props["focusSide"] is None
    assert "Focus Team does not match game_sheet.json; using neutral cards" in props["flags"]


def test_replay_without_moment_and_unmatched_goal_keep_game_score(tmp_path):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    plan["replays"][0].pop("moment_s")
    sheet["goals"]["away"] = [{"per": "1", "scorer": "9", "assists": []}]
    selection["goals"].insert(1, {"id": "away:1", "period": 1, "elapsed_s": 30,
                                   "chosen": None, "window": {"expected": 25}})
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {})
    bugs = [e for e in props["events"] if e["kind"] == "scorebug"]
    assert bugs[2]["data"] == {"score": [1, 0], "period": 1}
    assert bugs[3]["data"] == {"score": [1, 1], "period": 1}


def test_unreadable_sheet_clock_keeps_final_totals_and_flags_timing(tmp_path):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    sheet["goals"]["away"] = [{"per": "1", "scorer": "9", "assists": []}]
    selection["goals"].append({"id": "away:1", "period": 1, "elapsed_s": None, "chosen": None})
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {})
    assert next(e for e in props["events"] if e["kind"] == "final")["data"]["score"] == [2, 1]
    assert "away:1: goal timing unavailable; scorebug needs review" in props["flags"]


def test_selected_goal_with_missing_sheet_clock_uses_camera_scoring_order(tmp_path):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    selection["goals"][0]["elapsed_s"] = None
    selection["goals"][1]["period"] = 1
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {})
    bugs = [e for e in props["events"] if e["kind"] == "scorebug"]
    assert bugs[0]["data"]["score"] == [0, 0]
    assert bugs[1]["data"]["score"] == [1, 0]
    assert bugs[-1]["data"]["score"] == [2, 0]
    assert "home:1: sheet clock unavailable; score order uses camera timing" in props["flags"]


@pytest.mark.parametrize("bad", ["stale-sheet", "stale-audio", "bad-cue", "orphan-replay", "bad-fps", "null-frames"])
def test_props_reject_stale_or_invalid_inputs(tmp_path, bad):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    if bad == "stale-sheet":
        sheet["goals"]["home"].pop()
    elif bad == "stale-audio":
        audio["duration_s"] = 39
    elif bad == "bad-cue":
        audio["cues"][0]["t"] = -1
    elif bad == "orphan-replay":
        plan["replays"][0]["goal_id"] = "home:9"
    elif bad == "bad-fps":
        plan["fps"] = 60
    else:
        plan["clips"][0]["frames"] = None
    with pytest.raises(ValueError):
        rg.write_props(tmp_path, sheet, selection, plan, audio, {})


def test_period_wipe_rounded_cue_uses_new_clip_frame(tmp_path):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    plan["clips"] = [plan["clips"][0], plan["clips"][2]]
    plan["clips"][0]["frames"] = 31
    plan["replays"] = []
    audio["duration_s"] = 331 / 30
    audio["cues"] = [{"cue": "sfx_period_wipe", "t": 1.033}]
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {})
    card = next(e for e in props["events"] if e["kind"] == "period")
    assert card["startFrame"] == 31 and card["data"]["period"] == 2


@pytest.mark.parametrize("minutes,pim", [("2+10", 12), ("2:30", 2.5), ("GM", None), ("?", None), (0, 0), ("0", 0)])
def test_penalty_minutes_keep_cards_and_final_stats_usable(tmp_path, minutes, pim):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    sheet["penalties"]["home"][0]["minutes"] = minutes
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {})
    final = next(e for e in props["events"] if e["kind"] == "final")
    assert final["data"]["table"]["home"]["skaters"][0]["pim"] == pim
    assert next(e for e in props["events"] if e["kind"] == "penalty")["data"]["minutes"] == minutes
    assert ("home: penalty minutes unavailable for #8; PIM needs review" in props["flags"]) == (pim is None)


def test_later_play_keeps_goals_already_counted_without_camera_timing(tmp_path):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    sheet["goals"]["away"] = [{"per": "1", "scorer": "9", "assists": []}]
    selection["goals"].insert(1, {"id": "away:1", "period": 1, "elapsed_s": 30, "chosen": None})
    plan["clips"].append({"goal_id": "play:1", "kind": "play", "start_s": 130,
                           "moment_s": 135, "end_s": 140, "frames": 300, "speed": 1})
    audio["duration_s"] = 50
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {})
    bugs = [e for e in props["events"] if e["kind"] == "scorebug"]
    assert bugs[-2]["data"]["score"] == [2, 1]
    assert bugs[-1]["data"]["score"] == [2, 1]


def test_top_cards_stop_at_next_card_and_source_clip_boundary(tmp_path):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    audio["cues"][0]["t"] = 9
    audio["cues"].insert(1, {"cue": "sfx_penalty_card", "t": 9.5, "goal_id": "penalty:home:1"})
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {})
    cards = [e for e in props["events"] if e["kind"] in ("goal", "penalty")]
    assert cards[0]["durationFrames"] == 15
    assert cards[1]["durationFrames"] == 15


def test_cli_runs_alpha_renderer_then_overlay_with_audio_copy(tmp_path, monkeypatch):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    for name, data in [("game_sheet.json", sheet), ("selection.json", selection),
                       ("recap_assembly.json", plan), ("recap_audio.json", audio)]:
        (tmp_path / name).write_text(json.dumps(data))
    calls, published = [], []
    def run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "ffprobe":
            return SimpleNamespace(returncode=0, stdout=json.dumps({"streams": [
                {"codec_type": "video", "width": 1280, "height": 720,
                 "avg_frame_rate": "30/1", "nb_frames": "1200"}, {"codec_type": "audio"}]}))
        return SimpleNamespace(returncode=0, stdout="")
    monkeypatch.setattr(subprocess, "run", run)
    replace = rg.os.replace
    def publish(a, b):
        if b.suffix == ".mp4":
            published.append((a, b))
        else:
            replace(a, b)
    monkeypatch.setattr(rg.os, "replace", publish)
    assert rg.main([str(tmp_path)]) == 0
    assert calls[1][0] == "node" and calls[1][1].endswith("render.mjs")
    assert "recap_graphics_props.json" in calls[1][2]
    overlay = calls[2]
    assert overlay[0] == "ffmpeg"
    assert str(tmp_path / "A_Recap_audio.mp4") in overlay
    assert overlay[overlay.index("-c:a") + 1] == "copy"
    assert "overlay=" in overlay[overlay.index("-filter_complex") + 1]
    assert "format=rgb:eof_action" in overlay[overlay.index("-filter_complex") + 1]
    assert overlay[overlay.index("-color_primaries") + 1] == "bt709"
    assert overlay[overlay.index("-color_trc") + 1] == "bt709"
    assert published[-1][1] == tmp_path / "A_Recap_graphics.mp4"
    props = json.loads((tmp_path / rg.PROPS_FILE).read_text())
    assert (props["width"], props["height"]) == (1280, 720)
    report = json.loads((tmp_path / rg.REPORT_FILE).read_text())
    assert report["output"] == "A_Recap_graphics.mp4" and report["duration_s"] == 40


def test_cli_render_failure_does_not_publish_success(tmp_path, monkeypatch):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    for name, data in [("game_sheet.json", sheet), ("selection.json", selection),
                       ("recap_assembly.json", plan), ("recap_audio.json", audio)]:
        (tmp_path / name).write_text(json.dumps(data))
    (tmp_path / rg.PROPS_FILE).write_text('{"previous": true}\n')
    def run(argv, **kwargs):
        if argv[0] == "ffprobe":
            return SimpleNamespace(returncode=0, stdout=json.dumps({"streams": [
                {"codec_type": "video", "width": 1920, "height": 1080,
                 "avg_frame_rate": "30/1", "nb_frames": "1200"}]}))
        raise subprocess.CalledProcessError(1, argv)
    monkeypatch.setattr(subprocess, "run", run)
    assert rg.main([str(tmp_path)]) == 1
    assert not (tmp_path / rg.REPORT_FILE).exists()
    assert json.loads((tmp_path / rg.PROPS_FILE).read_text()) == {"previous": True}


def test_cli_rejects_truncated_overlay_before_publication(tmp_path, monkeypatch):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    for name, data in [("game_sheet.json", sheet), ("selection.json", selection),
                       ("recap_assembly.json", plan), ("recap_audio.json", audio)]:
        (tmp_path / name).write_text(json.dumps(data))
    published = []
    def run(argv, **kwargs):
        frames = "1199" if argv[-1].endswith(".tmp.mp4") else "1200"
        return SimpleNamespace(returncode=0, stdout=json.dumps({"streams": [
            {"codec_type": "video", "width": 1920, "height": 1080,
             "avg_frame_rate": "30/1", "nb_frames": frames}]}))
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(rg.os, "replace", lambda a, b: published.append((a, b)))
    assert rg.main([str(tmp_path)]) == 1
    assert not published and not (tmp_path / rg.REPORT_FILE).exists()


def test_graphics_stage_tracks_theme_and_mixed_video(tmp_path, monkeypatch):
    import recap_runner as rr
    ctx = rr.Context(tmp_path, {"focus_team": "ice", "teams": {"ice": {"name": "Ice Pak"}}}, rr.Reporter())
    (tmp_path / "recap_assembly.json").write_text(json.dumps({"output": "A_Recap.mp4"}))
    stage = next(s for s in rr.STAGES if s.name == "graphics")
    assert stage.needs == ("mix",)
    assert stage.outputs(ctx) == ["recap_graphics_props.json", "recap_graphics.json", "A_Recap_graphics.mp4"]
    before = stage.fingerprint(ctx)
    ctx.options["teams"]["ice"]["colors"] = ["#FF0000", "#FFFFFF"]
    assert before != stage.fingerprint(ctx)
    assert "A_Recap_audio.mp4" in before["inputs"]
    calls = []
    def run(argv, on_line=None):
        calls.append(argv)
        on_line("[graphics] flag: home: goalie saves unavailable in game_sheet.json")
        on_line("[graphics] A_Recap_graphics.mp4: 40.0s")
        return 0, []
    monkeypatch.setattr(ctx, "run_cmd", run)
    result = stage.run(ctx)
    assert result.state == "flagged" and len(result.flags) == 1
    assert calls[0][1].endswith("recap_graphics.py")
    assert json.loads(calls[0][-1])["teams"]["ice"]["colors"] == ["#FF0000", "#FFFFFF"]
