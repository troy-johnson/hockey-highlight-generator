import json
import subprocess
from pathlib import Path
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
    selection["goals"][0]["chosen"] = {"candidate_id": "whistle:0"}
    selection["goals"][1]["chosen"] = {"candidate_id": "whistle:1"}
    selection["plays"] = [{"id": "penalty:home:1", "candidate_id": "whistle:2"}]
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


def acoustic_proofs(plan, selection, signals):
    """Stub the measured onset at the bound time for isolated schedule tests."""
    from audio_mix import build_timeline
    from whistle_presentation import METHOD, presentation_piece, source_binding
    timeline = build_timeline(plan, selection=selection, audio_signals=signals)
    return [{"goal_id": e["goal_id"], "mapped_t": e["confirmation_out"], "onset_t": e["confirmation_out"],
             "method": METHOD, "piece": source_binding(presentation_piece(timeline, e["goal_id"], e["confirmation_out"]))}
            for e in timeline if e.get("confirmation_out") is not None]


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
        {"num": "30", "name": "C. Goalie", "sa": 0, "ga": 0, "saves": 0, "sv_pct": None}]
    assert props["events"][-1]["kind"] == "final"
    assert "FINAL overlaps a late card; review the closing timeline" in props["flags"]


def test_transition_panes_cover_exact_source_cuts_and_period_plate_holds(tmp_path):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    next(c for c in audio["cues"] if c["cue"] == "sfx_period_wipe")["t"] = 29.6
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {})
    panes = [e for e in props["events"] if e["kind"] == "replay_pane"]
    assert [(e["startFrame"], e["durationFrames"], e["data"]["cutFrame"], e["data"]["edge"])
            for e in panes] == [(292, 16, 300, "in"), (592, 16, 600, "out")]
    wipe = next(e for e in props["events"] if e["kind"] == "period_wipe")
    assert (wipe["startFrame"], wipe["durationFrames"], wipe["data"]["cutFrame"]) == (885, 30, 900)
    period = next(e for e in props["events"] if e["kind"] == "period")
    assert (period["startFrame"], period["durationFrames"]) == (915, 82)
    path = tmp_path / "transitions.json"
    path.write_text(json.dumps(props))
    renderer = Path(rg.__file__).parents[1] / "graphics" / "render.mjs"
    result = subprocess.run(["node", str(renderer), str(path), str(tmp_path), "--validate"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_consecutive_replays_share_one_pane_at_their_common_cut(tmp_path):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    plan["replays"].append(dict(plan["replays"][0]))
    plan["duration_s"] = audio["duration_s"] = 50
    audio["cues"] = []
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {})
    assert sum(e["kind"] == "replay_pane" and e["data"]["cutFrame"] == 600
               for e in props["events"]) == 1


def test_score_waits_for_goal_card_confirmation_and_open_uses_lockup(tmp_path):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    audio["cues"][0]["t"] = 7
    (tmp_path / "lockup.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    (tmp_path / "crest.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    props = rg.write_props(tmp_path, sheet, selection, plan, audio,
                           {"focus_team": "Ice Pak", "teams": {"ip": {"name": "Ice Pak",
                            "lockup": str(tmp_path / "lockup.svg"),
                            "crest": str(tmp_path / "crest.svg")}}})
    before = next(e for e in props["events"] if e["kind"] == "scorebug" and e["startFrame"] == 0)
    assert before["durationFrames"] == 210
    assert before["data"]["score"] == [0, 0]
    assert props["focusSide"] == "home"
    assert props["teams"]["home"]["lockup"] == "home_lockup.svg"
    # A transparent transition crest is copied alongside the other brand assets.
    assert props["teams"]["home"]["crest"] == "home_crest.svg"


def test_late_goal_card_remains_readable_during_its_own_replay(tmp_path):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    audio["cues"][0]["t"] = 9.5
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {})
    card = next(e for e in props["events"] if e["kind"] == "goal")
    assert card["startFrame"] == 285
    assert card["durationFrames"] == 120
    # The following play still starts at its accepted source/output position.
    assert next(e for e in props["events"] if e["kind"] == "penalty")["startFrame"] == 600


def test_final_follows_all_footage_and_uses_its_own_closing_section(tmp_path):
    import audio_mix as am
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    plan["closing"] = [{"kind": "final", "goal_id": "close", "frames": 150,
                        "speed": 1, "start_s": 0, "moment_s": 0, "parts": []}]
    plan["duration_s"] = audio["duration_s"] = 45
    timeline = am.build_timeline(plan)
    segments = am.plan_bed(timeline, selection["periods"], [], "test")
    assert [s["period"] for s in segments] == [1, 2]
    assert segments[-1]["out_end"] == 45
    cues = am.plan_cues(timeline, duration_s=45,
                        period_wipes=[s["boundary"] for s in segments[1:]])
    assert not any(c["cue"] == "sfx_period_wipe" and c["t"] == 40 for c in cues)
    final_cue = next(c for c in cues if c["cue"] == "sfx_final_card")
    assert final_cue["t"] == 40
    audio["cues"] = [c for c in audio["cues"] if c["cue"] != "sfx_final_card"] + [final_cue]
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {})
    final = next(e for e in props["events"] if e["kind"] == "final")
    assert final["startFrame"] == 1200
    assert final["durationFrames"] == 150
    assert props["durationFrames"] == 1350


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
    assert period["startFrame"] == 915 and period["data"] == {"period": 2, "score": [1, 0]}
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
                                        "secondary": "#222222", "line": "#5AA4D0", "logo": None}
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
    assert card["startFrame"] == 46 and card["data"]["period"] == 2


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


def test_short_rails_are_omitted_instead_of_flashing_over_other_plays(tmp_path):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    audio["cues"][0]["t"] = 9
    audio["cues"].insert(1, {"cue": "sfx_penalty_card", "t": 9.5, "goal_id": "penalty:home:1"})
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {})
    cards = [e for e in props["events"] if e["kind"] in ("goal", "penalty")]
    assert not any(c["startFrame"] in (270, 285) for c in cards)
    assert sum("insufficient readable time" in f for f in props["flags"]) == 1
    assert sum("cue is outside its source play" in f for f in props["flags"]) == 1


def test_cli_runs_alpha_renderer_then_overlay_with_audio_copy(tmp_path, monkeypatch, freeze_sheet):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    next(c for c in audio["cues"] if c["cue"] == "sfx_penalty_card")["t"] = 25
    signals = {"whistles": [{"t": 15}, {"t": 125}, {"t": 35}]}
    audio["presentation_whistles"] = acoustic_proofs(plan, selection, signals)
    for name, data in [("game_sheet.json", sheet), ("selection.json", selection),
                       ("recap_assembly.json", plan), ("recap_audio.json", audio)]:
        (tmp_path / name).write_text(json.dumps(data))
    calls, published = [], []
    freeze_sheet(sheet)
    (tmp_path / "coverage.json").write_text(json.dumps({"periods": selection["periods"]}))
    (tmp_path / "audio_signals.json").write_text(json.dumps({"whistles": [{"t": 15}, {"t": 125}, {"t": 35}]}))
    workspace = tmp_path.parent / (tmp_path.name + "-local-workspace")
    def run(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "ffprobe":
            return SimpleNamespace(returncode=0, stdout=json.dumps({"streams": [
                {"codec_type": "video", "width": 1280, "height": 720,
                 "avg_frame_rate": "30/1", "nb_frames": "1200"}, {"codec_type": "audio"}]}))
        if argv[0] == "ffmpeg":
            Path(argv[-1]).write_bytes(b"validated video")
        return SimpleNamespace(returncode=0, stdout="")
    monkeypatch.setattr(subprocess, "run", run)
    replace = rg.os.replace
    def publish(a, b):
        assert a.parent == b.parent  # Publication never renames across filesystems.
        if b.suffix == ".mp4":
            published.append((a, b))
        else:
            replace(a, b)
    monkeypatch.setattr(rg.os, "replace", publish)
    assert rg.main([str(tmp_path), "--workspace", str(workspace)]) == 0
    assert calls[1][0] == "node" and calls[1][1].endswith("render.mjs")
    assert "recap_graphics_props.json" in calls[1][2]
    assert Path(calls[1][2]).parent == workspace
    overlay = calls[2]
    assert overlay[0] == "ffmpeg"
    assert Path(overlay[-1]).parent == workspace
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


def test_cli_render_failure_does_not_publish_success(tmp_path, monkeypatch, freeze_sheet):
    import recap_graphics as rg
    system_temp = tmp_path.parent / (tmp_path.name + "-system-temp")
    monkeypatch.setattr(rg.tempfile, "gettempdir", lambda: str(system_temp))
    sheet, selection, plan, audio = game()
    next(c for c in audio["cues"] if c["cue"] == "sfx_penalty_card")["t"] = 25
    signals = {"whistles": [{"t": 15}, {"t": 125}, {"t": 35}]}
    audio["presentation_whistles"] = acoustic_proofs(plan, selection, signals)
    for name, data in [("game_sheet.json", sheet), ("selection.json", selection),
                       ("recap_assembly.json", plan), ("recap_audio.json", audio)]:
        (tmp_path / name).write_text(json.dumps(data))
    (tmp_path / rg.PROPS_FILE).write_text('{"previous": true}\n')
    freeze_sheet(sheet)
    (tmp_path / "coverage.json").write_text(json.dumps({"periods": selection["periods"]}))
    (tmp_path / "audio_signals.json").write_text(json.dumps({"whistles": [{"t": 15}, {"t": 125}, {"t": 35}]}))
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
    assert list((system_temp / "hockeyrecap-graphics").glob("*/recap_graphics_props.json"))


def test_cli_failed_share_copy_preserves_previous_video_and_reports(tmp_path, monkeypatch, freeze_sheet):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    for name, data in [("game_sheet.json", sheet), ("selection.json", selection),
                       ("recap_assembly.json", plan), ("recap_audio.json", audio)]:
        (tmp_path / name).write_text(json.dumps(data))
    previous = {"A_Recap_graphics.mp4": b"previous video", rg.PROPS_FILE: b"previous props",
                rg.REPORT_FILE: b"previous report"}
    for name, data in previous.items():
        (tmp_path / name).write_bytes(data)
    workspace = tmp_path.parent / (tmp_path.name + "-workspace")
    logo = tmp_path / "logo.png"
    logo.write_bytes(b"synthetic logo")
    options = {"teams": {"ice": {"name": "Ice Pak", "logo": str(logo)}}}
    freeze_sheet(sheet, options)
    def run(argv, **kwargs):
        if argv[0] == "ffprobe":
            return SimpleNamespace(returncode=0, stdout=json.dumps({"streams": [
                {"codec_type": "video", "width": 1920, "height": 1080,
                 "avg_frame_rate": "30/1", "nb_frames": "1200"}]}))
        if argv[0] == "node":
            assert (workspace / "public" / "home.png").read_bytes() == b"synthetic logo"
            assert not (tmp_path / ".recap_cache").exists()
        if argv[0] == "ffmpeg":
            Path(argv[-1]).write_bytes(b"new validated video")
        return SimpleNamespace(returncode=0)
    copyfile = rg.shutil.copyfile
    def copy(source, destination):
        if Path(source).name == rg.PROPS_FILE:
            raise OSError(12, "Cannot allocate memory")
        return copyfile(source, destination)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(rg.shutil, "copyfile", copy)
    assert rg.main([str(tmp_path), "--workspace", str(workspace),
                    "--options", json.dumps(options)]) == 1
    for name, data in previous.items():
        assert (tmp_path / name).read_bytes() == data
    assert not list(tmp_path.glob(".*.tmp"))


def test_cli_rejects_truncated_overlay_before_publication(tmp_path, monkeypatch, freeze_sheet):
    import recap_graphics as rg
    monkeypatch.setattr(rg.tempfile, "gettempdir", lambda: str(tmp_path.parent / (tmp_path.name + "-temp")))
    sheet, selection, plan, audio = game()
    for name, data in [("game_sheet.json", sheet), ("selection.json", selection),
                       ("recap_assembly.json", plan), ("recap_audio.json", audio)]:
        (tmp_path / name).write_text(json.dumps(data))
    published = []
    freeze_sheet(sheet)
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
    assert before == stage.fingerprint(ctx)  # Only the snapshot supplies team data.
    changed = stage.fingerprint(ctx)
    ctx.options["kit"] = "navy"
    assert changed != stage.fingerprint(ctx)
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


def kit_options(tmp_path, kit):
    (tmp_path / "navy_crest.png").write_bytes(b"png")
    return {"focus_team": "ice-pak", "perspective": "focus", "kit": kit,
            "teams": {"ice-pak": {"name": "Ice Pak", "short_name": "IPK",
                                   "colors": {"primary": "#A60735", "secondary": "#000000"},
                                   "kits": {"red": {"fill": "#A60735", "line": "#E8354F"},
                                            "navy": {"fill": "#102E52", "line": "#5AA4D0",
                                                     "crest": "navy_crest.png"},
                                            "white": {"fill": "#FFFFFF", "line": "#EEF3F6", "text": "#1E3160"}}},
                      "visitors": {"name": "Visitors", "colors": {"primary": "#7B2D36"}}},
            "_layers": {"config_dir": str(tmp_path)}}


@pytest.mark.parametrize("kit,fill,line,text", [
    ("red", "#A60735", "#E8354F", "#FFFFFF"),
    ("navy", "#102E52", "#5AA4D0", "#FFFFFF"),
    ("white", "#FFFFFF", "#EEF3F6", "#1E3160"),
])
def test_focus_team_kit_sets_fill_line_text_and_crest(tmp_path, kit, fill, line, text):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, kit_options(tmp_path, kit))
    home = props["teams"]["home"]
    assert (home["kit"], home["primary"], home["line"], home["text"]) == (kit, fill, line, text)
    assert ("crest" in home) == (kit == "navy")
    assert not any("kit" in f for f in props["flags"])


def test_unknown_kit_is_flagged_and_team_colours_stay():
    import recap_graphics as rg
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    sheet, selection, plan, audio = game()
    props = rg.write_props(tmp, sheet, selection, plan, audio, kit_options(tmp, "green"))
    assert props["teams"]["home"]["primary"] == "#A60735"
    assert "kit" not in props["teams"]["home"]
    assert "home: kit 'green' not in Team Config; using team colors" in props["flags"]


def test_line_colour_must_read_on_black_ice(tmp_path):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, kit_options(tmp_path, None))
    # Deep red (2.4:1) and maroon fail 3:1 on the graphite plate, so lines use frost grey.
    assert props["teams"]["home"]["line"] == "#9FB2C1"
    # Both would be frost grey, so the opponent changes to its secondary (white by default).
    assert props["teams"]["away"]["line"] == "#FFFFFF"
    assert rg.contrast("#E8354F", rg.PLATE) >= 3 > rg.contrast("#A60735", rg.PLATE)


def test_kit_line_that_fails_contrast_is_flagged(tmp_path):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    options = kit_options(tmp_path, "red")
    options["teams"]["ice-pak"]["kits"]["red"]["line"] = "#A60735"
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, options)
    assert props["teams"]["home"]["line"] == "#9FB2C1"
    assert "home: kit 'red' line #A60735 fails 3:1 on #0E1319; using #9FB2C1" in props["flags"]


@pytest.mark.parametrize("preview,status,valid", [
    (True, "design_sample", True), (False, "design_sample", False),
    (True, "confirmed", False),
])
def test_call_design_fixture_cannot_be_presented_as_a_game_decision(tmp_path, preview, status, valid):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {})
    props["designPreview"] = preview
    props["events"] = [{"kind": "call", "startFrame": 0, "durationFrames": 150,
                        "data": {"status": status, "result": "NO GOAL",
                                 "reason": "GOALTENDER INTERFERENCE",
                                 "penaltyDisposition": "NO PENALTY ON THE PLAY"}}]
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(props))
    renderer = Path(rg.__file__).parents[1] / "graphics" / "render.mjs"
    result = subprocess.run(["node", str(renderer), str(path), str(tmp_path), "--validate"],
                            capture_output=True, text=True)
    assert (result.returncode == 0) == valid, result.stderr


def test_final_lists_both_rosters_and_goalie_shots_from_the_sheet():
    import recap_graphics as rg
    sheet, *_ = game()
    sheet["rosters"]["away"] = {"30": "D. Keeper", "4": "E. Skater"}
    sheet["goalkeeping"] = {"home": [{"player": "30", "name": "C. Goalie"}],
                            "away": [{"player": "30"}]}
    sheet["shots_on_goal"] = {"home": [10, 4], "away": [6, 3]}
    flags = []
    final = rg.final_data(sheet, flags)
    # Rostered players without a stat are not listed; the goalie is not a skater row.
    assert final["table"]["away"]["skaters"] == []
    # Shots against come from the opponent's shots; GA from the verified goal rows.
    assert final["table"]["home"]["goalies"] == [
        {"num": "30", "name": "C. Goalie", "sa": 9, "ga": 0, "saves": 9, "sv_pct": 1.0}]
    assert final["table"]["away"]["goalies"] == [
        {"num": "30", "name": "D. Keeper", "sa": 14, "ga": 2, "saves": 12, "sv_pct": 0.857}]


def test_final_lists_at_most_ten_skaters_with_a_stat_each():
    import recap_graphics as rg
    sheet, *_ = game()
    sheet["rosters"]["home"] = {str(n): f"P{n}" for n in range(1, 16)}
    sheet["goals"]["home"] = [{"per": "1", "time": "5:00", "scorer": str(n), "assists": []}
                              for n in range(1, 13)]
    sheet["penalties"]["home"] = []
    final = rg.final_data(sheet, [])
    skaters = final["table"]["home"]["skaters"]
    assert len(skaters) == 10
    assert all(r["g"] or r["a"] or r["pim"] for r in skaters)
    # The score still counts every goal, not only the listed skaters.
    assert final["score"][0] == 12


def test_goalie_without_shots_keeps_goals_against_and_no_save_numbers():
    import recap_graphics as rg
    sheet, *_ = game()
    sheet["goalkeeping"] = {"home": [], "away": [{"player": "1", "name": "F. Keeper"}]}
    sheet["shots_on_goal"] = {"home": [10, None], "away": [6, 3]}
    final = rg.final_data(sheet, [])
    assert final["table"]["away"]["goalies"] == [
        {"num": "1", "name": "F. Keeper", "sa": None, "ga": 2, "saves": None, "sv_pct": None}]


def test_goalie_numbers_from_the_sheet_are_validated_not_trusted():
    import recap_graphics as rg
    sheet, *_ = game()
    sheet["goalkeeping"] = {"home": [{"player": 30, "name": "C. Goalie", "shots": "12", "ga": "?"}],
                            "away": [{"player": 1, "name": "F. Keeper", "shots": 1, "ga": 2}]}
    flags = []
    final = rg.final_data(sheet, flags)["table"]
    home = final["home"]["goalies"][0]
    assert (home["sa"], home["ga"], home["saves"]) == (12, len(sheet["goals"]["away"]), 12 - len(sheet["goals"]["away"]))
    away = final["away"]["goalies"][0]
    assert away["saves"] is None and away["sv_pct"] is None
    assert any("away: goalie #1 shots below goals against" in f for f in flags)


def test_accent_token_is_the_focus_kit_line_and_cyan_without_focus(tmp_path):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, kit_options(tmp_path, "red"))
    assert props["tokens"]["accent"] == "#E8354F"
    options = kit_options(tmp_path, "red")
    options["focus_team"] = None
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, options)
    assert props["tokens"]["accent"] == "#7FD3F0"


@pytest.mark.parametrize("secondary,line", [("#FFFFFF", "#FFFFFF"), ("#B01030", "#9FB2C1"),
                                            ("#F04060", "#9FB2C1")])
def test_opponent_line_that_clashes_with_the_kit_line_changes(tmp_path, secondary, line):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    options = kit_options(tmp_path, "red")
    options["teams"]["visitors"]["colors"] = {"primary": "#C8102E", "secondary": secondary}
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, options)
    assert props["teams"]["home"]["line"] == "#E8354F"
    assert props["teams"]["away"]["line"] == line
    assert props["teams"]["away"]["primary"] == "#C8102E"


def test_focus_team_matches_names_without_case(tmp_path):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    options = kit_options(tmp_path, "red")
    options["teams"]["ice-pak"]["name"] = "ICE PAK"
    options["focus_team"] = "ICE PAK"
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, options)
    assert props["focusSide"] == "home" and props["teams"]["home"]["kit"] == "red"


def test_kit_without_a_line_is_rejected_and_flagged(tmp_path):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    options = kit_options(tmp_path, "red")
    del options["teams"]["ice-pak"]["kits"]["red"]["line"]
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, options)
    assert "kit" not in props["teams"]["home"]
    assert "home: kit 'red' needs #RRGGBB fill and line; using team colors" in props["flags"]


def test_goal_without_confirmation_holds_score_through_its_replay(tmp_path):
    # The whistle gates the score as well as the event bar and card.
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    audio["cues"] = [c for c in audio["cues"] if c.get("goal_id") != "home:1"]
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {})
    bugs = [(b["startFrame"], b["durationFrames"], b["data"]["score"])
            for b in props["events"] if b["kind"] == "scorebug"]
    assert bugs[:2] == [(0, 300, [0, 0]), (300, 300, [0, 0])]


def test_best_play_replay_gets_a_replay_pane(tmp_path):
    import recap_graphics as rg
    sheet, selection, plan, audio = game()
    play = plan["clips"][1]
    plan["replays"].append(dict(play, kind="tight", start_s=34, frames=30))
    plan["duration_s"] = audio["duration_s"] = 41
    audio["cues"] = [dict(c, t=c["t"] + 1) if c["t"] >= 30 else c for c in audio["cues"]]
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {})
    replays = [(e["startFrame"], e["durationFrames"]) for e in props["events"] if e["kind"] == "replay"]
    assert (900, 30) in replays
