# tests/test_audio_mix.py — Recap audio mix (hhg-3r5.39, spec 002 §5.10)
import json
import math
from pathlib import Path

import numpy as np
import pytest

import audio_mix as am
import recap_runner as rr


def _entry(goal_id, kind="goal", start=0.0, end=10.0, moment=5.0, frames=None, speed=1.1,
           camera="cam1", play_kind=None, parts=None):
    return {"goal_id": goal_id, "kind": kind, "play_kind": play_kind, "camera": camera,
            "moment_s": moment, "start_s": start, "end_s": end, "speed": speed,
            "angle_deg": 0.0,
            "parts": parts if parts is not None else
            [{"file": "/x/GX010001.MP4", "seek_s": start, "duration_s": end - start}],
            "frames": frames if frames is not None else math.ceil((end - start) / speed * am.FPS)}


def _plan(clips, replays=()):
    return {"schema_version": 3, "speed": 1.1, "fps": am.FPS, "audio": "silent",
            "clips": list(clips), "replays": list(replays),
            "duration_s": (sum(c["frames"] for c in clips)
                           + sum(r["frames"] for r in replays)) / am.FPS,
            "output": "2025-11-09_A-vs-B_Recap.mp4", "flags": []}


# --- timeline ----------------------------------------------------------------

def test_build_timeline_render_order_and_out_positions():
    a = _entry("home:1", start=100.0, end=110.0, moment=105.0, frames=273)
    b = _entry("away:1", start=200.0, end=210.0, moment=205.0, frames=273)
    rep = {"goal_id": "home:1", "kind": "tight", "camera": "cam1", "start_s": 104.0,
           "end_s": 106.0, "moment_s": 105.0, "speed": 0.5, "frames": 60,
           "parts": [{"file": "/x/GX010001.MP4", "seek_s": 104.0, "duration_s": 2.0}]}
    tl = am.build_timeline(_plan([a, b], [rep]))
    assert [e["goal_id"] for e in tl] == ["home:1", "home:1", "away:1"]
    assert tl[0]["out_start"] == 0.0 and tl[0]["out_dur"] == 273 / am.FPS
    assert tl[1]["out_start"] == 273 / am.FPS and tl[1]["out_dur"] == 60 / am.FPS
    assert tl[2]["out_start"] == (273 + 60) / am.FPS
    # The goal moment lands proportionally inside the output clip.
    assert math.isclose(tl[2]["moment_out"], tl[2]["out_start"] + 5.0 / 1.1)
    # Parts carry their detection-timeline position for PA-mute mapping.
    assert tl[2]["parts"][0]["tl_start"] == 200.0
    assert tl[2]["parts"][0]["out_start"] == tl[2]["out_start"]


def test_build_timeline_multi_part_positions():
    e = _entry("away:1", start=100.0, end=112.0, parts=[
        {"file": "/x/a.MP4", "seek_s": 100.0, "duration_s": 8.0},
        {"file": "/x/b.MP4", "seek_s": 0.0, "duration_s": 4.0}])
    tl = am.build_timeline(_plan([e]))
    assert [p["tl_start"] for p in tl[0]["parts"]] == [100.0, 108.0]
    assert tl[0]["parts"][1]["out_start"] == 8.0 / 1.1


# --- PA music muting ---------------------------------------------------------

def test_goal_card_waits_for_matched_whistle_after_motion_peak():
    entry = _entry("away:1", start=100, end=110, moment=105, speed=1)
    selection = {"goals": [{"id": "away:1", "chosen": {"candidate_id": "whistle:1"}}]}
    signals = {"whistles": [{"t": 104}, {"t": 107}]}
    tl = am.build_timeline(_plan([entry]), selection=selection, audio_signals=signals)
    cues = am.plan_cues(tl, duration_s=10)
    assert next(c for c in cues if c["cue"] == "sfx_goal_card")["t"] == 7
    assert next(c for c in cues if c["cue"] == "horn")["t"] == 7
    assert tl[0]["moment_out"] == 5
    assert tl[0]["confirmation_out"] == 7
    signals["whistles"][1]["t"] = 104
    tl = am.build_timeline(_plan([entry]), selection=selection, audio_signals=signals)
    assert not any(c["cue"] == "sfx_goal_card" for c in am.plan_cues(tl, duration_s=10))


def test_goal_confirmation_omitted_from_both_live_and_replay_is_not_announced():
    entry = _entry("away:1", start=100, end=109, moment=106, speed=1)
    replay = _entry("away:1", kind="tight", start=104.25, end=108.25,
                    moment=106, speed=.5)
    selection = {"goals": [{"id": "away:1", "chosen": {"candidate_id": "whistle:0"}}]}
    signals = {"whistles": [{"t": 109.02}]}
    tl = am.build_timeline(_plan([entry], [replay]), selection=selection, audio_signals=signals)
    assert tl[0]["confirmation_out"] is None
    assert not any(c["cue"] == "sfx_goal_card" for c in am.plan_cues(tl, duration_s=17))


def test_goal_confirmation_past_cut_without_replay_does_not_announce_over_next_play():
    entry = _entry("away:1", start=100, end=109, moment=106, speed=1)
    following = _entry("home:1", start=200, end=209, moment=206, speed=1)
    selection = {"goals": [{"id": "away:1", "chosen": {"candidate_id": "whistle:0"}}]}
    tl = am.build_timeline(_plan([entry, following]), selection=selection,
                           audio_signals={"whistles": [{"t": 109.02}]})
    assert tl[0]["confirmation_out"] is None
    assert not any(c.get("goal_id") == "away:1" and c["cue"] in ("horn", "sfx_goal_card")
                   for c in am.plan_cues(tl, duration_s=18))

def _span(cam, start, end, state="mute"):
    return {"camera": cam, "start": start, "end": end, "state": state}


def test_mute_windows_intersect_the_part_in_source_time():
    entry = {"camera": "cam1", "start_s": 100.0, "end_s": 110.0,
             "parts": [{"file": "/x/a.MP4", "seek_s": 100.0, "duration_s": 10.0,
                        "tl_start": 100.0}]}
    spans = [_span("cam1", 103.0, 106.0), _span("cam1", 200.0, 201.0),
             _span("cam2", 104.0, 105.0), _span("cam1", 108.0, 112.0, "review")]
    assert am.mute_windows(entry, spans) == [(3.0, 6.0)]


# --- cues --------------------------------------------------------------------

def _cues(perspective, focus_side=None, entries=None, pp_goals=(), comic_ids=(),
          penalty_kinds=("penalty",), fight_kinds=("fight",), goal_counts=(2, 1),
          period_wipes=(15.0,)):
    entries = entries or [_entry("home:1", moment=5.0, frames=300),
                          _entry("away:1", start=10.0, end=20.0, moment=15.0, frames=300),
                          _entry("play:1", kind="play", play_kind="penalty",
                                 start=20.0, end=30.0, moment=25.0, frames=300)]
    tl = am.build_timeline(_plan(entries))
    return am.plan_cues(tl, perspective=perspective, focus_side=focus_side,
                        duration_s=30.0, pp_goals=set(pp_goals), comic_calls=list(comic_ids),
                        penalty_kinds=set(penalty_kinds), fight_kinds=set(fight_kinds),
                        goal_counts=goal_counts, period_wipes=period_wipes)


def test_horn_fires_on_all_goals_in_neutral():
    cues = [c for c in _cues("neutral") if c["cue"] == "horn"]
    assert [c["t"] for c in cues] == [5.0 / 1.1, 10.0 + 5.0 / 1.1]


def test_horn_fires_on_focus_team_goals_only():
    cues = [c for c in _cues("focus", focus_side="home") if c["cue"] == "horn"]
    assert [c["goal_id"] for c in cues] == ["home:1"]


def test_close_kind_follows_perspective_and_score():
    assert _cues("neutral")[-1]["cue"] == "close_neutral"
    assert _cues("focus", focus_side="home")[-1]["cue"] == "close_win"
    assert _cues("focus", focus_side="home", goal_counts=(1, 2))[-1]["cue"] == "close_loss"
    assert _cues("focus", focus_side="home", goal_counts=(1, 1))[-1]["cue"] == "close_neutral"
    assert _cues("focus", focus_side="away", goal_counts=(1, 2))[-1]["cue"] == "close_win"


def test_game_start_and_neutral_sting_at_open():
    cues = _cues("neutral")
    assert [c for c in cues if c["cue"] == "game_start"][0]["t"] == 0.0
    assert [c for c in cues if c["cue"] == "sting_neutral"][0]["t"] == 0.0
    assert not [c for c in _cues("focus", focus_side="home") if c["cue"] == "sting_neutral"]


def test_opening_sound_follows_skate_impact_hold_and_exit():
    opening = {"kind": "stinger", "frames": 135, "start_s": 0, "moment_s": 0,
               "speed": 1, "parts": [], "out_start": 0, "out_dur": 4.5}
    cues = am.plan_cues([opening], duration_s=4.5)
    hit = next(c for c in cues if c["cue"] == "game_start")
    assert hit["dur_s"] == 4.5
    audio = am.synth_placeholder("game_start", duration_s=4.5)
    assert audio.shape == (216000, 2)
    assert np.isfinite(audio).all()
    rms = lambda a, b: np.sqrt(np.mean(audio[int(a * am.SR):int(b * am.SR)] ** 2))
    assert rms(.86, .96) > 2 * rms(2.5, 3.0)
    assert rms(4.23, 4.47) > 2 * rms(2.5, 3.0)  # exit whoosh under the frost pane
    assert np.max(np.abs(audio)) <= .85


def test_opening_sound_marks_breakthrough_plate_and_frost_exit():
    audio = am.synth_placeholder("game_start", duration_s=4.5)
    rms = lambda a, b: np.sqrt(np.mean(audio[int(a * am.SR / 30):int(b * am.SR / 30)] ** 2))
    hold = rms(80, 105)
    assert rms(46, 49) > 2 * rms(42, 45)  # ice cracks as the crest breaks through
    assert rms(66, 69) > 2 * rms(58, 62)  # matchup plate lands
    assert rms(127, 134) > 1.5 * rms(108, 114)  # frost pane whoosh covers the hand-off
    assert rms(0, 3) < rms(4, 26)      # quiet room tone before the skate


def test_play_start_keeps_legacy_sound_without_a_visual_stinger():
    audio = am.synth_placeholder("game_start")
    assert audio.shape == (48000, 2)
    frequencies = np.fft.rfftfreq(len(audio), 1 / am.SR)
    peak = frequencies[np.argmax(np.abs(np.fft.rfft(audio[:, 0])))]
    assert peak == 60


def test_penalty_and_special_stings():
    cues = _cues("focus", focus_side="home", pp_goals={"away:1"}, comic_ids=["play:1"],
                 penalty_kinds=("penalty",), fight_kinds=("fight",))
    assert [c["t"] for c in cues if c["cue"] == "sting_penalty"] == [20.0]
    assert [c for c in cues if c["cue"] == "sting_power_play"]
    assert [c for c in cues if c["cue"] == "sting_comic_call"]
    assert not [c for c in cues if c["cue"] == "sting_fight"]


def test_sfx_hooks_are_timeline_driven():
    cues = [c["cue"] for c in _cues("neutral")]
    for sfx in ("sfx_goal_card", "sfx_period_wipe", "sfx_final_card", "sfx_penalty_card"):
        assert sfx in cues


def test_cue_events_carry_their_audio_length():
    cues = {c["cue"]: c for c in _cues("neutral")}
    for c in cues.values():
        assert c["dur_s"] > 0


# --- bed ---------------------------------------------------------------------

def _bed(i, bar_s=2.0):
    return {"id": f"bed-{i:02d}", "file": f"/x/bed-{i:02d}.wav", "duration_s": 60.0,
            "bpm": 240.0 / bar_s, "source": "pixabay", "content_id_safe": True, "tags": []}


def _periods():
    return [{"n": 1, "start": 0.0, "end": 1000.0}, {"n": 2, "start": 1000.0, "end": 2000.0}]


def test_bed_rotation_is_deterministic_and_varies_by_game():
    beds = [_bed(i) for i in range(4)]
    a = am.bed_index(beds, "2025-11-09|A-vs-B", period=1)
    assert a == am.bed_index(beds, "2025-11-09|A-vs-B", period=1)
    assert am.bed_index(beds, "2025-11-09|A-vs-B", period=2) == (a + 1) % 4
    assert len({am.bed_index(beds, f"g{i}", period=1) for i in range(6)}) > 1


def test_plan_bed_cuts_periods_on_bars():
    # Period-1 entries cover output 0..20 s, period 2 from 20 s; the bed changes on
    # the outgoing grid's first bar line at or after the boundary.
    entries = [_entry("home:1", start=10.0, end=20.0, moment=12.0, frames=330),
               _entry("home:2", start=30.0, end=40.0, moment=32.0, frames=330),
               _entry("away:1", start=1010.0, end=1020.0, moment=1012.0, frames=330)]
    tl = am.build_timeline(_plan(entries))
    bed = am.plan_bed(tl, _periods(), [_bed(0), _bed(1)], "k")
    assert [s["period"] for s in bed] == [1, 2]
    assert bed[0]["out_start"] == 0.0
    change = bed[1]["out_start"]
    assert change >= 22.0 and math.isclose(change % bed[0]["bar_s"], 0.0, abs_tol=1e-9)


def test_plan_bed_outside_periods_uses_the_nearest_period():
    entries = [_entry("home:1", start=900.0, end=910.0, moment=905.0, frames=330)]
    bed = am.plan_bed(am.build_timeline(_plan(entries)), _periods(), [_bed(0)], "k")
    assert bed[0]["period"] == 1


def test_bed_loop_region_is_whole_bars():
    assert am.loop_bars(duration_s=61.0, bar_s=2.0) == 60.0
    assert am.loop_bars(duration_s=3.0, bar_s=2.0) == 2.0
    assert am.loop_bars(duration_s=4.0, bar_s=2.0) == 4.0


def test_bar_seconds_prefers_beat_grid_over_bpm():
    assert am.bar_seconds({"bpm": 120.0}) == 2.0
    assert am.bar_seconds({"bpm": 120.0, "bar_s": 3.0}) == 3.0
    assert am.bar_seconds({"beats": [0.0, 0.5, 1.0, 1.5], "beats_per_bar": 4}) == 2.0


# --- ducking -----------------------------------------------------------------

def test_duck_envelope_dips_under_each_cue_then_recovers():
    n = am.SR * 6
    events = [{"t": 1.0, "dur_s": 1.0}]
    g = am.duck_envelope(n, am.SR, events, duck_db=6.0)
    assert math.isclose(float(g[0]), 1.0)
    assert math.isclose(float(g.min()), 10 ** (-6 / 20), rel_tol=0.05)
    assert float(g[-1]) > 0.99
    assert g.shape == (n,)


# --- placeholder synthesis ---------------------------------------------------

def test_placeholder_cues_are_bounded_and_finite():
    for cue in ("game_start", "horn", "sting_penalty", "close_win", "close_neutral",
                "sfx_goal_card", "sting_neutral"):
        buf = am.synth_placeholder(cue)
        assert buf.ndim == 2 and buf.shape[1] == 2 and len(buf) > 0
        assert np.isfinite(buf).all() and float(np.abs(buf).max()) <= 1.0


def test_placeholder_bed_has_bar_loudness_pattern():
    bed = am.synth_placeholder_bed()
    bar = int(am.SR * am.PLACEHOLDER_BED_BAR_S)
    assert len(bed) >= 4 * bar
    rms = [float(np.sqrt(np.mean(bed[i * bar:(i + 1) * bar] ** 2))) for i in range(4)]
    assert all(r > 0 for r in rms)


# --- manifest ----------------------------------------------------------------

def test_load_manifest_missing_is_placeholder_mode(tmp_path):
    man, flags = am.load_manifest(tmp_path / "nope.json")
    assert man is None and any("placeholder" in f for f in flags)


def test_load_manifest_validates_schema(tmp_path):
    (tmp_path / "bed-00.wav").write_bytes(b"riff")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1,
                                 "beds": [{**_bed(0), "file": "bed-00.wav"}], "cues": []}))
    man, flags = am.load_manifest(path)
    assert man["beds"][0]["id"] == "bed-00" and not flags
    path.write_text(json.dumps({"schema_version": 2, "beds": [], "cues": []}))
    man, flags = am.load_manifest(path)
    assert man is None and any("schema" in f for f in flags)


def test_load_manifest_reports_missing_media(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1,
                                "beds": [{**_bed(0), "file": "missing.wav"}],
                                "cues": [{"id": "horn", "file": "missing.wav",
                                          "duration_s": 2.0, "gain_db": 0.0}]}))
    man, flags = am.load_manifest(path)
    assert man is not None
    assert any("missing.wav" in f for f in flags)
    assert man["beds"] == [] and man["cues"] == []


# --- loudnorm ----------------------------------------------------------------

def test_parse_loudnorm_json_measurements():
    text = ("[Parsed_loudnorm_0 @ 0x1] i: -13.2 LUFS\n"
            "\n[Parsed_loudnorm_0 @ 0x1] {\n"
            '\t"input_i" : "-23.4",\n'
            '\t"input_tp" : "-6.5",\n'
            '\t"input_lra" : "10.2",\n'
            '\t"input_thresh" : "-33.9",\n'
            '\t"target_offset" : "0.4"\n}\n')
    m = am.parse_loudnorm_json(text)
    assert m == {"input_i": -23.4, "input_tp": -6.5, "input_lra": 10.2,
                 "input_thresh": -33.9, "target_offset": 0.4}


# --- helpers -----------------------------------------------------------------

def test_fit_to_length_trims_and_pads():
    buf = np.ones((10, 2), dtype=np.float32)
    assert len(am.fit_to_length(buf, 6)) == 6
    out = am.fit_to_length(buf, 14)
    assert len(out) == 14 and float(out[10:].max()) == 0.0


def test_mixed_output_name():
    assert am.mixed_output_name("2025-11-09_A-vs-B_Recap.mp4") == \
        "2025-11-09_A-vs-B_Recap_audio.mp4"


def test_atempo_filter_chains_below_half_speed():
    assert am.atempo_filter(1.1) == "atempo=1.100000"
    assert am.atempo_filter(0.5) == "atempo=0.500000"
    assert am.atempo_filter(0.3) == "atempo=0.5,atempo=0.600000"
    assert am.atempo_filter(0.1) == "atempo=0.5,atempo=0.500000"  # floor 0.25


def test_resolve_focus_side_matches_team_case_insensitively():
    sheet = {"teams": {"home": "ICE PAK", "away": "Ghost Pirates"}}
    side, msg = am.resolve_focus_side("focus", "Ice Pak", sheet)
    assert side == "home" and msg is None
    side, msg = am.resolve_focus_side("focus", "ghost pirates", sheet)
    assert side == "away" and msg is None
    side, msg = am.resolve_focus_side("focus", "No Such Team", sheet)
    assert side is None and "not on the Game Sheet" in msg
    side, msg = am.resolve_focus_side("neutral", "Ice Pak", sheet)
    assert side is None and msg is None


# --- runner wiring -----------------------------------------------------------

def test_mix_stage_wiring(tmp_path):
    stage = next(s for s in rr.STAGES if s.name == "mix")
    assert stage.needs == ("assembly",) and not stage.needs_two_cameras
    ctx = rr.Context(tmp_path, {}, rr.Reporter())
    # No plan yet: the mixed output name falls back to recap.mp4's stem.
    assert stage.outputs(ctx) == ["recap_audio.json", "recap_audio.mp4"]
    argv = rr.mix_argv(ctx)
    assert argv[1].endswith("audio_mix.py") and argv[2] == str(tmp_path)
    assert set(rr.mix_options(ctx)) == {"perspective", "focus_team", "comic_calls", "audio",
                                        "period_transitions"}


def test_period_changes_are_plain_cuts_unless_transitions_are_enabled():
    segments = [{"boundary": 0.0}, {"boundary": 40.0}, {"boundary": 80.0}]
    assert am.period_wipe_times(segments, {}) == []
    assert am.period_wipe_times(segments, {"period_transitions": True}) == [40.0, 80.0]
    ctx = rr.Context(Path("."), {}, rr.Reporter())
    assert rr.mix_options(ctx)["period_transitions"] is False


def test_mix_options_resolve_focus_team_name():
    ctx = rr.Context(Path("."), {"perspective": "focus", "focus_team": "icepak",
                                 "teams": {"icepak": {"name": "Ice Pak"}},
                                 "comic_calls": ["play:1"], "audio": {"bed_gain_db": -18}},
                     rr.Reporter())
    opts = rr.mix_options(ctx)
    assert opts["focus_team"] == "Ice Pak" and opts["perspective"] == "focus"
    assert opts["comic_calls"] == ["play:1"] and opts["audio"] == {"bed_gain_db": -18}


def test_mix_fingerprint_hashes_its_inputs(tmp_path):
    ctx = rr.Context(tmp_path, {"perspective": "neutral", "audio": {"manifest": str(tmp_path / "m.json")}},
                     rr.Reporter())
    (tmp_path / "recap_assembly.json").write_text(json.dumps({"output": "recap.mp4"}))
    for name in ("music_spans.json", "selection.json", "game_sheet.json", "recap.mp4"):
        (tmp_path / name).write_text("{}" if not name.endswith(".mp4") else "x")
    fp = next(s for s in rr.STAGES if s.name == "mix").fingerprint(ctx)
    assert set(fp["inputs"]) == {"recap_assembly.json", "music_spans.json", "selection.json",
                                 "game_sheet.json", "recap.mp4", "audio_signals.json", "play_review.json",
                                 "game_data.json", "coverage.json"}
    assert fp["options"] == rr.mix_options(ctx)
    assert "scripts" in fp


def _mix_result(tmp_path, monkeypatch, rc, lines):
    ctx = rr.Context(tmp_path, {}, rr.Reporter())

    def fake_run(argv, on_line=None):
        for ln in lines:
            if on_line:
                on_line(ln)
        return rc, lines

    monkeypatch.setattr(ctx, "run_cmd", fake_run)
    return rr._run_mix(ctx)


def test_mix_stage_flags_and_summary(tmp_path, monkeypatch):
    res = _mix_result(tmp_path, monkeypatch, 0, [
        "[mix] flag: placeholder cue audio: no Cue Library manifest; synthesized cues and bed",
        "[mix] 2025-11-09_A-vs-B_Recap_audio.mp4: 192.3s, 3 cue(s), 1 PA span muted, "
        "-14.0 LUFS integrated, -1.4 dBTP peak"])
    assert res.state == "flagged" and "placeholder" in res.flags[0]
    assert res.message.startswith("2025-11-09_A-vs-B_Recap_audio.mp4")
    assert _mix_result(tmp_path, monkeypatch, 0, ["[mix] x"]).state == "done"
    res = _mix_result(tmp_path, monkeypatch, 1, ["[ERROR] mix: recap_assembly.json is missing"])
    assert res.state == "failed" and not res.fatal


# --- game cues (hhg-3r5.110) ------------------------------------------------

def _replay(goal_id, start, frames=60, speed=.5):
    return {"goal_id": goal_id, "kind": "tight", "camera": "cam1", "start_s": start,
            "end_s": start + frames / am.FPS * speed, "moment_s": start, "speed": speed,
            "frames": frames, "parts": [{"file": "/x/GX010001.MP4", "seek_s": start,
                                         "duration_s": frames / am.FPS * speed}]}


def _game_timeline():
    clips = [_entry("home:1", start=100.0, end=110.0, moment=105.0, frames=300),
             _entry("away:1", start=200.0, end=210.0, moment=205.0, frames=300),
             _entry("penalty:home:1", kind="play", play_kind="penalty",
                    start=300.0, end=310.0, moment=305.0, frames=300),
             _entry("penalty:away:1", kind="play", play_kind="penalty",
                    start=400.0, end=410.0, moment=405.0, frames=300)]
    replays = [_replay("home:1", 104.0), _replay("away:1", 204.0),
               _replay("penalty:home:1", 304.0), _replay("penalty:away:1", 404.0)]
    return am.build_timeline(_plan(clips, replays))


def test_game_cue_ids_are_known_to_the_mixer_and_the_cue_library():
    import cue_library as cl
    assert am.GAME_CUES <= set(am.CUE_IDS) == set(cl.CUE_IDS)


def test_songs_follow_the_scoring_and_the_penalized_team():
    tl = _game_timeline()
    cues = am.plan_cues(tl, perspective="focus", focus_side="home", duration_s=48.0, game_cues=True)
    songs = {c["goal_id"]: c for c in cues if c["cue"].startswith("song_")}
    assert {k: c["cue"] for k, c in songs.items()} == {
        "home:1": "song_goal_focus", "away:1": "song_goal_opponent",
        "penalty:home:1": "song_penalty_focus", "penalty:away:1": "song_penalty_opponent"}
    # A goal song starts at the goal and ends with its replay (live 10 s + replay 2 s).
    home = songs["home:1"]
    assert home["t"] == pytest.approx(5.0 / 1.1)
    assert home["t"] + home["dur_s"] == pytest.approx(12.0)
    assert home["fade_out_s"] == am.SONG_FADE_S
    # A penalty song starts at the start of the penalty play without a whistle.
    assert songs["penalty:home:1"]["t"] == pytest.approx(24.0)
    assert songs["penalty:home:1"]["t"] + songs["penalty:home:1"]["dur_s"] == pytest.approx(36.0)


def test_songs_need_a_focus_team_and_game_cues():
    tl = _game_timeline()
    assert not any(c["cue"].startswith("song_") for c in
                   am.plan_cues(tl, perspective="neutral", duration_s=48.0, game_cues=True))
    assert not any(c["cue"].startswith("song_") or c["cue"] == "sfx_replay_pane" for c in
                   am.plan_cues(tl, perspective="focus", focus_side="home", duration_s=48.0))


def test_replay_pane_peaks_on_each_replay_cut():
    tl = _game_timeline()
    panes = [c["t"] for c in am.plan_cues(tl, duration_s=48.0, game_cues=True)
             if c["cue"] == "sfx_replay_pane"]
    cuts = [10.0, 12.0, 22.0, 24.0, 34.0, 36.0, 46.0]
    assert panes == pytest.approx([t - 8 / am.FPS for t in cuts])


def test_live_replay_continues_the_live_source_at_its_speed():
    clip = _entry("home:1", start=100.0, end=110.0, moment=105.0, frames=273)
    tl = am.build_timeline(_plan([clip], [_replay("home:1", 104.0), _replay("home:1", 106.0)]),
                           live_replay=True)
    first, second = tl[1], tl[2]
    assert first["parts"][0]["seek_s"] == pytest.approx(110.0)
    assert first["parts"][0]["duration_s"] == pytest.approx(2.2)
    assert second["parts"][0]["seek_s"] == pytest.approx(112.2)
    # A Replay keeps its kind and picture timing; only its sound changes.
    assert first["kind"] == "tight" and first["live_sound"] and first["speed"] == 1.1
    assert first["frames"] == 60 and first["out_dur"] == 2.0
    # The continued piece maps PA-mute spans in its own source time.
    assert first["parts"][0]["tl_start"] == pytest.approx(110.0)
    assert am.mute_windows(first, [_span("cam1", 111.0, 111.5)]) == [(1.0, 1.5)]
    # Without the option the Replay keeps its own slowed source.
    assert am.build_timeline(_plan([clip], [_replay("home:1", 104.0)]))[1]["parts"][0]["seek_s"] == 104.0


def test_live_replay_sound_is_not_lowered_like_slowed_replay_sound(monkeypatch):
    monkeypatch.setattr(am, "_entry_audio", lambda *_: np.ones((am.SR, 2), dtype=np.float32))
    buf = am.game_sound_entry({"kind": "tight", "out_dur": 1.0, "live_sound": True}, [])
    assert buf[am.SR // 2, 0] == 1


def test_live_replay_keeps_its_place_across_multi_part_clips():
    clip = _entry("away:1", start=100.0, end=112.0, parts=[
        {"file": "/x/a.MP4", "seek_s": 100.0, "duration_s": 8.0},
        {"file": "/x/b.MP4", "seek_s": 0.0, "duration_s": 4.0}])
    tl = am.build_timeline(_plan([clip], [_replay("away:1", 104.0)]), live_replay=True)
    part = tl[1]["parts"][0]
    assert part["file"] == "/x/b.MP4" and part["seek_s"] == pytest.approx(4.0)


def test_whistle_after_the_live_cut_maps_through_live_replay_sound():
    """Review finding: a whistle just after the live cut plays at live speed under the Replay."""
    clip = _entry("home:1", start=100.0, end=109.0, moment=106.0, speed=1)
    replay = _replay("home:1", 104.0, frames=360)        # 0.5x, source 104-110
    selection = {"goals": [{"id": "home:1", "chosen": {"candidate_id": "whistle:0"}}]}
    signals = {"whistles": [{"t": 109.02}]}
    slowed = am.build_timeline(_plan([clip], [replay]), selection=selection, audio_signals=signals)
    live = am.build_timeline(_plan([clip], [replay]), selection=selection, audio_signals=signals,
                             live_replay=True)
    assert slowed[0]["confirmation_out"] == pytest.approx(9.0 + 5.02 / .5)
    assert live[0]["confirmation_out"] == pytest.approx(9.02)


def test_period_cut_times_mark_live_pieces_that_change_period():
    tl = _game_timeline()
    periods = [{"n": 1, "start": 0.0, "end": 250.0}, {"n": 2, "start": 250.0, "end": 500.0}]
    assert am.period_cut_times(tl, periods) == pytest.approx([24.0])
    cues = am.plan_cues(tl, duration_s=48.0, game_cues=True, period_cuts=[24.0])
    # The plain-cut sound is its own cue: sfx_period_wipe makes graphics draw a period pane.
    assert [c["t"] for c in cues if c["cue"] == "sfx_period_cut"] == [24.0]
    assert not [c for c in cues if c["cue"] == "sfx_period_wipe"]


def _chapter(path, value, seconds=1.0):
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(am.SR)
        w.writeframes(np.full(int(am.SR * seconds) * 2, value, dtype="<i2").tobytes())
    return str(path)


def test_continued_sound_crosses_into_the_next_chapter(tmp_path):
    """Review finding: live sound past the chapter end reads the Recording's next chapter."""
    first = _chapter(tmp_path / "GX010001.MP4", 3000)
    second = _chapter(tmp_path / "GX020001.MP4", 6000)
    am.set_recordings({"recordings": {"cam1": [[first, second]]}})
    buf, missing = am.decode_continued({"file": first, "seek_s": .5, "duration_s": 1.0}, 1.0)
    assert missing == pytest.approx(0, abs=.01) and len(buf) == pytest.approx(am.SR, abs=480)
    assert buf[am.SR // 4, 0] == pytest.approx(3000 / 32768, rel=1e-3)
    assert buf[3 * am.SR // 4, 0] == pytest.approx(6000 / 32768, rel=1e-3)
    # A seek past the first chapter starts inside the second.
    buf, missing = am.decode_continued({"file": first, "seek_s": 1.2, "duration_s": .5}, 1.0)
    assert missing == pytest.approx(0, abs=.01) and buf[100, 0] == pytest.approx(6000 / 32768, rel=1e-3)
    # The last chapter: the shortfall is reported, and the mix flags it.
    buf, missing = am.decode_continued({"file": first, "seek_s": 1.5, "duration_s": 1.0}, 1.0)
    assert missing == pytest.approx(.5, abs=.01)
    notes = []
    am._entry_audio({"goal_id": "home:1", "speed": 1.0, "parts": [
        {"file": first, "seek_s": 1.5, "duration_s": 1.0, "continued": True}]}, [], notes)
    assert notes and "no next chapter" in notes[0]


def test_replay_pane_uses_the_library_length():
    """Review finding: a 0.25 s pane cut off Pane A's peak at 0.27 s."""
    cues = am.plan_cues(_game_timeline(), duration_s=48.0, game_cues=True,
                        cue_durations={"sfx_replay_pane": 1.0})
    pane = next(c for c in cues if c["cue"] == "sfx_replay_pane")
    assert pane["dur_s"] == 1.0 and pane["t"] == pytest.approx(10.0 - 8 / am.FPS)


def test_a_long_close_cue_never_starts_before_the_recap():
    """Review finding: without a FINAL piece a long close cue gave a negative start."""
    tl = am.build_timeline(_plan([_entry("home:1", frames=300)]))
    cues = am.plan_cues(tl, duration_s=10.0, game_cues=True, cue_durations={"close_neutral": 20.0})
    assert min(c["t"] for c in cues) >= 0
    assert next(c for c in cues if c["cue"] == "sfx_final_card")["t"] == 0.0



def test_continued_sound_stays_in_its_own_camera_recording(tmp_path):
    """Review round 2: in a flat folder another camera's file can carry the next chapter name."""
    first = _chapter(tmp_path / "GX010001.MP4", 3000)
    _chapter(tmp_path / "GX020001.MP4", 6000)                 # cam2's file, same GoPro name
    am.set_recordings({"recordings": {"cam1": [[first]], "cam2": [[str(tmp_path / "GX020001.MP4")]]}})
    buf, missing = am.decode_continued({"file": first, "seek_s": .5, "duration_s": 1.0}, 1.0)
    assert missing == pytest.approx(.5, abs=.01)
    am.set_recordings(None)
    assert am.next_chapter(first) is None


@pytest.mark.parametrize("probed", [1.5, 0.0])
def test_a_short_decode_moves_to_the_next_chapter_at_its_start(tmp_path, monkeypatch, probed):
    """Review round 2: count decoded audio, not the request; an unknown length still continues."""
    first = _chapter(tmp_path / "GX010001.MP4", 3000)
    second = _chapter(tmp_path / "GX020001.MP4", 6000)
    am.set_recordings({"recordings": {"cam1": [[first, second]]}})
    monkeypatch.setattr(am, "_media_duration", lambda path: probed if path == first else 1.0)
    buf, missing = am.decode_continued({"file": first, "seek_s": .5, "duration_s": 1.0}, 1.0)
    assert missing == pytest.approx(0, abs=.01) and len(buf) == pytest.approx(am.SR, abs=480)
    # The first 0.5 s is chapter 1's end; chapter 2 follows from its own start.
    assert buf[am.SR // 4, 0] == pytest.approx(3000 / 32768, rel=1e-3)
    assert buf[3 * am.SR // 4, 0] == pytest.approx(6000 / 32768, rel=1e-3)


def test_live_replay_keeps_the_pictured_time_for_period_lookups():
    """Review round 2: a Replay's period follows its picture, not its continued sound."""
    clips = [_entry("home:1", start=90.0, end=100.0, moment=95.0, speed=1),
             _entry("away:1", start=200.0, end=210.0, moment=205.0, speed=1)]
    replay = _replay("home:1", 94.0, frames=180)
    replay.pop("moment_s")                                   # production Replays omit it
    tl = am.build_timeline(_plan(clips, [replay]), live_replay=True)
    assert tl[1]["moment_s"] == 94.0 and tl[1]["start_s"] == pytest.approx(100.0)
    periods = [{"n": 1, "start": 0.0, "end": 100.0}, {"n": 2, "start": 100.0, "end": 300.0}]
    segments = am.plan_bed(tl, periods, [], "game")
    assert segments[1]["boundary"] >= 16.0 - 1e-9


def test_no_pane_sound_where_a_replay_exits_into_final():
    """Review round 2: graphics draw no pane at the FINAL cut, so no pane sound plays there."""
    final = {"kind": "final", "goal_id": "close", "start_s": 0, "moment_s": 0, "speed": 1,
             "frames": 150, "parts": []}
    plan = _plan([_entry("home:1", frames=300)], [_replay("home:1", 4.0)])
    plan["closing"] = [final]
    tl = am.build_timeline(plan)
    panes = [c["t"] for c in am.plan_cues(tl, duration_s=17.0, game_cues=True)
             if c["cue"] == "sfx_replay_pane"]
    assert panes == pytest.approx([10.0 - 8 / am.FPS])
