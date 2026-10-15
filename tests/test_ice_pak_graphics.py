"""Approved regional graphics: source timing, frozen data, and game sound."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

import audio_mix as am
import recap_graphics as rg
from test_recap_graphics import acoustic_proofs, game


def inputs():
    sheet, selection, plan, audio = game()
    selection["goals"][0]["chosen"] = {"candidate_id": "whistle:0"}
    selection["goals"][1]["chosen"] = {"candidate_id": "whistle:1"}
    selection["plays"] = [{"id": "penalty:home:1", "candidate_id": "whistle:2"}]
    signals = {"whistles": [{"t": 17.011}, {"t": 127}, {"t": 36.025}]}
    timeline = am.build_timeline(plan, selection=selection, audio_signals=signals)
    audio["cues"] = am.plan_cues(timeline, duration_s=40)
    audio["presentation_whistles"] = acoustic_proofs(plan, selection, signals)
    return sheet, selection, plan, audio, signals


def test_bars_cards_and_score_never_precede_fractional_whistle(tmp_path):
    sheet, selection, plan, audio, signals = inputs()
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {},
                           coverage={"periods": selection["periods"]}, audio_signals=signals)
    bar = next(e for e in props["events"] if e["kind"] == "event_bar")
    goal = next(e for e in props["events"] if e["kind"] == "goal")
    assert bar["startFrame"] == goal["startFrame"] == 211
    bugs = [e for e in props["events"] if e["kind"] == "scorebug"]
    assert bugs[0]["data"]["score"] == [0, 0]
    assert bugs[0]["durationFrames"] == 211
    assert bugs[1]["startFrame"] == 211 and bugs[1]["data"]["score"] == [1, 0]
    assert plan == game()[2]  # Graphics must not change cuts.


def test_penalty_waits_for_its_matched_whistle():
    _, _, _, audio, _ = inputs()
    cue = next(c for c in audio["cues"] if c["cue"] == "sfx_penalty_card")
    assert cue["t"] == pytest.approx(26.025)


@pytest.mark.parametrize("bad_time", [0, 5, 7.010])
def test_graphics_reject_stale_mix_before_the_whistle(tmp_path, bad_time):
    sheet, selection, plan, audio, signals = inputs()
    next(c for c in audio["cues"] if c["cue"] == "sfx_goal_card")["t"] = bad_time
    with pytest.raises(ValueError, match="whistle; rerun mix"):
        rg.write_props(tmp_path, sheet, selection, plan, audio, {}, audio_signals=signals)


def test_periods_use_coverage_not_stale_selection(tmp_path):
    sheet, selection, plan, audio, signals = inputs()
    coverage = {"periods": [{"n": 3, "start": 0, "end": 200}]}
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {}, coverage=coverage, audio_signals=signals)
    assert {e["data"]["period"] for e in props["events"] if e["kind"] == "scorebug"} == {3}


def test_no_whistle_does_not_create_announcements(tmp_path):
    sheet, selection, plan, _, _ = inputs()
    timeline = am.build_timeline(plan, selection=selection, audio_signals={"whistles": []})
    audio = {**game()[3], "cues": am.plan_cues(timeline, duration_s=40)}
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {}, audio_signals={"whistles": []})
    assert not any(e["kind"] in ("goal", "penalty", "event_bar") for e in props["events"])
    bugs = [e for e in props["events"] if e["kind"] == "scorebug"]
    assert bugs[0]["data"]["score"] == bugs[1]["data"]["score"] == [0, 0]


def test_whistle_just_past_live_cut_keeps_old_score_until_next_frame(tmp_path):
    sheet, selection, plan, audio = game()
    plan["clips"][0]["moment_s"] = 17
    plan["replays"][0].update(start_s=20, end_s=30)
    selection["goals"][0]["chosen"] = {"candidate_id": "whistle:0"}
    signals = {"whistles": [{"t": 20.02}]}
    timeline = am.build_timeline(plan, selection=selection, audio_signals=signals)
    audio["cues"] = am.plan_cues(timeline, duration_s=40)
    audio["presentation_whistles"] = acoustic_proofs(plan, selection, signals)
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {}, audio_signals=signals)
    bugs = [e for e in props["events"] if e["kind"] == "scorebug"]
    assert [(e["startFrame"], e["durationFrames"], e["data"]["score"]) for e in bugs[:3]] == [
        (0, 300, [0, 0]), (300, 1, [0, 0]), (301, 299, [1, 0])]


def test_frozen_final_keeps_pim_and_row_order(tmp_path, freeze_sheet):
    from game_data import graphics_inputs
    sheet, _, _, _ = game()
    sheet["penalties"]["home"] = [{"player": "8", "classification": "major", "minutes": 5},
                                    {"player": "8", "classification": "game misconduct", "minutes": 10}]
    data = freeze_sheet(sheet)
    frozen, options = graphics_inputs(data, {"teams": {"home": {"roster": {"8": "Changed name"}}}})
    final = rg.final_data(frozen, [])
    expected = [{"num": r["jersey_label"], **{k: r[k] for k in ("name", "g", "a", "pts", "pim")}}
                for r in data["stats"]["home"]]
    assert final["table"]["home"]["skaters"] == expected
    assert expected[0]["pim"] == 15
    assert expected[0]["name"] != "Changed name"


def test_frozen_final_does_not_drop_rows():
    rows = [{"num": str(i), "name": f"Player {i}", "g": 1, "a": 0, "pts": 1, "pim": 0} for i in range(17)]
    sheet = {"_final": {"score": [17, 0], "table": {"home": {"skaters": rows, "goalies": []}}}}
    assert rg.final_data(sheet, [])["table"]["home"]["skaters"] == rows


def test_replay_game_sound_is_eight_db_lower_and_fades_edges(monkeypatch):
    monkeypatch.setattr(am, "_entry_audio", lambda *_: np.ones((am.SR, 2), dtype=np.float32))
    entry = {"kind": "tight", "out_dur": 1.0}
    buf = am.game_sound_entry(entry, [])
    assert buf.shape == (am.SR, 2)
    assert buf[am.SR // 2, 0] == pytest.approx(10 ** (-8 / 20))
    assert np.max(np.abs(buf[[0, -1]])) == 0


def test_normalize_game_sound_targets_minus_twenty_lufs(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(am, "_measure_loudness", lambda *a, **k: calls.append(k) or {"input_i": -float("inf")})
    buf = np.zeros((480, 2), dtype=np.float32)
    assert am.normalize_game_sound(buf) is buf
    assert calls == [{"raw_input": True, "tp": -2.0, "target_i": -20.0}]


def test_white_score_flash_uses_dark_ink():
    module = Path(rg.__file__).parents[1] / "graphics/src/score_flash.mjs"
    result = subprocess.run(["node", "--input-type=module", "-e", f"""
      import {{scoreFlash}} from {json.dumps(module.as_uri())};
      console.log(JSON.stringify([scoreFlash('#FFFFFF', 1), scoreFlash('#FFFFFF', 0)]));
    """], capture_output=True, text=True, check=True)
    white, graphite = json.loads(result.stdout)
    assert white == {"background": "rgb(255,255,255)", "color": "#0E1319"}
    assert graphite["color"] == "#EEF3F6"


def test_final_sound_continues_last_source_and_fades(monkeypatch):
    seen = []
    def sound(entry, windows):
        seen.append((entry, windows))
        return np.ones((am.SR * 5, 2), dtype=np.float32)
    monkeypatch.setattr(am, "game_sound_entry", sound)
    previous = {"start_s": 10, "kind": "play", "camera": "cam1", "speed": 1.1,
                "parts": [{"file": "game.mp4", "seek_s": 100, "duration_s": 4}]}
    buf = am.closing_game_sound({"out_dur": 5}, previous, [])
    assert seen[0][0]["parts"][0]["seek_s"] == 104
    assert seen[0][0]["parts"][0]["duration_s"] == 5.5
    assert buf[am.SR, 0] == 1
    assert buf[-1, 0] == 0


def test_presentation_onset_waits_for_whistle_in_kept_camera_audio():
    from whistle_presentation import whistle_onset
    from scipy.signal import butter, sosfilt
    rng = np.random.default_rng(42)
    sr = am.SR
    mono = rng.normal(0, .0001, sr * 3)
    start, end = round(1.3 * sr), round(1.8 * sr)
    tone = sosfilt(butter(10, [2100, 2350], fs=sr, btype="bandpass", output="sos"),
                   rng.normal(0, 2, end - start))
    mono[start:end] += tone
    onset = whistle_onset(np.column_stack([mono, mono]), sr, 1.0, 3.0)
    assert 1.3 <= onset < 1.5


def test_presentation_does_not_invent_a_whistle_from_silence():
    from whistle_presentation import whistle_onset
    assert whistle_onset(np.zeros((am.SR * 2, 2)), am.SR, 1, 2) is None


def test_graphics_use_kept_audio_onset_proof_not_merged_camera_start(tmp_path):
    from whistle_presentation import METHOD
    sheet, selection, plan, audio, signals = inputs()
    for cue in audio["cues"]:
        if cue["cue"] == "sfx_goal_card" and cue["goal_id"] == "home:1":
            cue["t"] += .3
    record = next(r for r in audio["presentation_whistles"] if r["goal_id"] == "home:1")
    record["onset_t"] = 7.311
    # Other event cues have no acoustic proof and must not be accepted.
    audio["cues"] = [c for c in audio["cues"] if not c["cue"].startswith("sfx_") or c.get("goal_id") == "home:1"]
    props = rg.write_props(tmp_path, sheet, selection, plan, audio, {}, audio_signals=signals)
    bug = next(e for e in props["events"] if e["kind"] == "scorebug" and e["data"]["score"] == [1, 0])
    assert bug["startFrame"] == 220


@pytest.mark.parametrize("record", [
    {"goal_id": "home:1", "mapped_t": 7, "onset_t": 7.3, "method": "unknown"},
    {"goal_id": "home:1", "mapped_t": 6, "onset_t": 7.3, "method": "kept-audio-whistle-v1"},
    {"goal_id": "home:1", "mapped_t": 7, "onset_t": 6.9, "method": "kept-audio-whistle-v1"},
])
def test_stale_or_early_acoustic_proof_is_rejected(record):
    from whistle_presentation import verified_onset
    with pytest.raises(ValueError, match="rerun mix"):
        verified_onset({"presentation_whistles": [record]}, "home:1", 7)


def test_missing_acoustic_proof_cannot_restore_merged_camera_timing():
    from whistle_presentation import verified_onset
    assert verified_onset({}, "home:1", 7.011) is None


def test_acoustic_search_stops_at_the_exact_source_cut(monkeypatch):
    import whistle_presentation as wp
    called = []
    def onset(audio, sample_rate, approximate, end, *, start_limit=0, look_after=1.5):
        called.append((approximate, end, start_limit))
        return None
    monkeypatch.setattr(wp, "whistle_onset", onset)
    live = {"goal_id": "home:1", "kind": "goal", "start_s": 10, "end_s": 11,
            "speed": 1, "frames": 30, "out_start": 0, "out_dur": 1, "confirmation_out": .9, "parts": []}
    replay = {**live, "kind": "tight", "start_s": 3, "end_s": 4, "out_start": 1}
    wp.confirm_presentations([live, replay], np.zeros((am.SR * 2, 2)), am.SR)
    assert len(called) == 1
    assert called[0] == pytest.approx((.9, 1, .9))
    assert live["confirmation_out"] is None


def test_saved_whistle_proof_must_match_source_mapping(tmp_path):
    sheet, selection, plan, audio, signals = inputs()
    audio["presentation_whistles"][0]["piece"]["start_s"] += .5
    with pytest.raises(ValueError, match="proof is stale"):
        rg.write_props(tmp_path, sheet, selection, plan, audio, {}, audio_signals=signals)
