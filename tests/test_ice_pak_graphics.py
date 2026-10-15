"""Approved regional graphics: source timing, frozen data, and game sound."""
import copy
import json

import numpy as np
import pytest

import audio_mix as am
import recap_graphics as rg
from test_recap_graphics import game


def inputs():
    sheet, selection, plan, audio = game()
    selection["goals"][0]["chosen"] = {"candidate_id": "whistle:0"}
    selection["goals"][1]["chosen"] = {"candidate_id": "whistle:1"}
    selection["plays"] = [{"id": "penalty:home:1", "candidate_id": "whistle:2"}]
    signals = {"whistles": [{"t": 17.011}, {"t": 127}, {"t": 36.025}]}
    timeline = am.build_timeline(plan, selection=selection, audio_signals=signals)
    audio["cues"] = am.plan_cues(timeline, duration_s=40)
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
    selection["goals"][0]["chosen"] = {"candidate_id": "whistle:0"}
    signals = {"whistles": [{"t": 20.02}]}
    timeline = am.build_timeline(plan, selection=selection, audio_signals=signals)
    audio["cues"] = am.plan_cues(timeline, duration_s=40)
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
