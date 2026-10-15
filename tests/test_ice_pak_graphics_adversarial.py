"""Adversarial production contracts. All game data and assets stay in tmp_path."""

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import audio_mix as am
import recap_graphics as rg
import short_clips as short
from game_data import graphics_inputs


@pytest.fixture
def frozen_game(tmp_path, freeze_sheet):
    sheet = {
        "date": "2026-10-11",
        "teams": {"home": "Ice Pak", "away": "Visitors"},
        "rosters": {"home": {"17": "Alexandria Long-Surname-Skaterson"}, "away": {}},
        "goals": {"home": [{"per": "1", "time": "10:00", "scorer": "17", "assists": []}], "away": []},
        "penalties": {"home": [], "away": []},
    }
    data = freeze_sheet(sheet)
    sheet, options = graphics_inputs(data, {"_layers": {"config_dir": str(tmp_path)}})
    selection = {"schema_version": 2, "periods": [{"n": 1, "start": 0, "end": 100}],
                 "goals": [{"id": "home:1", "period": 1, "elapsed_s": 20,
                            "chosen": {"candidate_id": "whistle:0", "detection_s": 17}}], "plays": []}
    live = {"kind": "goal", "goal_id": "home:1", "start_s": 10, "end_s": 20,
            "moment_s": 17, "speed": 1, "frames": 300, "parts": []}
    replay = dict(live, kind="tight", start_s=14, end_s=24)
    closing = {"kind": "final", "goal_id": "close", "start_s": 0, "moment_s": 0,
               "speed": 1, "frames": 150, "parts": []}
    plan = {"schema_version": 3, "fps": 30, "output": "synthetic_Recap.mp4",
            "clips": [live], "replays": [replay], "closing": [closing], "duration_s": 25}
    return sheet, options, selection, plan


def props_for(tmp_path, fixture, whistle, *, cue_delay=0):
    sheet, options, selection, plan = fixture
    signals = {"whistles": [{"t": whistle}]}
    timeline = am.build_timeline(plan, selection=selection, audio_signals=signals)
    cues = am.plan_cues(timeline, duration_s=plan["duration_s"])
    for cue in cues:
        if cue["cue"] == "sfx_goal_card":
            cue["t"] += cue_delay
    audio = {"output": "synthetic_Recap_audio.mp4", "video": plan["output"],
              "duration_s": plan["duration_s"], "cues": cues}
    from test_recap_graphics import acoustic_proofs
    audio["presentation_whistles"] = acoustic_proofs(plan, selection, signals)
    return rg.write_props(tmp_path, sheet, selection, plan, audio, options, publish=False,
                          coverage={"periods": selection["periods"]}, audio_signals=signals)


@pytest.mark.parametrize("speed", [1, .5])
def test_whistle_past_live_cut_maps_through_replay_seek_and_speed(tmp_path, frozen_game, speed):
    """The live cut excludes the whistle; Replay first contains it much later."""
    plan = frozen_game[3]
    replay = plan["replays"][0]
    replay.update(speed=speed, frames=round((replay["end_s"] - replay["start_s"]) / speed * 30))
    plan["duration_s"] = sum(e["frames"] for e in plan["clips"] + plan["replays"] + plan["closing"]) / 30
    before = copy.deepcopy(plan)
    whistle = 20.02
    props = props_for(tmp_path, frozen_game, whistle)
    first_audible_whistle = 300 + math.ceil((whistle - replay["start_s"]) / speed * 30 - 1e-8)
    changed = next(e for e in props["events"] if e["kind"] == "scorebug" and e["data"]["score"] == [1, 0])
    assert changed["startFrame"] == first_audible_whistle, (
        "The score uses the expired live seek instead of the Replay seek and speed", changed, first_audible_whistle)
    assert all(e["startFrame"] >= first_audible_whistle for e in props["events"]
               if e["kind"] in ("goal", "event_bar"))
    assert plan == before


def test_score_uses_measured_whistle_not_a_late_cached_graphics_cue(tmp_path, frozen_game):
    """A valid but late mix cue must not replace the measured whistle origin."""
    whistle = 17.011
    props = props_for(tmp_path, frozen_game, whistle, cue_delay=.5)
    expected = math.ceil((whistle - 10) * 30)
    changed = next(e for e in props["events"] if e["kind"] == "scorebug" and e["data"]["score"] == [1, 0])
    assert changed["startFrame"] == expected
    assert next(e for e in props["events"] if e["kind"] == "event_bar")["startFrame"] == expected


def test_short_live_score_does_not_announce_at_the_flow_moment(tmp_path):
    """The Short path must use the whistle rule, not the old flow-moment rule."""
    entry = {"kind": "goal", "start_s": 10, "moment_s": 15, "speed": 1,
             "score_before": [0, 0], "score": [1, 0], "confirmation_out": 7.011}
    assert short.score_at(entry, 5) == [0, 0], "Short changes its score two seconds before the bound whistle"
    assert short.score_at(entry, 7) == [0, 0]
    assert short.score_at(entry, 211 / 30) == [1, 0]


@pytest.mark.parametrize("kind,caption", [("goal", "GOAL Alexandria Long-Surname-Skaterson"),
                                         ("penalty", "PENALTY Alexandria Long-Surname-Skaterson")])
def test_short_caption_is_absent_before_the_whistle(tmp_path, monkeypatch, kind, caption):
    """An event caption cannot appear at the start of its source play."""
    labels = []
    monkeypatch.setattr(short, "_text", lambda frame, text, *args, **kwargs: labels.append(str(text)))
    entry = {"kind": kind, "start_s": 10, "moment_s": 15, "speed": 1,
             "caption": caption, "score_before": [0, 0], "score": [1, 0], "confirmation_out": 7.011}
    short._overlay(np.zeros((1920, 1080, 3), dtype=np.uint8), entry,
                   {"teams": {"home": "Ice Pak", "away": "Visitors"}}, elapsed_s=0)
    assert not any(label.startswith(("GOAL ", "PENALTY ")) for label in labels), labels


def test_short_final_preserves_the_complete_frozen_surname(tmp_path, monkeypatch, frozen_game):
    labels = []
    monkeypatch.setattr(short, "_text", lambda frame, text, *args, **kwargs: labels.append(str(text)))
    sheet = frozen_game[0]
    final = rg.final_data(sheet, [])
    name = final["table"]["home"]["skaters"][0]["name"]
    short._overlay(np.zeros((1920, 1080, 3), dtype=np.uint8), dict(final, kind="final"),
                   {"teams": sheet["teams"]})
    assert any(name in label for label in labels), (name, labels)


def test_game_sound_keeps_mastered_opening_gain_when_output_peak_is_safe(tmp_path, monkeypatch, frozen_game):
    """The broadcast -14 LUFS correction must not attenuate mastered game sound."""
    sheet, options, selection, plan = frozen_game
    opening_path = tmp_path / "mastered_opening.mp4"
    opening_path.touch()
    plan["opening"] = [{"kind": "stinger", "goal_id": "open", "start_s": 0, "moment_s": 0,
                        "speed": 1, "frames": 139, "parts": [],
                        "opening_video": {"path": str(opening_path), "frames": 148}}]
    plan["duration_s"] += 139 / 30
    for name, value in {"recap_assembly.json": plan, "selection.json": selection,
                        "coverage.json": {"periods": selection["periods"]},
                        "music_spans.json": {"spans": []},
                        "audio_signals.json": {"whistles": [{"t": 17.011}]}}.items():
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / plan["output"]).touch()
    options["audio"] = {"manifest": str(tmp_path / "unused-manifest.json")}
    monkeypatch.setattr(am, "game_sound_entry", lambda e, w: np.zeros((round(e["out_dur"] * am.SR), 2), dtype=np.float32))
    monkeypatch.setattr(am, "normalize_game_sound", lambda buf: buf)
    monkeypatch.setattr(am, "decode_audio", lambda path, seek, duration, speed=1: np.full((round(duration * am.SR), 2), .2, dtype=np.float32))
    monkeypatch.setattr(am, "_probe_duration", lambda path: plan["duration_s"])
    encode_filters = []

    def measured_loudness(path, *, raw_input, **kwargs):
        # Model the measured effect of the unwanted broadcast gain correction.
        gain = -4.2 * encode_filters[-1].count("volume=") if encode_filters and not raw_input else 0
        return {"input_i": -10 + gain, "input_tp": -2 + gain,
                "input_lra": 7, "input_thresh": -30, "target_offset": 0}

    def subprocess_run(argv, **kwargs):
        if "-c:a" in argv:
            encode_filters.append(argv[argv.index("-af") + 1])
            Path(argv[-1]).touch()
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(am.subprocess, "run", subprocess_run)
    monkeypatch.setattr(am, "_measure_loudness", measured_loudness)
    assert am.main([str(tmp_path), "--options", json.dumps(options)]) == 0
    report = json.loads((tmp_path / "recap_audio.json").read_text())
    assert report["mode"] == "game_sound"
    assert len(encode_filters) == 1, encode_filters
    assert "volume=" not in encode_filters[0]


def _adversarial_whistle_audio(duration, onset, sr=am.SR):
    """Keep synthetic whistle samples entirely in memory."""
    mono = np.zeros(round(duration * sr), dtype=np.float32)
    if onset is not None:
        first = round(onset * sr)
        length = min(round(.6 * sr), len(mono) - first)
        mono[first:first + length] = .2 * np.sin(2 * np.pi * 2200 * np.arange(length) / sr)
    return np.column_stack([mono, mono])


@pytest.mark.parametrize("kept_onset", [None, 7.35], ids=["silent-camera", "later-kept-camera-whistle"])
def test_short_requires_kept_audio_confirmation_not_only_merged_timestamp(
        tmp_path, monkeypatch, frozen_game, kept_onset):
    """Short must not retain the merged timestamp that Recap now rejects."""
    from whistle_presentation import whistle_onset

    sheet, _, selection, original = frozen_game
    live = dict(original["clips"][0], camera="cam1", caption="GOAL " + rg.goal_data(sheet, "home:1")["scorer"],
                score_before=[0, 0], score=[1, 0],
                parts=[{"file": str(tmp_path / "camera.mp4"), "seek_s": 10, "duration_s": 10}])
    replay = dict(original["replays"][0], kind="replay", camera="cam1", score_before=[0, 0], score=[1, 0],
                  parts=[{"file": str(tmp_path / "camera.mp4"), "seek_s": 14, "duration_s": 10}])
    plan = {"opening": [], "clips": [live], "replays": [replay], "flags": [], "duration_s": 25,
            "teams": sheet["teams"], "final": dict(rg.final_data(sheet, []), kind="final", frames=150)}
    source = _adversarial_whistle_audio(10, kept_onset)
    measured = whistle_onset(source, am.SR, 7.011, 10)
    assert measured is None if kept_onset is None else measured >= kept_onset
    short.bind_short_whistles(plan, selection, {"whistles": [{"t": 17.011}]})
    monkeypatch.setattr(am, "_entry_audio", lambda entry, windows: source.copy() if entry["kind"] == "goal"
                        else np.zeros((round(entry["frames"] / 30 * am.SR), 2), dtype=np.float32))
    short.prepare_audio(plan, None, [])
    assert short.score_at(live, 211 / 30) == [0, 0], (
        "Short announces at the unverified merged time", live["confirmation_out"], measured)
    if kept_onset is None:
        assert short.score_at(replay, 0) == [0, 0]


def test_graphics_reject_acoustic_proof_from_another_kept_camera(tmp_path, frozen_game):
    """Equal output timestamps do not prove that the same source audio was checked."""
    from whistle_presentation import confirm_presentations

    sheet, options, selection, plan = frozen_game
    signals = {"whistles": [{"t": 17.011, "cameras": ["cam1", "cam2"]}]}
    for entry in plan["clips"] + plan["replays"]:
        entry.update(camera="cam1", parts=[{"file": str(tmp_path / "cam1.mp4"),
                                          "seek_s": entry["start_s"], "duration_s": 10}])
    timeline = am.build_timeline(plan, selection=selection, audio_signals=signals)
    proof = confirm_presentations(timeline, _adversarial_whistle_audio(25, 7.35), am.SR)
    assert proof[0]["onset_t"] > 7.35
    audio = {"output": "synthetic_Recap_audio.mp4", "video": plan["output"], "duration_s": 25,
             "presentation_whistles": proof, "cues": am.plan_cues(timeline, duration_s=25)}
    # A different camera can have a later audible whistle at the same merged event time.
    for entry in plan["clips"] + plan["replays"]:
        entry["camera"] = "cam2"
        entry["parts"][0]["file"] = str(tmp_path / "cam2.mp4")
    with pytest.raises(ValueError, match="proof|rerun mix"):
        rg.write_props(tmp_path, sheet, selection, plan, audio, options, publish=False,
                       coverage={"periods": selection["periods"]}, audio_signals=signals)


def test_production_graphics_cannot_announce_without_acoustic_proof(tmp_path, monkeypatch, frozen_game):
    """An older mix report must not silently bypass the new kept-audio requirement."""
    _, options, selection, plan = frozen_game
    root = tmp_path / "game"
    root.mkdir()
    (root / "game_data.json").write_text((tmp_path / "game_data.json").read_text())
    signals = {"whistles": [{"t": 17.011}]}
    timeline = am.build_timeline(plan, selection=selection, audio_signals=signals)
    audio = {"output": "synthetic_Recap_audio.mp4", "video": plan["output"], "duration_s": 25,
             "cues": am.plan_cues(timeline, duration_s=25)}
    for name, value in {"selection.json": selection, "recap_assembly.json": plan,
                        "recap_audio.json": audio, "coverage.json": {"periods": selection["periods"]},
                        "audio_signals.json": signals}.items():
        (root / name).write_text(json.dumps(value))
    monkeypatch.setattr(rg, "probe_video", lambda *args: {"width": 1920, "height": 1080})
    result = rg.main([str(root), "--options", json.dumps(options), "--props-only",
                      "--workspace", str(tmp_path / "workspace")])
    if result:
        assert result == 1
        assert not (root / rg.PROPS_FILE).exists()
    else:
        props = json.loads((root / rg.PROPS_FILE).read_text())
        assert not any(e["kind"] in ("goal", "penalty", "event_bar") for e in props["events"]), (
            "Production graphics announced a whistle without acoustic proof", props["events"])


def test_acoustic_delay_past_live_cut_is_checked_in_the_bound_replay(tmp_path, frozen_game):
    """A small source-time delay can become a large output delay across a Replay cut."""
    from whistle_presentation import confirm_presentations

    _, _, selection, plan = frozen_game
    plan["clips"][0].update(end_s=17.1, frames=213)
    plan["replays"][0].update(start_s=14, end_s=20, speed=.5, frames=360)
    plan["duration_s"] = (213 + 360 + 150) / 30
    timeline = am.build_timeline(plan, selection=selection, audio_signals={"whistles": [{"t": 17.0}]})
    assert timeline[0]["confirmation_out"] == 7
    # The kept camera's whistle is 0.3 seconds later, outside live but inside this Replay.
    audible = 213 / 30 + (17.3 - 14) / .5
    proof = confirm_presentations(timeline, _adversarial_whistle_audio(plan["duration_s"], audible), am.SR)
    onset = proof[0]["onset_t"]
    assert onset is not None and audible <= onset < audible + .2, (
        "The output-only search missed the bound whistle retained by Replay", proof, audible)
