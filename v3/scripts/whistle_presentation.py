"""Confirm presentation onset in kept audio, without changing Detection or Selection."""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.signal import resample_poly, stft

sys.path.insert(0, os.path.dirname(__file__))
from audio_signals import _runs, whistle_score

METHOD = "kept-audio-whistle-v1"


def whistle_onset(audio: np.ndarray, sample_rate: int, approximate: float,
                  end: float, *, start_limit: float = 0, look_after: float = 1.5) -> float | None:
    """Find a sustained whistle near an already bound event. Do not identify new events."""
    start = max(0, start_limit, approximate - .3)
    stop = min(len(audio) / sample_rate, end, approximate + look_after)
    if stop <= start:
        return None
    # The presentation check uses the existing 16 kHz whistle spectrum, not flow signals.
    step = sample_rate // 16000
    if step < 1 or sample_rate % 16000:
        raise ValueError("presentation audio must use a multiple of 16000 Hz")
    mono = resample_poly(audio[round(start * sample_rate):round(stop * sample_rate)].mean(axis=1), 1, step)
    if len(mono) < 512 or np.max(np.abs(mono)) < 1e-8:
        return None
    hz, times, spectrum = stft(mono, 16000, nperseg=512, noverlap=352, boundary=None, padded=False)
    power = np.abs(spectrum) ** 2
    score, _ = whistle_score(10 * np.log10(power + 1e-14), hz)
    energy = power[(hz >= 1800) & (hz <= 3000)].sum(axis=0)
    # Reject weak precursor noise. Require at least 80 ms of sustained band contrast.
    mask = (score > 6) & (energy > energy.max() * .2)
    for first, last in _runs(mask):
        if last - first < 8:
            continue
        # Use the window's right edge and a one-frame guard, never its leading edge.
        onset = max(approximate, start + float(times[first]) + .016 + 1 / 30)
        if onset < stop:
            return onset
    return None


def presentation_piece(timeline: list[dict], goal_id: str, when: float) -> dict | None:
    return next((p for p in timeline if p.get("goal_id") == goal_id and
                 p["out_start"] <= when < p["out_start"] + p["out_dur"]), None)


def source_binding(piece: dict) -> dict:
    """The exact source mapping that a saved onset proof covers."""
    return {"goal_id": piece["goal_id"], "kind": piece["kind"], "start_s": piece["start_s"],
            "end_s": piece["end_s"], "speed": piece.get("speed", 1), "frames": piece["frames"],
            "camera": piece.get("camera"),
            "out_start": piece["out_start"], "out_dur": piece["out_dur"],
            "parts": [{k: p[k] for k in ("file", "seek_s", "duration_s")} for p in piece.get("parts", [])]}


def confirm_presentations(timeline: list[dict], audio: np.ndarray, sample_rate: int) -> list[dict]:
    """Return the measured timing proof and update this mix's presentation schedule only."""
    records = []
    for index, entry in enumerate(timeline):
        if entry["kind"] != "goal" and not (entry["kind"] == "play" and entry.get("play_kind") == "penalty"):
            continue
        approximate = entry.get("confirmation_out")
        if approximate is None:
            continue
        mapped_piece = presentation_piece(timeline, entry["goal_id"], approximate)
        onset, piece = None, mapped_piece
        if mapped_piece:
            source_time = mapped_piece["start_s"] + (approximate - mapped_piece["out_start"]) * float(mapped_piece.get("speed") or 1)
            for retained in timeline[index:]:
                if retained.get("goal_id") != entry["goal_id"]:
                    break
                speed = float(retained.get("speed") or 1)
                first = max(source_time - .3, retained["start_s"])
                last = min(source_time + 1.5, retained["end_s"], retained["start_s"] + retained["out_dur"] * speed)
                if last <= first:
                    continue
                output_estimate = retained["out_start"] + (source_time - retained["start_s"]) / speed
                onset = whistle_onset(audio, sample_rate, output_estimate,
                                      retained["out_start"] + (last - retained["start_s"]) / speed,
                                      start_limit=max(approximate, retained["out_start"] + (first - retained["start_s"]) / speed),
                                      look_after=1.5 / speed)
                if onset is not None:
                    piece = retained
                    break
        records.append({"goal_id": entry["goal_id"], "mapped_t": approximate, "onset_t": onset, "method": METHOD,
                        "piece": source_binding(piece) if piece else None})
        entry["confirmation_out"] = onset
    return records


def verified_onset(audio_report: dict, goal_id: str, mapped: float | None,
                   *, timeline: list[dict] | None = None) -> float | None:
    """Check that a mix's onset proof still binds to the same source whistle."""
    if "presentation_whistles" not in audio_report:
        return None
    records = [r for r in audio_report["presentation_whistles"] if r["goal_id"] == goal_id]
    if not records:
        return None
    if len(records) != 1 or mapped is None:
        raise ValueError("whistle presentation proof does not match the source event; rerun mix")
    record = records[0]
    onset = record["onset_t"]
    mapped_piece = presentation_piece(timeline or [], goal_id, mapped)
    piece = presentation_piece(timeline or [], goal_id, onset) if onset is not None else mapped_piece
    source_time = (mapped_piece["start_s"] + (mapped - mapped_piece["out_start"]) * float(mapped_piece.get("speed") or 1)
                   if mapped_piece else None)
    onset_source = (piece["start_s"] + (onset - piece["out_start"]) * float(piece.get("speed") or 1)
                    if piece and onset is not None else None)
    if (record["method"] != METHOD or not np.isfinite(record["mapped_t"]) or
            abs(record["mapped_t"] - mapped) > 1e-6 or
            mapped_piece is None or piece is None or record.get("piece") != source_binding(piece) or
            onset is not None and (not np.isfinite(onset) or onset < mapped or
                                   not source_time <= onset_source <= source_time + 1.5 or
                                   onset >= piece["out_start"] + piece["out_dur"])):
        raise ValueError("whistle presentation proof is stale; rerun mix")
    return onset
