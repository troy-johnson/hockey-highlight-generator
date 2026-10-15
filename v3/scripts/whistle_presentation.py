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
                  end: float) -> float | None:
    """Find a sustained whistle near an already bound event. Do not identify new events."""
    start = max(0, approximate - .3)
    stop = min(len(audio) / sample_rate, end, approximate + 1.5)
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


def confirm_presentations(timeline: list[dict], audio: np.ndarray, sample_rate: int) -> list[dict]:
    """Return the measured timing proof and update this mix's presentation schedule only."""
    records = []
    for index, entry in enumerate(timeline):
        if entry["kind"] != "goal" and not (entry["kind"] == "play" and entry.get("play_kind") == "penalty"):
            continue
        approximate = entry.get("confirmation_out")
        if approximate is None:
            continue
        end = entry["out_start"] + entry["out_dur"]
        for piece in timeline[index + 1:]:
            if piece.get("goal_id") != entry["goal_id"]:
                break
            end = piece["out_start"] + piece["out_dur"]
        onset = whistle_onset(audio, sample_rate, approximate, end)
        records.append({"goal_id": entry["goal_id"], "mapped_t": approximate, "onset_t": onset, "method": METHOD})
        entry["confirmation_out"] = onset
    return records


def verified_onset(audio_report: dict, goal_id: str, mapped: float | None) -> float | None:
    """Check that a mix's onset proof still binds to the same source whistle."""
    if "presentation_whistles" not in audio_report:
        return mapped
    records = [r for r in audio_report["presentation_whistles"] if r["goal_id"] == goal_id]
    if not records:
        return None
    if len(records) != 1 or mapped is None:
        raise ValueError("whistle presentation proof does not match the source event; rerun mix")
    record = records[0]
    onset = record["onset_t"]
    if (record["method"] != METHOD or not np.isfinite(record["mapped_t"]) or
            abs(record["mapped_t"] - mapped) > 1e-6 or
            onset is not None and (not np.isfinite(onset) or not mapped <= onset <= mapped + 1.5)):
        raise ValueError("whistle presentation proof is stale; rerun mix")
    return onset
