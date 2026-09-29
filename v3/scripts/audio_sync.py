# v3/scripts/audio_sync.py
"""
Check the camera sync offset against the two cameras' rink audio (hhg-38a.12).

GoPro timecode is each camera's own clock and can be wrong by many seconds
(Ghost Pirates, 2025-11-09: 20 s apart while the timecodes were 1 frame apart).
Both cameras hear the same whistles, sticks and boards, so the offset can be
measured directly: spectral-flux onset envelopes (100 Hz) of 20 s windows from
cam2 are matched against cam1 with normalized cross-correlation. The offset is
trusted only when several windows agree.

Offset convention (same as gopro_meta.compute_sync): offset_s = cam2 start
minus cam1 start, so cam1_audio(t + offset_s) == cam2_audio(t).
"""
from __future__ import annotations

import subprocess

import numpy as np
from scipy import signal

SR = 8000                 # Hz, analysis sample rate
HOP = 80                  # samples -> 100 Hz onset envelope
ENV_SR = SR / HOP
LOAD_S = 700.0            # seconds of audio read from the start of each camera
QUERY_S = 20.0            # length of each cam2 window
N_WINDOWS = 5
MAX_OFFSET_S = 60.0       # matches gopro_meta's plausibility limit
AGREE_S = 0.1             # windows within this of the median agree
MIN_AGREE = 3             # windows that must agree for a confident offset
DISAGREE_S = 0.5          # timecode vs audio difference that switches to audio


def load_audio(path: str, seconds: float = LOAD_S) -> np.ndarray:
    """Decode the first `seconds` of mono audio at SR as float32."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-t", f"{seconds:.1f}", "-i", path,
           "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    if not raw:
        raise RuntimeError(f"No audio decoded from {path}")
    return np.frombuffer(raw, dtype=np.float32).copy()


def onset_envelope(x: np.ndarray) -> np.ndarray:
    """Spectral-flux onset envelope at 100 Hz, locally normalized (3 s window)."""
    f, _, z = signal.stft(x, SR, nperseg=512, noverlap=512 - HOP, boundary=None)
    mag = np.log1p(1000 * np.abs(z))
    flux = np.maximum(np.diff(mag, axis=1), 0)
    e = flux[(f >= 300) & (f <= 3500)].sum(0)
    k = int(3 * ENV_SR)
    mu = np.convolve(e, np.ones(k) / k, "same")
    sd = np.sqrt(np.convolve((e - mu) ** 2, np.ones(k) / k, "same")) + 1e-9
    return ((e - mu) / sd).astype(np.float32)


def _ncc(long: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Normalized cross-correlation of q against every position in long."""
    n = len(q)
    q = (q - q.mean()) / (q.std() + 1e-9)
    c = signal.fftconvolve(long, q[::-1], mode="valid")
    cs = np.concatenate([[0.0], np.cumsum(long, dtype=np.float64)])
    cs2 = np.concatenate([[0.0], np.cumsum(long.astype(np.float64) ** 2)])
    mu = (cs[n:] - cs[:-n]) / n
    var = np.maximum((cs2[n:] - cs2[:-n]) / n - mu ** 2, 1e-12)
    return c / (n * np.sqrt(var))


def measure_offset(cam1: np.ndarray, cam2: np.ndarray) -> dict:
    """Measure offset_s (cam2 start - cam1 start) from two audio arrays at SR."""
    e1, e2 = onset_envelope(cam1), onset_envelope(cam2)
    q = int(QUERY_S * ENV_SR)
    m = int(MAX_OFFSET_S * ENV_SR)
    lo, hi = m, min(len(e2), len(e1)) - m - q
    lags: list[float] = []
    if hi > lo:
        for t in np.linspace(lo, hi, N_WINDOWS).astype(int):
            region = e1[t - m: t + m + q]
            r = _ncc(region, e2[t: t + q])
            k = int(np.argmax(r))
            frac = 0.0
            if 0 < k < len(r) - 1:                    # parabolic peak interpolation
                a, b, c = r[k - 1], r[k], r[k + 1]
                den = a - 2 * b + c
                frac = 0.5 * (a - c) / den if den != 0 else 0.0
            lags.append((k + frac - m) / ENV_SR)
    if not lags:
        return {"offset_s": 0.0, "confident": False, "n_agree": 0, "n_windows": 0}
    lags_arr = np.array(lags)
    med = float(np.median(lags_arr))
    agree = lags_arr[np.abs(lags_arr - med) <= AGREE_S]
    return {"offset_s": round(float(agree.mean()), 3), "confident": len(agree) >= MIN_AGREE,
            "n_agree": int(len(agree)), "n_windows": len(lags)}


def _apply_offset(sync: dict, offset: float) -> None:
    sync["offset_s"] = offset
    sync["cam1_detect_offset_s"] = offset if offset > 0 else 0.0
    sync["cam2_detect_offset_s"] = -offset if offset < 0 else 0.0


def verify_sync_with_audio(sync: dict, cam1_first: str, cam2_first: str) -> dict:
    """
    Check a timecode-based sync dict against rink audio and return it updated.

    Confident audio that disagrees with timecode by more than DISAGREE_S wins
    (sync_method 'audio'); agreement gives 'timecode+audio'; an inconclusive or
    failed check keeps timecode and adds a warning for review.
    """
    sync = dict(sync)
    sync["warnings"] = list(sync.get("warnings", []))
    sync["timecode_offset_s"] = sync["offset_s"]
    try:
        r = measure_offset(load_audio(cam1_first), load_audio(cam2_first))
    except Exception as exc:  # no audio, unreadable file: keep timecode, flag it
        sync["warnings"].append(f"[WARN] Sync audio check failed ({exc}); using {sync['sync_method']}")
        return sync
    sync["audio_offset_s"] = r["offset_s"]
    sync["audio_agree"] = f"{r['n_agree']}/{r['n_windows']}"
    if not r["confident"]:
        sync["warnings"].append(
            f"[WARN] Sync audio check inconclusive ({sync['audio_agree']} windows agree); using {sync['sync_method']}")
        return sync
    diff = r["offset_s"] - sync["offset_s"]
    if abs(diff) > DISAGREE_S:
        sync["warnings"].append(
            f"[WARN] Timecode and audio disagree by {diff:+.2f}s; using audio offset {r['offset_s']:.3f}s")
        _apply_offset(sync, r["offset_s"])
        sync["sync_method"] = "audio"
    else:
        sync["sync_method"] = f"{sync['sync_method']}+audio"
    return sync
