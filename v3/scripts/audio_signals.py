# v3/scripts/audio_signals.py
"""
Whistles, stoppages and PA music spans from rink audio (hhg-3r5.29).

    python audio_signals.py <game_folder> [--no-music]

Reads cam1_concat.txt / cam2_concat.txt and writes, in the Game Folder:

  audio_signals.json   whistles and stoppages
  music_spans.json     PA music spans (YAMNet), one list for both cameras

Timeline: all times are seconds on the detection timeline, the same time base
as events.csv and markers.csv. That is cam1 concat time; cam2 is moved by the
sync offset with the manifests' '# seek' and '# recording' comments, as in
v2/scripts/signals.py.

Whistle (measured on wild_03012026, see docs/research/whistle-detection.md):
a referee whistle is a flat band about 250 Hz wide between 2.0 and 2.5 kHz,
0.3 to 2.5 s long. Per 10-ms frame, the score is the mean level of the best
250-Hz band in 1.8-3.0 kHz minus the louder of its two flank bands (dB).
Runs above 6 dB that peak at 12 dB or more with a steady frequency are
whistles. A scoreboard horn has a harmonic stack and an unsteady best band.

Stoppage: rink sound (skates, sticks, puck) stops when play stops and comes
back at the faceoff. Audio activity is the positive spectral flux in
0.5-3.5 kHz, averaged over 3 s and ranked within the game (0 = quietest).
A whistle starts a stoppage when the activity of both cameras stays low just
after it; the stoppage ends when the activity rises again.

PA music: YAMNet (TFLite, optional ML stack) class 132 "Music", with the rule
from docs/research/pa-music-detection.md (hhg-3r5.18).

Exit codes: 0 = written; 1 = no camera audio could be read.
Music problems (ML stack missing, model not available) are flags in
music_spans.json; whistles and stoppages are still written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "v2", "scripts"))

TIMELINE = "detection (cam1 concat time; cam2 moved by the sync offset)"

# Feature extraction ---------------------------------------------------------
SR = 16000                  # decode rate (Hz); whistles are below 3 kHz
N_FFT = 512                 # 31.25 Hz bins
HOP = 160                   # 10 ms
RATE = SR / HOP             # feature frames per second (100)
CHUNK_S = 30                # seconds of audio read from ffmpeg at a time
FEATURES_VERSION = 1        # change when the cached features change

WHISTLE_BAND_HZ = (1800.0, 3000.0)
BAND_W_HZ = 250.0
FLUX_BAND_HZ = (500.0, 3500.0)

# YAMNet (MediaPipe float32 export): 0.975 s window, 0.48 s hop, 16 kHz
YAMNET_URL = "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite"
YAMNET_SHA256 = "4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf"
YAMNET_ENV = "HHG_YAMNET_MODEL"
YAMNET_WIN = 15600
YAMNET_HOP = 7680
YAMNET_HOP_S = YAMNET_HOP / SR
YAMNET_MUSIC = 132
CACHE_HOME = Path.home() / ".cache" / "hockey-highlight-generator"

# Whistle rule ---------------------------------------------------------------
# A run is kept when it is steady (hz_sd <= steady_sd_hz) or strong
# (peak >= strong_db), and always within peak_db, max_sd_hz and hz_range.
# Tuned by ear on wild_03012026 (docs/research/whistle-detection.md).
WHISTLE = dict(on_db=6.0, min_s=0.3, gap_s=0.3, peak_db=10.0, max_sd_hz=300.0,
               steady_sd_hz=150.0, strong_db=12.0,
               hz_range=(1900.0, 2700.0), smooth_frames=9, merge_s=0.5)

# Stoppage rule --------------------------------------------------------------
STOPPAGE = dict(smooth_s=3.0, look_from_s=1.0, look_to_s=10.0, low=0.35, high=0.60,
                restart_s=2.0, max_s=120.0)

# Music rule (hhg-3r5.18) ----------------------------------------------------
MUSIC = dict(start=0.15, end=0.075, min_s=3.0, pad_before_s=1.0, pad_after_s=2.0,
             review_low=0.05, review_min_s=3.0, median_s=3)


# ---------------------------------------------------------------------------
# Pure logic: whistles
# ---------------------------------------------------------------------------

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(d[::2].tolist(), d[1::2].tolist()))


def whistle_score(power_db: np.ndarray, freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Band contrast per frame. power_db: (bins, frames). Returns (score dB, Hz of
    the best band) per frame. The score is the mean of a 250-Hz band minus the
    louder of the two bands next to it.
    """
    df = float(freqs[1] - freqs[0])
    w = max(1, int(round(BAND_W_HZ / df)))
    band = uniform_filter1d(power_db, w, axis=0, mode="nearest")
    lo = np.roll(band, w + 2, axis=0)
    hi = np.roll(band, -(w + 2), axis=0)
    contrast = band - np.maximum(lo, hi)
    sel = np.flatnonzero((freqs >= WHISTLE_BAND_HZ[0]) & (freqs <= WHISTLE_BAND_HZ[1]))
    sub = contrast[sel]
    k = sub.argmax(axis=0)
    cols = np.arange(sub.shape[1])
    return sub[k, cols].astype(np.float32), freqs[sel][k].astype(np.float32)


def find_whistles(score: np.ndarray, hz: np.ndarray, rate: float = RATE, t0: float = 0.0,
                  rule: dict | None = None) -> list[dict]:
    """
    Whistles from one camera's score and best-band frequency.
    Returns dicts: start, end, peak_db, hz, hz_sd (times in s from t0).
    """
    r = dict(WHISTLE, **(rule or {}))
    score = np.nan_to_num(np.asarray(score, np.float32), nan=0.0)
    if len(score) == 0:
        return []
    s = median_filter(score, size=r["smooth_frames"], mode="nearest")
    runs = _runs(s > r["on_db"])
    merged: list[list[int]] = []
    gap = int(round(r["gap_s"] * rate))
    for a, b in runs:
        if merged and a - merged[-1][1] < gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        if (b - a) / rate < r["min_s"]:
            continue
        seg_hz = hz[a:b][s[a:b] > r["on_db"]]
        peak = float(s[a:b].max())
        f_med = float(np.median(seg_hz))
        f_sd = float(np.std(seg_hz))
        if peak < r["peak_db"] or f_sd > r["max_sd_hz"] or not (r["hz_range"][0] <= f_med <= r["hz_range"][1]):
            continue
        if f_sd > r["steady_sd_hz"] and peak < r["strong_db"]:
            continue
        out.append({"start": round(t0 + a / rate, 2), "end": round(t0 + b / rate, 2),
                    "peak_db": round(peak, 1), "hz": round(f_med), "hz_sd": round(f_sd)})
    return out


def merge_camera_whistles(per_cam: dict[str, list[dict]], merge_s: float = WHISTLE["merge_s"]) -> list[dict]:
    """
    One list of whistles for the game. Whistles of different cameras whose
    times overlap or are within merge_s are one whistle. Single-camera
    whistles are kept. Each whistle lists its cameras.
    """
    items = sorted(((w["start"], w["end"], cam, w) for cam, ws in per_cam.items() for w in ws))
    out: list[dict] = []
    for start, end, cam, w in items:
        if out and start <= out[-1]["end"] + merge_s:
            cur = out[-1]
            cur["end"] = max(cur["end"], end)
            if cam not in cur["cameras"]:
                cur["cameras"].append(cam)
            if w["peak_db"] > cur["peak_db"]:
                cur.update(peak_db=w["peak_db"], hz=w["hz"])
        else:
            out.append({"t": start, "start": start, "end": end, "peak_db": w["peak_db"], "hz": w["hz"],
                        "cameras": [cam]})
    for w in out:
        w["cameras"].sort()
    return out


# ---------------------------------------------------------------------------
# Pure logic: audio activity and stoppages
# ---------------------------------------------------------------------------

def activity_rank(flux: np.ndarray, rate: float = RATE, smooth_s: float = STOPPAGE["smooth_s"]) -> np.ndarray:
    """3-s mean of the flux, ranked within the game to 0..1. NaN (no audio) stays NaN."""
    flux = np.asarray(flux, np.float64)
    ok = np.isfinite(flux)
    out = np.full(len(flux), np.nan, np.float32)
    if not ok.any():
        return out
    sm = uniform_filter1d(np.where(ok, flux, 0.0), max(1, int(round(smooth_s * rate))), mode="nearest")
    vals = sm[ok]
    # Equal values get the same rank (mean of their positions), so a long
    # stretch of constant sound does not climb from 0 to 1 over time.
    out[ok] = (rankdata(vals, method="average") - 1) / max(1, len(vals) - 1)
    return out


def combine_activity(ranks: list[np.ndarray]) -> np.ndarray:
    """Mean of the cameras' ranked activity; a camera without audio at a time is left out."""
    ranks = [r for r in ranks if r is not None and len(r)]
    if not ranks:
        return np.zeros(0, np.float32)
    n = max(len(r) for r in ranks)
    stack = np.full((len(ranks), n), np.nan, np.float32)
    for i, r in enumerate(ranks):
        stack[i, :len(r)] = r
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(stack, axis=0).astype(np.float32)


def find_stoppages(whistles: list[dict], activity: np.ndarray, rate: float = RATE,
                   rule: dict | None = None) -> list[dict]:
    """
    Stoppages from whistles and combined activity (0..1 per frame).

    A whistle starts a stoppage when the mean activity from look_from_s to
    look_to_s after the whistle ends is below `low`. The stoppage ends at the
    first time activity stays above `high` for restart_s (the faceoff), or at
    max_s. A later whistle inside a stoppage joins it.
    Sets whistle["stoppage"] to the stoppage index or None, and whistle["role"]
    to "stoppage" (starts one), "in_stoppage" (for example the faceoff
    whistle), or "no_stoppage".
    """
    r = dict(STOPPAGE, **(rule or {}))
    act = np.asarray(activity, np.float32)
    n = len(act)
    out: list[dict] = []
    for w in whistles:
        w["stoppage"] = None
        w["role"] = "no_stoppage"
        if out and w["start"] <= out[-1]["end"]:
            out[-1]["whistles"].append(w["t"])
            w["stoppage"] = len(out) - 1
            w["role"] = "in_stoppage"
            continue
        a = int(round((w["end"] + r["look_from_s"]) * rate))
        b = int(round((w["end"] + r["look_to_s"]) * rate))
        if a >= n:
            continue
        win = act[a:min(b, n)]
        win = win[np.isfinite(win)]
        if len(win) == 0:
            continue
        level = float(win.mean())
        if level >= r["low"]:
            continue
        cap = min(n, int(round((w["start"] + r["max_s"]) * rate)))
        need = max(1, int(round(r["restart_s"] * rate)))
        end_i, found = cap, False
        above = np.nan_to_num(act[a:cap], nan=0.0) > r["high"]
        for s0, s1 in _runs(above):
            if s1 - s0 >= need:
                end_i, found = a + s0, True
                break
        w["stoppage"] = len(out)
        w["role"] = "stoppage"
        out.append({"start": w["t"], "end": round(end_i / rate, 2), "restart_found": found,
                    "activity_after": round(level, 3), "whistles": [w["t"]]})
    return out


# ---------------------------------------------------------------------------
# Pure logic: PA music spans
# ---------------------------------------------------------------------------

def music_bins(frame_scores: np.ndarray, hop_s: float = YAMNET_HOP_S, win_s: float = YAMNET_WIN / SR,
               t0: float = 0.0) -> np.ndarray:
    """Average YAMNet frame scores into 1-s bins by frame centre. Bin i covers [t0+i, t0+i+1)."""
    fs = np.asarray(frame_scores, np.float64)
    if len(fs) == 0:
        return np.zeros(0, np.float32)
    centres = np.arange(len(fs)) * hop_s + win_s / 2
    idx = np.floor(centres).astype(int)
    n = int(idx.max()) + 1
    ok = np.isfinite(fs)
    sums = np.bincount(idx[ok], weights=fs[ok], minlength=n)
    cnt = np.bincount(idx[ok], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
    return out.astype(np.float32)


def music_spans(bins: np.ndarray, camera: str, t0: float = 0.0, rule: dict | None = None) -> list[dict]:
    """
    PA music spans from 1-s bins (research hhg-3r5.18):
    3-s median filter; a span starts at >= start and ends below end; spans
    shorter than min_s are dropped; pad 1 s before and 2 s after; state
    'mute'. Runs of bins in [review_low, start) for review_min_s or more,
    outside mute spans, are state 'review' (not muted).
    """
    r = dict(MUSIC, **(rule or {}))
    b = np.nan_to_num(np.asarray(bins, np.float32), nan=0.0)
    if len(b) == 0:
        return []
    m = median_filter(b, size=int(r["median_s"]), mode="nearest")
    spans, on, s0 = [], False, 0
    for i, v in enumerate(m):
        if not on and v >= r["start"]:
            on, s0 = True, i
        elif on and v < r["end"]:
            spans.append((s0, i))
            on = False
    if on:
        spans.append((s0, len(m)))
    out = []
    muted = np.zeros(len(m), bool)
    for a, e in spans:
        if e - a < r["min_s"]:
            continue
        # Mark the padded span so no 'review' span overlaps a 'mute' span.
        muted[max(0, a - int(np.ceil(r["pad_before_s"]))):e + int(np.ceil(r["pad_after_s"]))] = True
        span = {"camera": camera, "start": round(max(0.0, t0 + a - r["pad_before_s"]), 2),
                "end": round(t0 + e + r["pad_after_s"], 2), "peak": round(float(m[a:e].max()), 3),
                "state": "mute"}
        if out and span["start"] <= out[-1]["end"]:  # padding made two spans touch
            out[-1].update(end=span["end"], peak=max(out[-1]["peak"], span["peak"]))
        else:
            out.append(span)
    for a, e in _runs((m >= r["review_low"]) & (m < r["start"]) & ~muted):
        if e - a >= r["review_min_s"]:
            out.append({"camera": camera, "start": round(t0 + a, 2), "end": round(t0 + e, 2),
                        "peak": round(float(m[a:e].max()), 3), "state": "review"})
    return sorted(out, key=lambda s: s["start"])


# ---------------------------------------------------------------------------
# YAMNet model
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def yamnet_model_path(download: bool = True) -> Path:
    """
    Path of yamnet.tflite. $HHG_YAMNET_MODEL wins; else the file in
    ~/.cache/hockey-highlight-generator/, downloaded from the MediaPipe model
    bucket on first use and checked against its SHA-256.
    """
    env = os.environ.get(YAMNET_ENV)
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"{YAMNET_ENV}={env} is not a file")
        return p
    p = CACHE_HOME / "yamnet.tflite"
    if p.is_file() and _sha256(p) == YAMNET_SHA256:
        return p
    if not download:
        raise FileNotFoundError(f"{p} missing")
    CACHE_HOME.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part")
    print(f"[audio] downloading YAMNet model from {YAMNET_URL}", flush=True)
    urllib.request.urlretrieve(YAMNET_URL, tmp)
    if _sha256(tmp) != YAMNET_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("downloaded yamnet.tflite has the wrong SHA-256")
    os.replace(tmp, p)
    return p


class _Yamnet:
    def __init__(self, model: Path):
        from ai_edge_litert.interpreter import Interpreter
        self.it = Interpreter(model_path=str(model))
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]["index"]
        self.out = self.it.get_output_details()[0]["index"]

    def music(self, frame: np.ndarray) -> float:
        self.it.set_tensor(self.inp, frame.astype(np.float32))
        self.it.invoke()
        return float(self.it.get_tensor(self.out)[0][YAMNET_MUSIC])


def load_music_model() -> tuple[Path | None, str | None]:
    """(model path, None) or (None, reason) when music detection is unavailable."""
    try:
        import ai_edge_litert.interpreter  # noqa: F401
    except ImportError:
        return None, "PA music detection needs the optional ML stack (pip install -r requirements-ml.txt)"
    try:
        path = yamnet_model_path()
    except Exception as exc:  # noqa: BLE001 - any fetch problem is a flag
        return None, f"PA music detection: YAMNet model unavailable ({exc})"
    try:
        _Yamnet(path)  # a bad file must not stop whistle detection later
    except Exception as exc:  # noqa: BLE001
        return None, f"PA music detection: YAMNet model could not be loaded ({exc})"
    return path, None


# ---------------------------------------------------------------------------
# Decoding and features
# ---------------------------------------------------------------------------

class _Features:
    """Streaming features for one Recording: whistle score/Hz, flux (100 Hz), YAMNet music (0.48 s)."""

    def __init__(self, yamnet: _Yamnet | None):
        self.win = np.hanning(N_FFT + 2)[1:-1].astype(np.float32)
        self.freqs = np.fft.rfftfreq(N_FFT, 1 / SR)
        self.flux_sel = (self.freqs >= FLUX_BAND_HZ[0]) & (self.freqs <= FLUX_BAND_HZ[1])
        self.buf = np.zeros(0, np.float32)
        self.ybuf = np.zeros(0, np.float32)
        self.prev = None
        self.yamnet = yamnet
        self.score, self.hz, self.flux, self.music = [], [], [], []

    def feed(self, x: np.ndarray) -> None:
        self.buf = np.concatenate([self.buf, x])
        if len(self.buf) >= N_FFT:
            n = (len(self.buf) - N_FFT) // HOP + 1
            fr = np.lib.stride_tricks.sliding_window_view(self.buf, N_FFT)[::HOP][:n]
            Z = np.fft.rfft(fr * self.win, axis=1).T / self.win.sum()
            mag = np.abs(Z)
            s, f = whistle_score(10 * np.log10(mag ** 2 + 1e-14), self.freqs)
            self.score.append(s)
            self.hz.append(f)
            lm = np.log1p(1000 * mag[self.flux_sel])
            prev = lm[:, :1] if self.prev is None else self.prev
            d = np.diff(np.concatenate([prev, lm], axis=1), axis=1)
            self.flux.append(np.maximum(d, 0).sum(axis=0).astype(np.float32))
            self.prev = lm[:, -1:]
            self.buf = self.buf[n * HOP:]
        if self.yamnet is not None:
            self.ybuf = np.concatenate([self.ybuf, x])
            while len(self.ybuf) >= YAMNET_WIN:
                self.music.append(self.yamnet.music(self.ybuf[:YAMNET_WIN]))
                self.ybuf = self.ybuf[YAMNET_HOP:]

    def result(self) -> dict:
        cat = lambda a: np.concatenate(a).astype(np.float32) if a else np.zeros(0, np.float32)  # noqa: E731
        out = {"score": cat(self.score), "hz": cat(self.hz), "flux": cat(self.flux)}
        if self.yamnet is not None:
            out["music"] = np.asarray(self.music, np.float32)
        return out


def _decode(input_args: list[str], feats: _Features, progress=None) -> float:
    """Decode mono 16 kHz audio with ffmpeg and feed it to feats. Returns seconds decoded."""
    argv = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", *input_args,
            "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    # stderr goes to a file, not a pipe: a full stderr pipe would block ffmpeg.
    with tempfile.TemporaryFile() as errf:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=errf)
        done = 0
        step = CHUNK_S * SR * 4
        try:
            assert proc.stdout is not None
            while True:
                raw = proc.stdout.read(step)
                if not raw:
                    break
                x = np.frombuffer(raw[: len(raw) // 4 * 4], np.float32)
                feats.feed(x)
                done += len(x)
                if progress:
                    progress(done / SR)
            rc = proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        errf.seek(0)
        err = errf.read().decode(errors="replace").strip()
    # A partial decode is an error: it must not be cached or used as a whole camera.
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed (exit {rc}) after {done / SR:.0f} s: {err[-300:]}")
    if done == 0:
        raise RuntimeError(f"ffmpeg decoded no audio{': ' + err[-300:] if err else ''}")
    return done / SR


def _feature_key(files: list[str], seek: float, music_id: str | None) -> str:
    """Cache key. music_id is the SHA-256 of the YAMNet file, or None for no music features."""
    from signals import _file_identity
    payload = {"version": FEATURES_VERSION, "sr": SR, "n_fft": N_FFT, "hop": HOP,
               "whistle_band": WHISTLE_BAND_HZ, "band_w": BAND_W_HZ, "flux_band": FLUX_BAND_HZ,
               "files": [_file_identity(f) for f in files], "seek": round(seek, 3),
               "music": music_id}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def _recording_features(manifest_lines: list[str], base_dir: str, input_args: list[str],
                        model: Path | None, cache_dir: str | None, progress, music_id: str | None = None) -> dict:
    """
    Features of one Recording, from the cache when its files and settings are
    unchanged. With a model, only a cache with music features from the same
    model file is used. Without a model, a cache with music is also used, but
    its music features are dropped.
    """
    from signals import _manifest_inputs
    files, seek = _manifest_inputs(manifest_lines, base_dir)
    if model is not None and music_id is None:
        music_id = _sha256(model)
    write_path, read_paths = None, []
    if cache_dir:
        try:
            write_path = os.path.join(cache_dir, f"audio_{_feature_key(files, seek, music_id if model else None)}.npz")
            read_paths = [write_path]
            if model is None:
                read_paths.append(os.path.join(cache_dir, f"audio_{_feature_key(files, seek, YAMNET_SHA256)}.npz"))
        except OSError:
            write_path, read_paths = None, []
        for p in read_paths:
            if os.path.exists(p):
                try:
                    with np.load(p) as z:
                        feats = {k: z[k] for k in z.files}
                except Exception:  # noqa: BLE001 - damaged cache file: compute again
                    continue
                if model is None:
                    feats.pop("music", None)
                elif "music" not in feats:
                    continue
                print(f"[audio] cache hit: {os.path.basename(files[0]) if files else '?'}", flush=True)
                return feats
    feats = _Features(_Yamnet(model) if model else None)
    _decode(input_args, feats, progress)
    out = feats.result()
    if write_path and cache_dir:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = write_path + ".tmp.npz"
            np.savez(tmp, **out)
            os.replace(tmp, write_path)
        except OSError as exc:
            print(f"[WARN] could not write audio cache {write_path} ({exc})", flush=True)
    return out


def camera_features(manifest: str, model: Path | None, cache_dir: str | None, progress=None,
                    music_id: str | None = None) -> dict:
    """
    Features of one camera on the detection timeline. Each Recording block is
    placed at its '# recording' start, as in signals.extract_signals; gaps
    are NaN (score, flux, music) so they count as no audio.
    """
    from signals import _recording_blocks, concat_input_args, write_temp_manifest, _remove_temp_manifest
    base = os.path.dirname(os.path.abspath(manifest))
    blocks = _recording_blocks(manifest)
    with open(manifest) as f:
        lines = f.read().splitlines()
    if len(blocks) < 2:
        parts = [(0.0, _recording_features(lines, base, concat_input_args(manifest), model, cache_dir, progress,
                                         music_id))]
    else:
        parts = []
        done_before = [0.0]
        for start, files in blocks:
            header = ["ffconcat version 1.0"] + ([f"# seek {-start:.3f}"] if start < 0 else [])
            tmp = write_temp_manifest(header + files, base, "audio_recording_")
            try:
                before = done_before[0]
                prog = (lambda s, b=before: progress(b + s)) if progress else None
                feats = _recording_features(header + files, base, concat_input_args(tmp), model, cache_dir, prog,
                                            music_id)
            finally:
                _remove_temp_manifest(tmp)
            done_before[0] += len(feats["score"]) / RATE
            parts.append((max(start, 0.0), feats))
    return _place(parts, model is not None)


def _place(parts: list[tuple[float, dict]], with_music: bool) -> dict:
    n = max(int(round(t * RATE)) + len(f["score"]) for t, f in parts)
    out = {k: np.full(n, np.nan, np.float32) for k in ("score", "hz", "flux")}
    music = None
    if with_music and all("music" in f for _, f in parts):
        nm = max(int(round(t / YAMNET_HOP_S)) + len(f["music"]) for t, f in parts)
        music = np.full(nm, np.nan, np.float32)
    for t, f in parts:
        i = int(round(t * RATE))
        for k in out:
            out[k][i:i + len(f[k])] = f[k]
        if music is not None:
            j = int(round(t / YAMNET_HOP_S))
            music[j:j + len(f["music"])] = f["music"]
    if music is not None:
        out["music"] = music
    return out


# ---------------------------------------------------------------------------
# Flow (extra field only)
# ---------------------------------------------------------------------------

def _cached_flow(root: Path, cam: str, manifest: str, fps: int, width: int,
                 with_audio: bool) -> np.ndarray | None:
    """Net+slot flow of one camera from the detection signal cache, or None when not cached."""
    try:
        from signals import _manifest_inputs, _recording_blocks, load_rois, signal_cache_key
    except Exception:  # noqa: BLE001
        return None
    try:
        rois = load_rois(str(root / "rois.json"))[f"camera_{cam[-1]}"]
    except Exception:  # noqa: BLE001
        return None
    base = os.path.dirname(os.path.abspath(manifest))
    blocks = _recording_blocks(manifest)
    with open(manifest) as f:
        lines = f.read().splitlines()
    groups = [(0.0, lines)] if len(blocks) < 2 else [
        (start, ["ffconcat version 1.0"] + ([f"# seek {-start:.3f}"] if start < 0 else []) + fl)
        for start, fl in blocks]
    placed = []
    for start, ls in groups:
        files, seek = _manifest_inputs(ls, base)
        try:
            key = signal_cache_key(files, seek, rois, fps, width, with_audio)
        except OSError:
            return None
        p = root / ".recap_cache" / "signals" / f"signals_{key}.npz"
        if not p.exists():
            return None
        try:
            with np.load(p) as z:
                placed.append((int(round(max(start, 0.0) * fps)), (z["net"] + z["slot"]).astype(np.float32)))
        except Exception:  # noqa: BLE001 - damaged cache: flow is an extra field only
            return None
    n = max(i + len(a) for i, a in placed)
    out = np.zeros(n, np.float32)
    for i, a in placed:
        out[i:i + len(a)] = a
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyse(root: Path, use_music: bool = True, cache: bool = True, fps: int = 12, width: int = 1280,
            flow_audio: bool = False) -> tuple[dict, dict]:
    flags: list[str] = []
    music_flags: list[str] = []
    cams = [c for c in ("cam1", "cam2") if (root / f"{c}_concat.txt").exists()]
    model = None
    if use_music:
        model, why = load_music_model()
        if why:
            music_flags.append(why)
    else:
        music_flags.append("PA music detection turned off (--no-music)")
    music_id = _sha256(model) if model else None
    cache_dir = str(root / ".recap_cache" / "audio") if cache else None
    total = {}
    try:
        from recordings import _duration
        ch = json.loads((root / "chapters.json").read_text())
        for c in cams:
            total[c] = sum(_duration(p) for rec in ch.get("recordings", {}).get(c, []) for p in rec)
    except Exception:  # noqa: BLE001 - progress only
        pass
    lock = threading.Lock()
    done = {c: 0.0 for c in cams}

    def prog(cam):
        def f(s):
            with lock:
                done[cam] = s
                tot = sum(total.values())
                print(f"[audio] decoded {sum(done.values()):.0f}/{tot:.0f} s", flush=True)
        return f

    feats: dict[str, dict] = {}

    def job(cam):
        try:
            return cam, camera_features(str(root / f"{cam}_concat.txt"), model, cache_dir, prog(cam),
                                        music_id), None
        except Exception as exc:  # noqa: BLE001 - a bad camera is a flag
            return cam, None, str(exc)

    with ThreadPoolExecutor(max_workers=max(1, len(cams))) as ex:
        for cam, f, err in ex.map(job, cams):
            if err:
                flags.append(f"{cam}: audio could not be read ({err})")
            else:
                feats[cam] = f

    per_cam = {c: find_whistles(f["score"], f["hz"]) for c, f in feats.items()}
    whistles = merge_camera_whistles(per_cam)
    act = combine_activity([activity_rank(f["flux"]) for f in feats.values()])
    stoppages = find_stoppages(whistles, act)
    flows = {}
    for c in cams:
        try:
            flows[c] = _cached_flow(root, c, str(root / f"{c}_concat.txt"), fps, width, flow_audio)
        except Exception:  # noqa: BLE001 - flow is an extra field only
            flows[c] = None
    for s in stoppages:
        for c, flow in flows.items():
            if flow is None or not len(flow):
                continue
            a, b = int(s["start"] * fps), int(s["end"] * fps)
            if a < len(flow):
                q = float(np.median(flow))
                s.setdefault("flow_vs_median", {})[c] = round(float(np.mean(flow[a:max(a + 1, b)])) / max(q, 1e-9), 2)
    try:
        sync = json.loads((root / "sync_info.json").read_text())
        offset = sync.get("offset_s")
    except (OSError, json.JSONDecodeError):
        offset = None
    if len(feats) < len(cams) or len(cams) < 2:
        flags.append(f"stoppages use audio from {', '.join(sorted(feats)) or 'no camera'} only")

    signals_out = {
        "timeline": TIMELINE, "sync_offset_s": offset, "cameras": sorted(feats),
        "whistles": whistles, "stoppages": stoppages,
        "per_camera_whistles": {c: len(v) for c, v in per_cam.items()},
        "settings": {"whistle": WHISTLE, "stoppage": STOPPAGE}, "flags": flags,
    }
    spans = []
    for c, f in feats.items():
        if "music" in f:
            spans += music_spans(music_bins(f["music"]), c)
    if model and not any("music" in f for f in feats.values()):
        music_flags.append("PA music detection: no camera audio")
    music_out = {
        "timeline": TIMELINE, "model": {"name": "YAMNet float32 (MediaPipe)", "class": "Music (132)",
                                        "sha256": music_id, "path": str(model)} if model else None,
        "spans": sorted(spans, key=lambda s: (s["start"], s["camera"])),
        "muted_s": round(sum(s["end"] - s["start"] for s in spans if s["state"] == "mute"), 1),
        "settings": MUSIC, "flags": music_flags,
        "available": bool(model) and any("music" in f for f in feats.values()),
    }
    return signals_out, music_out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Whistles, stoppages and PA music spans from rink audio.")
    ap.add_argument("game_folder")
    ap.add_argument("--no-music", action="store_true", help="skip YAMNet PA music detection")
    ap.add_argument("--no-cache", action="store_true", help="do not read or write .recap_cache/audio")
    ap.add_argument("--fps", type=int, default=12, help="detection fps (to find cached flow signals)")
    ap.add_argument("--width", type=int, default=1280, help="detection width (to find cached flow signals)")
    ap.add_argument("--flow_audio", action="store_true", help="detection ran with --audio_weight > 0")
    args = ap.parse_args(argv)
    root = Path(args.game_folder).resolve()
    sig, mus = analyse(root, use_music=not args.no_music, cache=not args.no_cache, fps=args.fps,
                       width=args.width, flow_audio=args.flow_audio)
    if not sig["cameras"]:
        print("[audio] no camera audio could be read: " + "; ".join(sig["flags"]), flush=True)
        return 1
    for name, obj in (("audio_signals.json", sig), ("music_spans.json", mus)):
        tmp = root / (name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2) + "\n")
        os.replace(tmp, root / name)
    n_stop = len(sig["stoppages"])
    print(f"[audio] {len(sig['whistles'])} whistles, {n_stop} stoppages, "
          f"{sum(s['state'] == 'mute' for s in mus['spans'])} music spans ({mus['muted_s']} s muted)", flush=True)
    for f in sig["flags"] + mus["flags"]:
        print(f"[audio] flag: {f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
