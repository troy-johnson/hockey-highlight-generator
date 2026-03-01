# v2/scripts/detect_events.py
#
# V2 hockey highlight detection engine.
#
# Improvements over V1:
#   - Optical flow (Farneback) instead of raw frame-differencing
#   - Audio RMS fused into the detection signal (tunable weight)
#   - Rolling percentile threshold per 5-minute segment, floored at global P70
#     → adapts to intensity variation across game periods
#   - Scoring formula weights exposed as CLI args
#   - --debug_plot saves a matplotlib PNG of the fused signal + peaks
#
# CLI interface is backwards-compatible with V1 run_detect.sh.
#
# Usage:
#   python detect_events.py cam1.mp4 cam2.mp4 rois.json --verbose
#
# Key args (V1-compatible):
#   --fps            analysis sampling fps (default 12)
#   --width          analysis scale width (default 1280)
#   --thresh_pct     peak threshold percentile within each segment (default 92)
#   --min_sep_s      minimum seconds between peaks per camera (default 6)
#   --merge_gap_s    merge windows if gap <= this (default 2, recommend 0)
#   --max_win_s      max window duration for normal plays (default 25)
#   --max_big_win_s  max window duration for big plays (default 32)
#   --big_color_min  color threshold to treat as big play (default Orange)
#   --replay_markers add a REPLAY marker after big plays
#   --replay_offset_s seconds after window end for REPLAY marker (default 1.0)
#   --cooldown_s     suppress windows within this gap of a prior kept one (default 0)
#   --out_csv        debug events CSV path (default events.csv)
#   --out_markers    Resolve markers CSV path (default markers.csv)
#   --timeline_fps   Resolve timeline fps for frame conversion (default 60)
#   --verbose        print progress logs
#
# New V2 args:
#   --flow_weight    weight for the optical-flow signal component (default 0.8)
#   --audio_weight   weight for the audio-RMS signal component (default 0.2)
#                    set to 0 to disable audio entirely
#   --w_attack       scoring weight for peak attack energy (default 0.55)
#   --w_rebound      scoring weight for rebound activity (default 0.25)
#   --w_drop         scoring weight for post-peak drop (default 0.20)
#   --debug_plot     save a PNG of the fused signal with detected peaks marked

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

# signals.py lives next to this file; add script dir to path so it imports cleanly
sys.path.insert(0, os.path.dirname(__file__))
from signals import extract_signals, load_rois  # noqa: E402


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Window:
    start_s: float
    end_s: float
    peak_s: float
    cam: int
    peak_val: float


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def moving_avg(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    k = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, k, mode="same")


def normalize_p99(x: np.ndarray) -> np.ndarray:
    """Divide by the 99th percentile so the signal lives roughly in [0, 1]."""
    p = float(np.percentile(x, 99)) if x.size > 0 else 1.0
    return (x / (p + 1e-6)).astype(np.float32)


def fuse_signals(
    net_flow: np.ndarray,
    slot_flow: np.ndarray,
    audio_rms: np.ndarray,
    flow_weight: float,
    audio_weight: float,
) -> np.ndarray:
    """
    Fuse optical-flow ROI energy and audio RMS into a single attack signal.

    Flow component: weighted blend of net (70%) and slot (30%) flow,
    normalized to P99 so its scale is comparable to audio.
    Audio component: audio RMS normalized to P99.
    Final signal: flow_weight * norm_flow + audio_weight * norm_audio.
    """
    flow = 0.7 * net_flow + 0.3 * slot_flow
    norm_flow = normalize_p99(flow)

    if audio_weight > 0 and audio_rms.size == norm_flow.size:
        norm_audio = normalize_p99(audio_rms)
        return (flow_weight * norm_flow + audio_weight * norm_audio).astype(np.float32)

    return (flow_weight * norm_flow).astype(np.float32)


# ---------------------------------------------------------------------------
# Rolling percentile threshold
# ---------------------------------------------------------------------------

def rolling_threshold(
    signal: np.ndarray,
    fps: int,
    thresh_pct: float,
    segment_s: float = 300.0,
    floor_pct: float = 70.0,
) -> np.ndarray:
    """
    Compute a per-sample threshold using non-overlapping 5-minute segments.

    For each segment: segment_threshold = percentile(segment, thresh_pct).
    Global floor: floor_threshold = percentile(whole_signal, floor_pct).
    Per-sample threshold = max(segment_threshold, floor_threshold).

    This makes the detector adapt to quiet periods (e.g., early in period 1)
    without getting flooded by false positives during high-energy stretches,
    while the floor prevents thresholds from collapsing to near-zero on dead ice.

    Returns float32 array same length as signal.
    """
    n = len(signal)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    floor = float(np.percentile(signal, floor_pct))
    seg_len = max(1, int(round(segment_s * fps)))

    thresholds = np.empty(n, dtype=np.float32)
    i = 0
    while i < n:
        seg = signal[i : i + seg_len]
        seg_thresh = float(np.percentile(seg, thresh_pct))
        t = max(seg_thresh, floor)
        thresholds[i : i + seg_len] = t
        i += seg_len

    return thresholds


# ---------------------------------------------------------------------------
# Peak / window detection
# ---------------------------------------------------------------------------

def peak_windows(
    attack: np.ndarray,
    fps: int,
    thresh_pct: float,
    min_sep_s: float,
    cam: int,
) -> list[Window]:
    """
    Detect peaks in `attack` using the rolling percentile threshold.
    Returns one 13-second Window (7s before, 6s after) per peak.
    """
    if attack.size < 10:
        return []

    thresholds = rolling_threshold(attack, fps, thresh_pct)
    distance = max(1, int(round(min_sep_s * fps)))

    # find_peaks accepts an array for `height` — each peak must exceed its
    # corresponding threshold value.
    peaks, _ = find_peaks(attack, height=thresholds, distance=distance)

    out: list[Window] = []
    for p in peaks:
        t = p / fps
        out.append(
            Window(
                start_s=t - 7.0,
                end_s=t + 6.0,
                peak_s=t,
                cam=cam,
                peak_val=float(attack[p]),
            )
        )
    return out


def merge_intervals(
    intervals: list[tuple[float, float]], gap_s: float
) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe + gap_s:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def window_slice(
    series: np.ndarray, fps: int, start_s: float, end_s: float
) -> tuple[np.ndarray, int, int]:
    a = max(0, int(np.floor(start_s * fps)))
    b = min(len(series), int(np.ceil(end_s * fps)))
    if b <= a:
        return np.array([], dtype=series.dtype), a, b
    return series[a:b], a, b


def score_window(
    cam_attack: np.ndarray,
    fps: int,
    start_s: float,
    end_s: float,
) -> dict:
    """
    Score a window using three components:
      max_attack  — peak signal value (normalized by caller)
      rebound     — fraction of secondary peaks >= 80% of max within window [0, 1]
      drop        — post-peak signal drop in the following 2 seconds
    """
    seg, a, b = window_slice(cam_attack, fps, start_s, end_s)
    if seg.size == 0:
        return {"max_attack": 0.0, "rebound": 0.0, "drop": 0.0}

    max_attack = float(seg.max())

    if seg.size > 5:
        pk, _ = find_peaks(
            seg,
            height=max_attack * 0.80,
            distance=max(1, int(0.7 * fps)),
        )
        rebound = float(min(3, len(pk))) / 3.0
    else:
        rebound = 0.0

    peak_local = int(np.argmax(seg))
    peak_idx = a + peak_local
    after_a = peak_idx
    after_b = min(len(cam_attack), peak_idx + int(2.0 * fps))
    if after_b > after_a and after_a < len(cam_attack):
        drop = float(max_attack - float(cam_attack[after_a:after_b].min()))
    else:
        drop = 0.0

    return {"max_attack": max_attack, "rebound": rebound, "drop": drop}


# ---------------------------------------------------------------------------
# Cooldown suppression
# ---------------------------------------------------------------------------

def apply_cooldown(
    windows: list[tuple[float, float, float, int, float]],
    cooldown_s: float,
) -> list[tuple[float, float, float, int, float]]:
    """
    windows: list of (start_s, end_s, score, primary_cam, conf)
    Keeps higher-scoring windows when two start within `cooldown_s` of each other.
    """
    if cooldown_s <= 0:
        return windows

    by_score = sorted(windows, key=lambda r: -r[2])
    kept: list[tuple[float, float, float, int, float]] = []
    kept_starts: list[float] = []

    for r in by_score:
        s = r[0]
        if all(abs(s - ks) >= cooldown_s for ks in kept_starts):
            kept.append(r)
            kept_starts.append(s)

    return sorted(kept, key=lambda r: r[0])


# ---------------------------------------------------------------------------
# Scoring / colors
# ---------------------------------------------------------------------------

def score_to_color(score: float) -> str:
    if score > 1.60:
        return "Red"
    if score > 1.30:
        return "Orange"
    if score > 1.00:
        return "Yellow"
    return "Blue"


def color_rank(c: str) -> int:
    order = {"Blue": 0, "Yellow": 1, "Orange": 2, "Red": 3}
    return order.get(c, 0)


def sec_to_frame(sec: float, timeline_fps: int) -> int:
    return int(round(sec * float(timeline_fps)))


# ---------------------------------------------------------------------------
# Debug plot
# ---------------------------------------------------------------------------

def save_debug_plot(
    attack1: np.ndarray,
    attack2: np.ndarray,
    peaks1_idx: list[int],
    peaks2_idx: list[int],
    thresholds1: np.ndarray,
    thresholds2: np.ndarray,
    fps: int,
    out_path: str,
) -> None:
    """
    Save a PNG showing the fused attack signals for both cameras with detected
    peaks and rolling thresholds marked.
    Matplotlib is imported lazily so it is never required at import time.
    """
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    t1 = np.arange(len(attack1)) / fps
    t2 = np.arange(len(attack2)) / fps

    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=False)

    for ax, attack, thresholds, peaks_idx, label in [
        (axes[0], attack1, thresholds1, peaks1_idx, "cam1"),
        (axes[1], attack2, thresholds2, peaks2_idx, "cam2"),
    ]:
        t = np.arange(len(attack)) / fps
        ax.plot(t, attack, linewidth=0.6, color="steelblue", label="fused signal")
        ax.plot(t, thresholds, linewidth=1.0, color="orange", linestyle="--", label="rolling threshold")
        if peaks_idx:
            ax.plot(
                [p / fps for p in peaks_idx],
                [attack[p] for p in peaks_idx],
                "rv",
                markersize=6,
                label=f"peaks ({len(peaks_idx)})",
            )
        ax.set_title(f"{label} — fused attack signal")
        ax.set_ylabel("energy (normalized)")
        ax.set_xlabel("time (s)")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, linewidth=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="V2 hockey highlight detection (optical flow + audio RMS)"
    )
    # ── Positional ──────────────────────────────────────────────────────────
    ap.add_argument("cam1", help="camera 1 MP4")
    ap.add_argument("cam2", help="camera 2 MP4")
    ap.add_argument("rois", help="rois.json from roi_picker")

    # ── V1-compatible args ───────────────────────────────────────────────────
    ap.add_argument("--fps", type=int, default=12, help="analysis fps (sample rate)")
    ap.add_argument("--width", type=int, default=1280, help="analysis scale width")
    ap.add_argument("--thresh_pct", type=float, default=92.0,
                    help="peak threshold percentile per 5-min segment (90-97 typical)")
    ap.add_argument("--min_sep_s", type=float, default=6.0,
                    help="min seconds between peaks per camera")
    ap.add_argument("--merge_gap_s", type=float, default=2.0,
                    help="merge windows if gap <= this")
    ap.add_argument("--max_win_s", type=float, default=25.0,
                    help="max duration for normal plays")
    ap.add_argument("--max_big_win_s", type=float, default=32.0,
                    help="max duration for big plays (>= big_color_min)")
    ap.add_argument("--big_color_min", default="Orange",
                    choices=["Yellow", "Orange", "Red"],
                    help="color threshold to treat as big play")
    ap.add_argument("--cooldown_s", type=float, default=0.0,
                    help="suppress windows that start within this many seconds of a prior kept window")
    ap.add_argument("--replay_markers", action="store_true",
                    help="add a REPLAY marker after big plays")
    ap.add_argument("--replay_offset_s", type=float, default=1.0,
                    help="seconds after window end to place REPLAY marker")
    ap.add_argument("--out_csv", default="events.csv",
                    help="debug events CSV (sorted by score)")
    ap.add_argument("--out_markers", default="markers.csv",
                    help="Resolve markers CSV (Frame,Name,Note,Color,Duration)")
    ap.add_argument("--timeline_fps", type=int, default=60,
                    help="Resolve timeline fps for markers.csv")
    ap.add_argument("--verbose", action="store_true")

    # ── V2 new args ──────────────────────────────────────────────────────────
    ap.add_argument("--flow_weight", type=float, default=0.8,
                    help="weight for optical-flow signal component (default 0.8)")
    ap.add_argument("--audio_weight", type=float, default=0.2,
                    help="weight for audio-RMS signal component (default 0.2; 0 = disable)")
    ap.add_argument("--w_attack", type=float, default=0.55,
                    help="scoring weight: peak attack energy (default 0.55)")
    ap.add_argument("--w_rebound", type=float, default=0.25,
                    help="scoring weight: rebound activity (default 0.25)")
    ap.add_argument("--w_drop", type=float, default=0.20,
                    help="scoring weight: post-peak signal drop (default 0.20)")
    ap.add_argument("--debug_plot", action="store_true",
                    help="save a PNG of the fused signal with detected peaks "
                         "(saved alongside --out_csv)")

    args = ap.parse_args()
    verbose = args.verbose

    # ── Validate scoring weights ─────────────────────────────────────────────
    w_sum = args.w_attack + args.w_rebound + args.w_drop
    if abs(w_sum - 1.0) > 0.01:
        print(
            f"[WARN] Scoring weights sum to {w_sum:.3f} (expected ~1.0). "
            "Results will be relative but colors may shift.",
            flush=True,
        )

    log("[INFO] Loading ROIs...", verbose)
    rois = load_rois(args.rois)

    # ── Extract signals ──────────────────────────────────────────────────────
    log("[INFO] Extracting signals from cam1...", verbose)
    t0 = time.time()
    net1, slot1, audio1 = extract_signals(
        args.cam1, rois["camera_1"], args.fps, args.width, verbose
    )

    log("[INFO] Extracting signals from cam2...", verbose)
    net2, slot2, audio2 = extract_signals(
        args.cam2, rois["camera_2"], args.fps, args.width, verbose
    )
    log(f"[INFO] Signal extraction done in {time.time() - t0:.1f}s", verbose)

    # ── Fuse signals ─────────────────────────────────────────────────────────
    log("[INFO] Fusing flow + audio signals...", verbose)
    attack1_raw = fuse_signals(net1, slot1, audio1, args.flow_weight, args.audio_weight)
    attack2_raw = fuse_signals(net2, slot2, audio2, args.flow_weight, args.audio_weight)

    # Smooth ~0.5 seconds to reduce single-frame noise spikes
    smooth_win = max(1, int(round(0.5 * args.fps)))
    attack1s = moving_avg(attack1_raw, smooth_win)
    attack2s = moving_avg(attack2_raw, smooth_win)

    # ── Rolling thresholds ───────────────────────────────────────────────────
    thresholds1 = rolling_threshold(attack1s, args.fps, args.thresh_pct)
    thresholds2 = rolling_threshold(attack2s, args.fps, args.thresh_pct)

    # ── Peak detection ───────────────────────────────────────────────────────
    log("[INFO] Detecting peaks...", verbose)
    w1 = peak_windows(attack1s, args.fps, args.thresh_pct, args.min_sep_s, cam=1)
    w2 = peak_windows(attack2s, args.fps, args.thresh_pct, args.min_sep_s, cam=2)
    log(f"  cam1 peaks: {len(w1)}", verbose)
    log(f"  cam2 peaks: {len(w2)}", verbose)

    # ── Debug plot ───────────────────────────────────────────────────────────
    if args.debug_plot:
        plot_path = os.path.splitext(args.out_csv)[0] + "_debug.png"
        log(f"[INFO] Saving debug plot to {plot_path}...", verbose)
        peaks1_idx = [int(round(w.peak_s * args.fps)) for w in w1]
        peaks2_idx = [int(round(w.peak_s * args.fps)) for w in w2]
        save_debug_plot(
            attack1s, attack2s,
            peaks1_idx, peaks2_idx,
            thresholds1, thresholds2,
            args.fps,
            plot_path,
        )
        log(f"[INFO] Debug plot saved: {plot_path}", verbose)

    # ── Merge windows across cameras ─────────────────────────────────────────
    intervals = [(w.start_s, w.end_s) for w in (w1 + w2)]
    merged = merge_intervals(intervals, gap_s=args.merge_gap_s)
    log(f"[INFO] Merged into {len(merged)} candidate windows (pre-sanitize)", verbose)

    # Clamp start to 0, ensure minimum 0.5 s duration
    sanitized: list[tuple[float, float]] = []
    for s, e in merged:
        s = max(0.0, float(s))
        e = max(s + 0.5, float(e))
        sanitized.append((s, e))
    merged = sanitized

    # ── Normalize for scoring — use P99 of the fused (smoothed) signal ──────
    p99_1 = float(np.percentile(attack1s, 99)) if attack1s.size else 1.0
    p99_2 = float(np.percentile(attack2s, 99)) if attack2s.size else 1.0

    # ── Score windows + choose primary camera ────────────────────────────────
    log("[INFO] Scoring windows + selecting primary camera...", verbose)
    rows: list[tuple[float, float, float, int, float]] = []

    for s, e in merged:
        st1 = score_window(attack1s, args.fps, s, e)
        st2 = score_window(attack2s, args.fps, s, e)

        m1 = st1["max_attack"]
        m2 = st2["max_attack"]

        primary = 1 if m1 >= m2 else 2
        conf = abs(m1 - m2) / (m1 + m2 + 1e-6)

        if primary == 1:
            s_attack = max(0.0, m1 / (p99_1 + 1e-6))
            rebound = st1["rebound"]
            drop_norm = max(0.0, st1["drop"] / (p99_1 + 1e-6))
        else:
            s_attack = max(0.0, m2 / (p99_2 + 1e-6))
            rebound = st2["rebound"]
            drop_norm = max(0.0, st2["drop"] / (p99_2 + 1e-6))

        score = args.w_attack * s_attack + args.w_rebound * rebound + args.w_drop * drop_norm
        score *= (0.85 + 0.15 * conf)

        # Penalize windows with poor camera agreement
        if conf < 0.15:
            score *= 0.85

        # Clamp window duration by color class
        color = score_to_color(score)
        is_big = color_rank(color) >= color_rank(args.big_color_min)
        max_len = float(args.max_big_win_s) if is_big else float(args.max_win_s)
        if (e - s) > max_len:
            e = s + max_len

        rows.append((float(s), float(e), float(score), int(primary), float(conf)))

    log(
        f"[INFO] Window caps: small<={args.max_win_s:.1f}s, "
        f"big({args.big_color_min}+)<={args.max_big_win_s:.1f}s",
        verbose,
    )

    # ── Cooldown suppression ─────────────────────────────────────────────────
    rows_sorted = sorted(rows, key=lambda r: r[0])
    rows_cd = apply_cooldown(rows_sorted, float(args.cooldown_s))
    if args.cooldown_s > 0:
        log(
            f"[INFO] Cooldown suppression: {len(rows_sorted)} -> {len(rows_cd)} "
            f"(cooldown={args.cooldown_s:.1f}s)",
            verbose,
        )

    # ── Write events.csv (debug, sorted by score desc) ───────────────────────
    log(f"[INFO] Writing outputs: {args.out_csv}, {args.out_markers}", verbose)
    with open(args.out_csv, "w", encoding="utf-8") as f:
        f.write("start_s,end_s,score,primary_cam,confidence\n")
        for r in sorted(rows_cd, key=lambda x: -x[2]):
            f.write(f"{r[0]:.3f},{r[1]:.3f},{r[2]:.4f},{r[3]},{r[4]:.4f}\n")

    # ── Write markers.csv (Resolve import format) ─────────────────────────────
    with open(args.out_markers, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Name", "Note", "Color", "Duration"])

        for s, e, score, primary, conf in rows_cd:
            color = score_to_color(score)
            in_frame = sec_to_frame(s, args.timeline_fps)
            win_s = max(0.0, e - s)
            dur_frames = max(1, int(round(win_s * args.timeline_fps)))
            name = f"PLAY cam{primary}"
            note = f"score={score:.2f} conf={conf:.2f} win={win_s:.1f}s"
            writer.writerow([in_frame, name, note, color, dur_frames])

            if args.replay_markers:
                is_big = color_rank(color) >= color_rank(args.big_color_min)
                if is_big:
                    r_s = e + float(args.replay_offset_s)
                    r_frame = sec_to_frame(r_s, args.timeline_fps)
                    r_note = f"after={args.replay_offset_s:.1f}s from end | {name} | {note}"
                    writer.writerow([r_frame, "REPLAY", r_note, color, 1])

    log(f"[INFO] Marker count: {len(rows_cd)}", verbose)
    log("[DONE] V2 detection complete.", verbose)


if __name__ == "__main__":
    main()
