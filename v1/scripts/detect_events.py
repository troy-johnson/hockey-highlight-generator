# scripts/detect_events.py
#
# Generate Resolve timeline markers for hockey highlight candidates from two synced camera MP4s.
# - Downscales + samples frames with ffmpeg (fast on Apple Silicon)
# - Uses simple frame-diff motion energy in ROIs (net + slot) to find peaks
# - Merges peaks across cameras into unified event windows
# - Chooses a suggested primary camera per event
# - Writes events.csv + markers.csv (Resolve marker CSV import)
#
# Usage:
#   python scripts/detect_events.py cam1.mp4 cam2.mp4 rois.json --verbose
#
# Args:
#   --fps            analysis sampling fps (default 12)
#   --width          analysis scale width (default 1280)
#   --thresh_pct     percentile threshold for peak detection (default 92)
#   --min_sep_s      minimum seconds between peaks per camera (default 6)
#   --merge_gap_s    merge windows if within this gap (default 2) (recommend 0 once stable)
#   --max_win_s      maximum window duration for normal plays (default 25)
#   --max_big_win_s  maximum window duration for big plays (default 32)
#   --big_color_min  color threshold to treat as big play (default Orange)
#   --replay_markers add a REPLAY marker after big plays
#   --replay_offset_s seconds after window end to place REPLAY marker (default 1.0)
#   --cooldown_s     suppress windows that start within this many seconds of a prior kept window (default 0)
#   --out_csv        output debug events CSV path (default events.csv)
#   --out_markers    output Resolve markers CSV path (default markers.csv)
#   --timeline_fps   Resolve timeline fps for seconds->frames conversion (default 60)
#   --verbose        print progress logs

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from scipy.signal import find_peaks


# -------------------------
# Logging
# -------------------------
def log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg, flush=True)


# -------------------------
# Data structures
# -------------------------
@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    w: int
    h: int

    def clamp_to(self, width: int, height: int) -> "ROI":
        x = max(0, min(self.x, width - 1))
        y = max(0, min(self.y, height - 1))
        w = max(1, min(self.w, width - x))
        h = max(1, min(self.h, height - y))
        return ROI(x, y, w, h)


@dataclass
class Window:
    start_s: float
    end_s: float
    peak_s: float
    cam: int
    peak_val: float


# -------------------------
# ROIs
# -------------------------
def load_rois(path: str) -> dict:
    """
    Expected rois.json:
    {
      "camera_1": {"net":[x,y,w,h], "slot":[x,y,w,h]},
      "camera_2": {"net":[x,y,w,h], "slot":[x,y,w,h]}
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def to_roi(lst) -> ROI:
        if lst is None or len(lst) != 4:
            raise ValueError(f"Bad ROI {lst} in {path}")
        return ROI(int(lst[0]), int(lst[1]), int(lst[2]), int(lst[3]))

    return {
        "camera_1": {"net": to_roi(data["camera_1"]["net"]), "slot": to_roi(data["camera_1"]["slot"])},
        "camera_2": {"net": to_roi(data["camera_2"]["net"]), "slot": to_roi(data["camera_2"]["slot"])},
    }


# -------------------------
# ffmpeg helpers
# -------------------------
def ffprobe_dims(video_path: str) -> tuple[int, int]:
    probe = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            video_path,
        ]
    )
    j = json.loads(probe)
    w = int(j["streams"][0]["width"])
    h = int(j["streams"][0]["height"])
    return w, h


def ffmpeg_gray_frames(video_path: str, fps: int, width: int) -> tuple[Iterator[np.ndarray], int, int]:
    """
    Yields grayscale frames as uint8 arrays.
    Uses ffmpeg to:
      - sample to fps
      - scale to width (preserve aspect)
      - format gray
      - output rawvideo to stdout
    """
    ow, oh = ffprobe_dims(video_path)
    scale_h = int(round(oh * (width / ow)))
    if scale_h % 2 == 1:
        scale_h += 1  # make even

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        video_path,
        "-vf",
        f"fps={fps},scale={width}:{scale_h},format=gray",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    if p.stdout is None:
        raise RuntimeError("ffmpeg stdout not available")

    frame_size = width * scale_h  # gray8: 1 byte per pixel

    def gen() -> Iterator[np.ndarray]:
        while True:
            buf = p.stdout.read(frame_size)
            if len(buf) < frame_size:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape((scale_h, width))

    return gen(), width, scale_h


# -------------------------
# Signal helpers
# -------------------------
def moving_avg(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    k = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, k, mode="same")


def compute_energy_series(
    video_path: str, rois: dict, fps: int, width: int, verbose: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (net_energy, slot_energy, full_energy), one sample per (1/fps) second.
    Note: First frame is used as prev; output arrays are length = sampled_frames - 1.
    """
    log(f"[INFO] Processing {video_path}", verbose)
    t0 = time.time()

    frames, w, h = ffmpeg_gray_frames(video_path, fps=fps, width=width)

    net_roi: ROI = rois["net"].clamp_to(w, h)
    slot_roi: ROI = rois["slot"].clamp_to(w, h)

    prev = None
    net: list[float] = []
    slot: list[float] = []
    full: list[float] = []

    sampled = 0
    for fr in frames:
        sampled += 1
        if verbose and sampled % 500 == 0:
            log(f"  sampled frames read: {sampled}", verbose)

        if prev is None:
            prev = fr
            continue

        diff = np.abs(fr.astype(np.int16) - prev.astype(np.int16)).astype(np.uint8)
        prev = fr

        net.append(float(diff[net_roi.y : net_roi.y + net_roi.h, net_roi.x : net_roi.x + net_roi.w].sum()))
        slot.append(float(diff[slot_roi.y : slot_roi.y + slot_roi.h, slot_roi.x : slot_roi.x + slot_roi.w].sum()))
        full.append(float(diff.sum()))

    dt = time.time() - t0
    log(f"[INFO] Finished {video_path} in {dt:.1f}s (sampled={sampled})", verbose)

    return (
        np.array(net, dtype=np.float32),
        np.array(slot, dtype=np.float32),
        np.array(full, dtype=np.float32),
    )


def peak_windows(attack: np.ndarray, fps: int, thresh_pct: float, min_sep_s: float, cam: int) -> list[Window]:
    if attack.size < 10:
        return []

    thresh = np.percentile(attack, thresh_pct)
    distance = max(1, int(round(min_sep_s * fps)))

    peaks, _ = find_peaks(attack, height=thresh, distance=distance)
    out: list[Window] = []
    for p in peaks:
        t = p / fps
        # base window: 7s before, 6s after peak
        out.append(Window(start_s=t - 7.0, end_s=t + 6.0, peak_s=t, cam=cam, peak_val=float(attack[p])))
    return out


def merge_intervals(intervals: list[tuple[float, float]], gap_s: float) -> list[tuple[float, float]]:
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


def window_slice(series: np.ndarray, fps: int, start_s: float, end_s: float) -> tuple[np.ndarray, int, int]:
    a = max(0, int(np.floor(start_s * fps)))
    b = min(len(series), int(np.ceil(end_s * fps)))
    if b <= a:
        return np.array([], dtype=series.dtype), a, b
    return series[a:b], a, b


def score_window(cam_attack: np.ndarray, fps: int, start_s: float, end_s: float) -> dict:
    seg, a, b = window_slice(cam_attack, fps, start_s, end_s)
    if seg.size == 0:
        return {"max_attack": 0.0, "rebound": 0.0, "drop": 0.0}

    max_attack = float(seg.max())

    # rebound-ish: number of secondary peaks within window at >= 80% max
    if seg.size > 5:
        pk, _ = find_peaks(seg, height=max_attack * 0.80, distance=max(1, int(0.7 * fps)))
        rebound = float(min(3, len(pk))) / 3.0
    else:
        rebound = 0.0

    # drop after peak (next 2 seconds)
    peak_local = int(np.argmax(seg))
    peak_idx = a + peak_local
    after_a = peak_idx
    after_b = min(len(cam_attack), peak_idx + int(2.0 * fps))
    if after_b > after_a and after_a < len(cam_attack):
        drop = float(max_attack - float(cam_attack[after_a:after_b].min()))
    else:
        drop = 0.0

    return {"max_attack": max_attack, "rebound": rebound, "drop": drop}


def apply_cooldown(
    windows: list[tuple[float, float, float, int, float]],
    cooldown_s: float,
) -> list[tuple[float, float, float, int, float]]:
    """
    windows: list of (start_s, end_s, score, primary_cam, conf)
    Keeps higher-score windows when they start within cooldown of each other.
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


def score_to_color(score: float) -> str:
    # Resolve marker colors (subset we use)
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


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cam1", help="camera 1 MP4 (post trim + post 110% speed)")
    ap.add_argument("cam2", help="camera 2 MP4 (post trim + post 110% speed)")
    ap.add_argument("rois", help="rois.json from roi_picker")
    ap.add_argument("--fps", type=int, default=12, help="analysis fps (sample rate)")
    ap.add_argument("--width", type=int, default=1280, help="analysis scale width")
    ap.add_argument("--thresh_pct", type=float, default=92.0, help="peak threshold percentile (90-97 typical)")
    ap.add_argument("--min_sep_s", type=float, default=6.0, help="min seconds between peaks per camera")
    ap.add_argument("--merge_gap_s", type=float, default=2.0, help="merge windows if gap <= this")
    ap.add_argument("--max_win_s", type=float, default=25.0, help="max duration for normal plays")
    ap.add_argument("--max_big_win_s", type=float, default=32.0, help="max duration for big plays (>= big_color_min)")
    ap.add_argument(
        "--big_color_min",
        default="Orange",
        choices=["Yellow", "Orange", "Red"],
        help="color threshold to treat as big play",
    )
    ap.add_argument("--cooldown_s", type=float, default=0.0, help="suppress windows close together (seconds)")
    ap.add_argument("--replay_markers", action="store_true", help="add a REPLAY marker after big plays")
    ap.add_argument("--replay_offset_s", type=float, default=1.0, help="seconds after window end to place REPLAY marker")
    ap.add_argument("--out_csv", default="events.csv", help="debug events CSV (sorted by score)")
    ap.add_argument("--out_markers", default="markers.csv", help="Resolve markers CSV (Frame,Name,Note,Color,Duration)")
    ap.add_argument("--timeline_fps", type=int, default=60, help="Resolve timeline fps for markers.csv")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    verbose = args.verbose

    log("[INFO] Loading ROIs...", verbose)
    rois = load_rois(args.rois)

    # Compute energy series per camera
    net1, slot1, _ = compute_energy_series(args.cam1, rois["camera_1"], args.fps, args.width, verbose)
    net2, slot2, _ = compute_energy_series(args.cam2, rois["camera_2"], args.fps, args.width, verbose)

    log("[INFO] Building attack signals...", verbose)
    attack1 = 0.7 * net1 + 0.3 * slot1
    attack2 = 0.7 * net2 + 0.3 * slot2

    # Smooth ~0.5 seconds
    smooth_win = max(1, int(round(0.5 * args.fps)))
    attack1s = moving_avg(attack1, smooth_win)
    attack2s = moving_avg(attack2, smooth_win)

    # Detect peaks / windows per camera
    log("[INFO] Detecting peaks...", verbose)
    w1 = peak_windows(attack1s, args.fps, args.thresh_pct, args.min_sep_s, cam=1)
    w2 = peak_windows(attack2s, args.fps, args.thresh_pct, args.min_sep_s, cam=2)
    log(f"  cam1 peaks: {len(w1)}", verbose)
    log(f"  cam2 peaks: {len(w2)}", verbose)

    # Merge windows across cameras
    intervals = [(w.start_s, w.end_s) for w in (w1 + w2)]
    merged = merge_intervals(intervals, gap_s=args.merge_gap_s)
    log(f"[INFO] Merged into {len(merged)} candidate windows (pre-sanitize)", verbose)

    # Sanitize only (no length clamp yet)
    sanitized: list[tuple[float, float]] = []
    for (s, e) in merged:
        s = max(0.0, float(s))
        e = max(s + 0.5, float(e))
        sanitized.append((s, e))
    merged = sanitized

    # Normalization (per-camera) - use P99
    p99_1 = float(np.percentile(attack1s, 99)) if attack1s.size else 1.0
    p99_2 = float(np.percentile(attack2s, 99)) if attack2s.size else 1.0

    rows: list[tuple[float, float, float, int, float]] = []

    log("[INFO] Scoring windows + selecting primary camera...", verbose)
    for (s, e) in merged:
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

        score = 0.55 * s_attack + 0.25 * rebound + 0.20 * drop_norm
        score *= (0.85 + 0.15 * conf)

        if conf < 0.15:
            score *= 0.85

        # Per-window clamp: small plays capped at max_win_s, big plays capped at max_big_win_s
        color = score_to_color(score)
        is_big = color_rank(color) >= color_rank(args.big_color_min)
        max_len = float(args.max_big_win_s) if is_big else float(args.max_win_s)
        if (e - s) > max_len:
            e = s + max_len

        rows.append((float(s), float(e), float(score), int(primary), float(conf)))

    log(
        f"[INFO] Window caps: small<={float(args.max_win_s):.1f}s, big({args.big_color_min}+)<={float(args.max_big_win_s):.1f}s",
        verbose,
    )

    # Cooldown suppression
    before = len(rows)
    rows_time_sorted = sorted(rows, key=lambda r: r[0])
    rows_cd = apply_cooldown(rows_time_sorted, float(args.cooldown_s))
    after = len(rows_cd)
    if args.cooldown_s > 0:
        log(f"[INFO] Cooldown suppression: {before} -> {after} (cooldown={args.cooldown_s:.1f}s)", verbose)

    # Write debug events.csv (sorted by score desc)
    log(f"[INFO] Writing outputs: {args.out_csv}, {args.out_markers}", verbose)
    with open(args.out_csv, "w", encoding="utf-8") as f:
        f.write("start_s,end_s,score,primary_cam,confidence\n")
        for r in sorted(rows_cd, key=lambda x: -x[2]):
            f.write(f"{r[0]:.3f},{r[1]:.3f},{r[2]:.4f},{r[3]},{r[4]:.4f}\n")

    # Write Resolve markers CSV (timeline markers)
    # Resolve: Timeline -> Import Markers from CSV
    with open(args.out_markers, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Frame", "Name", "Note", "Color", "Duration"])

        for (s, e, score, primary, conf) in rows_cd:
            color = score_to_color(score)

            in_frame = sec_to_frame(s, args.timeline_fps)
            win_s = max(0.0, e - s)
            dur_frames = max(1, int(round(win_s * args.timeline_fps)))

            name = f"PLAY cam{primary}"
            note = f"score={score:.2f} conf={conf:.2f} win={win_s:.1f}s"

            # RANGED marker for the play window
            w.writerow([in_frame, name, note, color, dur_frames])

            # Optional replay marker (point)
            if args.replay_markers:
                is_big = color_rank(color) >= color_rank(args.big_color_min)
                if is_big:
                    r_s = e + float(args.replay_offset_s)
                    r_frame = sec_to_frame(r_s, args.timeline_fps)
                    r_name = "REPLAY"
                    r_note = f"after={args.replay_offset_s:.1f}s from end | {name} | {note}"
                    w.writerow([r_frame, r_name, r_note, color, 1])

    log(f"[INFO] Marker count: {len(rows_cd)}", verbose)
    log("[DONE] Detection complete.", verbose)


if __name__ == "__main__":
    main()