# v2/scripts/signals.py
#
# Signal extraction for the V2 hockey highlight detection pipeline.
#
# Returns three energy signals per camera, all aligned to the same frame
# timeline at the analysis fps:
#   - net_flow   : mean optical flow magnitude in the net ROI
#   - slot_flow  : mean optical flow magnitude in the slot ROI
#   - audio_rms  : audio RMS energy, resampled to frame timestamps
#
# Usage (from detect_events.py):
#   from signals import load_rois, extract_signals
#   rois = load_rois("rois.json")
#   net_flow, slot_flow, audio_rms = extract_signals(
#       "cam1.mp4", rois["camera_1"], fps=12, width=1280
#   )

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Generator

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ROI loading
# ---------------------------------------------------------------------------

def load_rois(path: str) -> dict:
    """
    Load rois.json produced by roi_picker.py.

    Expected format:
    {
      "camera_1": {"net": [x,y,w,h], "slot": [x,y,w,h]},
      "camera_2": {"net": [x,y,w,h], "slot": [x,y,w,h]}
    }
    Returns {"camera_1": {"net": ROI, "slot": ROI}, "camera_2": {...}}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def to_roi(lst) -> ROI:
        if lst is None or len(lst) != 4:
            raise ValueError(f"Bad ROI {lst!r} in {path}")
        return ROI(int(lst[0]), int(lst[1]), int(lst[2]), int(lst[3]))

    return {
        "camera_1": {
            "net": to_roi(data["camera_1"]["net"]),
            "slot": to_roi(data["camera_1"]["slot"]),
        },
        "camera_2": {
            "net": to_roi(data["camera_2"]["net"]),
            "slot": to_roi(data["camera_2"]["slot"]),
        },
    }


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def ffprobe_dims(video_path: str) -> tuple[int, int]:
    """Return (width, height) of the first video stream."""
    probe = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            video_path,
        ]
    )
    j = json.loads(probe)
    w = int(j["streams"][0]["width"])
    h = int(j["streams"][0]["height"])
    return w, h


def _ffmpeg_gray_frames(
    video_path: str, fps: int, width: int
) -> tuple[Generator[np.ndarray, None, None], int, int]:
    """
    Yield grayscale (uint8) frames via ffmpeg at `fps` and `width`.
    Returns (generator, out_width, out_height).
    ffmpeg does all scaling/resampling — Python only slices numpy views.
    """
    ow, oh = ffprobe_dims(video_path)
    scale_h = int(round(oh * (width / ow)))
    if scale_h % 2 == 1:
        scale_h += 1  # ffmpeg requires even dimensions

    cmd = [
        "ffmpeg", "-v", "error",
        "-i", video_path,
        "-vf", f"fps={fps},scale={width}:{scale_h},format=gray",
        "-f", "rawvideo",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    if proc.stdout is None:
        raise RuntimeError("ffmpeg stdout not available")

    frame_size = width * scale_h  # gray8: 1 byte per pixel

    def _gen() -> Generator[np.ndarray, None, None]:
        while True:
            buf = proc.stdout.read(frame_size)
            if len(buf) < frame_size:
                proc.stdout.close()
                proc.wait()
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape((scale_h, width))

    return _gen(), width, scale_h


# ---------------------------------------------------------------------------
# Optical flow helpers
# ---------------------------------------------------------------------------

# Farneback parameters — tuned for 1280-wide hockey frames at 12 fps.
# pyr_scale=0.5, levels=3, winsize=15 gives a good speed/accuracy tradeoff.
_FARNEBACK_PARAMS: dict = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


def _flow_magnitude(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel optical flow magnitude between two grayscale frames.
    Returns float32 array of shape (H, W).
    """
    flow = cv2.calcOpticalFlowFarneback(prev, curr, None, **_FARNEBACK_PARAMS)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag  # float32, H×W


def _roi_mean(mag: np.ndarray, roi: ROI) -> float:
    """Return mean flow magnitude within an ROI. Returns 0.0 for empty patches."""
    patch = mag[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
    if patch.size == 0:
        return 0.0
    return float(patch.mean())


# ---------------------------------------------------------------------------
# Audio RMS extraction
# ---------------------------------------------------------------------------

_AUDIO_SAMPLE_RATE = 22050  # Hz — low enough for speed, ample for crowd RMS
_AUDIO_WINDOW_S = 0.5       # RMS window size in seconds


def extract_audio_rms(video_path: str, fps: int, n_frames: int) -> np.ndarray:
    """
    Extract audio RMS energy from `video_path`, windowed at 0.5 s intervals,
    then interpolated to match the video signal length (n_frames samples).

    Returns float32 array of length n_frames.
    Returns zeros if the video has no audio stream.
    """
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32)

    # Decode the entire audio stream as mono PCM int16.
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(_AUDIO_SAMPLE_RATE),
        "-ac", "1",
        "-f", "s16le",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw = proc.stdout.read()
    proc.wait()

    if not raw:
        # No audio stream — return zeros so detect_events can set audio_weight=0
        return np.zeros(n_frames, dtype=np.float32)

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

    # Compute RMS in non-overlapping 0.5 s windows.
    window_samples = max(1, int(_AUDIO_WINDOW_S * _AUDIO_SAMPLE_RATE))
    n_windows = max(1, len(audio) // window_samples)

    rms = np.zeros(n_windows, dtype=np.float32)
    for i in range(n_windows):
        chunk = audio[i * window_samples : (i + 1) * window_samples]
        if chunk.size > 0:
            rms[i] = float(np.sqrt(np.mean(chunk ** 2)))

    # Timestamps for the center of each RMS window.
    rms_times = (np.arange(n_windows, dtype=np.float32) + 0.5) * _AUDIO_WINDOW_S

    # Timestamps of each video signal sample (aligned with optical flow output,
    # which starts at frame index 1 since frame 0 is consumed as `prev`).
    video_times = np.arange(n_frames, dtype=np.float32) / fps

    # Linear interpolation, clamping at boundaries.
    out = np.interp(video_times, rms_times, rms).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Main public interface
# ---------------------------------------------------------------------------

def extract_signals(
    video_path: str,
    rois: dict,
    fps: int,
    width: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract three energy signals from a single camera video.

    Parameters
    ----------
    video_path : str
        Path to the camera MP4.
    rois : dict
        {"net": ROI, "slot": ROI} — as returned by load_rois()["camera_N"].
    fps : int
        Analysis frame rate (frames extracted by ffmpeg).
    width : int
        Scale width for analysis frames (height computed to preserve aspect).
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    net_flow : np.ndarray, float32
        Mean optical flow magnitude in the net ROI, one value per frame pair.
    slot_flow : np.ndarray, float32
        Mean optical flow magnitude in the slot ROI, one value per frame pair.
    audio_rms : np.ndarray, float32
        Audio RMS energy interpolated to match net_flow / slot_flow length.

    All three arrays have the same length = (total_sampled_frames - 1).
    """
    if verbose:
        print(f"[signals] Processing {video_path}", flush=True)
    t0 = time.time()

    frames, w, h = _ffmpeg_gray_frames(video_path, fps=fps, width=width)

    net_roi: ROI = rois["net"].clamp_to(w, h)
    slot_roi: ROI = rois["slot"].clamp_to(w, h)

    prev: np.ndarray | None = None
    net_vals: list[float] = []
    slot_vals: list[float] = []
    sampled = 0

    for fr in frames:
        sampled += 1
        if verbose and sampled % 500 == 0:
            print(f"  [signals] frames processed: {sampled}", flush=True)

        if prev is None:
            prev = fr
            continue

        mag = _flow_magnitude(prev, fr)
        prev = fr

        net_vals.append(_roi_mean(mag, net_roi))
        slot_vals.append(_roi_mean(mag, slot_roi))

    n_frames = len(net_vals)

    if verbose:
        dt = time.time() - t0
        print(
            f"[signals] Flow done: {video_path} in {dt:.1f}s "
            f"({sampled} frames sampled, {n_frames} flow values)",
            flush=True,
        )

    net_flow = np.array(net_vals, dtype=np.float32)
    slot_flow = np.array(slot_vals, dtype=np.float32)

    if verbose:
        print(f"[signals] Extracting audio RMS for {video_path}...", flush=True)

    audio_rms = extract_audio_rms(video_path, fps=fps, n_frames=n_frames)

    if verbose:
        print(
            f"[signals] Audio done: {n_frames} samples, "
            f"RMS range [{audio_rms.min():.1f}, {audio_rms.max():.1f}]",
            flush=True,
        )

    return net_flow, slot_flow, audio_rms
