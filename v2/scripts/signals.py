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

import atexit
import json
import os
import subprocess
import tempfile
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


_TEMP_MANIFESTS: list[str] = []


def cleanup_temp_manifests() -> None:
    """Delete the temporary concat manifests this process wrote (also runs at exit)."""
    while _TEMP_MANIFESTS:
        try:
            os.remove(_TEMP_MANIFESTS.pop())
        except OSError:
            pass


atexit.register(cleanup_temp_manifests)


def write_temp_manifest(lines: list[str], base_dir: str, prefix: str) -> str:
    """
    Write a temporary concat manifest. Relative 'file' entries are resolved
    against base_dir (the original manifest's folder), because ffmpeg resolves
    them against the manifest's own location. Removed by cleanup_temp_manifests.
    """
    out = []
    for line in lines:
        s = line.strip()
        if s.startswith("file "):
            p = s[5:].strip().strip("'\"")
            if not os.path.isabs(p):
                p = os.path.join(base_dir, p)
            line = f"file '{p}'"
        out.append(line)
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(out) + "\n")
    _TEMP_MANIFESTS.append(path)
    return path


def concat_input_args(video_path: str) -> list[str]:
    """
    Return the ffmpeg input arguments for a video path or a concat manifest.

    A camera sync offset is applied with '-ss' before the concat input. It is
    read from a '# seek <s>' comment (written by gopro_meta) or, for older
    manifests, from an 'inpoint' line, which is stripped: concat 'inpoint' on
    GoPro HEVC applies only about 1/3 of the offset (hhg-38a.11).
    """
    if not video_path.endswith(".txt"):
        return ["-i", video_path]
    with open(video_path) as f:
        lines = f.read().splitlines()
    seek = 0.0
    for line in lines:
        parts = line.strip().split()
        if parts[:2] == ["#", "seek"] and len(parts) == 3:
            seek = float(parts[2])
        elif parts[:1] == ["inpoint"] and len(parts) == 2:
            seek = float(parts[1])
    path = video_path
    if any(line.strip().startswith("inpoint") for line in lines):
        path = write_temp_manifest([l for l in lines if not l.strip().startswith("inpoint")],
                                   os.path.dirname(os.path.abspath(video_path)), "concat_noinpoint_")
    args = ["-ss", f"{seek:.3f}"] if seek > 0.0 else []
    return args + ["-f", "concat", "-safe", "0", "-i", path]


def _ffmpeg_gray_frames(
    video_path: str, fps: int, width: int
) -> tuple[Generator[np.ndarray, None, None], int, int]:
    """
    Yield grayscale (uint8) frames via ffmpeg at `fps` and `width`.
    Accepts either a single MP4 path or an ffmpeg concat manifest (.txt).
    Returns (generator, out_width, out_height).
    """
    is_concat = video_path.endswith(".txt")  # NEW

    if is_concat:  # NEW — get dims from first file in manifest
        first_file = None
        with open(video_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("file "):
                    first_file = line[5:].strip("'\"")
                    if not os.path.isabs(first_file):     # ffmpeg resolves entries against the manifest's folder
                        first_file = os.path.join(os.path.dirname(os.path.abspath(video_path)), first_file)
                    break
        if first_file is None:
            raise RuntimeError(f"No file entries found in concat manifest: {video_path}")
        ow, oh = ffprobe_dims(first_file)
    else:
        ow, oh = ffprobe_dims(video_path)

    scale_h = int(round(oh * (width / ow)))
    if scale_h % 2 == 1:
        scale_h += 1

    cmd = ["ffmpeg", "-v", "error", *concat_input_args(video_path)]
    cmd += [
        "-vf", f"fps={fps},scale={width}:{scale_h},format=gray",
        "-f", "rawvideo",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    if proc.stdout is None:
        raise RuntimeError("ffmpeg stdout not available")

    frame_size = width * scale_h

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
        *concat_input_args(video_path),
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

def _extract_single_signals(
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


def _recording_blocks(manifest: str) -> list[tuple[float, list[str]]]:
    """Parse '# recording <start>' blocks of a concat manifest (empty if none)."""
    blocks: list[tuple[float, list[str]]] = []
    with open(manifest) as f:
        for line in f.read().splitlines():
            parts = line.strip().split()
            if parts[:2] == ["#", "recording"] and len(parts) == 3:
                blocks.append((float(parts[2]), []))
            elif blocks and line.strip().startswith("file "):
                blocks[-1][1].append(line.strip())
    return blocks


def extract_signals(
    video_path: str,
    rois: dict,
    fps: int,
    width: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract net flow, slot flow and audio RMS for one camera (see
    _extract_single_signals). A manifest with several '# recording <start>'
    blocks is extracted one Recording at a time, and each result is placed at
    its start on the detection timeline, with zeros in the gaps, so both
    cameras stay aligned (hhg-38a.13).
    """
    blocks = _recording_blocks(video_path) if video_path.endswith(".txt") else []
    if len(blocks) < 2:
        return _extract_single_signals(video_path, rois, fps=fps, width=width, verbose=verbose)
    placed = []
    for start, files in blocks:
        header = ["ffconcat version 1.0"] + ([f"# seek {-start:.3f}"] if start < 0 else [])
        tmp = write_temp_manifest(header + files, os.path.dirname(os.path.abspath(video_path)), "recording_")
        placed.append((int(round(max(start, 0.0) * fps)), _extract_single_signals(tmp, rois, fps=fps, width=width, verbose=verbose)))
    n = max(i + len(sig[0]) for i, sig in placed)
    out = [np.zeros(n, dtype=np.float32) for _ in range(3)]
    for i, sig in placed:
        for k in range(3):
            out[k][i:i + len(sig[k])] = sig[k]
    return out[0], out[1], out[2]
