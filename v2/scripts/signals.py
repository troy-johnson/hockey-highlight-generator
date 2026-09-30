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
import hashlib
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


def _remove_temp_manifest(path: str) -> None:
    """Delete one temporary manifest now instead of at exit."""
    try:
        os.remove(path)
    except OSError:
        pass
    if path in _TEMP_MANIFESTS:
        _TEMP_MANIFESTS.remove(path)


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


def _videotoolbox_available() -> bool:
    """True when this ffmpeg has the VideoToolbox decoder and scale_vt filter (macOS)."""
    global _VT_CACHE
    if _VT_CACHE is None:
        try:
            accels = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True).stdout
            filters = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
            _VT_CACHE = "videotoolbox" in accels and "scale_vt" in filters
        except OSError:
            _VT_CACHE = False
    return _VT_CACHE


_VT_CACHE: bool | None = None


def _use_hwaccel() -> bool:
    """Hardware decode unless HHG_HWACCEL=0 or VideoToolbox is missing (4K HEVC decode is ~1.7x faster)."""
    return os.environ.get("HHG_HWACCEL", "1") != "0" and _videotoolbox_available()


class HardwareDecodeFailed(RuntimeError):
    """The VideoToolbox decoder stopped with an error after producing frames."""


def _ffmpeg_gray_frames(
    video_path: str, fps: int, width: int, hw: bool | None = None
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

    frame_size = width * scale_h

    def _cmd(hw: bool) -> list[str]:
        cmd = ["ffmpeg", "-v", "error"]
        if hw:
            cmd += ["-hwaccel", "videotoolbox", "-hwaccel_output_format", "videotoolbox_vld"]
            vf = f"fps={fps},scale_vt=w={width}:h={scale_h},hwdownload,format=nv12,format=gray"
        else:
            vf = f"fps={fps},scale={width}:{scale_h},format=gray"
        return cmd + [*concat_input_args(video_path), "-vf", vf, "-f", "rawvideo", "pipe:1"]

    def _read(cmd: list[str], state: dict) -> Generator[np.ndarray, None, None]:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        if proc.stdout is None:
            raise RuntimeError("ffmpeg stdout not available")
        while True:
            buf = proc.stdout.read(frame_size)
            if len(buf) < frame_size:
                proc.stdout.close()
                state["rc"] = proc.wait()
                return
            yield np.frombuffer(buf, dtype=np.uint8).reshape((scale_h, width))

    def _gen() -> Generator[np.ndarray, None, None]:
        use_hw = _use_hwaccel() if hw is None else hw
        state: dict = {}
        n = 0
        for fr in _read(_cmd(use_hw), state):
            n += 1
            yield fr
        if use_hw and n == 0:          # hardware path produced nothing: decode in software instead
            use_hw, state, n = False, {}, 0
            for fr in _read(_cmd(False), state):
                n += 1
                yield fr
        if state.get("rc"):
            if use_hw:                 # frames already consumed are incomplete: the caller restarts in software
                raise HardwareDecodeFailed(f"VideoToolbox decode of {video_path} failed after {n} frames")
            # Software errors usually mean a truncated file (battery died mid-chapter): keep what decoded.
            print(f"[WARN] ffmpeg decode of {video_path} ended with an error after {n} frames; "
                  "keeping the frames decoded so far", flush=True)

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


_CROP_PAD = 48  # px around the ROIs so Farneback's pyramid sees context at the edges


def _roi_crop(net: ROI, slot: ROI, width: int, height: int, pad: int = _CROP_PAD) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1): the union of the net and slot ROIs plus padding, clamped to the frame.
    Flow is computed only here: ROI means match full-frame flow within ~1% at ~4x the speed."""
    x0 = max(0, min(net.x, slot.x) - pad)
    y0 = max(0, min(net.y, slot.y) - pad)
    x1 = min(width, max(net.x + net.w, slot.x + slot.w) + pad)
    y1 = min(height, max(net.y + net.h, slot.y + slot.h) + pad)
    return x0, y0, x1, y1


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

def _flow_values(video_path: str, rois: dict, fps: int, width: int, verbose: bool, hw: bool | None):
    """Per-frame-pair net and slot flow means for one video (flow on the ROI crop only)."""
    frames, w, h = _ffmpeg_gray_frames(video_path, fps=fps, width=width, hw=hw)

    net_roi: ROI = rois["net"].clamp_to(w, h)
    slot_roi: ROI = rois["slot"].clamp_to(w, h)
    x0, y0, x1, y1 = _roi_crop(net_roi, slot_roi, w, h)
    net_roi = ROI(net_roi.x - x0, net_roi.y - y0, net_roi.w, net_roi.h)
    slot_roi = ROI(slot_roi.x - x0, slot_roi.y - y0, slot_roi.w, slot_roi.h)

    prev: np.ndarray | None = None
    net_vals: list[float] = []
    slot_vals: list[float] = []
    sampled = 0

    for fr in frames:
        fr = np.ascontiguousarray(fr[y0:y1, x0:x1])
        sampled += 1
        if verbose and sampled % 500 == 0:
            print(f"  [signals] frames processed: {sampled} ({os.path.basename(video_path)})", flush=True)

        if prev is None:
            prev = fr
            continue

        mag = _flow_magnitude(prev, fr)
        prev = fr

        net_vals.append(_roi_mean(mag, net_roi))
        slot_vals.append(_roi_mean(mag, slot_roi))
    return net_vals, slot_vals, sampled, w, h


def _extract_single_signals(
    video_path: str,
    rois: dict,
    fps: int,
    width: int,
    verbose: bool = False,
    with_audio: bool = True,
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

    try:
        net_vals, slot_vals, sampled, w, h = _flow_values(video_path, rois, fps, width, verbose, hw=None)
    except HardwareDecodeFailed as exc:
        print(f"[WARN] {exc}; decoding this camera again in software", flush=True)
        net_vals, slot_vals, sampled, w, h = _flow_values(video_path, rois, fps, width, verbose, hw=False)

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

    if not with_audio:  # audio weight 0: skip decoding the whole file again
        return net_flow, slot_flow, np.zeros(n_frames, dtype=np.float32)

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


# ---------------------------------------------------------------------------
# Signal cache (hhg-3r5.77)
# ---------------------------------------------------------------------------
# One .npz file per Recording keeps the three signal arrays (a few hundred KB
# per hour of footage). No frames or proxies are cached. The key holds the
# identity (path, size, mtime) of every input file, every setting that
# changes the signals, the decode path, and a hash of this file, so a change
# to the extraction code makes the old entries unused.

SIGNALS_VERSION = 1
_CODE_HASH: str | None = None


def _code_hash() -> str:
    """Hash of signals.py; a code change gives new cache keys."""
    global _CODE_HASH
    if _CODE_HASH is None:
        with open(os.path.abspath(__file__), "rb") as f:
            _CODE_HASH = hashlib.sha256(f.read()).hexdigest()[:16]
    return _CODE_HASH


def _manifest_inputs(lines: list[str], base_dir: str) -> tuple[list[str], float]:
    """Input file paths and seek value of manifest lines ('# seek' or a legacy 'inpoint')."""
    files: list[str] = []
    seek = 0.0
    for line in lines:
        t = line.strip()
        parts = t.split()
        if parts[:2] == ["#", "seek"] and len(parts) == 3:
            seek = float(parts[2])
        elif parts[:1] == ["inpoint"] and len(parts) == 2:
            seek = float(parts[1])
        elif t.startswith("file "):
            path = t[5:].strip().strip("'\"")
            files.append(path if os.path.isabs(path) else os.path.join(base_dir, path))
    return files, seek


def _file_identity(path: str) -> list:
    st = os.stat(path)
    return [os.path.abspath(path), st.st_size, st.st_mtime_ns]


def signal_cache_key(files: list[str], seek: float, rois: dict, fps: int, width: int,
                     with_audio: bool) -> str:
    """Hash of the inputs and settings that change one Recording's signals."""
    def roi(r):
        return [r.x, r.y, r.w, r.h] if isinstance(r, ROI) else r
    payload = {
        "version": SIGNALS_VERSION,
        "code": _code_hash(),
        "hwaccel": _use_hwaccel(),
        "files": [_file_identity(f) for f in files],
        "seek": round(seek, 3),
        "rois": {k: roi(v) for k, v in sorted(rois.items())},
        "fps": fps, "width": width, "with_audio": bool(with_audio),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def _cached_single_signals(video_path: str, lines: list[str] | None, base_dir: str, rois: dict,
                           fps: int, width: int, verbose: bool, with_audio: bool, cache_dir: str | None):
    """_extract_single_signals with an optional cache lookup around it."""
    if not cache_dir:
        return _extract_single_signals(video_path, rois, fps=fps, width=width, verbose=verbose, with_audio=with_audio)
    if lines is None:
        if video_path.endswith(".txt"):
            with open(video_path) as f:
                lines = f.read().splitlines()
            files, seek = _manifest_inputs(lines, os.path.dirname(os.path.abspath(video_path)))
        else:
            files, seek = [video_path], 0.0
    else:
        files, seek = _manifest_inputs(lines, base_dir)
    try:
        key = signal_cache_key(files, seek, rois, fps, width, with_audio)
    except OSError:  # an input is missing: let ffmpeg report it
        return _extract_single_signals(video_path, rois, fps=fps, width=width, verbose=verbose, with_audio=with_audio)
    path = os.path.join(cache_dir, f"signals_{key}.npz")
    if os.path.exists(path):
        try:
            with np.load(path) as z:
                sig = (z["net"].astype(np.float32), z["slot"].astype(np.float32), z["audio"].astype(np.float32))
            print(f"[signals] cache hit: {os.path.basename(files[0]) if files else video_path} "
                  f"({len(sig[0])} flow values)", flush=True)
            return sig
        except Exception as exc:  # damaged cache file: compute again
            print(f"[WARN] signal cache file {path} unreadable ({exc}); computing again", flush=True)
    sig = _extract_single_signals(video_path, rois, fps=fps, width=width, verbose=verbose, with_audio=with_audio)
    tmp = path + ".tmp.npz"
    try:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(tmp, net=sig[0], slot=sig[1], audio=sig[2])
        os.replace(tmp, path)
    except OSError as exc:  # the cache must not stop detection
        print(f"[WARN] could not write signal cache file {path} ({exc})", flush=True)
    return sig


def extract_signals(
    video_path: str,
    rois: dict,
    fps: int,
    width: int,
    verbose: bool = False,
    with_audio: bool = True,
    cache_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract net flow, slot flow and audio RMS for one camera (see
    _extract_single_signals). A manifest with several '# recording <start>'
    blocks is extracted one Recording at a time, and each result is placed at
    its start on the detection timeline, with zeros in the gaps, so both
    cameras stay aligned (hhg-38a.13).

    cache_dir: when set, each Recording's signals are kept there and used
    again while its input files and settings are unchanged (hhg-3r5.77).
    """
    blocks = _recording_blocks(video_path) if video_path.endswith(".txt") else []
    if len(blocks) < 2:
        return _cached_single_signals(video_path, None, "", rois, fps, width, verbose, with_audio, cache_dir)
    placed = []
    for start, files in blocks:
        header = ["ffconcat version 1.0"] + ([f"# seek {-start:.3f}"] if start < 0 else [])
        tmp = write_temp_manifest(header + files, os.path.dirname(os.path.abspath(video_path)), "recording_")
        try:
            placed.append((int(round(max(start, 0.0) * fps)),
                           _cached_single_signals(tmp, header + files, os.path.dirname(os.path.abspath(video_path)), rois, fps, width, verbose, with_audio, cache_dir)))
        finally:
            _remove_temp_manifest(tmp)
    n = max(i + len(sig[0]) for i, sig in placed)
    out = [np.zeros(n, dtype=np.float32) for _ in range(3)]
    for i, sig in placed:
        for k in range(3):
            out[k][i:i + len(sig[k])] = sig[k]
    return out[0], out[1], out[2]
