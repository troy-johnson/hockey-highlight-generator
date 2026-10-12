"""Read small frame samples into a separate cache for published-clip matching."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    duration = float(info["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"{path}: invalid duration")
    return {"width": int(stream["width"]), "height": int(stream["height"]), "duration_s": duration}


def read_frame(path: Path, t: float, width=640) -> np.ndarray:
    if not math.isfinite(t) or t < 0:
        raise ValueError("frame time must be finite and nonnegative")
    info = probe(path)
    height = round(width * info["height"] / info["width"] / 2) * 2
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t:.6f}", "-i", str(path),
         "-frames:v", "1", "-vf", f"scale={width}:{height},format=gray", "-f", "rawvideo", "pipe:1"],
        capture_output=True, check=True)
    if len(result.stdout) != width * height:
        raise ValueError(f"{path}: no complete frame at {t:.3f} seconds")
    return np.frombuffer(result.stdout, np.uint8).reshape(height, width)


def sample_recording(files: list[Path], cache: Path, *, fps=1.0, width=320,
                     keyframes_only=True, start_s=0.0, duration_s=None) -> tuple[np.ndarray, np.ndarray, Path]:
    """Cache approximate coarse samples or fully decoded local refinement samples."""
    if (not files or not math.isfinite(fps) or fps <= 0 or width < 32
            or not math.isfinite(start_s) or start_s < 0
            or duration_s is not None and (not math.isfinite(duration_s) or duration_s <= 0)):
        raise ValueError("invalid frame sampling settings")
    files = [p.resolve(strict=True) for p in files]
    info = [probe(p) for p in files]
    total = sum(p["duration_s"] for p in info)
    height = round(width * info[0]["height"] / info[0]["width"] / 2) * 2
    settings = {"version": 1, "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "fps": fps, "width": width, "height": height, "keyframes_only": keyframes_only,
                "start_s": start_s, "duration_s": duration_s,
                "files": [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in files]}
    identity = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:24]
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / f"frames_{identity}.npz"
    if output.exists():
        with np.load(output, allow_pickle=False) as data:
            frames, times = data["frames"], data["times"]
        if (frames.ndim != 3 or frames.shape[1:] != (height, width) or frames.dtype != np.uint8
                or times.ndim != 1 or len(times) != len(frames) or not len(times)
                or not np.isfinite(times).all()
                or not np.allclose(times, start_s + np.arange(len(times)) / fps)):
            raise ValueError(f"invalid frame cache: {output}")
        return frames, times, output
    if start_s >= total:
        raise ValueError("frame sample starts beyond the Recording")
    with tempfile.TemporaryDirectory(prefix="frame_match_", dir=cache) as directory:
        manifest = Path(directory) / "input.txt"
        manifest.write_text("ffconcat version 1.0\n" + "".join(
            "file '" + str(p).replace("'", "'\\''") + "'\n" for p in files))
        command = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "2"]
        if keyframes_only:
            command += ["-skip_frame", "nokey"]
        command += ["-ss", f"{start_s:.6f}", "-f", "concat", "-safe", "0", "-i", str(manifest)]
        if duration_s is not None:
            command += ["-t", f"{duration_s:.6f}"]
        command += ["-an", "-vf", f"fps={fps},scale={width}:{height},format=gray", "-f", "rawvideo", "pipe:1"]
        result = subprocess.run(command, capture_output=True, check=True)
        frame_size = height * width
        if not result.stdout or len(result.stdout) % frame_size:
            raise ValueError("frame extraction returned incomplete data")
        frames = np.frombuffer(result.stdout, np.uint8).reshape(-1, height, width)
        expected = min(total - start_s, duration_s) if duration_s is not None else total - start_s
        if abs(len(frames) / fps - expected) > max(2.0, 2 / fps):
            raise ValueError("frame extraction did not cover the requested duration")
        times = start_s + np.arange(len(frames), dtype=np.float64) / fps
        temporary = Path(directory) / "frames.npz"
        np.savez_compressed(temporary, frames=frames, times=times)
        os.replace(temporary, output)
    return frames, times, output
