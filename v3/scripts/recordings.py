# v3/scripts/recordings.py
"""
Cameras and Recordings from GoPro files (hhg-38a.13).

- A camera is identified by the serial in the 'CASN' GPMF atom, which HERO
  cameras write into every MP4 (near the end of the file, in the moov box).
- A Recording is one continuous capture; the camera splits it into chapters
  named G<X|H><chapter:2><file number:4>.MP4. Chapters of one Recording share
  the file number. A camera that is stopped and restarted gets a new number.
- A Recording is 'black' when the lens was covered (sampled frames nearly black).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np

_NAME = re.compile(r"^G[XH](\d{2})(\d{4})\.MP4$", re.IGNORECASE)
_SCAN_BYTES = 8_000_000
_SERIAL = re.compile(r"^[A-Z0-9]{8,24}$")
BLACK_MEAN_LUMA = 16.0      # 0..255; a covered lens reads well below this
BLACK_SAMPLES = 9          # every sample must be dark: covered-then-uncovered footage is kept


def parse_gopro_name(name: str) -> tuple[int, int] | None:
    """Return (chapter, file_number) for a GoPro chapter file name, else None."""
    m = _NAME.match(name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def group_recordings(paths: list[str]) -> list[list[str]]:
    """Group chapter paths into Recordings, ordered by file number then chapter.
    Files without GoPro names form one Recording, sorted by name (older layouts)."""
    named = [(parse_gopro_name(Path(p).name), p) for p in paths]
    if any(key is None for key, _ in named):
        return [sorted(paths)]
    groups: dict[int, list[tuple[int, str]]] = {}
    for (chapter, number), p in named:
        groups.setdefault(number, []).append((chapter, p))
    return [[p for _, p in sorted(groups[n])] for n in sorted(groups)]


def read_camera_serial(path: str) -> str | None:
    """Read the camera serial from the 'CASN' atom (searched at both ends of the file)."""
    size = Path(path).stat().st_size
    with open(path, "rb") as f:
        chunks = [f.read(min(size, _SCAN_BYTES))]
        if size > _SCAN_BYTES:
            f.seek(size - _SCAN_BYTES)
            chunks.append(f.read())
    for data in chunks:
        i = data.find(b"CASN")
        while 0 <= i and i + 8 <= len(data):
            # GPMF key: 4-byte key, type char 'c', sample size, repeat count (big-endian 16 bit)
            type_char, sample_size = data[i + 4], data[i + 5]
            repeat = int.from_bytes(data[i + 6:i + 8], "big")
            if type_char == ord("c") and sample_size == 1:
                raw = data[i + 8:i + 8 + repeat]
                serial = raw.split(b"\0", 1)[0].decode("ascii", "ignore").strip()
                if _SERIAL.match(serial):
                    return serial
            i = data.find(b"CASN", i + 4)             # invalid match: keep searching
    return None


def _duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    return float(out) if out else 0.0


def is_black_recording(recording: list[str]) -> bool:
    """
    True when every frame sampled across the Recording is nearly black (lens
    covered the whole time). A Recording that is dark for a while and then shows
    play is kept: the dark part simply has no motion.
    """
    durations = [_duration(p) for p in recording]
    total = sum(durations)
    lumas = []
    for t in np.linspace(0.05, 0.95, BLACK_SAMPLES) * total:
        for path, d in zip(recording, durations):
            if t <= d:
                break
            t -= d
        raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t:.1f}", "-i", path, "-frames:v", "1",
                              "-vf", "scale=64:36,format=gray", "-f", "rawvideo", "pipe:1"], capture_output=True).stdout
        if raw:
            lumas.append(float(np.frombuffer(raw, np.uint8).mean()))
    return bool(lumas) and max(lumas) < BLACK_MEAN_LUMA
