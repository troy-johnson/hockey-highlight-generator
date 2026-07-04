# V3 Highlight Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the manual GoPro export + scrubbing workflow with `hockeydetect game_folder` → `compile_reel.py` → pre-cut multicam candidate reel in Resolve.

**Architecture:** Two command-line scripts (`discover.py`, `gopro_meta.py`) handle chapter discovery and timecode sync; `signals.py` gains concat manifest support so detection runs directly on raw GoPro chapters; a new Resolve script (`compile_reel.py`) imports chapters, creates a multicam clip, and assembles all detected events on Track 1 with angles pre-switched.

**Tech Stack:** Python 3.10+, ffmpeg/ffprobe, OpenCV/NumPy/SciPy (unchanged), DaVinciResolveScript (Resolve-side only), pytest.

---

## File map

| Action | Path | Responsibility |
|---|---|---|
| Create | `v3/scripts/discover.py` | Scan `cam1/`/`cam2/` subfolders, validate, write `chapters.json` |
| Create | `v3/scripts/gopro_meta.py` | Extract timecodes via ffprobe, compute offset, write `sync_info.json` + concat manifests |
| Create | `v3/resolve_scripts/compile_reel.py` | Resolve script: import chapters → multicam → place clips on Track 1 |
| Modify | `v2/scripts/signals.py` | `_ffmpeg_gray_frames()` accepts `.txt` concat manifest path |
| Modify | `run_detect.sh` | Folder mode: call v3 discovery + metadata scripts, pass concat manifests to detection |
| Create | `tests/conftest.py` | Add `v2/scripts` and `v3/scripts` to `sys.path` |
| Create | `tests/test_discover.py` | Unit tests for `discover.py` |
| Create | `tests/test_gopro_meta.py` | Unit tests for `gopro_meta.py` |
| Create | `tests/test_signals_concat.py` | Unit test for concat manifest support in `signals.py` |
| Create | `tests/test_compile_reel_logic.py` | Unit tests for pure-Python functions in `compile_reel.py` |

---

## Task 1: Test scaffold + `discover.py`

**Files:**
- Create: `tests/conftest.py`
- Create: `v3/scripts/discover.py`
- Create: `tests/test_discover.py`

- [ ] **Step 1.1 — Create directory structure**

```bash
mkdir -p v3/scripts v3/resolve_scripts tests
```

- [ ] **Step 1.2 — Create `pytest.ini`** (project root)

```ini
[pytest]
testpaths = tests
```

- [ ] **Step 1.3 — Create `tests/conftest.py`**

```python
# tests/conftest.py
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "v2", "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "v3", "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "v3", "resolve_scripts"))
```

- [ ] **Step 1.4 — Install pytest into the venv**

```bash
source .venv/bin/activate
pip install pytest
```

- [ ] **Step 1.5 — Write failing tests**

```python
# tests/test_discover.py
import json
import pytest
from pathlib import Path
from discover import discover


def _setup(tmp_path, cam1_files=None, cam2_files=None):
    if cam1_files is not None:
        (tmp_path / "cam1").mkdir()
        for f in cam1_files:
            (tmp_path / "cam1" / f).touch()
    if cam2_files is not None:
        (tmp_path / "cam2").mkdir()
        for f in cam2_files:
            (tmp_path / "cam2" / f).touch()
    return str(tmp_path)


def test_finds_and_sorts_chapters(tmp_path):
    folder = _setup(tmp_path,
        cam1_files=["GOPRO1802.MP4", "GOPRO1801.MP4"],
        cam2_files=["GOPRO1901.MP4"])
    result = discover(folder)
    assert result["cam1"] == [
        str(tmp_path / "cam1" / "GOPRO1801.MP4"),
        str(tmp_path / "cam1" / "GOPRO1802.MP4"),
    ]
    assert result["cam2"] == [str(tmp_path / "cam2" / "GOPRO1901.MP4")]


def test_writes_chapters_json(tmp_path):
    folder = _setup(tmp_path,
        cam1_files=["GOPRO1801.MP4"], cam2_files=["GOPRO1901.MP4"])
    discover(folder)
    data = json.loads((tmp_path / "chapters.json").read_text())
    assert "cam1" in data and "cam2" in data
    assert len(data["cam1"]) == 1


def test_ignores_non_mp4(tmp_path):
    folder = _setup(tmp_path,
        cam1_files=["GOPRO1801.MP4", "notes.txt", "thumb.jpg"],
        cam2_files=["GOPRO1901.MP4"])
    result = discover(folder)
    assert len(result["cam1"]) == 1


def test_case_insensitive_extension(tmp_path):
    folder = _setup(tmp_path,
        cam1_files=["GOPRO1801.mp4"],
        cam2_files=["GOPRO1901.MP4"])
    result = discover(folder)
    assert len(result["cam1"]) == 1


def test_missing_cam1_exits(tmp_path):
    folder = _setup(tmp_path, cam2_files=["GOPRO1901.MP4"])
    with pytest.raises(SystemExit, match="cam1"):
        discover(folder)


def test_missing_cam2_exits(tmp_path):
    folder = _setup(tmp_path, cam1_files=["GOPRO1801.MP4"])
    with pytest.raises(SystemExit, match="cam2"):
        discover(folder)


def test_empty_cam1_exits(tmp_path):
    (tmp_path / "cam1").mkdir()
    (tmp_path / "cam2").mkdir()
    (tmp_path / "cam2" / "GOPRO1901.MP4").touch()
    with pytest.raises(SystemExit, match="No .MP4"):
        discover(str(tmp_path))
```

- [ ] **Step 1.6 — Run tests, confirm they fail**

```bash
pytest tests/test_discover.py -v
```
Expected: `ModuleNotFoundError: No module named 'discover'`

- [ ] **Step 1.7 — Implement `v3/scripts/discover.py`**

```python
# v3/scripts/discover.py
from __future__ import annotations

import json
import sys
from pathlib import Path


def discover(game_folder: str) -> dict[str, list[str]]:
    """
    Scan game_folder/cam1/ and game_folder/cam2/ for MP4 chapter files.
    Returns {"cam1": [...sorted paths...], "cam2": [...sorted paths...]}.
    Writes chapters.json to game_folder as a side effect.
    Raises SystemExit with a descriptive message on any validation failure.
    """
    root = Path(game_folder)
    result: dict[str, list[str]] = {}

    for cam in ("cam1", "cam2"):
        subfolder = root / cam
        if not subfolder.is_dir():
            sys.exit(f"[ERROR] {cam}/ subfolder not found in {game_folder}")
        chapters = sorted(
            str(p) for p in subfolder.iterdir()
            if p.suffix.upper() == ".MP4"
        )
        if not chapters:
            sys.exit(f"[ERROR] No .MP4 files found in {cam}/")
        result[cam] = chapters

    (root / "chapters.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: discover.py <game_folder>")
    chapters = discover(sys.argv[1])
    print(f"[discover] cam1: {len(chapters['cam1'])} chapters, "
          f"cam2: {len(chapters['cam2'])} chapters")
    print(f"[discover] Wrote chapters.json")
```

- [ ] **Step 1.8 — Run tests, confirm they pass**

```bash
pytest tests/test_discover.py -v
```
Expected: 7 passed

- [ ] **Step 1.9 — Commit**

```bash
git add v3/ tests/conftest.py tests/test_discover.py
git commit -m "feat: add discover.py with tests — chapter discovery for v3 pipeline"
```

---

## Task 2: `gopro_meta.py`

**Files:**
- Create: `v3/scripts/gopro_meta.py`
- Create: `tests/test_gopro_meta.py`

- [ ] **Step 2.1 — Write failing tests**

```python
# tests/test_gopro_meta.py
import json
import pytest
from unittest.mock import patch
from pathlib import Path
from gopro_meta import (
    timecode_to_seconds,
    extract_chapter_time,
    compute_sync,
    write_concat_manifest,
)

# ---------------------------------------------------------------------------
# timecode_to_seconds
# ---------------------------------------------------------------------------

def test_timecode_zero():
    assert timecode_to_seconds("00:00:00:00") == pytest.approx(0.0)

def test_timecode_one_second():
    assert timecode_to_seconds("00:00:01:00") == pytest.approx(1.0)

def test_timecode_frames():
    # 30 frames at 60fps = 0.5s
    assert timecode_to_seconds("00:00:00:30", fps=60) == pytest.approx(0.5)

def test_timecode_hours():
    assert timecode_to_seconds("01:00:00:00") == pytest.approx(3600.0)

def test_timecode_full():
    # 10:30:02:12 at 60fps = 37802 + 12/60 = 37802.2s
    assert timecode_to_seconds("10:30:02:12", fps=60) == pytest.approx(37802.2)

# ---------------------------------------------------------------------------
# extract_chapter_time — mocks ffprobe
# ---------------------------------------------------------------------------

_FFPROBE_WITH_TIMECODE = json.dumps({
    "streams": [],
    "format": {"tags": {"timecode": "10:30:00:00",
                        "creation_time": "2024-01-15T10:30:00.000000Z"}}
})

_FFPROBE_NO_TIMECODE = json.dumps({
    "streams": [],
    "format": {"tags": {"creation_time": "2024-01-15T10:30:00.500000Z"}}
})

def test_extract_uses_timecode_when_present(monkeypatch):
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **k: _FFPROBE_WITH_TIMECODE.encode()
    )
    seconds, source = extract_chapter_time("/fake/GOPRO1801.MP4")
    assert source == "timecode"
    assert seconds == pytest.approx(timecode_to_seconds("10:30:00:00"))

def test_extract_falls_back_to_creation_time(monkeypatch):
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **k: _FFPROBE_NO_TIMECODE.encode()
    )
    seconds, source = extract_chapter_time("/fake/GOPRO1801.MP4")
    assert source == "creation_time"
    # 10:30:00.5 → 37800.5s into the day
    assert seconds == pytest.approx(37800.5, abs=1.0)

# ---------------------------------------------------------------------------
# compute_sync
# ---------------------------------------------------------------------------

def _make_sync(cam1_s, cam2_s, cam1_src="timecode", cam2_src="timecode"):
    with patch("gopro_meta.extract_chapter_time") as mock:
        mock.side_effect = [(cam1_s, cam1_src), (cam2_s, cam2_src)]
        return compute_sync(["/fake/cam1/A.MP4"], ["/fake/cam2/B.MP4"])

def test_sync_cam2_later():
    # cam2 starts 2s after cam1 → cam1_detect_offset_s = 2.0
    info = _make_sync(37800.0, 37802.0)
    assert info["offset_s"] == pytest.approx(2.0)
    assert info["cam1_detect_offset_s"] == pytest.approx(2.0)
    assert info["cam2_detect_offset_s"] == pytest.approx(0.0)

def test_sync_cam1_later():
    # cam1 starts 1.5s after cam2 → cam2_detect_offset_s = 1.5
    info = _make_sync(37801.5, 37800.0)
    assert info["offset_s"] == pytest.approx(-1.5)
    assert info["cam1_detect_offset_s"] == pytest.approx(0.0)
    assert info["cam2_detect_offset_s"] == pytest.approx(1.5)

def test_sync_tiny_offset_treated_as_zero():
    info = _make_sync(37800.0, 37800.05)  # 0.05s < 0.1 threshold
    assert info["offset_s"] == pytest.approx(0.0)
    assert info["cam1_detect_offset_s"] == pytest.approx(0.0)
    assert info["cam2_detect_offset_s"] == pytest.approx(0.0)

def test_sync_implausible_offset_exits():
    with pytest.raises(SystemExit, match="implausibly large"):
        _make_sync(37800.0, 37800.0 + 61.0)

def test_sync_method_recorded():
    info = _make_sync(37800.0, 37800.5, "timecode", "timecode")
    assert info["sync_method"] == "timecode"

def test_sync_mixed_method_recorded():
    info = _make_sync(37800.0, 37800.5, "timecode", "creation_time")
    assert info["sync_method"] == "mixed"

# ---------------------------------------------------------------------------
# write_concat_manifest
# ---------------------------------------------------------------------------

def test_manifest_no_inpoint(tmp_path):
    out = str(tmp_path / "cam1_concat.txt")
    write_concat_manifest(["/a/GOPRO1801.MP4", "/a/GOPRO1802.MP4"], 0.0, out)
    content = Path(out).read_text()
    assert "ffconcat version 1.0" in content
    assert "inpoint" not in content
    assert "/a/GOPRO1801.MP4" in content
    assert "/a/GOPRO1802.MP4" in content

def test_manifest_with_inpoint(tmp_path):
    out = str(tmp_path / "cam1_concat.txt")
    write_concat_manifest(["/a/GOPRO1801.MP4", "/a/GOPRO1802.MP4"], 2.5, out)
    lines = Path(out).read_text().splitlines()
    # inpoint should follow the first file line
    file_idx = next(i for i, l in enumerate(lines) if "GOPRO1801" in l)
    assert "inpoint 2.500" in lines[file_idx + 1]
    # Second file should NOT have inpoint
    file2_idx = next(i for i, l in enumerate(lines) if "GOPRO1802" in l)
    if file2_idx + 1 < len(lines):
        assert "inpoint" not in lines[file2_idx + 1]
```

- [ ] **Step 2.2 — Run tests, confirm they fail**

```bash
pytest tests/test_gopro_meta.py -v
```
Expected: `ModuleNotFoundError: No module named 'gopro_meta'`

- [ ] **Step 2.3 — Implement `v3/scripts/gopro_meta.py`**

```python
# v3/scripts/gopro_meta.py
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Time conversion helpers
# ---------------------------------------------------------------------------

def timecode_to_seconds(tc: str, fps: int = 60) -> float:
    """Convert HH:MM:SS:FF timecode string to float seconds."""
    hh, mm, ss, ff = (int(x) for x in tc.split(":"))
    return hh * 3600.0 + mm * 60.0 + ss + ff / fps


def _creation_time_to_seconds(iso_str: str) -> float:
    """Parse ISO 8601 creation_time to seconds-since-midnight (UTC)."""
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return dt.hour * 3600.0 + dt.minute * 60.0 + dt.second + dt.microsecond / 1_000_000


# ---------------------------------------------------------------------------
# ffprobe extraction
# ---------------------------------------------------------------------------

def extract_chapter_time(chapter_path: str) -> tuple[float, str]:
    """
    Extract start time from the first chapter file using ffprobe.
    Returns (seconds_since_midnight, source) where source is
    "timecode" or "creation_time".
    Raises SystemExit if no usable metadata is found.
    """
    raw = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream_tags=timecode",
            "-show_entries", "format_tags=timecode,creation_time",
            "-of", "json",
            chapter_path,
        ]
    )
    data = json.loads(raw)

    # Try timecode from format tags first, then stream tags
    tc = (
        data.get("format", {}).get("tags", {}).get("timecode")
        or next(
            (s.get("tags", {}).get("timecode")
             for s in data.get("streams", [])
             if s.get("tags", {}).get("timecode")),
            None,
        )
    )
    if tc:
        try:
            return timecode_to_seconds(tc), "timecode"
        except Exception:
            pass  # fall through to creation_time

    ct = data.get("format", {}).get("tags", {}).get("creation_time")
    if ct:
        try:
            return _creation_time_to_seconds(ct), "creation_time"
        except Exception:
            pass

    sys.exit(f"[ERROR] No usable timecode or creation_time metadata in {chapter_path}")


# ---------------------------------------------------------------------------
# Sync computation
# ---------------------------------------------------------------------------

def compute_sync(
    cam1_chapters: list[str],
    cam2_chapters: list[str],
) -> dict:
    """
    Extract timecode from first chapter of each camera, compute alignment offset.
    Returns sync_info dict matching the spec schema.
    Raises SystemExit on implausible offset (> 60s).
    """
    cam1_s, cam1_src = extract_chapter_time(cam1_chapters[0])
    cam2_s, cam2_src = extract_chapter_time(cam2_chapters[0])

    raw_offset = cam2_s - cam1_s

    # Treat tiny offsets as zero
    if abs(raw_offset) < 0.1:
        raw_offset = 0.0

    if abs(raw_offset) > 60.0:
        sys.exit(
            f"[ERROR] Sync offset {raw_offset:.1f}s is implausibly large — "
            "likely a metadata error. Check GoPro timecode sync."
        )

    if raw_offset > 0:          # cam2 started later → skip cam1 start
        cam1_detect_offset_s = raw_offset
        cam2_detect_offset_s = 0.0
    elif raw_offset < 0:        # cam1 started later → skip cam2 start
        cam1_detect_offset_s = 0.0
        cam2_detect_offset_s = abs(raw_offset)
    else:
        cam1_detect_offset_s = 0.0
        cam2_detect_offset_s = 0.0

    sources = {cam1_src, cam2_src}
    if sources == {"timecode"}:
        sync_method = "timecode"
    elif sources == {"creation_time"}:
        sync_method = "creation_time"
    else:
        sync_method = "mixed"
        print(
            f"[WARN] Sync sources differ (cam1={cam1_src}, cam2={cam2_src}). "
            "Accuracy may be reduced.",
            flush=True,
        )

    return {
        "offset_s": raw_offset,
        "cam1_detect_offset_s": cam1_detect_offset_s,
        "cam2_detect_offset_s": cam2_detect_offset_s,
        "sync_method": sync_method,
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# Concat manifest writer
# ---------------------------------------------------------------------------

def write_concat_manifest(
    chapters: list[str],
    detect_offset_s: float,
    out_path: str,
) -> None:
    """
    Write an ffmpeg concat manifest.
    If detect_offset_s > 0, adds an inpoint on the first file to align cameras.
    """
    lines = ["ffconcat version 1.0"]
    for i, ch in enumerate(chapters):
        lines.append(f"file '{ch}'")
        if i == 0 and detect_offset_s > 0.0:
            lines.append(f"inpoint {detect_offset_s:.3f}")
    Path(out_path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(game_folder: str) -> None:
    chapters_path = Path(game_folder) / "chapters.json"
    if not chapters_path.exists():
        sys.exit(f"[ERROR] chapters.json not found in {game_folder}. Run discover.py first.")

    chapters = json.loads(chapters_path.read_text())

    print("[gopro_meta] Extracting timecodes...", flush=True)
    sync = compute_sync(chapters["cam1"], chapters["cam2"])

    # Write sync_info.json
    sync_path = Path(game_folder) / "sync_info.json"
    sync_path.write_text(json.dumps(sync, indent=2))
    print(f"[gopro_meta] sync_info.json written (offset={sync['offset_s']:.3f}s, "
          f"method={sync['sync_method']})", flush=True)

    # Write concat manifests (with inpoint for the earlier camera)
    for cam, key in (("cam1", "cam1_detect_offset_s"), ("cam2", "cam2_detect_offset_s")):
        out = str(Path(game_folder) / f"{cam}_concat.txt")
        write_concat_manifest(chapters[cam], sync[key], out)
        print(f"[gopro_meta] {cam}_concat.txt written ({len(chapters[cam])} chapters, "
              f"offset={sync[key]:.3f}s)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: gopro_meta.py <game_folder>")
    main(sys.argv[1])
```

- [ ] **Step 2.4 — Run tests, confirm they pass**

```bash
pytest tests/test_gopro_meta.py -v
```
Expected: 13 passed

- [ ] **Step 2.5 — Commit**

```bash
git add v3/scripts/gopro_meta.py tests/test_gopro_meta.py
git commit -m "feat: add gopro_meta.py — timecode extraction, sync computation, concat manifests"
```

---

## Task 3: `signals.py` concat manifest support

**Files:**
- Modify: `v2/scripts/signals.py` (lines 105–140, `_ffmpeg_gray_frames`)
- Create: `tests/test_signals_concat.py`

- [ ] **Step 3.1 — Write failing test**

```python
# tests/test_signals_concat.py
import subprocess
from unittest.mock import patch, MagicMock
from signals import _ffmpeg_gray_frames


def _make_popen_mock():
    mock_proc = MagicMock()
    mock_proc.stdout.read.return_value = b""
    return mock_proc


def test_mp4_path_no_concat_flags(tmp_path, monkeypatch):
    """Single MP4 path must NOT include -f concat flags."""
    captured = []

    def fake_popen(cmd, **kwargs):
        captured.append(cmd)
        return _make_popen_mock()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("signals.ffprobe_dims", lambda p: (1920, 1080))

    try:
        gen, w, h = _ffmpeg_gray_frames("video.mp4", fps=12, width=640)
        next(gen, None)
    except Exception:
        pass

    assert captured, "Popen was not called"
    cmd = captured[0]
    assert "-f" not in cmd or "concat" not in cmd


def test_txt_path_adds_concat_flags(tmp_path, monkeypatch):
    """A .txt concat manifest path must add -f concat -safe 0 before -i."""
    manifest = tmp_path / "cam1_concat.txt"
    manifest.write_text("ffconcat version 1.0\nfile '/fake/GOPRO1801.MP4'\n")

    captured = []

    def fake_popen(cmd, **kwargs):
        captured.append(cmd)
        return _make_popen_mock()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr("signals.ffprobe_dims", lambda p: (1920, 1080))

    try:
        gen, w, h = _ffmpeg_gray_frames(str(manifest), fps=12, width=640)
        next(gen, None)
    except Exception:
        pass

    assert captured, "Popen was not called"
    cmd = captured[0]
    assert "-f" in cmd
    concat_idx = cmd.index("-f")
    assert cmd[concat_idx + 1] == "concat"
    assert "-safe" in cmd
    assert "0" in cmd[cmd.index("-safe") + 1 :]


def test_txt_path_calls_ffprobe_on_first_chapter(tmp_path, monkeypatch):
    """For a concat manifest, ffprobe_dims is called on the first listed file."""
    manifest = tmp_path / "cam1_concat.txt"
    manifest.write_text(
        "ffconcat version 1.0\nfile '/fake/GOPRO1801.MP4'\nfile '/fake/GOPRO1802.MP4'\n"
    )
    probed = []

    def fake_ffprobe(path):
        probed.append(path)
        return (1920, 1080)

    monkeypatch.setattr("signals.ffprobe_dims", fake_ffprobe)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _make_popen_mock())

    try:
        gen, w, h = _ffmpeg_gray_frames(str(manifest), fps=12, width=640)
        next(gen, None)
    except Exception:
        pass

    assert probed and probed[0] == "/fake/GOPRO1801.MP4"
```

- [ ] **Step 3.2 — Run test, confirm it fails**

```bash
pytest tests/test_signals_concat.py -v
```
Expected: `FAILED` — `.txt` test will fail because concat flags aren't added yet.

- [ ] **Step 3.3 — Modify `v2/scripts/signals.py`**

Replace the `_ffmpeg_gray_frames` function body. The key changes are highlighted with `# NEW`:

```python
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
                    break
        if first_file is None:
            raise RuntimeError(f"No file entries found in concat manifest: {video_path}")
        ow, oh = ffprobe_dims(first_file)
    else:
        ow, oh = ffprobe_dims(video_path)

    scale_h = int(round(oh * (width / ow)))
    if scale_h % 2 == 1:
        scale_h += 1

    cmd = ["ffmpeg", "-v", "error"]
    if is_concat:  # NEW
        cmd += ["-f", "concat", "-safe", "0"]
    cmd += [
        "-i", video_path,
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
```

- [ ] **Step 3.4 — Run tests, confirm they pass**

```bash
pytest tests/test_signals_concat.py -v
```
Expected: 3 passed

- [ ] **Step 3.5 — Confirm existing signals behavior unchanged**

```bash
pytest tests/ -v
```
Expected: all tests pass (no regressions)

- [ ] **Step 3.6 — Commit**

```bash
git add v2/scripts/signals.py tests/test_signals_concat.py
git commit -m "feat: signals.py accepts ffmpeg concat manifests for chapter-based detection"
```

---

## Task 4: `run_detect.sh` V3 folder mode

**Files:**
- Modify: `run_detect.sh`

No unit tests for shell — this task ends with a manual smoke test.

- [ ] **Step 4.1 — Add `SCRIPTS_V3` variable and update folder mode**

Edit `run_detect.sh`. Add after the existing `SCRIPTS=` line:

```bash
SCRIPTS_V3="$REPO_DIR/v3/scripts"
```

Replace the entire `# ---- Folder mode ----` block (lines 59–100 in the current file) with:

```bash
# ---- Folder mode ----
if [ "$MODE" = "folder" ]; then
  if [ ! -d "$PROJECT_DIR" ]; then
    echo "[ERROR] Folder does not exist: $PROJECT_DIR"
    exit 1
  fi

  echo ""
  echo "Project folder: $PROJECT_DIR"

  # V3: discover chapters, write chapters.json
  echo "[INFO] Discovering GoPro chapters..."
  python "$SCRIPTS_V3/discover.py" "$PROJECT_DIR"

  # V3: extract timecodes, write sync_info.json + concat manifests
  echo "[INFO] Extracting timecodes and building sync..."
  python "$SCRIPTS_V3/gopro_meta.py" "$PROJECT_DIR"

  CAM1="$PROJECT_DIR/cam1_concat.txt"
  CAM2="$PROJECT_DIR/cam2_concat.txt"
  ROIS="$PROJECT_DIR/rois.json"
  OUT_CSV="$PROJECT_DIR/events.csv"
  OUT_MARKERS="$PROJECT_DIR/markers.csv"
  OUT_FCPXML="$PROJECT_DIR/markers.fcpxml"
  OUT_EDL="$PROJECT_DIR/markers.edl"

  echo "Cam1 (concat): $CAM1"
  echo "Cam2 (concat): $CAM2"

  # ROI picker needs a real video file — use first chapter from chapters.json
  if [ ! -f "$ROIS" ]; then
    echo ""
    echo "[INFO] rois.json not found. Launching ROI picker..."
    FIRST_CAM1=$(python -c "import json; print(json.load(open('$PROJECT_DIR/chapters.json'))['cam1'][0])")
    FIRST_CAM2=$(python -c "import json; print(json.load(open('$PROJECT_DIR/chapters.json'))['cam2'][0])")
    python "$SCRIPTS/roi_picker.py" \
      "$FIRST_CAM1" \
      "$FIRST_CAM2" \
      --out "$ROIS"

    if [ ! -f "$ROIS" ]; then
      echo "[ERROR] ROI picker did not create: $ROIS"
      exit 1
    fi
  fi
fi
```

File mode block (lines 103–137) is **unchanged**.

- [ ] **Step 4.2 — Verify shell syntax**

```bash
bash -n run_detect.sh
```
Expected: no output (clean parse)

- [ ] **Step 4.3 — Manual smoke test (requires real game folder)**

If a game folder with `cam1/` and `cam2/` subfolders is available:

```bash
hockeydetect /path/to/game_folder
```

Verify outputs exist:
```bash
ls /path/to/game_folder/chapters.json \
      /path/to/game_folder/sync_info.json \
      /path/to/game_folder/cam1_concat.txt \
      /path/to/game_folder/cam2_concat.txt \
      /path/to/game_folder/events.csv
```

If no game folder is available, skip the manual test — unit tests in Tasks 1–3 cover the components.

- [ ] **Step 4.4 — Commit**

```bash
git add run_detect.sh
git commit -m "feat: run_detect.sh folder mode now uses v3 chapter discovery + sync"
```

---

## Task 5: `compile_reel.py` — pure Python logic

**Files:**
- Create: `v3/resolve_scripts/compile_reel.py` (logic functions only, no Resolve imports yet)
- Create: `tests/test_compile_reel_logic.py`

- [ ] **Step 5.1 — Write failing tests**

```python
# tests/test_compile_reel_logic.py
import csv
import pytest
from pathlib import Path
from compile_reel import (
    score_to_color,
    Event,
    load_events,
    select_events,
    calc_source_frames,
)

# ---------------------------------------------------------------------------
# score_to_color
# ---------------------------------------------------------------------------

def test_color_red():
    assert score_to_color(1.61) == "Red"

def test_color_orange():
    assert score_to_color(1.31) == "Orange"

def test_color_yellow():
    assert score_to_color(1.01) == "Yellow"

def test_color_blue():
    assert score_to_color(0.99) == "Blue"

def test_color_boundary_red():
    assert score_to_color(1.60) == "Orange"  # > 1.60 required for Red

# ---------------------------------------------------------------------------
# load_events
# ---------------------------------------------------------------------------

def _write_events_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_s", "end_s", "score", "primary_cam", "confidence"])
        for r in rows:
            writer.writerow(r)

def test_load_events_basic(tmp_path):
    csv_path = tmp_path / "events.csv"
    _write_events_csv(csv_path, [
        [10.0, 25.0, 1.7, 1, 0.9],
        [60.0, 80.0, 1.1, 2, 0.6],
    ])
    events = load_events(str(csv_path))
    assert len(events) == 2
    assert events[0].start_s == pytest.approx(10.0)
    assert events[0].primary_cam == 1
    assert events[0].color == "Red"
    assert events[1].color == "Yellow"

def test_load_events_empty(tmp_path):
    csv_path = tmp_path / "events.csv"
    _write_events_csv(csv_path, [])
    assert load_events(str(csv_path)) == []

# ---------------------------------------------------------------------------
# select_events
# ---------------------------------------------------------------------------

def _make_event(start_s, end_s, score, cam=1):
    return Event(start_s=start_s, end_s=end_s, score=score,
                 primary_cam=cam, confidence=0.8, color=score_to_color(score))

def test_select_all_under_cap():
    events = [_make_event(0, 10, 1.7), _make_event(20, 30, 1.1)]
    result = select_events(events, max_reel_s=900)
    assert len(result) == 2

def test_select_sorted_by_start_s():
    events = [_make_event(30, 40, 1.5), _make_event(0, 10, 1.2)]
    result = select_events(events, max_reel_s=900)
    assert result[0].start_s == 0

def test_select_drops_blue_first():
    events = [
        _make_event(0, 10, 1.7),    # Red, 10s
        _make_event(20, 320, 0.5),  # Blue, 300s — makes total 310s > 100s cap
    ]
    result = select_events(events, max_reel_s=100)
    assert len(result) == 1
    assert result[0].color == "Red"

def test_select_drops_lowest_score_blue_first():
    events = [
        _make_event(0, 10, 0.9),    # Blue, score=0.9
        _make_event(20, 30, 0.4),   # Blue, score=0.4 — dropped first
        _make_event(40, 50, 1.7),   # Red, 10s
    ]
    result = select_events(events, max_reel_s=25)
    scores = {e.score for e in result}
    assert 0.4 not in scores  # lowest-score Blue dropped

def test_select_never_drops_red():
    # 100 Red events × 15s each = 1500s, well over 100s cap
    events = [_make_event(i * 20, i * 20 + 15, 1.8) for i in range(100)]
    result = select_events(events, max_reel_s=100)
    assert all(e.color == "Red" for e in result)
    assert len(result) == 100

# ---------------------------------------------------------------------------
# calc_source_frames
# ---------------------------------------------------------------------------

def test_calc_basic():
    e = _make_event(10.0, 20.0, 1.5)
    # detect_offset=5s, preroll=3s, postroll=2s, fps=60
    # src_in  = 5 + 10 - 3 = 12s → 720 frames
    # src_out = 5 + 20 + 2 = 27s → 1620 frames
    in_f, out_f = calc_source_frames(e, detect_offset_s=5.0, timeline_fps=60)
    assert in_f == 720
    assert out_f == 1620

def test_calc_clamps_start_to_zero():
    e = _make_event(1.0, 10.0, 1.5)
    # detect_offset=0, preroll=3s → would give -2s → clamp to 0
    in_f, out_f = calc_source_frames(e, detect_offset_s=0.0, timeline_fps=60)
    assert in_f == 0

def test_calc_no_offset():
    e = _make_event(30.0, 45.0, 1.5)
    # detect_offset=0, preroll=3s, postroll=2s, fps=60
    # src_in  = 0 + 30 - 3 = 27s → 1620 frames
    # src_out = 0 + 45 + 2 = 47s → 2820 frames
    in_f, out_f = calc_source_frames(e, detect_offset_s=0.0, timeline_fps=60)
    assert in_f == 1620
    assert out_f == 2820
```

- [ ] **Step 5.2 — Run tests, confirm they fail**

```bash
pytest tests/test_compile_reel_logic.py -v
```
Expected: `ModuleNotFoundError: No module named 'compile_reel'`

- [ ] **Step 5.3 — Implement pure logic functions in `v3/resolve_scripts/compile_reel.py`**

```python
# v3/resolve_scripts/compile_reel.py
#
# Resolve Workspace Script: import GoPro chapters → multicam clip → highlight reel
#
# Run from inside DaVinci Resolve: Workspace → Scripts → compile_reel
# Requires: template project open, target timeline active.
#
# Pure logic functions (load_events, select_events, calc_source_frames) are
# unit-testable without Resolve. The _resolve_assemble() function requires
# a live Resolve instance.

from __future__ import annotations

import csv
import json
import os
import sys
import traceback
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Color / scoring (mirrors detect_events.py — keep in sync)
# ---------------------------------------------------------------------------

def score_to_color(score: float) -> str:
    if score > 1.60:
        return "Red"
    if score > 1.30:
        return "Orange"
    if score > 1.00:
        return "Yellow"
    return "Blue"


_COLOR_PRIORITY = {"Blue": 0, "Yellow": 1, "Orange": 2, "Red": 3}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Event:
    start_s: float
    end_s: float
    score: float
    primary_cam: int
    confidence: float
    color: str

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


# ---------------------------------------------------------------------------
# Pure logic — unit testable
# ---------------------------------------------------------------------------

def load_events(events_csv: str) -> list[Event]:
    """Parse events.csv into a list of Event objects."""
    events: list[Event] = []
    with open(events_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            score = float(row["score"])
            events.append(Event(
                start_s=float(row["start_s"]),
                end_s=float(row["end_s"]),
                score=score,
                primary_cam=int(row["primary_cam"]),
                confidence=float(row["confidence"]),
                color=score_to_color(score),
            ))
    return events


def select_events(events: list[Event], max_reel_s: float = 900.0) -> list[Event]:
    """
    Return all events sorted by start_s, capped at max_reel_s total duration.
    Drops lowest-priority/lowest-score events first: Blue → Yellow → Orange.
    Red events are never dropped regardless of cap.
    """
    total = sum(e.duration_s for e in events)
    if total <= max_reel_s:
        return sorted(events, key=lambda e: e.start_s)

    # Sort drop candidates: lowest priority first, then lowest score within tier
    droppable = sorted(
        [e for e in events if e.color != "Red"],
        key=lambda e: (_COLOR_PRIORITY[e.color], e.score),
    )

    kept = list(events)
    for candidate in droppable:
        if sum(e.duration_s for e in kept) <= max_reel_s:
            break
        kept.remove(candidate)

    return sorted(kept, key=lambda e: e.start_s)


def calc_source_frames(
    event: Event,
    detect_offset_s: float,
    timeline_fps: int,
    preroll_s: float = 3.0,
    postroll_s: float = 2.0,
) -> tuple[int, int]:
    """
    Convert a detection-relative event to recording-relative source frame numbers.

    detect_offset_s: seconds into the camera recording where detection t=0 falls
                     (from sync_info.json cam1_detect_offset_s / cam2_detect_offset_s).
    Returns (source_in_frame, source_out_frame).
    """
    src_start_s = max(0.0, detect_offset_s + event.start_s - preroll_s)
    src_end_s = detect_offset_s + event.end_s + postroll_s
    return (
        int(round(src_start_s * timeline_fps)),
        int(round(src_end_s * timeline_fps)),
    )


# ---------------------------------------------------------------------------
# Resolve assembly — requires live Resolve instance (not unit-testable)
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[compile_reel] {msg}", flush=True)


def _get_resolve():
    try:
        import DaVinciResolveScript as dvr
        return dvr.scriptapp("Resolve")
    except Exception:
        return None


def _resolve_assemble(
    game_folder: str,
    events: list[Event],
    sync_info: dict,
    max_reel_s: float = 900.0,
) -> None:
    """
    Main Resolve assembly. Called only when running inside Resolve.
    Verify MediaPool.CreateMultiCamClip() signature against
    Help → Developer → Scripting in your installed Resolve version.
    """
    resolve = _get_resolve()
    if resolve is None:
        _log("ERROR: Resolve scripting API not available. Run from Workspace → Scripts.")
        return

    pm = resolve.GetProjectManager()
    proj = pm.GetCurrentProject() if pm else None
    tl = proj.GetCurrentTimeline() if proj else None
    if proj is None or tl is None:
        _log("ERROR: No active project/timeline. Open your template project first.")
        return

    _log(f"Active timeline: {tl.GetName()}")

    # --- Import chapter files into a named media pool bin ---
    folder_name = os.path.basename(game_folder.rstrip("/\\"))
    bin_name = f"Game Footage — {folder_name}"
    media_pool = proj.GetMediaPool()
    root_folder = media_pool.GetRootFolder()
    game_bin = media_pool.AddSubFolder(root_folder, bin_name)
    if game_bin is None:
        _log(f"WARN: could not create bin '{bin_name}', importing to root")
        game_bin = root_folder
    media_pool.SetCurrentFolder(game_bin)

    chapters_path = os.path.join(game_folder, "chapters.json")
    with open(chapters_path) as f:
        chapters = json.load(f)

    all_chapter_paths = chapters["cam1"] + chapters["cam2"]
    _log(f"Importing {len(all_chapter_paths)} chapter files...")
    imported = media_pool.ImportMedia(all_chapter_paths)
    if not imported:
        _log("ERROR: ImportMedia returned nothing. Check file paths.")
        return
    _log(f"Imported {len(imported)} items.")

    # --- Separate cam1 / cam2 items by source path ---
    cam1_set = set(chapters["cam1"])
    cam2_set = set(chapters["cam2"])
    cam1_items = [i for i in imported if i.GetClipProperty("File Path") in cam1_set]
    cam2_items = [i for i in imported if i.GetClipProperty("File Path") in cam2_set]

    # --- Create multicam clip (verify API in Resolve scripting docs) ---
    _log("Creating multicam clip...")
    multicam_item = None
    try:
        # NOTE: CreateMultiCamClip signature varies by Resolve version.
        # Verify against Help → Developer → Scripting before running.
        multicam_item = media_pool.CreateMultiCamClip(
            cam1_items + cam2_items,
            {
                "name": f"Multicam — {folder_name}",
                "syncType": "timecode",
                "videoTrackCount": 2,
                "audioTrackCount": 2,
            },
        )
    except Exception as exc:
        _log(f"WARN: CreateMultiCamClip failed ({exc}). Falling back to dual-track placement.")

    # --- Select events ---
    selected = select_events(events, max_reel_s=max_reel_s)
    _log(f"Events selected: {len(selected)} / {len(events)} "
         f"(total duration: {sum(e.duration_s for e in selected):.0f}s)")

    timeline_fps = 60
    try:
        fps_str = proj.GetSetting("timelineFrameRate")
        if fps_str:
            timeline_fps = int(float(fps_str))
    except Exception:
        pass

    # --- Find stinger end on Track 1 ---
    existing = tl.GetItemListInTrack("video", 1) or []
    if existing:
        stinger_end = max(
            item.GetStart() + item.GetDuration() for item in existing
        )
    else:
        stinger_end = 0
    _log(f"Stinger end frame: {stinger_end} (placing clips after this)")

    # --- Place clips ---
    if multicam_item is None:
        _log("Multicam creation failed. Cannot place clips without multicam item.")
        return

    record_frame = stinger_end
    placed = 0
    for event in selected:
        cam_offset = (
            sync_info["cam1_detect_offset_s"]
            if event.primary_cam == 1
            else sync_info["cam2_detect_offset_s"]
        )
        src_in, src_out = calc_source_frames(
            event, cam_offset, timeline_fps
        )

        result = media_pool.AppendToTimeline([{
            "mediaPoolItem": multicam_item,
            "startFrame": src_in,
            "endFrame": src_out,
            "trackIndex": 1,
            "recordFrame": record_frame,
            "mediaType": 1,
        }])

        if result:
            placed_item = result[0]
            # Switch to primary_cam angle (angle index is 1-based)
            try:
                placed_item.SetCurrentVideoItem(event.primary_cam)
            except Exception:
                pass  # angle switching may not be available in all builds
            record_frame += src_out - src_in
            placed += 1

    total_s = record_frame / timeline_fps - stinger_end / timeline_fps
    _log(f"Placed {placed} clips. Total reel duration: {total_s:.0f}s "
         f"({total_s / 60:.1f} min)")


# ---------------------------------------------------------------------------
# Entry point (called by Resolve)
# ---------------------------------------------------------------------------

def main() -> None:
    resolve = _get_resolve()
    if resolve is None:
        print("[compile_reel] ERROR: Must be run from inside DaVinci Resolve "
              "(Workspace → Scripts).", flush=True)
        return

    fusion = resolve.Fusion()
    game_folder = None
    if fusion:
        try:
            game_folder = str(fusion.RequestDir(
                "Pick the game folder (contains events.csv and chapters.json)"
            ))
        except Exception:
            pass

    if not game_folder or not os.path.isdir(game_folder):
        _log("No folder selected. Aborting.")
        return

    events_csv = os.path.join(game_folder, "events.csv")
    sync_json  = os.path.join(game_folder, "sync_info.json")

    for path in (events_csv, sync_json):
        if not os.path.isfile(path):
            _log(f"ERROR: required file not found: {path}")
            _log("Run hockeydetect on this folder first.")
            return

    events   = load_events(events_csv)
    sync_info = json.loads(open(sync_json).read())

    _log(f"Loaded {len(events)} events from events.csv")
    _resolve_assemble(game_folder, events, sync_info)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[compile_reel] FATAL ERROR:", flush=True)
        print(traceback.format_exc(), flush=True)
```

- [ ] **Step 5.4 — Run tests, confirm they pass**

```bash
pytest tests/test_compile_reel_logic.py -v
```
Expected: 16 passed

- [ ] **Step 5.5 — Run full test suite**

```bash
pytest tests/ -v
```
Expected: all tests pass

- [ ] **Step 5.6 — Commit**

```bash
git add v3/resolve_scripts/compile_reel.py tests/test_compile_reel_logic.py
git commit -m "feat: compile_reel.py — event logic core (tested) + Resolve assembly"
```

---

## Task 6: Integration and index updates

**Files:**
- Modify: `docs/specs/index.md`

- [ ] **Step 6.1 — Update spec status to approved**

In `docs/specs/001-v3-highlight-pipeline-2026-04-25.md`, change:

```
**Status:** Revised — pending final user approval
```
to:
```
**Status:** Approved — implementation plan at docs/plans/2026-04-25-v3-highlight-pipeline.md
```

- [ ] **Step 6.2 — Add plans to index**

Create `docs/plans/index.md`:

```markdown
# Plans Index

| # | Title | File | Date | Status | Spec |
|---|---|---|---|---|---|
| 001 | V3 Highlight Pipeline | [2026-04-25-v3-highlight-pipeline.md](2026-04-25-v3-highlight-pipeline.md) | 2026-04-25 | In progress | [spec 001](../specs/001-v3-highlight-pipeline-2026-04-25.md) |
```

- [ ] **Step 6.3 — Final commit**

```bash
git add docs/specs/001-v3-highlight-pipeline-2026-04-25.md docs/plans/
git commit -m "docs: mark spec 001 approved, add plans index"
```

---

## Manual integration checklist (post-implementation)

Run these after all tasks are complete to verify the full pipeline end-to-end.

**Detection stage:**
- [ ] `hockeydetect /path/to/game_folder` with real GoPro chapters completes without error
- [ ] `chapters.json` lists correct files in sorted order
- [ ] `sync_info.json` has `offset_s` within expected range (< 5s for synced GoPros)
- [ ] `cam1_concat.txt` and `cam2_concat.txt` exist; first file in manifest matches first chapter
- [ ] `events.csv` is non-empty with expected columns
- [ ] File mode still works: `hockeydetect cam1.mp4 cam2.mp4` (backwards compatibility)

**Resolve assembly:**
- [ ] Open template project in Resolve, activate the main timeline
- [ ] Run `compile_reel` from Workspace → Scripts
- [ ] Pick game folder when prompted
- [ ] Verify "Game Footage — \<folder\>" bin appears in Media Pool containing chapter files
- [ ] Verify multicam clip is created in the bin (if `CreateMultiCamClip` is supported — check Resolve scripting docs first)
- [ ] Verify clips appear on Track 1 after the stinger
- [ ] Spot-check 2-3 clips: confirm multicam angle matches `primary_cam` in `events.csv`
- [ ] Check Resolve console for total reel duration (expect 5-15 min)
