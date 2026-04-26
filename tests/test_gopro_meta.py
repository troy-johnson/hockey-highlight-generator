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
    assert seconds == pytest.approx(37800.5, abs=1.0)

_FFPROBE_NO_METADATA = json.dumps({
    "streams": [],
    "format": {"tags": {}}
})

def test_extract_raises_when_no_metadata(monkeypatch):
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **k: _FFPROBE_NO_METADATA.encode()
    )
    with pytest.raises(ValueError, match="No usable timecode"):
        extract_chapter_time("/fake/GOPRO1801.MP4")

# ---------------------------------------------------------------------------
# compute_sync
# ---------------------------------------------------------------------------

def _make_sync(cam1_s, cam2_s, cam1_src="timecode", cam2_src="timecode"):
    with patch("gopro_meta.extract_chapter_time") as mock:
        mock.side_effect = [(cam1_s, cam1_src), (cam2_s, cam2_src)]
        return compute_sync(["/fake/cam1/A.MP4"], ["/fake/cam2/B.MP4"])

def test_sync_cam2_later():
    info = _make_sync(37800.0, 37802.0)
    assert info["offset_s"] == pytest.approx(2.0)
    assert info["cam1_detect_offset_s"] == pytest.approx(2.0)
    assert info["cam2_detect_offset_s"] == pytest.approx(0.0)

def test_sync_cam1_later():
    info = _make_sync(37801.5, 37800.0)
    assert info["offset_s"] == pytest.approx(-1.5)
    assert info["cam1_detect_offset_s"] == pytest.approx(0.0)
    assert info["cam2_detect_offset_s"] == pytest.approx(1.5)

def test_sync_tiny_offset_treated_as_zero():
    info = _make_sync(37800.0, 37800.05)
    assert info["offset_s"] == pytest.approx(0.0)
    assert info["cam1_detect_offset_s"] == pytest.approx(0.0)
    assert info["cam2_detect_offset_s"] == pytest.approx(0.0)

def test_sync_implausible_offset_raises():
    with pytest.raises(ValueError, match="implausibly large"):
        _make_sync(37800.0, 37800.0 + 61.0)

def test_sync_method_timecode():
    info = _make_sync(37800.0, 37800.5, "timecode", "timecode")
    assert info["sync_method"] == "timecode"

def test_sync_method_mixed():
    info = _make_sync(37800.0, 37800.5, "timecode", "creation_time")
    assert info["sync_method"] == "mixed"

def test_sync_method_creation_time():
    info = _make_sync(37800.0, 37800.5, "creation_time", "creation_time")
    assert info["sync_method"] == "creation_time"

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
    file_idx = next(i for i, l in enumerate(lines) if "GOPRO1801" in l)
    assert "inpoint 2.500" in lines[file_idx + 1]
    file2_idx = next(i for i, l in enumerate(lines) if "GOPRO1802" in l)
    if file2_idx + 1 < len(lines):
        assert "inpoint" not in lines[file2_idx + 1]
