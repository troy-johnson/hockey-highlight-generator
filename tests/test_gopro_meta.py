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
    _check_chapter_continuity,
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
    "streams": [{"r_frame_rate": "60/1"}],
    "format": {
        "duration": "3600.0",
        "tags": {
            "timecode": "10:30:00:00",
            "creation_time": "2024-01-15T10:30:00.000000Z",
        },
    },
})

_FFPROBE_NO_TIMECODE = json.dumps({
    "streams": [{"r_frame_rate": "60/1"}],
    "format": {
        "duration": "3600.0",
        "tags": {"creation_time": "2024-01-15T10:30:00.500000Z"},
    },
})

_FFPROBE_NO_METADATA = json.dumps({
    "streams": [],
    "format": {"tags": {}},
})


def test_extract_uses_timecode_when_present(monkeypatch):
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **k: _FFPROBE_WITH_TIMECODE.encode(),
    )
    seconds, source, raw_tc, duration = extract_chapter_time("/fake/GOPRO1801.MP4")
    assert source == "timecode"
    assert seconds == pytest.approx(timecode_to_seconds("10:30:00:00"))
    assert raw_tc == "10:30:00:00"
    assert duration == pytest.approx(3600.0)


def test_extract_falls_back_to_creation_time(monkeypatch):
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **k: _FFPROBE_NO_TIMECODE.encode(),
    )
    seconds, source, raw_tc, duration = extract_chapter_time("/fake/GOPRO1801.MP4")
    assert source == "creation_time"
    assert seconds == pytest.approx(37800.5, abs=1.0)
    assert raw_tc is None
    assert duration == pytest.approx(3600.0)


def test_extract_raises_when_no_metadata(monkeypatch):
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **k: _FFPROBE_NO_METADATA.encode(),
    )
    with pytest.raises(ValueError, match="No usable timecode"):
        extract_chapter_time("/fake/GOPRO1801.MP4")


def test_extract_uses_fps_from_stream(monkeypatch):
    """fps is read from r_frame_rate and applied to timecode frame conversion."""
    ffprobe_data = json.dumps({
        "streams": [{"r_frame_rate": "30/1"}],
        "format": {
            "duration": "600.0",
            "tags": {"timecode": "00:00:00:15"},  # 15 frames at 30fps = 0.5s
        },
    })
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda *a, **k: ffprobe_data.encode(),
    )
    seconds, source, raw_tc, _ = extract_chapter_time("/fake/GOPRO1801.MP4")
    assert source == "timecode"
    assert raw_tc == "00:00:00:15"
    # At 30fps: 15/30 = 0.5s (vs 15/60 = 0.25s if wrong fps)
    assert seconds == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_sync
# ---------------------------------------------------------------------------

def _make_sync(cam1_s, cam2_s, cam1_src="timecode", cam2_src="timecode"):
    cam1_tc = "10:30:00:00" if cam1_src == "timecode" else None
    cam2_tc = "10:30:00:05" if cam2_src == "timecode" else None
    with patch("gopro_meta.extract_chapter_time") as mock:
        mock.side_effect = [
            (cam1_s, cam1_src, cam1_tc, 3600.0),
            (cam2_s, cam2_src, cam2_tc, 3600.0),
        ]
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


def test_sync_includes_start_timecodes_for_timecode_source():
    info = _make_sync(37800.0, 37802.0, "timecode", "timecode")
    assert info["cam1_start_timecode"] == "10:30:00:00"
    assert info["cam2_start_timecode"] == "10:30:00:05"


def test_sync_start_timecodes_are_none_for_creation_time_source():
    info = _make_sync(37800.0, 37802.0, "creation_time", "creation_time")
    assert info["cam1_start_timecode"] is None
    assert info["cam2_start_timecode"] is None


# ---------------------------------------------------------------------------
# _check_chapter_continuity
# ---------------------------------------------------------------------------

def test_continuity_no_warning_for_contiguous_chapters():
    # chapter 1: starts at 3600.0s, duration 600.0s → chapter 2 expected at 4200.0s
    # chapter 2: actual starts at 4200.0s — perfectly contiguous
    metas = [
        (3600.0, "timecode", "01:00:00:00", 600.0),
        (4200.0, "timecode", "01:10:00:00", 600.0),
    ]
    warnings = _check_chapter_continuity(metas, "cam1")
    assert warnings == []


def test_continuity_warns_on_chapter_gap():
    # chapter 2 starts 10s after expected — gap of 10s
    metas = [
        (3600.0, "timecode", "01:00:00:00", 600.0),
        (4210.0, "timecode", "01:10:10:00", 600.0),
    ]
    warnings = _check_chapter_continuity(metas, "cam1")
    assert len(warnings) == 1
    assert "cam1" in warnings[0]
    assert "chapter 2" in warnings[0]
    assert "gap" in warnings[0]
    assert "10.0s" in warnings[0]


def test_continuity_warns_on_chapter_overlap():
    # chapter 2 starts 10s before expected — overlap of 10s
    metas = [
        (3600.0, "timecode", "01:00:00:00", 600.0),
        (4190.0, "timecode", "01:09:50:00", 600.0),
    ]
    warnings = _check_chapter_continuity(metas, "cam2")
    assert len(warnings) == 1
    assert "cam2" in warnings[0]
    assert "overlap" in warnings[0]


def test_continuity_skips_check_when_previous_duration_is_zero():
    metas = [
        (3600.0, "timecode", "01:00:00:00", 0.0),  # duration unknown
        (4200.0, "timecode", "01:10:00:00", 600.0),
    ]
    warnings = _check_chapter_continuity(metas, "cam1")
    assert warnings == []


def test_continuity_small_gap_below_threshold_is_silent():
    # 3s gap — below the 5s threshold
    metas = [
        (3600.0, "timecode", "01:00:00:00", 600.0),
        (4203.0, "timecode", "01:10:03:00", 600.0),
    ]
    warnings = _check_chapter_continuity(metas, "cam1")
    assert warnings == []


def test_compute_sync_propagates_continuity_warnings():
    # cam1 has 2 chapters with a 10s gap between them
    with patch("gopro_meta.extract_chapter_time") as mock:
        mock.side_effect = [
            (3600.0, "timecode", "01:00:00:00", 600.0),   # cam1 ch1
            (4210.0, "timecode", "01:10:10:00", 600.0),   # cam1 ch2 — gap
            (3600.0, "timecode", "01:00:00:00", 600.0),   # cam2 ch1
        ]
        info = compute_sync(
            ["/fake/cam1/A.MP4", "/fake/cam1/B.MP4"],
            ["/fake/cam2/C.MP4"],
        )

    assert len(info["warnings"]) == 1
    assert "cam1" in info["warnings"][0]
    assert "gap" in info["warnings"][0]


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
