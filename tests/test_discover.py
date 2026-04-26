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


def test_empty_cam2_exits(tmp_path):
    (tmp_path / "cam1").mkdir()
    (tmp_path / "cam1" / "GOPRO1801.MP4").touch()
    (tmp_path / "cam2").mkdir()
    with pytest.raises(SystemExit, match="No .MP4"):
        discover(str(tmp_path))


def test_both_cameras_missing_exits(tmp_path):
    with pytest.raises(SystemExit, match="Both cam1/ and cam2/ are required"):
        discover(str(tmp_path))
