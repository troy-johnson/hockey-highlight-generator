# tests/test_discover.py
import json
import pytest
from pathlib import Path
import discover as _D
from discover import discover


@pytest.fixture(autouse=True)
def _no_black_check(monkeypatch):
    monkeypatch.setattr(_D, "is_black_recording", lambda rec: False)


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
    with pytest.raises(ValueError, match="cam1"):
        discover(folder)


def test_missing_cam2_exits(tmp_path):
    folder = _setup(tmp_path, cam1_files=["GOPRO1801.MP4"])
    with pytest.raises(ValueError, match="cam2"):
        discover(folder)


def test_empty_cam1_exits(tmp_path):
    (tmp_path / "cam1").mkdir()
    (tmp_path / "cam2").mkdir()
    (tmp_path / "cam2" / "GOPRO1901.MP4").touch()
    with pytest.raises(ValueError, match="No .MP4"):
        discover(str(tmp_path))


def test_empty_cam2_exits(tmp_path):
    (tmp_path / "cam1").mkdir()
    (tmp_path / "cam1" / "GOPRO1801.MP4").touch()
    (tmp_path / "cam2").mkdir()
    with pytest.raises(ValueError, match="No .MP4"):
        discover(str(tmp_path))


def test_empty_folder_exits(tmp_path):
    with pytest.raises(ValueError, match="No .MP4 files found"):
        discover(str(tmp_path))


def test_nonexistent_folder_raises(tmp_path):
    with pytest.raises(ValueError, match="Game folder not found"):
        discover(str(tmp_path / "does_not_exist"))


def test_old_output_videos_are_not_camera_files(tmp_path):
    folder = _setup(tmp_path, cam1_files=["GX010017.MP4", "cam1.mp4"], cam2_files=["GX010038.MP4", "recap.mp4"])
    (tmp_path / "cam1" / ".GX019999.MP4").touch()
    result = discover(folder)
    assert [Path(p).name for p in result["cam1"]] == ["GX010017.MP4"]
    assert [Path(p).name for p in result["cam2"]] == ["GX010038.MP4"]


def test_is_old_output_video():
    assert _D.is_old_output_video("cam1.mp4")
    assert _D.is_old_output_video("CAM2.MP4")
    assert _D.is_old_output_video("highlights_final.mp4")
    assert _D.is_old_output_video("goal_overlay.mp4")
    assert not _D.is_old_output_video("GX010017.MP4")


def test_non_strict_accepts_one_camera(tmp_path):
    folder = _setup(tmp_path, cam1_files=["GX010017.MP4"])
    result = discover(folder, strict=False)
    assert [Path(p).name for p in result["cam1"]] == ["GX010017.MP4"]
    assert "cam2" not in result or not result.get("cam2")
    assert result["flags"]


def test_non_strict_all_black_camera_is_a_flag(tmp_path, monkeypatch):
    folder = _setup(tmp_path, cam1_files=["GX010017.MP4"], cam2_files=["GX010038.MP4"])
    monkeypatch.setattr(_D, "is_black_recording", lambda rec: "GX010038" in rec[0])
    result = discover(folder, strict=False)
    assert "cam2" in result["missing"]
    assert any("cam2" in f for f in result["flags"])
    with pytest.raises(ValueError):
        discover(folder)                     # strict mode keeps the old error


def test_non_strict_no_usable_camera_raises(tmp_path, monkeypatch):
    folder = _setup(tmp_path, cam1_files=["GX010017.MP4"], cam2_files=["GX010038.MP4"])
    monkeypatch.setattr(_D, "is_black_recording", lambda rec: True)
    with pytest.raises(ValueError, match="No usable camera"):
        discover(folder, strict=False)


def _flat(tmp_path, monkeypatch, files):
    """files: {name: (serial or None, size)}; serials come from a fake CASN reader."""
    for name, (_serial, size) in files.items():
        (tmp_path / name).write_bytes(b"x" * size)
    serial_of = {str(tmp_path / n): s for n, (s, _) in files.items()}
    monkeypatch.setattr(_D, "read_camera_serial", lambda p: serial_of[str(p)])
    return str(tmp_path)


def test_strict_error_names_files_without_serial(tmp_path, monkeypatch):
    folder = _flat(tmp_path, monkeypatch, {"GX010017.MP4": ("S1", 10), "clip.MP4": (None, 10)})
    with pytest.raises(ValueError, match=r"found 1 .*1 file\(s\) without a camera serial were ignored: clip.MP4"):
        discover(folder)


def test_flat_two_cameras_ignore_file_without_serial(tmp_path, monkeypatch):
    folder = _flat(tmp_path, monkeypatch, {"GX010017.MP4": ("S1", 10), "GX010038.MP4": ("S2", 10),
                                          "clip.MP4": (None, 10)})
    result = discover(folder)
    assert result["serials"] == {"cam1": "S1", "cam2": "S2"}
    assert any(e["reason"] == "no camera serial" for e in result["excluded"])


def test_non_strict_more_than_two_cameras_keeps_all_and_flags(tmp_path, monkeypatch):
    folder = _flat(tmp_path, monkeypatch, {"GX010017.MP4": ("S1", 50), "GX010038.MP4": ("S2", 5),
                                          "GX010050.MP4": ("S3", 40)})
    result = discover(folder, strict=False)
    assert result["serials"] == {"cam1": "S1", "cam2": "S3", "cam3": "S2"}
    assert [Path(p).name for p in result["cam3"]] == ["GX010038.MP4"]
    assert any("cam3" in f and "S2" in f for f in result["flags"])
    with pytest.raises(ValueError, match="found 3"):
        discover(folder)                     # strict mode (hockeydetect) unchanged


def test_recap_output_video_pattern():
    assert _D.is_old_output_video("2026-03-01_Ice-Pak-vs-Wild_Recap.mp4")
    assert _D.is_old_output_video("recap.mp4")
    assert not _D.is_old_output_video("GX010017.MP4")
