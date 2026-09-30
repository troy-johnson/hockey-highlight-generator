# tests/test_recordings.py — cameras and Recordings from GoPro files (hhg-38a.13)
import json
import pytest
from pathlib import Path

import discover as D
from recordings import parse_gopro_name, group_recordings, read_camera_serial


def test_parse_gopro_name():
    assert parse_gopro_name("GX010016.MP4") == (1, 16)
    assert parse_gopro_name("GX020016.mp4") == (2, 16)
    assert parse_gopro_name("GH030007.MP4") == (3, 7)
    assert parse_gopro_name("GOPRO1801.MP4") is None


def test_group_recordings_by_file_number_and_chapter():
    paths = ["/x/GX020016.MP4", "/x/GX010037.MP4", "/x/GX010016.MP4", "/x/GX030016.MP4"]
    assert group_recordings(paths) == [
        ["/x/GX010016.MP4", "/x/GX020016.MP4", "/x/GX030016.MP4"],
        ["/x/GX010037.MP4"],
    ]


def test_group_recordings_unknown_names_form_one_recording():
    assert group_recordings(["/x/GOPRO1802.MP4", "/x/GOPRO1801.MP4"]) == [["/x/GOPRO1801.MP4", "/x/GOPRO1802.MP4"]]


def test_read_camera_serial_from_casn_atom(tmp_path):
    f = tmp_path / "GX010008.MP4"
    f.write_bytes(b"\0" * 5000 + b"CASN" + bytes.fromhex("6301000f") + b"C35042501472\0\0\0" + b"\0" * 100)
    assert read_camera_serial(str(f)) == "C35042501472"


def test_read_camera_serial_missing(tmp_path):
    f = tmp_path / "x.MP4"; f.write_bytes(b"\0" * 1000)
    assert read_camera_serial(str(f)) is None


def _flat(tmp_path, files):
    for name in files: (tmp_path / name).touch()
    return str(tmp_path)


def test_flat_folder_groups_cameras_by_serial(tmp_path, monkeypatch):
    serial = {"GX010007.MP4": "B", "GX010008.MP4": "B", "GX010026.MP4": "A", "GX010027.MP4": "A", "GX020027.MP4": "A"}
    monkeypatch.setattr(D, "read_camera_serial", lambda p: serial[Path(p).name])
    monkeypatch.setattr(D, "is_black_recording", lambda rec: False)
    folder = _flat(tmp_path, list(serial) + ["IMG_1.jpg", "GL010008.LRV"])
    out = D.discover(folder)
    assert out["serials"] == {"cam1": "A", "cam2": "B"}                       # cameras ordered by serial
    assert out["recordings"]["cam1"] == [[str(tmp_path / "GX010026.MP4")], [str(tmp_path / "GX010027.MP4"), str(tmp_path / "GX020027.MP4")]]
    assert out["cam2"] == [str(tmp_path / "GX010007.MP4"), str(tmp_path / "GX010008.MP4")]
    assert json.loads((tmp_path / "chapters.json").read_text())["recordings"] == out["recordings"]


def test_black_recordings_are_excluded(tmp_path, monkeypatch):
    serial = {"GX010007.MP4": "B", "GX010008.MP4": "B", "GX010026.MP4": "A", "GX010027.MP4": "A"}
    monkeypatch.setattr(D, "read_camera_serial", lambda p: serial[Path(p).name])
    monkeypatch.setattr(D, "is_black_recording", lambda rec: Path(rec[0]).name in ("GX010007.MP4", "GX010026.MP4"))
    out = D.discover(_flat(tmp_path, list(serial)))
    assert out["cam1"] == [str(tmp_path / "GX010027.MP4")]
    assert out["cam2"] == [str(tmp_path / "GX010008.MP4")]
    assert {e["reason"] for e in out["excluded"]} == {"black"} and len(out["excluded"]) == 2


def test_flat_folder_needs_two_cameras(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "read_camera_serial", lambda p: "A")
    monkeypatch.setattr(D, "is_black_recording", lambda rec: False)
    with pytest.raises(ValueError, match="2 cameras"):
        D.discover(_flat(tmp_path, ["GX010001.MP4"]))


def test_all_black_camera_is_an_error(tmp_path, monkeypatch):
    serial = {"GX010007.MP4": "B", "GX010026.MP4": "A"}
    monkeypatch.setattr(D, "read_camera_serial", lambda p: serial[Path(p).name])
    monkeypatch.setattr(D, "is_black_recording", lambda rec: Path(rec[0]).name == "GX010007.MP4")
    with pytest.raises(ValueError, match="no usable"):
        D.discover(_flat(tmp_path, list(serial)))


# ---------------------------------------------------------------------------
# Review findings (GPT-6.1 review of #18)
# ---------------------------------------------------------------------------

import recordings as R


def test_invalid_first_casn_does_not_hide_a_valid_serial(tmp_path):
    f = tmp_path / "GX010008.MP4"
    bad = b"CASN" + bytes.fromhex("6301000f") + b"\0" * 15          # empty payload
    good = b"CASN" + bytes.fromhex("6301000f") + b"C3504250147247\0"
    f.write_bytes(b"\0" * 100 + bad + b"\0" * 100 + good + b"\0" * 100)
    assert read_camera_serial(str(f)) == "C3504250147247"


def _fake_frames(monkeypatch, lumas):
    it = iter(lumas)
    def fake_run(cmd, **kw):
        class P: pass
        p = P()
        if cmd[0] == "ffprobe":
            p.stdout = "600.0\n"
        else:
            p.stdout = bytes([next(it)]) * (64 * 36)
        return p
    monkeypatch.setattr(R.subprocess, "run", fake_run)


def test_mixed_dark_and_bright_recording_is_kept(monkeypatch):
    _fake_frames(monkeypatch, [3, 3, 3, 3, 3, 3, 120, 130, 125])
    assert R.is_black_recording(["/x/GX010001.MP4"]) is False


def test_consistently_dark_recording_is_black(monkeypatch):
    _fake_frames(monkeypatch, [3] * 9)
    assert R.is_black_recording(["/x/GX010001.MP4"]) is True
