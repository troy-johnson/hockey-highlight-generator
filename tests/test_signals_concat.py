# tests/test_signals_concat.py
import subprocess
from unittest.mock import MagicMock
from signals import _ffmpeg_gray_frames


def _make_popen_mock():
    mock_proc = MagicMock()
    mock_proc.stdout.read.return_value = b""
    return mock_proc


def test_mp4_path_no_concat_flags(monkeypatch):
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
    assert "0" in cmd[cmd.index("-safe") + 1:]


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
