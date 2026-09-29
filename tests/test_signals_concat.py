# tests/test_signals_concat.py
import subprocess
from unittest.mock import MagicMock
import pytest
import signals as _signals
from signals import _ffmpeg_gray_frames


@pytest.fixture(autouse=True)
def _software_decode(monkeypatch):
    # These tests check the command shape; keep the one-time VideoToolbox probe out of the Popen mocks.
    monkeypatch.setattr(_signals, "_videotoolbox_available", lambda: False)


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


# ---------------------------------------------------------------------------
# Seek handling (hhg-38a.11): offsets become -ss before the concat input
# ---------------------------------------------------------------------------

from signals import concat_input_args


def test_concat_input_args_uses_seek_comment(tmp_path):
    m = tmp_path / "cam1_concat.txt"
    m.write_text("ffconcat version 1.0\n# seek 20.000\nfile '/fake/A.MP4'\n")
    args = concat_input_args(str(m))
    assert args[:2] == ["-ss", "20.000"]
    assert args[args.index("-f") + 1] == "concat"
    assert args[-2] == "-i"


def test_concat_input_args_converts_legacy_inpoint(tmp_path):
    m = tmp_path / "cam1_concat.txt"
    m.write_text("ffconcat version 1.0\nfile '/fake/A.MP4'\ninpoint 20.000\n")
    args = concat_input_args(str(m))
    assert args[:2] == ["-ss", "20.000"]
    cleaned = args[-1]
    assert cleaned != str(m)
    text = open(cleaned).read()
    assert "inpoint" not in text and "/fake/A.MP4" in text


def test_concat_input_args_no_offset(tmp_path):
    m = tmp_path / "cam1_concat.txt"
    m.write_text("ffconcat version 1.0\nfile '/fake/A.MP4'\n")
    assert concat_input_args(str(m)) == ["-f", "concat", "-safe", "0", "-i", str(m)]


def test_concat_input_args_plain_mp4():
    assert concat_input_args("video.mp4") == ["-i", "video.mp4"]


def test_gray_frames_puts_seek_before_input(tmp_path, monkeypatch):
    m = tmp_path / "cam1_concat.txt"
    m.write_text("ffconcat version 1.0\n# seek 5.000\nfile '/fake/A.MP4'\n")
    captured = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **k: captured.append(cmd) or _make_popen_mock())
    monkeypatch.setattr("signals.ffprobe_dims", lambda p: (1920, 1080))
    gen, w, h = _ffmpeg_gray_frames(str(m), fps=12, width=640)
    next(gen, None)
    cmd = captured[0]
    assert cmd.index("-ss") < cmd.index("-i")
    assert "inpoint" not in " ".join(cmd)


def test_audio_rms_uses_same_seek(tmp_path, monkeypatch):
    from signals import extract_audio_rms
    m = tmp_path / "cam1_concat.txt"
    m.write_text("ffconcat version 1.0\n# seek 5.000\nfile '/fake/A.MP4'\n")
    captured = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **k: captured.append(cmd) or _make_popen_mock())
    extract_audio_rms(str(m), fps=12, n_frames=10)
    cmd = captured[0]
    assert cmd[cmd.index("-ss") + 1] == "5.000" and cmd.index("-ss") < cmd.index("-i")
    assert cmd[cmd.index("-f") + 1] == "concat"


# Review findings (GPT-6.1 review of #17): temporary manifests keep relative paths and are cleaned up.

def test_legacy_manifest_conversion_keeps_relative_paths_working(tmp_path):
    (tmp_path / "chapters").mkdir()
    m = tmp_path / "cam1_concat.txt"
    m.write_text("ffconcat version 1.0\nfile 'chapters/A.MP4'\ninpoint 3.000\n")
    args = concat_input_args(str(m))
    text = open(args[-1]).read()
    assert f"file '{tmp_path / 'chapters' / 'A.MP4'}'" in text


def test_temporary_manifests_are_removed_by_cleanup(tmp_path):
    import signals
    m = tmp_path / "cam1_concat.txt"
    m.write_text("ffconcat version 1.0\nfile '/fake/A.MP4'\ninpoint 3.000\n")
    tmp = concat_input_args(str(m))[-1]
    assert __import__("os").path.exists(tmp)
    signals.cleanup_temp_manifests()
    assert not __import__("os").path.exists(tmp)


def test_dimension_probe_resolves_relative_manifest_paths(tmp_path, monkeypatch):
    """Second review of #17: the probe must not depend on the process's working directory."""
    import signals
    m = tmp_path / "cam1_concat.txt"
    m.write_text("ffconcat version 1.0\nfile 'chapters/A.MP4'\n")
    probed = []
    monkeypatch.setattr(signals, "ffprobe_dims", lambda p: probed.append(p) or (1920, 1080))
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **k: _make_popen_mock())
    gen, w, h = signals._ffmpeg_gray_frames(str(m), fps=12, width=640)
    assert probed == [str(tmp_path / "chapters" / "A.MP4")]


# ---------------------------------------------------------------------------
# Recording blocks are placed on the game timeline (hhg-38a.13)
# ---------------------------------------------------------------------------

def test_extract_signals_places_recording_blocks(tmp_path, monkeypatch):
    import numpy as np
    import signals
    m = tmp_path / "cam2_concat.txt"
    m.write_text("ffconcat version 1.0\n# recording -2.000\nfile '/b/R1.MP4'\n# recording 10.000\nfile '/b/R2.MP4'\n")
    seen = []

    def fake_single(path, rois, fps, width, verbose=False):
        text = open(path).read(); seen.append(text)
        n = 60 if "R1" in text else 24          # R1: 5 s after its 2 s seek, R2: 2 s (12 fps)
        v = 1.0 if "R1" in text else 2.0
        return (np.full(n, v, np.float32),) * 3

    monkeypatch.setattr(signals, "_extract_single_signals", fake_single)
    net, slot, audio = signals.extract_signals(str(m), {"net": None, "slot": None}, fps=12, width=640)
    assert "# seek 2.000" in seen[0] and "R2" not in seen[0]
    assert "# seek" not in seen[1] and "R2" in seen[1]
    assert len(net) == 10 * 12 + 24
    assert net[:60].tolist() == [1.0] * 60 and net[60:120].tolist() == [0.0] * 60 and net[120:].tolist() == [2.0] * 24


def test_recording_block_manifests_are_removed_after_extraction(tmp_path, monkeypatch):
    import numpy as np, os, signals
    m = tmp_path / "cam2_concat.txt"
    m.write_text("ffconcat version 1.0\n# recording 0.000\nfile '/b/R1.MP4'\n# recording 10.000\nfile '/b/R2.MP4'\n")
    used = []
    monkeypatch.setattr(signals, "_extract_single_signals",
                        lambda path, rois, fps, width, verbose=False, **kw: used.append(path) or (np.zeros(2, np.float32),) * 3)
    signals.extract_signals(str(m), {}, fps=12, width=640)
    assert len(used) == 2 and not any(os.path.exists(p) for p in used)
