# tests/test_signal_cache.py — signals kept per Recording (hhg-3r5.77)
import os
import time

import numpy as np
import pytest

import signals as S


@pytest.fixture
def fake_extract(monkeypatch):
    calls = []

    def fake(video_path, rois, fps, width, verbose=False, with_audio=True):
        if video_path.endswith(".txt"):
            with open(video_path) as f:
                n = sum(1 for l in f if l.startswith("file "))
        else:
            n = 1
        calls.append(video_path)
        v = np.full(n * 10, float(len(calls)), dtype=np.float32)
        return v, v, v

    monkeypatch.setattr(S, "_extract_single_signals", fake)
    return calls


ROIS = {"net": S.ROI(1, 2, 3, 4), "slot": S.ROI(5, 6, 7, 8)}


def _game(tmp_path):
    for n in ("GX010017.MP4", "GX020017.MP4", "GX010018.MP4"):
        (tmp_path / n).write_bytes(b"x" * 10)
    m = tmp_path / "cam1_concat.txt"
    m.write_text("ffconcat version 1.0\n# recording 0.000\nfile 'GX010017.MP4'\nfile 'GX020017.MP4'\n"
                 "# recording 30.000\nfile 'GX010018.MP4'\n")
    return m


def test_unchanged_input_uses_cache(tmp_path, fake_extract):
    m = _game(tmp_path)
    cache = tmp_path / "cache"
    a = S.extract_signals(str(m), ROIS, 1, 640, cache_dir=str(cache))
    assert len(fake_extract) == 2                      # two Recordings
    assert len(list(cache.glob("signals_*.npz"))) == 2
    b = S.extract_signals(str(m), ROIS, 1, 640, cache_dir=str(cache))
    assert len(fake_extract) == 2                      # nothing computed again
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_changed_file_recomputes_only_its_recording(tmp_path, fake_extract):
    m = _game(tmp_path)
    cache = str(tmp_path / "cache")
    S.extract_signals(str(m), ROIS, 1, 640, cache_dir=cache)
    later = time.time() + 100
    os.utime(tmp_path / "GX010018.MP4", (later, later))
    S.extract_signals(str(m), ROIS, 1, 640, cache_dir=cache)
    assert len(fake_extract) == 3


def test_changed_settings_recompute(tmp_path, fake_extract):
    m = _game(tmp_path)
    cache = str(tmp_path / "cache")
    S.extract_signals(str(m), ROIS, 1, 640, cache_dir=cache)
    S.extract_signals(str(m), ROIS, 2, 640, cache_dir=cache)         # fps
    assert len(fake_extract) == 4
    moved = {"net": S.ROI(9, 2, 3, 4), "slot": ROIS["slot"]}
    S.extract_signals(str(m), moved, 2, 640, cache_dir=cache)         # ROI
    assert len(fake_extract) == 6


def test_no_cache_dir_always_computes(tmp_path, fake_extract):
    m = _game(tmp_path)
    S.extract_signals(str(m), ROIS, 1, 640)
    S.extract_signals(str(m), ROIS, 1, 640)
    assert len(fake_extract) == 4
    assert not (tmp_path / "cache").exists()


def test_single_video_cache(tmp_path, fake_extract):
    v = tmp_path / "GX010017.MP4"
    v.write_bytes(b"x")
    cache = str(tmp_path / "cache")
    S.extract_signals(str(v), ROIS, 1, 640, cache_dir=cache)
    S.extract_signals(str(v), ROIS, 1, 640, cache_dir=cache)
    assert len(fake_extract) == 1


def test_cache_key_changes_with_seek_and_audio(tmp_path):
    f = tmp_path / "a.MP4"
    f.write_bytes(b"x")
    k = S.signal_cache_key([str(f)], 0.0, ROIS, 12, 1280, False)
    assert k == S.signal_cache_key([str(f)], 0.0, ROIS, 12, 1280, False)
    assert k != S.signal_cache_key([str(f)], 1.5, ROIS, 12, 1280, False)
    assert k != S.signal_cache_key([str(f)], 0.0, ROIS, 12, 1280, True)
