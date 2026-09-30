# tests/test_audio_sync.py — camera offset from rink audio (hhg-38a.12)
import numpy as np
import pytest
from unittest.mock import patch

from audio_sync import SR, measure_offset, verify_sync_with_audio


def _rink(seconds, seed):
    """Synthetic rink audio: low noise plus random sharp transients (stick/boards/whistles)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.01, int(seconds * SR)).astype(np.float32)
    for t in rng.uniform(0, seconds, int(seconds * 1.5)):
        i = int(t * SR); n = int(0.03 * SR)
        x[i:i + n] += rng.normal(0, 0.3, len(x[i:i + n])) * np.exp(-np.arange(len(x[i:i + n])) / (0.008 * SR))
    return x


def _pair(true_offset, seconds=700, seed=1):
    """cam1 starts first; cam2 starts `true_offset` s later: cam2(t) = cam1(t + offset)."""
    world = _rink(seconds + 120, seed)
    s1 = int(60 * SR); s2 = s1 + int(true_offset * SR)
    cam1 = world[s1: s1 + int(seconds * SR)]
    cam2 = world[s2: s2 + int(seconds * SR)] * 0.6 + _rink(seconds, seed + 99) * 0.3
    return cam1, cam2


@pytest.mark.parametrize("true_offset", [20.0, -7.3, 0.0, 41.25])
def test_measure_offset_recovers_known_offset(true_offset):
    cam1, cam2 = _pair(true_offset)
    r = measure_offset(cam1, cam2)
    assert r["confident"]
    assert r["offset_s"] == pytest.approx(true_offset, abs=0.02)


def test_measure_offset_not_confident_for_unrelated_audio():
    r = measure_offset(_rink(700, 5), _rink(700, 6))
    assert not r["confident"]


def _timecode_sync(offset):
    return {"offset_s": offset, "cam1_detect_offset_s": max(offset, 0.0), "cam2_detect_offset_s": max(-offset, 0.0),
            "sync_method": "timecode", "warnings": []}


def test_verify_adopts_audio_when_timecode_disagrees():
    with patch("audio_sync.measure_offset", return_value={"offset_s": 20.0, "confident": True, "n_agree": 5, "n_windows": 5}), \
         patch("audio_sync.load_audio", return_value=np.zeros(10, np.float32)):
        s = verify_sync_with_audio(_timecode_sync(0.0), "/a.MP4", "/b.MP4")
    assert s["sync_method"] == "audio"
    assert s["offset_s"] == pytest.approx(20.0)
    assert s["cam1_detect_offset_s"] == pytest.approx(20.0) and s["cam2_detect_offset_s"] == 0.0
    assert s["timecode_offset_s"] == 0.0
    assert any("disagree" in w for w in s["warnings"])


def test_verify_negative_audio_offset_skips_cam2():
    with patch("audio_sync.measure_offset", return_value={"offset_s": -12.5, "confident": True, "n_agree": 4, "n_windows": 5}), \
         patch("audio_sync.load_audio", return_value=np.zeros(10, np.float32)):
        s = verify_sync_with_audio(_timecode_sync(0.0), "/a.MP4", "/b.MP4")
    assert s["cam1_detect_offset_s"] == 0.0 and s["cam2_detect_offset_s"] == pytest.approx(12.5)


def test_verify_keeps_timecode_when_audio_agrees():
    with patch("audio_sync.measure_offset", return_value={"offset_s": 2.2, "confident": True, "n_agree": 5, "n_windows": 5}), \
         patch("audio_sync.load_audio", return_value=np.zeros(10, np.float32)):
        s = verify_sync_with_audio(_timecode_sync(2.0), "/a.MP4", "/b.MP4")
    assert s["sync_method"] == "timecode+audio"
    assert s["offset_s"] == pytest.approx(2.0)
    assert s["audio_offset_s"] == pytest.approx(2.2)


def test_verify_keeps_timecode_and_warns_when_inconclusive():
    with patch("audio_sync.measure_offset", return_value={"offset_s": 9.0, "confident": False, "n_agree": 1, "n_windows": 5}), \
         patch("audio_sync.load_audio", return_value=np.zeros(10, np.float32)):
        s = verify_sync_with_audio(_timecode_sync(0.0), "/a.MP4", "/b.MP4")
    assert s["sync_method"] == "timecode"
    assert s["offset_s"] == 0.0
    assert any("inconclusive" in w for w in s["warnings"])


def test_verify_survives_audio_load_failure():
    with patch("audio_sync.load_audio", side_effect=RuntimeError("no audio")):
        s = verify_sync_with_audio(_timecode_sync(0.0), "/a.MP4", "/b.MP4")
    assert s["sync_method"] == "timecode"
    assert any("audio check failed" in w for w in s["warnings"])


# Review findings (GPT-6.1 review of #17): low-information audio must never be "confident".

def test_silent_audio_is_not_confident():
    z = np.zeros(int(700 * SR), np.float32)
    r = measure_offset(z, z)
    assert not r["confident"]


def test_lags_at_the_search_edge_are_rejected(monkeypatch):
    import audio_sync as A
    # a peak pinned to the edge of the search range is not a measurement
    monkeypatch.setattr(A, "_ncc", lambda region, q: np.concatenate([[1.0], np.zeros(len(region) - len(q))]))
    cam1, cam2 = _pair(5.0)
    assert not A.measure_offset(cam1, cam2)["confident"]


# Second review of #17: audio is read across a Recording's chapters, not only its first file.

def test_load_audio_reads_a_recording_as_one_stream(monkeypatch, tmp_path):
    import audio_sync as A
    captured = []

    class R:
        stdout = np.zeros(8, np.float32).tobytes()

    def fake_run(cmd, **kw):
        captured.append(cmd)
        if "-f" in cmd and "concat" in cmd:
            manifest = cmd[cmd.index("-i") + 1]
            captured.append(open(manifest).read())
        return R()

    monkeypatch.setattr(A.subprocess, "run", fake_run)
    A.load_audio(["/r/GX010001.MP4", "/r/GX020001.MP4"], seconds=10, start=600.0)
    cmd, manifest = captured
    assert cmd.index("-ss") < cmd.index("-f") < cmd.index("-i")
    assert cmd[cmd.index("-ss") + 1] == "600.000"
    assert "file '/r/GX010001.MP4'" in manifest and "file '/r/GX020001.MP4'" in manifest
