"""Keep partial and stale frame samples out of published-clip matching."""
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3" / "scripts"))
import answer_key_frames as F


def fixture(tmp_path, monkeypatch):
    raw = tmp_path / "game" / "chapter.MP4"
    raw.parent.mkdir()
    raw.write_bytes(b"raw source")
    monkeypatch.setattr(F, "probe", lambda _: {"width": 64, "height": 36, "duration_s": 5.0})
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=np.zeros((5, 18, 32), np.uint8).tobytes())
    monkeypatch.setattr(F.subprocess, "run", run)
    return raw, tmp_path / "cache", calls


def test_frame_cache_reuses_identical_inputs_without_touching_source(tmp_path, monkeypatch):
    raw, cache, calls = fixture(tmp_path, monkeypatch)
    frames, times, path = F.sample_recording([raw], cache, width=32)
    again, repeated, second = F.sample_recording([raw], cache, width=32)
    assert len(calls) == 1
    assert "nokey" in calls[0]
    assert path == second
    assert np.array_equal(frames, again)
    assert np.array_equal(times, repeated)
    assert raw.read_bytes() == b"raw source"
    assert list(raw.parent.iterdir()) == [raw]


def test_changed_raw_identity_does_not_reuse_frame_cache(tmp_path, monkeypatch):
    raw, cache, calls = fixture(tmp_path, monkeypatch)
    first = F.sample_recording([raw], cache, width=32)[2]
    raw.write_bytes(b"changed raw input")
    second = F.sample_recording([raw], cache, width=32)[2]
    assert first != second
    assert len(calls) == 2


@pytest.mark.parametrize("body", [b"partial", bytes(32 * 18)])
def test_partial_samples_are_not_cached(tmp_path, monkeypatch, body):
    raw, cache, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(F.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=body))
    with pytest.raises(ValueError):
        F.sample_recording([raw], cache, width=32)
    assert not list(cache.glob("*.npz"))


def test_corrupt_sample_blocks_without_starting_decode(tmp_path, monkeypatch):
    raw, cache, calls = fixture(tmp_path, monkeypatch)
    _, _, path = F.sample_recording([raw], cache, width=32)
    np.savez(path, frames=np.zeros((5, 18, 32), np.uint8), times=np.full(5, np.nan))
    with pytest.raises(ValueError, match="invalid frame cache"):
        F.sample_recording([raw], cache, width=32)
    assert len(calls) == 1


def test_decode_failure_does_not_leave_a_cache(tmp_path, monkeypatch):
    raw, cache, _ = fixture(tmp_path, monkeypatch)
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(F.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        F.sample_recording([raw], cache, width=32)
    assert not list(cache.glob("*.npz"))
