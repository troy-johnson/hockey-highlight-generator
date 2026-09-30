# tests/test_signals_perf.py — detection speed-ups (hhg-3r5.19): ROI-crop flow, VideoToolbox decode, parallel cameras
import subprocess
import numpy as np
import pytest
from unittest.mock import MagicMock

import signals
from signals import ROI, _roi_crop


def test_roi_crop_covers_both_rois_with_padding():
    net, slot = ROI(380, 180, 410, 240), ROI(380, 60, 410, 140)
    assert _roi_crop(net, slot, 1280, 720, pad=48) == (332, 12, 838, 468)


def test_roi_crop_is_clamped_to_frame():
    assert _roi_crop(ROI(0, 0, 100, 100), ROI(1200, 650, 80, 70), 1280, 720, pad=48) == (0, 0, 1280, 720)


def test_crop_flow_matches_full_frame_roi_means():
    rng = np.random.default_rng(0)
    base = (rng.random((720 + 40, 1280 + 40)) * 255).astype(np.uint8)
    import cv2
    base = cv2.GaussianBlur(base, (0, 0), 3)
    frames = [np.ascontiguousarray(base[20 + d: 740 + d, 20 + 2 * d: 1300 + 2 * d]) for d in range(0, 8, 2)]
    net, slot = ROI(380, 180, 410, 240), ROI(380, 60, 410, 140)
    x0, y0, x1, y1 = _roi_crop(net, slot, 1280, 720)
    for a, b in zip(frames, frames[1:]):
        full = signals._flow_magnitude(a, b)
        crop = signals._flow_magnitude(np.ascontiguousarray(a[y0:y1, x0:x1]), np.ascontiguousarray(b[y0:y1, x0:x1]))
        for r in (net, slot):
            f = signals._roi_mean(full, r)
            c = signals._roi_mean(crop, ROI(r.x - x0, r.y - y0, r.w, r.h))
            assert c == pytest.approx(f, rel=0.03)


def _popen_capture(captured, data=b""):
    def fake(cmd, **kw):
        captured.append(cmd)
        p = MagicMock()
        buf = [data]
        p.stdout.read.side_effect = lambda n: buf.pop(0) if buf else b""
        return p
    return fake


def test_hwaccel_command_when_available(monkeypatch):
    captured = []
    monkeypatch.setattr(subprocess, "Popen", _popen_capture(captured))
    monkeypatch.setattr(signals, "ffprobe_dims", lambda p: (3840, 2160))
    monkeypatch.setattr(signals, "_videotoolbox_available", lambda: True)
    monkeypatch.delenv("HHG_HWACCEL", raising=False)
    gen, w, h = signals._ffmpeg_gray_frames("v.mp4", fps=12, width=1280)
    list(gen)
    cmd = captured[0]
    assert cmd[cmd.index("-hwaccel") + 1] == "videotoolbox" and cmd.index("-hwaccel") < cmd.index("-i")
    assert "scale_vt=w=1280:h=720" in cmd[cmd.index("-vf") + 1]


def test_software_when_disabled(monkeypatch):
    captured = []
    monkeypatch.setattr(subprocess, "Popen", _popen_capture(captured))
    monkeypatch.setattr(signals, "ffprobe_dims", lambda p: (3840, 2160))
    monkeypatch.setattr(signals, "_videotoolbox_available", lambda: True)
    monkeypatch.setenv("HHG_HWACCEL", "0")
    gen, w, h = signals._ffmpeg_gray_frames("v.mp4", fps=12, width=1280)
    list(gen)
    assert "-hwaccel" not in captured[0]
    assert "scale=1280:720" in captured[0][captured[0].index("-vf") + 1]


def test_falls_back_to_software_when_hardware_yields_nothing(monkeypatch):
    frame = bytes(1280 * 720)
    calls = []

    def fake(cmd, **kw):
        calls.append(cmd)
        p = MagicMock()
        buf = [] if "-hwaccel" in cmd else [frame]
        p.stdout.read.side_effect = lambda n: buf.pop(0) if buf else b""
        return p

    monkeypatch.setattr(subprocess, "Popen", fake)
    monkeypatch.setattr(signals, "ffprobe_dims", lambda p: (3840, 2160))
    monkeypatch.setattr(signals, "_videotoolbox_available", lambda: True)
    monkeypatch.delenv("HHG_HWACCEL", raising=False)
    gen, w, h = signals._ffmpeg_gray_frames("v.mp4", fps=12, width=1280)
    out = list(gen)
    assert len(out) == 1 and len(calls) == 2 and "-hwaccel" not in calls[1]


def test_extract_both_runs_each_camera_once(monkeypatch):
    import detect_events as DE
    seen = []
    monkeypatch.setattr(DE, "extract_signals", lambda path, rois, fps, width, verbose=False, with_audio=True: seen.append(path) or (np.ones(3),) * 3)
    monkeypatch.setenv("HHG_PARALLEL", "0")
    (a, b) = DE.extract_both("c1.txt", "c2.txt", {"camera_1": 1, "camera_2": 2}, 12, 1280, False)
    assert sorted(seen) == ["c1.txt", "c2.txt"] and len(a) == 3 and len(b) == 3


def test_audio_is_skipped_when_not_needed(monkeypatch):
    frame = np.zeros((720, 1280), np.uint8)
    monkeypatch.setattr(signals, "_ffmpeg_gray_frames", lambda p, fps, width, hw=None: (iter([frame, frame, frame]), 1280, 720))
    called = []
    monkeypatch.setattr(signals, "extract_audio_rms", lambda *a, **k: called.append(1) or np.ones(2, np.float32))
    rois = {"net": ROI(380, 180, 410, 240), "slot": ROI(380, 60, 410, 140)}
    net, slot, audio = signals.extract_signals("v.mp4", rois, fps=12, width=1280, with_audio=False)
    assert not called and audio.tolist() == [0.0, 0.0] and len(net) == 2
    signals.extract_signals("v.mp4", rois, fps=12, width=1280)
    assert called


# Review finding (GPT-6.1 review of #19): a mid-stream hardware failure must not silently truncate detection.

def _decoder(monkeypatch, hw_frames, hw_rc, sw_frames, sw_rc=0):
    frame = bytes(1280 * 720)
    calls = []

    def fake(cmd, **kw):
        hw = "-hwaccel" in cmd
        calls.append("hw" if hw else "sw")
        p = MagicMock()
        buf = [frame] * (hw_frames if hw else sw_frames)
        p.stdout.read.side_effect = lambda n: buf.pop(0) if buf else b""
        p.wait.return_value = hw_rc if hw else sw_rc
        p.returncode = hw_rc if hw else sw_rc
        return p

    monkeypatch.setattr(subprocess, "Popen", fake)
    monkeypatch.setattr(signals, "ffprobe_dims", lambda p: (3840, 2160))
    monkeypatch.setattr(signals, "_videotoolbox_available", lambda: True)
    monkeypatch.delenv("HHG_HWACCEL", raising=False)
    return calls


def test_midstream_hardware_failure_restarts_in_software(monkeypatch):
    calls = _decoder(monkeypatch, hw_frames=2, hw_rc=1, sw_frames=5)
    rois = {"net": ROI(380, 180, 410, 240), "slot": ROI(380, 60, 410, 140)}
    net, slot, audio = signals.extract_signals("v.mp4", rois, fps=12, width=1280, with_audio=False)
    assert calls == ["hw", "sw"]
    assert len(net) == 4                      # 5 software frames -> 4 flow values; the partial hardware run is discarded


def test_software_decode_error_keeps_frames_and_warns(monkeypatch, capsys):
    _decoder(monkeypatch, hw_frames=0, hw_rc=1, sw_frames=3, sw_rc=1)
    rois = {"net": ROI(380, 180, 410, 240), "slot": ROI(380, 60, 410, 140)}
    net, _, _ = signals.extract_signals("v.mp4", rois, fps=12, width=1280, with_audio=False)
    assert len(net) == 2
    assert "ended with an error" in capsys.readouterr().out
