# tests/test_audio_signals.py — whistles, stoppages, PA music spans (hhg-3r5.29)
import numpy as np

import audio_signals as A

RATE = A.RATE


def _score(total_s, events, base=2.0):
    """Score and Hz arrays at 100 frames/s. events: (start_s, dur_s, db, hz, hz_jitter)."""
    n = int(total_s * RATE)
    score = np.full(n, base, np.float32)
    hz = np.full(n, 2900.0, np.float32)
    rng = np.random.default_rng(0)
    for start, dur, db, f, jit in events:
        a, b = int(start * RATE), int((start + dur) * RATE)
        score[a:b] = db
        hz[a:b] = f + rng.uniform(-jit, jit, b - a)
    return score, hz


# --- whistle score --------------------------------------------------------------

def test_whistle_score_finds_narrow_band():
    freqs = np.arange(257) * (A.SR / A.N_FFT)
    db = np.full((257, 5), -80.0)
    band = (freqs >= 2200) & (freqs <= 2450)
    db[band, 2] = -50.0
    score, hz = A.whistle_score(db, freqs)
    assert score[2] > 20 and 2150 <= hz[2] <= 2500
    assert abs(score[0]) < 1e-6


# --- whistle picking ------------------------------------------------------------

def test_find_whistles_keeps_steady_tone():
    s, h = _score(20, [(5.0, 0.8, 20.0, 2300, 20)])
    ws = A.find_whistles(s, h)
    assert len(ws) == 1
    w = ws[0]
    assert abs(w["start"] - 5.0) < 0.1 and abs(w["end"] - 5.8) < 0.1
    assert w["peak_db"] == 20.0 and abs(w["hz"] - 2300) < 30 and w["hz_sd"] < 40


def test_find_whistles_rejects_short_weak_wobbly_and_off_band():
    s, h = _score(40, [
        (2.0, 0.1, 20.0, 2300, 10),    # too short
        (8.0, 1.0, 9.0, 2300, 10),     # too weak (peak < 10 dB)
        (11.0, 0.25, 20.0, 2300, 10),  # shorter than 0.3 s
        (14.0, 1.0, 20.0, 2300, 600),  # frequency not steady (horn, voice)
        (20.0, 1.0, 20.0, 2900, 10),   # outside 1.9-2.7 kHz
    ])
    assert A.find_whistles(s, h) == []


def test_find_whistles_less_steady_tone_must_be_strong():
    # Uniform jitter of 350 Hz gives a frequency SD of about 200 Hz.
    weak, h = _score(20, [(5.0, 1.0, 11.0, 2300, 350)])
    assert A.find_whistles(weak, h) == []
    strong, h = _score(20, [(5.0, 1.0, 14.0, 2300, 350)])
    assert len(A.find_whistles(strong, h)) == 1
    steady_weak, h = _score(20, [(5.0, 1.0, 11.0, 2300, 10)])
    assert len(A.find_whistles(steady_weak, h)) == 1


def test_find_whistles_merges_short_gap():
    s, h = _score(20, [(5.0, 0.5, 20.0, 2300, 10), (5.6, 0.5, 20.0, 2300, 10)])
    ws = A.find_whistles(s, h)
    assert len(ws) == 1 and ws[0]["end"] > 6.0


def test_find_whistles_uses_time_offset():
    s, h = _score(10, [(1.0, 0.8, 20.0, 2300, 10)])
    assert abs(A.find_whistles(s, h, t0=100.0)[0]["start"] - 101.0) < 0.1


def test_merge_camera_whistles():
    per = {"cam1": [{"start": 10.0, "end": 10.8, "peak_db": 15.0, "hz": 2300}],
           "cam2": [{"start": 10.3, "end": 11.0, "peak_db": 22.0, "hz": 2310},
                    {"start": 30.0, "end": 30.5, "peak_db": 14.0, "hz": 2100}]}
    ws = A.merge_camera_whistles(per)
    assert len(ws) == 2
    assert ws[0]["cameras"] == ["cam1", "cam2"] and ws[0]["t"] == 10.0 and ws[0]["end"] == 11.0
    assert ws[0]["peak_db"] == 22.0 and ws[0]["hz"] == 2310
    assert ws[1]["cameras"] == ["cam2"]


# --- activity and stoppages -----------------------------------------------------

def test_activity_rank_keeps_nan_and_ranks_0_to_1():
    flux = np.concatenate([np.linspace(0, 1, 1000), [np.nan] * 10])
    r = A.activity_rank(flux)
    assert np.isnan(r[-1]) and np.nanmin(r) == 0.0 and np.nanmax(r) == 1.0


def test_combine_activity_ignores_missing_camera():
    a = np.array([0.2, 0.4, np.nan], np.float32)
    b = np.array([0.4, np.nan], np.float32)
    assert np.allclose(A.combine_activity([a, b])[:2], [0.3, 0.4])
    assert np.isnan(A.combine_activity([a, b])[2])


def _activity(total_s, quiet):
    act = np.full(int(total_s * RATE), 0.8, np.float32)
    for a, b in quiet:
        act[int(a * RATE):int(b * RATE)] = 0.1
    return act


def _w(t, dur=0.6):
    return {"t": t, "start": t, "end": t + dur}


def test_whistle_then_quiet_is_stoppage_ending_at_restart():
    act = _activity(120, [(10.5, 40.0)])
    ws = [_w(10.0), _w(38.0), _w(70.0)]
    st = A.find_stoppages(ws, act)
    assert len(st) == 1
    s = st[0]
    assert s["start"] == 10.0 and abs(s["end"] - 40.0) < 0.05 and s["restart_found"]
    assert s["whistles"] == [10.0, 38.0]
    assert [w["role"] for w in ws] == ["stoppage", "in_stoppage", "no_stoppage"]
    assert ws[2]["stoppage"] is None


def test_stoppage_without_restart_is_capped():
    act = _activity(400, [(10.5, 400)])
    st = A.find_stoppages([_w(10.0)], act)
    assert not st[0]["restart_found"] and abs(st[0]["end"] - (10.0 + A.STOPPAGE["max_s"])) < 0.05


def test_short_loud_blip_is_not_restart():
    act = _activity(120, [(10.5, 60.0)])
    act[int(30 * RATE):int(31 * RATE)] = 0.9  # 1 s, restart needs 2 s
    st = A.find_stoppages([_w(10.0)], act)
    assert abs(st[0]["end"] - 60.0) < 0.05


# --- PA music spans -------------------------------------------------------------

def test_music_bins_average_by_frame_centre():
    bins = A.music_bins(np.array([0.2, 0.4, 0.6, 0.8]), hop_s=0.5, win_s=1.0)
    # centres 0.5, 1.0, 1.5, 2.0 -> bins 0: [0.2], 1: [0.4, 0.6], 2: [0.8]
    assert np.allclose(bins, [0.2, 0.5, 0.8])


def test_music_span_rule_pads_and_drops_short():
    b = np.zeros(60, np.float32)
    b[10:20] = 0.5      # 10 s music -> mute
    b[30:32] = 0.5      # 2 s -> dropped (and too short for median)
    spans = A.music_spans(b, "cam1")
    mute = [s for s in spans if s["state"] == "mute"]
    assert len(mute) == 1
    assert mute[0]["start"] == 9.0 and mute[0]["end"] == 22.0 and mute[0]["camera"] == "cam1"
    assert mute[0]["peak"] == 0.5


def test_music_span_hysteresis_and_review():
    b = np.zeros(60, np.float32)
    b[5:10] = 0.3
    b[10:15] = 0.1      # between end (0.075) and start (0.15): stays on
    b[40:45] = 0.1      # review band for 5 s, never reaches start
    spans = A.music_spans(b, "cam2", t0=100.0)
    assert [(s["state"], s["start"], s["end"]) for s in spans] == [("mute", 104.0, 117.0),
                                                                     ("review", 140.0, 145.0)]


def test_music_padded_spans_that_touch_are_joined():
    b = np.zeros(40, np.float32)
    b[5:10] = 0.5
    b[11:16] = 0.5      # 1-s gap; median fills it -> one span
    b[18:23] = 0.5      # 2-s gap; padding (2 s + 1 s) makes them touch
    mute = [s for s in A.music_spans(b, "cam1") if s["state"] == "mute"]
    assert len(mute) == 1 and mute[0]["start"] == 4.0 and mute[0]["end"] == 25.0


def test_yamnet_env_override(tmp_path, monkeypatch):
    f = tmp_path / "m.tflite"
    f.write_bytes(b"x")
    monkeypatch.setenv(A.YAMNET_ENV, str(f))
    assert A.yamnet_model_path(download=False) == f


# --- review follow-ups (PR #24) -------------------------------------------------

def test_activity_rank_gives_equal_values_equal_rank():
    r = A.activity_rank(np.zeros(10000))
    assert np.allclose(r, r[0])
    # Constant sound is never "low then restart": no stoppage.
    assert A.find_stoppages([_w(10.0)], r) == []


def test_review_span_does_not_overlap_padded_mute_span():
    b = np.zeros(40, np.float32)
    b[3:6] = 0.1
    b[6:12] = 0.5
    spans = A.music_spans(b, "cam1")
    mute = [s for s in spans if s["state"] == "mute"]
    review = [s for s in spans if s["state"] == "review"]
    assert len(mute) == 1
    assert all(r["end"] <= mute[0]["start"] or r["start"] >= mute[0]["end"] for r in review)


def test_place_puts_recordings_at_start_with_nan_gap():
    f = {"score": np.ones(100, np.float32), "hz": np.ones(100, np.float32),
         "flux": np.ones(100, np.float32), "music": np.ones(4, np.float32)}
    out = A._place([(0.0, f), (5.0, f)], with_music=True)
    assert len(out["score"]) == 600
    assert np.isnan(out["score"][100:500]).all() and (out["score"][500:] == 1).all()
    j = int(round(5.0 / A.YAMNET_HOP_S))
    assert np.isnan(out["music"][4:j]).all() and (out["music"][j:j + 4] == 1).all()


def _fake_recording_features(calls):
    def f(lines, base, input_args, model, cache_dir, progress, music_id=None):
        calls.append((list(lines), list(input_args)))
        return {k: np.ones(100, np.float32) for k in ("score", "hz", "flux")}
    return f


def test_camera_features_single_recording_seeks_before_input(tmp_path, monkeypatch):
    m = tmp_path / "cam2_concat.txt"
    m.write_text("ffconcat version 1.0\n# seek 0.574\nfile 'a.MP4'\n")
    calls = []
    monkeypatch.setattr(A, "_recording_features", _fake_recording_features(calls))
    A.camera_features(str(m), None, None)
    args = calls[0][1]
    assert args[args.index("-ss") + 1] == "0.574" and args.index("-ss") < args.index("-i")


def test_camera_features_places_recording_blocks(tmp_path, monkeypatch):
    m = tmp_path / "cam1_concat.txt"
    m.write_text("ffconcat version 1.0\n# recording -0.5\nfile 'a.MP4'\n"
                 "# recording 10.0\nfile 'b.MP4'\n")
    calls = []
    monkeypatch.setattr(A, "_recording_features", _fake_recording_features(calls))
    out = A.camera_features(str(m), None, None)
    # First block starts before the timeline: seek 0.5 s into it, place at 0.
    assert "# seek 0.500" in calls[0][0] and not any("seek" in ln for ln in calls[1][0])
    assert len(out["score"]) == 1100
    assert (out["score"][:100] == 1).all() and np.isnan(out["score"][100:1000]).all()
    assert (out["score"][1000:] == 1).all()


def _fake_popen(monkeypatch, script):
    import subprocess
    import sys
    real = subprocess.Popen
    monkeypatch.setattr(A.subprocess, "Popen",
                        lambda argv, **kw: real([sys.executable, "-c", script], **kw))


def test_decode_partial_failure_raises_and_large_stderr_does_not_hang(monkeypatch):
    _fake_popen(monkeypatch, (
        "import sys\n"
        "sys.stderr.write('e' * 300000); sys.stderr.flush()\n"
        "sys.stdout.buffer.write(b'\\0' * 64000 * 4)\n"
        "sys.exit(1)\n"))
    import pytest
    with pytest.raises(RuntimeError, match="exit 1"):
        A._decode([], A._Features(None))


def test_decode_no_audio_raises(monkeypatch):
    _fake_popen(monkeypatch, "pass")
    import pytest
    with pytest.raises(RuntimeError, match="no audio"):
        A._decode([], A._Features(None))


def _cache_setup(tmp_path, monkeypatch):
    (tmp_path / "a.MP4").write_bytes(b"x")
    lines = ["ffconcat version 1.0", "file 'a.MP4'"]
    cache = tmp_path / "cache"
    cache.mkdir()
    files = [str(tmp_path / "a.MP4")]
    decoded = []

    class FakeYamnet:
        def __init__(self, model):
            pass

    def fake_decode(args, feats, progress=None):
        decoded.append(True)
        feats.score, feats.hz, feats.flux = ([np.zeros(5, np.float32)] for _ in range(3))
        if feats.yamnet is not None:
            feats.music = [0.5]
        return 1.0

    monkeypatch.setattr(A, "_Yamnet", FakeYamnet)
    monkeypatch.setattr(A, "_decode", fake_decode)
    return lines, cache, files, decoded


def test_cache_without_music_is_not_used_when_model_is_present(tmp_path, monkeypatch):
    lines, cache, files, decoded = _cache_setup(tmp_path, monkeypatch)
    old = {k: np.zeros(5, np.float32) for k in ("score", "hz", "flux")}
    np.savez(cache / f"audio_{A._feature_key(files, 0.0, None)}.npz", **old)
    out = A._recording_features(lines, str(tmp_path), [], tmp_path / "m.tflite", str(cache), None,
                                music_id="abc")
    assert decoded and "music" in out
    assert (cache / f"audio_{A._feature_key(files, 0.0, 'abc')}.npz").exists()


def test_cache_with_music_is_used_without_model_and_music_dropped(tmp_path, monkeypatch):
    lines, cache, files, decoded = _cache_setup(tmp_path, monkeypatch)
    old = {k: np.zeros(5, np.float32) for k in ("score", "hz", "flux", "music")}
    np.savez(cache / f"audio_{A._feature_key(files, 0.0, A.YAMNET_SHA256)}.npz", **old)
    out = A._recording_features(lines, str(tmp_path), [], None, str(cache), None)
    assert not decoded and "music" not in out


def test_other_model_file_gets_its_own_cache_key():
    assert A._feature_key([], 0.0, "abc") != A._feature_key([], 0.0, A.YAMNET_SHA256)


def test_bad_model_file_is_a_flag_not_a_crash(tmp_path, monkeypatch):
    import pytest
    pytest.importorskip("ai_edge_litert")
    bad = tmp_path / "bad.tflite"
    bad.write_bytes(b"not a model")
    monkeypatch.setattr(A, "yamnet_model_path", lambda download=True: bad)
    path, why = A.load_music_model()
    assert path is None and "could not be loaded" in why


def _game(tmp_path):
    for c in ("cam1", "cam2"):
        (tmp_path / f"{c}_concat.txt").write_text("ffconcat version 1.0\n")
    return tmp_path


def test_one_bad_camera_is_a_flag_and_other_camera_is_used(tmp_path, monkeypatch):
    score, hz = _score(60, [(10.0, 1.0, 20.0, 2200.0, 5.0)])

    def fake(manifest, model, cache_dir, progress=None, music_id=None):
        if "cam1" in manifest:
            raise RuntimeError("ffmpeg failed (exit 1)")
        return {"score": score, "hz": hz, "flux": np.ones(len(score), np.float32)}

    monkeypatch.setattr(A, "camera_features", fake)
    sig, mus = A.analyse(_game(tmp_path), use_music=False)
    assert sig["cameras"] == ["cam2"] and len(sig["whistles"]) == 1
    assert any("cam1: audio could not be read" in f for f in sig["flags"])


def test_broken_flow_cache_does_not_stop_analysis(tmp_path, monkeypatch):
    score, hz = _score(60, [(10.0, 1.0, 20.0, 2200.0, 5.0)])
    monkeypatch.setattr(A, "camera_features", lambda *a, **k: {
        "score": score, "hz": hz, "flux": np.ones(len(score), np.float32)})

    def boom(*a, **k):
        raise ValueError("corrupt npz")

    monkeypatch.setattr(A, "_cached_flow", boom)
    sig, _ = A.analyse(_game(tmp_path), use_music=False)
    assert len(sig["whistles"]) == 1
