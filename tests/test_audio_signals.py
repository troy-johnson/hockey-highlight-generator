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
