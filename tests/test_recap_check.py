# tests/test_recap_check.py — hockeyrecap check (hhg-3r5.62)
import recap_check as rc


def rec(name, start, duration, lumas, chapters=None):
    n = len(lumas)
    samples = [[duration * (0.02 + 0.96 * i / max(n - 1, 1)), l] for i, l in enumerate(lumas)]
    chapters = chapters or [{"path": f"/g/{name}", "start": start, "clock": "timecode", "duration": duration}]
    return {"chapters": chapters, "duration": duration, "start": start, "samples": samples}


BRIGHT = [80.0] * 9
T0 = 19 * 3600.0   # 19:00:00


def test_clean_game_is_ok():
    m = {"cameras": {"cam1": [rec("GX010017.MP4", T0, 3600, BRIGHT)],
                     "cam2": [rec("GX010038.MP4", T0 + 3, 3600, BRIGHT)]}}
    report = rc.analyze(m)
    assert report["ok"], report["findings"]
    assert report["cameras"]["cam1"]["footage_s"] == 3600


def test_covered_lens_recording():
    m = {"cameras": {"cam1": [rec("GX010017.MP4", T0 - 600, 300, [3.0] * 9),
                              rec("GX020017.MP4", T0, 3600, BRIGHT)],
                     "cam2": [rec("GX010038.MP4", T0, 3600, BRIGHT)]}}
    report = rc.analyze(m)
    kinds = [(f["kind"], f["camera"]) for f in report["findings"]]
    assert ("covered_lens", "cam1") in kinds
    assert report["cameras"]["cam1"]["usable_recordings"] == 1
    assert not report["ok"]


def test_partly_covered_recording():
    lumas = [80, 80, 80, 4, 4, 80, 80, 80, 80]
    m = {"cameras": {"cam1": [rec("GX010017.MP4", T0, 3600, lumas)],
                     "cam2": [rec("GX010038.MP4", T0, 3600, BRIGHT)]}}
    report = rc.analyze(m)
    assert [f["kind"] for f in report["findings"]] == ["partly_covered"]
    assert report["recordings"]["cam1"][0]["covered"] == "partial"


def test_camera_stopped_early():
    m = {"cameras": {"cam1": [rec("GX010017.MP4", T0, 3600, BRIGHT)],
                     "cam2": [rec("GX010038.MP4", T0, 2400, BRIGHT)]}}
    report = rc.analyze(m)
    early = [f for f in report["findings"] if f["kind"] == "stopped_early"]
    assert len(early) == 1 and early[0]["camera"] == "cam2"
    assert "20:00" in early[0]["text"]


def test_camera_started_late_and_recording_gap():
    m = {"cameras": {"cam1": [rec("GX010017.MP4", T0, 3600, BRIGHT)],
                     "cam2": [rec("GX010038.MP4", T0 + 600, 1200, BRIGHT),
                              rec("GX010039.MP4", T0 + 2400, 1200, BRIGHT)]}}
    kinds = {(f["kind"], f["camera"]) for f in rc.analyze(m)["findings"]}
    assert ("started_late", "cam2") in kinds
    assert ("recording_gap", "cam2") in kinds


def test_chapter_gap_and_missing_camera():
    ch = [{"path": "/g/GX010017.MP4", "start": T0, "clock": "timecode", "duration": 1000},
          {"path": "/g/GX020017.MP4", "start": T0 + 1060, "clock": "timecode", "duration": 1000}]
    m = {"cameras": {"cam1": [rec("GX010017.MP4", T0, 2000, BRIGHT, chapters=ch)]}}
    kinds = [f["kind"] for f in rc.analyze(m)["findings"]]
    assert "missing_camera" in kinds and "chapter_gap" in kinds


def test_ignored_files_do_not_fail_the_check():
    m = {"cameras": {"cam1": [rec("a.MP4", T0, 60, BRIGHT)], "cam2": [rec("b.MP4", T0, 60, BRIGHT)]},
         "excluded": [{"path": "/g/x.MP4", "reason": "no camera serial"}]}
    report = rc.analyze(m)
    assert report["ok"]
    assert report["findings"][0]["kind"] == "ignored_file"


def test_dark_spans():
    assert rc.dark_spans([[10, 80], [20, 3], [30, 3], [40, 80]], 50) == [(10, 40)]
    assert rc.dark_spans([[10, 3], [20, 80]], 50) == [(0.0, 20)]
    assert rc.dark_spans([[10, 80], [20, 3]], 50) == [(10, 50)]


def test_sample_count_bounds():
    assert rc._sample_count(60) == rc.MIN_SAMPLES
    assert rc._sample_count(3600) == 30
    assert rc._sample_count(99999) == rc.MAX_SAMPLES
