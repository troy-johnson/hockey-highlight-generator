import json
from pathlib import Path

import numpy as np
import pytest

import detection_regression as R


def clip(cid="g1", t=10, cam="cam1", goal=True):
    return {"id": cid, "reference_s": t, "camera": cam, "goal": goal}


def event(eid="e1", start=9, end=11, cam="cam1"):
    return {"id": eid, "start_s": start, "end_s": end, "camera": cam}


def key(clips=None, complete=False):
    return {"schema_version": 1, "timeline_kind": "detection", "clips_complete": complete,
            "timeline": {"cam1": [{"start_s": 0.0, "seek_s": 0.0, "files": ["cam1.MP4"]}],
                         "cam2": [{"start_s": 0.0, "seek_s": 0.0, "files": ["cam2.MP4"]}]},
            "clips": clips if clips is not None else [clip()]}


def test_half_open_no_tolerance():
    assert R.contains(event(start=10), clip())
    assert not R.contains(event(end=10), clip())
    assert not R.contains(event(start=10.001), clip())


def test_recall_camera_errors_duplicates_and_unknowns():
    k = key([clip(), clip("g2", 20, "cam2"), clip("g3", 30), clip("save", 40, goal=False)])
    events = [event(), event("duplicate"), event("wrong", 19, 21),
              event("save", 39, 41), event("unknown", 50, 60)]
    m = R.evaluate(k, events)
    assert m["goal_recall"] == 2 / 3
    assert [g["id"] for g in m["missed_goals"]] == ["g3"]
    assert m["correct_clip_matches"] == 2
    assert m["incorrect_clip_matches"] == 2
    assert {x["reason"] for x in m["incorrect_matches"]} == {"duplicate", "wrong_camera"}
    assert m["unjudged_events"] == 1
    k["clips_complete"] = True
    assert R.evaluate(k, events)["incorrect_clip_matches"] == 3


def test_maximum_matching_not_greedy():
    clips = [clip("a", 10), clip("b", 11)]
    events = [event("wide", 9, 12), event("narrow", 9, 10.5)]
    assert R.match_clips(clips, events) == {0: 1, 1: 0}
    assert R.match_clips(clips, events) == R.match_clips(clips, events)


def test_one_event_cannot_match_two_clips_but_recall_counts_both():
    m = R.evaluate(key([clip("a", 10), clip("b", 11)]), [event(end=12)])
    assert m["goals_found"] == 2
    assert m["correct_clip_matches"] == 1


def test_zero_goals_and_empty_events():
    assert R.evaluate(key([]), [event()])["goal_recall"] is None
    m = R.evaluate(key(), [])
    assert m["goal_recall"] == 0
    assert m["missed_goal_count"] == 1


def test_known_non_goals_have_separate_coverage_and_do_not_label_unknown_events():
    k = key([clip(), clip("save", 40, goal=False), clip("miss", 60, goal=False)])
    events = [event(), event("save", 39, 41), event("unknown", 50, 55)]
    m = R.evaluate(k, events)
    assert m["goals_total"] == 1
    assert m["goals_found"] == 1
    assert m["non_goals_total"] == 2
    assert m["non_goals_found"] == 1
    assert m["non_goal_recall"] == 0.5
    assert m["goal_clip_matches"] == 1
    assert m["non_goal_clip_matches"] == 1
    assert [c["id"] for c in m["missed_non_goals"]] == ["miss"]
    assert m["incorrect_clip_matches"] == 0
    assert m["unjudged_event_ids"] == ["unknown"]
    assert "precision" not in m


def test_non_goal_coverage_still_requires_the_correct_camera_for_a_clip_match():
    m = R.evaluate(key([clip("save", 40, "cam2", goal=False)]), [event("wrong", 39, 41)])
    assert m["goals_total"] == 0
    assert m["goal_recall"] is None
    assert m["non_goals_found"] == 1
    assert m["non_goal_clip_matches"] == 0
    assert m["incorrect_matches"][0]["reason"] == "wrong_camera"


def test_non_goal_coverage_is_unavailable_without_verified_negative_labels():
    m = R.evaluate(key(), [event("unknown", 50, 55)])
    assert m["non_goals_total"] == 0
    assert m["non_goal_recall"] is None
    assert m["missed_non_goals"] == []


@pytest.mark.parametrize("change", [
    lambda k: k["clips"].append(clip()),
    lambda k: k["clips"][0].update(reference_s=float("nan")),
    lambda k: k["clips"][0].update(camera="cam3"),
    lambda k: k["clips"][0].pop("goal"),
    lambda k: k.update(clips_complete="yes"),
    lambda k: k.update(timeline_kind="raw"),
])
def test_invalid_keys_fail(tmp_path, change):
    k = key()
    change(k)
    path = tmp_path / "key.json"
    path.write_text(json.dumps(k))
    with pytest.raises((ValueError, KeyError)):
        R.load_key(path)


@pytest.mark.parametrize("line", ["nan,20,1", "20,19,1", "1,2,3"])
def test_invalid_events_fail(tmp_path, line):
    p = tmp_path / "events.csv"
    p.write_text("start_s,end_s,primary_cam\n" + line + "\n")
    with pytest.raises(ValueError):
        R.read_events(p)


def result(events, runtime=2, signature="same"):
    return {"comparison_key": signature, "metrics": R.evaluate(key(), events),
            "runtime_s": runtime, "runtime_kind": "cached_signals", "run_id": "old"}


def test_comparison_first_change_and_incompatible():
    old, new = result([]), result([event()], 3)
    assert R.compare(new, None)["status"] == "first_result"
    diff = R.compare(new, old)
    assert diff["delta"]["goal_recall"] == 1
    assert diff["delta"]["correct_clip_matches"] == 1
    assert diff["recovered_goals"] == ["g1"]
    assert diff["new_clip_matches"] == ["g1"]
    assert diff["runtime_delta_s"] == 1
    assert R.compare(old, new)["newly_missed_goals"] == ["g1"]
    new["runtime_kind"] = "processing"
    assert R.compare(new, old)["runtime_delta_s"] is None
    new["comparison_key"] = "changed-key"
    assert R.compare(new, old)["status"] == "incomparable"


def test_comparison_accepts_saved_reports_without_new_label_coverage_metrics():
    old, new = result([]), result([event()])
    for name in ("non_goals_total", "non_goals_found", "non_goal_recall", "missed_non_goal_count",
                 "missed_non_goals", "goal_clip_matches", "non_goal_clip_matches"):
        old["metrics"].pop(name)
    comparison = R.compare(new, old)
    assert comparison["status"] == "compared"
    assert comparison["delta"]["goals_found"] == 1
    assert comparison["delta"]["non_goals_found"] is None
    assert comparison["delta"]["goal_clip_matches"] is None


def test_cli_reports_non_goals_as_coverage_not_classification(capsys):
    current = result([event(), event("save", 39, 41)])
    current["metrics"] = R.evaluate(key([clip(), clip("save", 40, goal=False)]),
                                    [event(), event("save", 39, 41)])
    current.update(game_id="synthetic", status="ok", candidate_source="flow",
                   comparison={"status": "first_result"})
    R.print_report({"games": [current]})
    output = capsys.readouterr().out
    assert "Known non-goals: 1/1" in output
    assert "not classification accuracy" in output


def make_suite(tmp_path, monkeypatch):
    keys, games = tmp_path / "keys", tmp_path / "games"
    keys.mkdir()
    games.mkdir()
    suite = keys / "suite.json"
    suite.write_text(json.dumps({"schema_version": 1, "games": [
        {"id": "synthetic", "folder": "synthetic", "answer_key": "key.json"}]}))
    (keys / "key.json").write_text(json.dumps(key()))
    game = games / "synthetic"
    game.mkdir()
    roi = {"net": [1, 2, 3, 4], "slot": [5, 6, 7, 8]}
    (game / "rois.json").write_text(json.dumps({"camera_1": roi, "camera_2": roi}))
    cache = game / ".recap_cache" / "signals"
    cache.mkdir(parents=True)
    monkeypatch.setenv("HHG_HWACCEL", "0")
    rois = R.signals.load_rois(str(game / "rois.json"))
    for n in (1, 2):
        video = game / f"cam{n}.MP4"
        video.write_bytes(b"synthetic; never decode")
        (game / f"cam{n}_concat.txt").write_text(f"ffconcat version 1.0\nfile '{video}'\n")
        cache_key = R.signals.signal_cache_key([str(video)], 0.0, rois[f"camera_{n}"], 12, 1280, False)
        a = np.zeros(600, dtype=np.float32)
        a[120:124] = 10
        np.savez(cache / f"signals_{cache_key}.npz", net=a, slot=a, audio=a * 0)
    return suite, games, game, tmp_path / "results"


def test_saved_runs_reuse_cache_keep_keys_and_game_outputs(monkeypatch, tmp_path):
    suite, games, game, results = make_suite(tmp_path, monkeypatch)
    (game / "events.csv").write_text("user output")
    before = {p: p.read_bytes() for root in (suite.parent, game) for p in root.rglob("*") if p.is_file()}
    def forbidden(*args, **kwargs):
        pytest.fail("footage extraction must not run")
    monkeypatch.setattr(R.signals, "extract_signals", forbidden)
    first = R.run_suite(suite, games, results)
    assert first["games"][0]["status"] == "ok"
    assert first["games"][0]["metrics"]["goals_found"] == 1
    assert first["games"][0]["metrics"]["correct_clip_matches"] == 1
    first_file = results / first["run_id"] / "result.json"
    saved = first_file.read_bytes()
    second = R.run_suite(suite, games, results)
    assert second["games"][0]["comparison"]["status"] == "compared"
    assert second["games"][0]["comparison"]["delta"]["event_count"] == 0
    assert first_file.read_bytes() == saved
    assert all(p.read_bytes() == value for p, value in before.items())
    config = json.loads(suite.read_text())
    config["detection"] = {"thresh_pct": 96}
    suite.write_text(json.dumps(config))
    third = R.run_suite(suite, games, results)
    assert third["games"][0]["settings"]["thresh_pct"] == 96
    assert third["games"][0]["comparison"]["status"] == "compared"


def test_missing_cache_blocks_and_does_not_hide_last_success(monkeypatch, tmp_path):
    suite, games, game, results = make_suite(tmp_path, monkeypatch)
    first = R.run_suite(suite, games, results)
    cache_file = next((game / ".recap_cache" / "signals").glob("*.npz"))
    cache_file.write_bytes(b"damaged")
    def forbidden(*args, **kwargs):
        pytest.fail("footage extraction must not run")
    monkeypatch.setattr(R.signals, "extract_signals", forbidden)
    second = R.run_suite(suite, games, results)
    assert second["games"][0]["status"] == "blocked"
    assert "approval" in second["games"][0]["error"]
    assert R.previous_results(results)["synthetic"]["run_id"] == first["run_id"]


def test_approved_processing_replaces_invalid_cache_without_false_regression(monkeypatch, tmp_path):
    suite, games, game, results = make_suite(tmp_path, monkeypatch)
    first = R.run_suite(suite, games, results)
    assert first["games"][0]["metrics"]["goals_found"] == 1
    generated = results / "cache" / "synthetic"
    generated.mkdir(parents=True)
    expected = {}
    for path in (game / ".recap_cache" / "signals").glob("*.npz"):
        expected[path.name] = path.read_bytes()
        with np.load(path) as z:
            arrays = {name: np.full_like(z[name], np.nan) for name in ("net", "slot", "audio")}
        np.savez(generated / path.name, **arrays)
        path.unlink()
    calls = []
    def synthetic_extract(*args, **kwargs):
        calls.append(args[0])
        signal = np.zeros(600, dtype=np.float32)
        signal[120:124] = 10
        return signal, signal, signal * 0
    monkeypatch.setattr(R.signals, "_extract_single_signals", synthetic_extract)
    second = R.run_suite(suite, games, results, allow_processing=True)
    saved = json.loads((results / second["run_id"] / "result.json").read_text())
    current = saved["games"][0]
    assert current["status"] == "ok"
    assert current["metrics"]["goals_found"] == 1
    assert current["comparison"]["delta"]["goal_recall"] == 0
    assert len(calls) == 2
    for name in expected:
        with np.load(generated / name) as z:
            assert all(np.isfinite(z[k]).all() for k in ("net", "slot", "audio"))
    third = R.run_suite(suite, games, results)
    assert third["games"][0]["status"] == "ok"
    assert third["games"][0]["metrics"]["goals_found"] == 1
    assert len(calls) == 2


@pytest.mark.parametrize("invalid", ["nan", "empty", "mismatched_lengths", "two_dimensions"])
def test_invalid_fresh_signals_block_without_replacing_previous_success(monkeypatch, tmp_path, invalid):
    suite, games, game, results = make_suite(tmp_path, monkeypatch)
    first = R.run_suite(suite, games, results)
    previous_file = results / first["run_id"] / "result.json"
    previous_bytes = previous_file.read_bytes()
    for path in (game / ".recap_cache" / "signals").glob("*.npz"):
        path.unlink()
    arrays = [np.zeros(600, dtype=np.float32) for _ in range(3)]
    if invalid == "nan":
        arrays[0][:] = np.nan
    elif invalid == "empty":
        arrays = [np.array([], dtype=np.float32)] * 3
    elif invalid == "mismatched_lengths":
        arrays[1] = arrays[1][:-1]
    else:
        arrays[0] = arrays[0].reshape(2, 300)
    monkeypatch.setattr(R.signals, "_extract_single_signals", lambda *args, **kwargs: tuple(arrays))
    second = R.run_suite(suite, games, results, allow_processing=True)
    saved = json.loads((results / second["run_id"] / "result.json").read_text())
    assert saved["games"][0]["status"] == "blocked"
    assert "invalid signal arrays" in saved["games"][0]["error"]
    assert previous_file.read_bytes() == previous_bytes
    assert R.previous_results(results)["synthetic"]["run_id"] == first["run_id"]
    assert not list((results / "cache" / "synthetic").rglob("*.npz"))
    assert not list((results / "cache" / "synthetic").rglob("*.txt"))


def test_cache_recording_placement_and_seek_rules(monkeypatch, tmp_path):
    monkeypatch.setenv("HHG_HWACCEL", "0")
    roi = {"net": R.signals.ROI(1, 2, 3, 4), "slot": R.signals.ROI(5, 6, 7, 8)}
    for name in ("a", "b"):
        (tmp_path / f"{name}.MP4").write_bytes(b"x")
    manifest = tmp_path / "cam.txt"
    manifest.write_text("ffconcat version 1.0\n# recording -2.000\nfile 'a.MP4'\n"
                        "# recording 5.000\nfile 'b.MP4'\n")
    parts = R.manifest_parts(manifest)
    assert parts[0]["seek_s"] == 2
    settings = {"fps": 1, "width": 1280, "audio_weight": 0}
    for i, part in enumerate(parts, 1):
        cache_key = R.signals.signal_cache_key(part["files"], part["seek_s"], roi, 1, 1280, False)
        a = np.full(2, i, dtype=np.float32)
        np.savez(tmp_path / f"signals_{cache_key}.npz", net=a, slot=a, audio=a)
    arrays, misses = R.load_camera_cache(parts, roi, settings, [tmp_path])
    assert not misses
    assert arrays[0].tolist() == [1, 1, 0, 0, 0, 2, 2]
    settings["fps"] = 2
    assert len(R.load_camera_cache(parts, roi, settings, [tmp_path])[1]) == 2


def test_approved_processing_reuses_valid_recordings(monkeypatch, tmp_path):
    monkeypatch.setenv("HHG_HWACCEL", "0")
    roi = {"net": R.signals.ROI(1, 2, 3, 4), "slot": R.signals.ROI(5, 6, 7, 8)}
    parts = []
    for i in range(2):
        p = tmp_path / f"{i}.MP4"
        p.write_bytes(b"x")
        parts.append({"start_s": i * 5.0, "seek_s": 0.0, "files": [str(p)]})
    settings = {"fps": 1, "width": 1280, "audio_weight": 0}
    k = R.signals.signal_cache_key(parts[0]["files"], 0.0, roi, 1, 1280, False)
    a = np.ones(2, dtype=np.float32)
    np.savez(tmp_path / f"signals_{k}.npz", net=a, slot=a, audio=a)
    calls = []
    def fake(manifest, *args, **kwargs):
        calls.append(Path(manifest).read_text())
        return (a * 2,) * 3
    monkeypatch.setattr(R.signals, "extract_signals", fake)
    cache = tmp_path / "generated"
    arrays, misses = R.load_camera_cache(parts, roi, settings, [tmp_path], processing_cache=cache)
    assert len(calls) == 1 and "1.MP4" in calls[0]
    assert len(misses) == 1
    assert arrays[0].tolist() == [1, 1, 0, 0, 0, 2, 2]
    assert not list(cache.glob("*.txt"))


@pytest.mark.parametrize("target", ["keys", "game", "game-child", "parent"])
def test_results_cannot_overwrite_inputs(monkeypatch, tmp_path, target):
    suite, games, game, results = make_suite(tmp_path, monkeypatch)
    output = {"keys": suite.parent, "game": game, "game-child": game / "results", "parent": tmp_path}[target]
    with pytest.raises(ValueError, match="separate"):
        R.run_suite(suite, games, output)


def test_suite_continues_after_missing_game(monkeypatch, tmp_path):
    suite, games, game, results = make_suite(tmp_path, monkeypatch)
    config = json.loads(suite.read_text())
    config["games"].insert(0, {"id": "missing", "folder": "missing", "answer_key": "key.json"})
    suite.write_text(json.dumps(config))
    report = R.run_suite(suite, games, results)
    assert [g["status"] for g in report["games"]] == ["blocked", "ok"]


def test_cli_runs_and_reports(monkeypatch, tmp_path, capsys):
    suite, games, game, results = make_suite(tmp_path, monkeypatch)
    args = ["--suite", str(suite), "--games-root", str(games), "--results", str(results)]
    assert R.main(args) == 0
    assert "runtime" in capsys.readouterr().out
    for p in (game / ".recap_cache" / "signals").glob("*.npz"):
        p.write_bytes(b"invalid")
    assert R.main(args) == 2
    assert "BLOCKED" in capsys.readouterr().out


def test_ghost_key_has_only_verified_references():
    path = Path(__file__).parent / "answer_keys" / "ghostpirates_11092025.json"
    k = R.load_key(path)
    assert len(k["clips"]) == 15
    assert all(c["goal"] for c in k["clips"])
    assert not k["clips_complete"]
    assert k["published_clip_count"] == 20
    assert "chosen" not in k


def test_yetis_key_has_nine_visually_verified_goals_and_both_cameras():
    path = Path(__file__).parent / "answer_keys" / "yetis_11022025.json"
    k = R.load_key(path)
    goals = [c for c in k["clips"] if c["goal"]]
    assert len(goals) == 9
    assert not k["clips_complete"]
    assert k["verified_goal_count"] == 9
    assert k["source"]["recap_url"] == "https://www.youtube.com/watch?v=67MCRG0OGzk"
    assert len(k["source"]["recap_sha256"]) == 64
    assert k["timeline"] == {
        "cam1": [{"start_s": 0.0, "seek_s": 0.0, "files": ["GX010025.MP4", "GX020025.MP4"]}],
        "cam2": [{"start_s": 0.0, "seek_s": 9.813, "files": ["GX010006.MP4", "GX020006.MP4"]}],
    }
    by_id = {c["id"]: c for c in goals}
    assert set(by_id) == {"yetis:1", *(f"icepak:{n}" for n in range(1, 9))}
    assert {c["id"] for c in goals if c["camera"] == "cam1"} == {"icepak:5", "icepak:6"}
    for c in goals:
        evidence = c["verified_source"]
        assert evidence["verification"] == "published goal banner and paired raw frame visually checked"
        assert len(evidence["paired_image_sha256"]) == 64
        assert c["reference_s"] == pytest.approx(evidence["raw_s"] - (9.813 if c["camera"] == "cam2" else 0))
        assert "predicted_detection_s" not in c
        assert "motion" not in c and "refinement" not in c and "chosen" not in c
    registry = json.loads((path.parent / "suite.json").read_text())
    assert registry["games"][0]["id"] == "ghostpirates_11092025"
    assert any(g["id"] == "yetis_11022025" and g["answer_key"] == path.name for g in registry["games"])


def test_yetis_non_goals_keep_obstructed_outcome_unlabeled_and_preserve_sync():
    k = R.load_key(Path(__file__).parent / "answer_keys" / "yetis_11022025.json")
    negatives = [c for c in k["clips"] if not c["goal"]]
    assert len(negatives) == k["verified_non_goal_count"] == 2
    assert not k["clips_complete"]
    assert {c["id"] for c in negatives} == {"non_goal:006", "non_goal:010"}
    assert any("edl:008" in limit and "unverified" in limit for limit in k["limits"])
    assert len(k["source"]["non_goal_review_sha256"]) == 64
    for c in negatives:
        evidence = c["verified_source"]
        assert evidence["review_rate_fps"] == 4
        assert evidence["non_goal_evidence"]
        assert len(evidence["paired_image_sha256"]) == 64
        assert c["reference_s"] == pytest.approx(evidence["raw_s"] - (9.813 if c["camera"] == "cam2" else 0))
        assert evidence["review_span_raw_s"][0] < evidence["chapter_s"] < evidence["review_span_raw_s"][1]
        assert "predicted_detection_s" not in c and "play_type" not in c


def test_flyers_key_excludes_obstructed_goal_without_inventing_camera_labels():
    path = Path(__file__).parent / "answer_keys" / "flyers_12142025.json"
    k = R.load_key(path)
    assert k["verified_goal_count"] == 2
    assert k["published_goal_count"] == 3
    assert not k["clips_complete"]
    assert {c["id"] for c in k["clips"]} == {"icepak:1", "flyers:1"}
    assert all(c["goal"] and c["camera"] == "cam2" for c in k["clips"])
    assert "flyers:2" not in {c["id"] for c in k["clips"]}
    assert any("obstructed" in text for text in k["limits"])
    for c in k["clips"]:
        assert c["reference_s"] == pytest.approx(c["verified_source"]["raw_s"])
        assert len(c["verified_source"]["paired_image_sha256"]) == 64
    registry = json.loads((path.parent / "suite.json").read_text())
    entry = next(g for g in registry["games"] if g["id"] == "flyers_12142025")
    assert entry["folder"] == "flyers_121142025"
    assert entry["answer_key"] == path.name


def test_blb_key_has_three_visually_verified_goals_and_explicit_folder_mapping():
    path = Path(__file__).parent / "answer_keys" / "blb_01042026.json"
    k = R.load_key(path)
    assert k["verified_goal_count"] == k["published_goal_count"] == 3
    assert not k["clips_complete"]
    goals = [c for c in k["clips"] if c["goal"]]
    assert {c["id"] for c in goals} == {"blb:1", "icepak:1", "icepak:2"}
    assert all(c["camera"] == "cam1" for c in goals)
    assert k["timeline"]["cam2"][0]["seek_s"] == 1.331
    assert k["source"]["recap_url"] == "https://www.youtube.com/watch?v=OndBGzB3Nwg"
    for c in goals:
        assert c["reference_s"] == pytest.approx(c["verified_source"]["raw_s"])
        assert len(c["verified_source"]["paired_image_sha256"]) == 64
        assert "motion" not in c and "refinement" not in c
    registry = json.loads((path.parent / "suite.json").read_text())
    entry = next(g for g in registry["games"] if g["id"] == "blb_01042026")
    assert entry["folder"] == "buzzlightbeer_01042025"
    assert entry["answer_key"] == path.name


def test_blb_non_goals_have_independent_play_review_and_verified_sync_mapping():
    path = Path(__file__).parent / "answer_keys" / "blb_01042026.json"
    k = R.load_key(path)
    negatives = [c for c in k["clips"] if not c["goal"]]
    assert len(negatives) == k["verified_non_goal_count"] == 3
    assert not k["clips_complete"]
    assert {c["id"] for c in negatives} == {"non_goal:003", "non_goal:006", "non_goal:009"}
    assert len(k["source"]["non_goal_review_sha256"]) == 64
    for c in negatives:
        evidence = c["verified_source"]
        assert evidence["review_rate_fps"] == 4
        assert evidence["non_goal_evidence"]
        assert "context sheets visually reviewed" in evidence["verification"]
        assert len(evidence["paired_image_sha256"]) == 64
        assert c["reference_s"] == pytest.approx(evidence["raw_s"] - (1.331 if c["camera"] == "cam2" else 0))
        assert evidence["review_span_raw_s"][0] < evidence["chapter_s"] < evidence["review_span_raw_s"][1]
        assert "predicted_detection_s" not in c and "play_type" not in c


def test_blue_devils_key_keeps_unverified_seventh_icepak_goal_out():
    path = Path(__file__).parent / "answer_keys" / "bluedevils_10192025.json"
    k = R.load_key(path)
    assert k["verified_goal_count"] == 10
    assert k["published_goal_count"] == 11
    assert not k["clips_complete"]
    assert len(k["clips"]) == 10
    by_id = {c["id"]: c for c in k["clips"]}
    assert "icepak:7" not in by_id
    assert set(by_id) == {*(f"icepak:{n}" for n in (1, 2, 3, 4, 5, 6, 8)),
                          *(f"bluedevils:{n}" for n in range(1, 4))}
    assert {c["id"] for c in k["clips"] if c["camera"] == "cam1"} == {
        "bluedevils:1", "bluedevils:3", "icepak:4", "icepak:5", "icepak:6"}
    assert k["timeline"]["cam2"][0]["seek_s"] == 13.151
    for c in k["clips"]:
        raw = c["verified_source"]["raw_s"]
        assert c["reference_s"] == pytest.approx(raw - (13.151 if c["camera"] == "cam2" else 0))
        assert len(c["verified_source"]["paired_image_sha256"]) == 64
        assert "predicted_detection_s" not in c
    registry = json.loads((path.parent / "suite.json").read_text())
    assert any(g["id"] == "bluedevils_10192025" and g["answer_key"] == path.name for g in registry["games"])


def test_opposing_camera_peaks_keep_their_correct_clip_matches(monkeypatch, tmp_path):
    suite, games, game, results = make_suite(tmp_path, monkeypatch)
    answer = key([clip("earlier", 10.2, "cam2"), clip("later", 20.2, "cam1")])
    (suite.parent / "key.json").write_text(json.dumps(answer))
    rois = R.signals.load_rois(str(game / "rois.json"))
    for n, peak, height in ((1, 20, 4), (2, 10, 10)):
        values = np.zeros(600, dtype=np.float32)
        values[peak * 12:peak * 12 + 4] = height
        video = game / f"cam{n}.MP4"
        cache_key = R.signals.signal_cache_key([str(video)], 0.0, rois[f"camera_{n}"], 12, 1280, False)
        np.savez(game / ".recap_cache" / "signals" / f"signals_{cache_key}.npz",
                 net=values, slot=values, audio=values * 0)
    legacy = R.run_suite(suite, games, results)
    assert legacy["games"][0]["metrics"]["correct_clip_matches"] == 1
    config = json.loads(suite.read_text())
    config["detection"] = {"preserve_peaks": True}
    suite.write_text(json.dumps(config))
    report = R.run_suite(suite, games, results)
    metrics = report["games"][0]["metrics"]
    assert metrics["goals_found"] == 2
    assert metrics["correct_clip_matches"] == 2
    assert metrics["incorrect_clip_matches"] == 0
    assert report["games"][0]["comparison"]["delta"]["correct_clip_matches"] == 1


def test_nearby_peaks_survive_duration_caps_and_start_time_cooldown(monkeypatch, tmp_path):
    suite, games, game, results = make_suite(tmp_path, monkeypatch)
    (suite.parent / "key.json").write_text(json.dumps(key([
        clip("earlier", 12.2), clip("later", 18.2)])))
    config = json.loads(suite.read_text())
    config["detection"] = {"min_sep_s": 6, "max_win_s": 4, "max_big_win_s": 4}
    suite.write_text(json.dumps(config))
    rois = R.signals.load_rois(str(game / "rois.json"))
    for n in (1, 2):
        values = np.zeros(600, dtype=np.float32)
        if n == 1:
            values[144:148] = 10
            values[216:220] = 4
        cache_key = R.signals.signal_cache_key([str(game / f"cam{n}.MP4")], 0.0,
                                               rois[f"camera_{n}"], 12, 1280, False)
        np.savez(game / ".recap_cache" / "signals" / f"signals_{cache_key}.npz",
                 net=values, slot=values, audio=values * 0)
    legacy = R.run_suite(suite, games, results)
    assert legacy["games"][0]["metrics"]["goals_found"] == 0
    config["detection"]["preserve_peaks"] = True
    suite.write_text(json.dumps(config))
    report = R.run_suite(suite, games, results)
    assert report["games"][0]["metrics"]["goals_found"] == 2
    assert report["games"][0]["metrics"]["correct_clip_matches"] == 2
    assert report["games"][0]["metrics"]["event_count"] == 2
    generated = results / report["run_id"] / "synthetic"
    assert all(e["end_s"] - e["start_s"] <= 4.001 for e in R.read_events(generated / "events.csv"))
    assert (generated / "markers.csv").read_text().splitlines()[0] == "Frame,Name,Note,Color,Duration"


def test_whole_window_camera_score_can_hide_an_earlier_attack(monkeypatch, tmp_path):
    suite, games, game, results = make_suite(tmp_path, monkeypatch)
    (suite.parent / "key.json").write_text(json.dumps(key([
        clip("earlier", 15.2, "cam1"), clip("later", 20.2, "cam2")])))
    rois = R.signals.load_rois(str(game / "rois.json"))
    fused = []
    for n, peak, height in ((1, 15, 4), (2, 20, 8)):
        values = np.ones(1200, dtype=np.float32)
        values[peak * 12:peak * 12 + 4] = height
        cache_key = R.signals.signal_cache_key([str(game / f"cam{n}.MP4")], 0.0,
                                               rois[f"camera_{n}"], 12, 1280, False)
        np.savez(game / ".recap_cache" / "signals" / f"signals_{cache_key}.npz",
                 net=values, slot=values, audio=values * 0)
        attack = R.detect_events.fuse_signals(values, values, values * 0, 0.8, 0)
        fused.append(R.detect_events.moving_avg(attack, 6))
    report = R.run_suite(suite, games, results, detection_overrides={"preserve_peaks": True})
    metrics = report["games"][0]["metrics"]
    assert metrics["correct_clip_matches"] == 2
    rows = R.read_events(results / report["run_id"] / "synthetic" / "events.csv")
    earlier = next(row for row in rows if row["camera"] == "cam1")
    assert earlier["start_s"] <= 15.2 < 20.2 < earlier["end_s"]
    maxima = [R.detect_events.score_window(a, 12, earlier["start_s"], earlier["end_s"])["max_attack"]
              for a in fused]
    assert maxima[1] > maxima[0]
    reassigned = [{**row, "camera": "cam2"} for row in rows]
    assert R.evaluate(key([clip("earlier", 15.2, "cam1"), clip("later", 20.2, "cam2")]),
                      reassigned)["correct_clip_matches"] == 1


def test_cli_candidate_experiment_does_not_change_keys_or_defaults(monkeypatch, tmp_path, capsys):
    suite, games, game, results = make_suite(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in suite.parent.iterdir()}
    defaults = json.loads(json.dumps(R.DEFAULTS["detection"]))
    assert R.main(["--suite", str(suite), "--games-root", str(games), "--results", str(results),
                   "--set", "detection.preserve_peaks=true", "--set", "detection.min_sep_s=6",
                   "--set", "detection.thresh_pct=90"]) == 0
    saved = R.previous_results(results)["synthetic"]
    assert saved["settings"]["preserve_peaks"] is True
    assert saved["settings"]["min_sep_s"] == 6
    assert saved["settings"]["thresh_pct"] == 90
    assert "cached_signals" in capsys.readouterr().out
    assert R.DEFAULTS["detection"] == defaults
    assert all(p.read_bytes() == content for p, content in before.items())


def make_whistle_suite(tmp_path, monkeypatch):
    import audio_signals as A
    import coverage as C
    import selection as S

    suite, games, game, results = make_suite(tmp_path, monkeypatch)
    monkeypatch.setattr(C, "_duration", lambda path: 50.0)
    rois = R.signals.load_rois(str(game / "rois.json"))
    audio_cache = game / ".recap_cache" / "audio"
    audio_cache.mkdir()
    for n in (1, 2):
        values = np.zeros(600, dtype=np.float32)
        if n == 1:
            values[120:124] = 0.5
            values[300:360] = 20
        files = [str(game / f"cam{n}.MP4")]
        cache_key = R.signals.signal_cache_key(files, 0.0, rois[f"camera_{n}"], 12, 1280, False)
        np.savez(game / ".recap_cache" / "signals" / f"signals_{cache_key}.npz",
                 net=values, slot=values, audio=values * 0)
        feature_key = A._feature_key(files, 0.0, None)
        np.savez(audio_cache / f"audio_{feature_key}.npz", flux=np.zeros(5000, dtype=np.float32))
    audio = {"timeline": S.TIMELINE, "sync_offset_s": 0.0,
             "whistles": [{"t": 11.2, "role": "stoppage", "stoppage": 0},
                          {"t": 12.0, "role": "in_stoppage", "stoppage": 0}],
             "stoppages": [{"start": 11.2, "end": 31.2, "whistles": [11.2, 12.0]}]}
    (game / "audio_signals.json").write_text(json.dumps(audio))
    (game / "sync_info.json").write_text(json.dumps({"offset_s": 0.0}))
    structure = {"timeline": S.TIMELINE, "sync_offset_s": 0.0,
                 "coverage": {cam: {"spans": [[0, 50]]} for cam in ("cam1", "cam2")}}
    (game / "coverage.json").write_text(json.dumps(structure))
    (suite.parent / "key.json").write_text(json.dumps(key([clip(t=10.2)])))
    return suite, games, game, results


def test_whistle_source_recovers_weak_goal_and_groups_repeated_whistles(monkeypatch, tmp_path):
    suite, games, game, results = make_whistle_suite(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for root in (suite.parent, game) for p in root.rglob("*") if p.is_file()}
    baseline = R.run_suite(suite, games, results)
    assert baseline["games"][0]["metrics"]["goals_found"] == 0
    report = R.run_suite(suite, games, results, candidate_source="whistles")
    current = report["games"][0]
    assert current["status"] == "ok"
    assert current["metrics"]["goals_found"] == 1
    assert current["metrics"]["correct_clip_matches"] == 1
    assert current["metrics"]["incorrect_clip_matches"] == 0
    assert current["metrics"]["event_count"] == 1
    assert current["candidate_source"] == "whistles"
    assert current["evaluated_events_file"] == "whistle_events.csv"
    assert current["comparison"]["delta"]["goals_found"] == 1
    generated = results / report["run_id"] / "synthetic"
    answer = R.load_key(suite.parent / "key.json")
    assert R.evaluate(answer, R.read_events(generated / "events.csv"))["goals_found"] == 0
    assert R.evaluate(answer, R.read_events(generated / "whistle_events.csv"))["goals_found"] == 1
    assert all(p.read_bytes() == content for p, content in before.items())


def test_whistle_source_reports_zero_recall_when_valid_inputs_have_no_candidates(monkeypatch, tmp_path):
    suite, games, game, results = make_whistle_suite(tmp_path, monkeypatch)
    first = R.run_suite(suite, games, results, candidate_source="whistles")
    assert first["games"][0]["status"] == "ok"
    audio_path = game / "audio_signals.json"
    audio = json.loads(audio_path.read_text())
    audio["whistles"] = []
    audio["stoppages"] = []
    audio_path.write_text(json.dumps(audio))
    report = R.run_suite(suite, games, results, candidate_source="whistles")
    current = report["games"][0]
    assert current["status"] == "ok"
    assert current["metrics"]["goals_found"] == 0
    assert current["metrics"]["goal_recall"] == 0
    assert current["metrics"]["event_count"] == 0
    assert [c["id"] for c in current["metrics"]["missed_goals"]] == ["g1"]
    assert current["comparison"]["newly_missed_goals"] == ["g1"]
    assert R.previous_results(results)["synthetic"]["run_id"] == report["run_id"]
    generated = results / report["run_id"] / "synthetic"
    assert R.read_events(generated / "whistle_events.csv") == []
    assert json.loads((generated / "whistle_candidates.json").read_text())["candidates"] == []


def test_whistle_source_keeps_distinct_nearby_stoppages(monkeypatch, tmp_path):
    suite, games, game, results = make_whistle_suite(tmp_path, monkeypatch)
    (suite.parent / "key.json").write_text(json.dumps(key([clip("first", 10.2), clip("second", 16.2)])))
    rois = R.signals.load_rois(str(game / "rois.json"))
    cache_key = R.signals.signal_cache_key([str(game / "cam1.MP4")], 0.0, rois["camera_1"], 12, 1280, False)
    source = game / ".recap_cache" / "signals" / f"signals_{cache_key}.npz"
    with np.load(source) as z:
        arrays = {k: z[k].copy() for k in ("net", "slot", "audio")}
    arrays["net"][192:196] = 0.7
    arrays["slot"][192:196] = 0.7
    np.savez(source, **arrays)
    audio = json.loads((game / "audio_signals.json").read_text())
    audio["whistles"] = [{"t": 11.2, "role": "stoppage", "stoppage": 0},
                          {"t": 17.2, "role": "stoppage", "stoppage": 1}]
    audio["stoppages"] = [{"start": 11.2, "end": 12.2, "whistles": [11.2]},
                           {"start": 17.2, "end": 18.2, "whistles": [17.2]}]
    (game / "audio_signals.json").write_text(json.dumps(audio))
    report = R.run_suite(suite, games, results, candidate_source="whistles")
    assert report["games"][0]["metrics"]["goals_found"] == 2
    assert report["games"][0]["metrics"]["correct_clip_matches"] == 2
    assert report["games"][0]["metrics"]["event_count"] == 2


def test_whistle_source_reports_a_goal_missed_when_only_a_delayed_seed_exists(monkeypatch, tmp_path):
    suite, games, game, results = make_whistle_suite(tmp_path, monkeypatch)
    answer_path = suite.parent / "key.json"
    answer_before = answer_path.read_bytes()
    audio_path = game / "audio_signals.json"
    audio = json.loads(audio_path.read_text())
    audio["whistles"] = [{"t": 33.47, "role": "no_stoppage", "stoppage": None}]
    audio["stoppages"] = []
    audio_path.write_text(json.dumps(audio))
    missed = R.run_suite(suite, games, results, candidate_source="whistles")["games"][0]
    assert missed["status"] == "ok"
    assert missed["metrics"]["goals_found"] == 0
    assert [c["id"] for c in missed["metrics"]["missed_goals"]] == ["g1"]
    assert missed["metrics"]["event_count"] == 1
    assert missed["metrics"]["incorrect_clip_matches"] == 0
    assert missed["metrics"]["unjudged_events"] == 1
    # This control supplies a synthetic nearby whistle, not a generated goal label.
    audio["whistles"].append({"t": 11.2, "role": "no_stoppage", "stoppage": None})
    audio_path.write_text(json.dumps(audio))
    recovered = R.run_suite(suite, games, results, candidate_source="whistles")["games"][0]
    assert recovered["metrics"]["goals_found"] == 1
    assert recovered["metrics"]["correct_clip_matches"] == 1
    assert recovered["comparison"]["recovered_goals"] == ["g1"]
    assert answer_path.read_bytes() == answer_before


@pytest.mark.parametrize("bad_input", ["audio_timeline", "coverage_offset", "all_offsets", "missing_audio", "invalid_audio"])
def test_whistle_source_blocks_bad_cached_evidence(monkeypatch, tmp_path, bad_input):
    suite, games, game, results = make_whistle_suite(tmp_path, monkeypatch)
    first = R.run_suite(suite, games, results, candidate_source="whistles")
    assert first["games"][0]["status"] == "ok"
    if bad_input == "audio_timeline":
        path = game / "audio_signals.json"
        data = json.loads(path.read_text())
        data["timeline"] = "recap"
        path.write_text(json.dumps(data))
    elif bad_input == "coverage_offset":
        path = game / "coverage.json"
        data = json.loads(path.read_text())
        data["sync_offset_s"] = 9
        path.write_text(json.dumps(data))
    elif bad_input == "all_offsets":
        for name, field in (("audio_signals.json", "sync_offset_s"), ("coverage.json", "sync_offset_s"),
                            ("sync_info.json", "offset_s")):
            path = game / name
            data = json.loads(path.read_text())
            data[field] = 9
            path.write_text(json.dumps(data))
    else:
        for path in (game / ".recap_cache" / "audio").glob("*.npz"):
            if bad_input == "missing_audio":
                path.unlink()
            else:
                np.savez(path, flux=np.full(5000, np.nan, dtype=np.float32))
    report = R.run_suite(suite, games, results, candidate_source="whistles")
    assert report["games"][0]["status"] == "blocked"
    assert R.previous_results(results)["synthetic"]["run_id"] == first["run_id"]


def test_combined_source_recovers_flow_only_goal_and_keeps_original_pools(monkeypatch, tmp_path):
    suite, games, game, results = make_whistle_suite(tmp_path, monkeypatch)
    answer_path = suite.parent / "key.json"
    answer_path.write_text(json.dumps(key([clip("weak", 10.2), clip("flow_only", 25.2)])))
    before = {p: p.read_bytes() for root in (suite.parent, game) for p in root.rglob("*") if p.is_file()}
    report = R.run_suite(suite, games, results, candidate_source="combined",
                         detection_overrides={"preserve_peaks": True})
    current = report["games"][0]
    assert current["status"] == "ok"
    assert current["metrics"]["goals_found"] == 2
    assert current["metrics"]["correct_clip_matches"] == 2
    assert current["candidate_source"] == "combined"
    assert current["evaluated_events_file"] == "combined_events.csv"
    generated = results / report["run_id"] / "synthetic"
    answer = R.load_key(answer_path)
    assert R.evaluate(answer, R.read_events(generated / "events.csv"))["goals_found"] == 1
    assert R.evaluate(answer, R.read_events(generated / "whistle_events.csv"))["goals_found"] == 1
    assert (generated / "combined_candidates.json").exists()
    assert "combined_candidates.py" in current["candidate_input_hashes"]
    assert all(p.read_bytes() == content for p, content in before.items())


def test_combined_source_blocks_missing_audio_without_extracting(monkeypatch, tmp_path):
    suite, games, game, results = make_whistle_suite(tmp_path, monkeypatch)
    for path in (game / ".recap_cache" / "audio").glob("*.npz"):
        path.unlink()
    report = R.run_suite(suite, games, results, candidate_source="combined")
    assert report["games"][0]["status"] == "blocked"


def test_combined_source_reports_empty_valid_pools_as_zero_recall(monkeypatch, tmp_path):
    suite, games, game, results = make_whistle_suite(tmp_path, monkeypatch)
    for path in (game / ".recap_cache" / "signals").glob("*.npz"):
        zero = np.zeros(600, dtype=np.float32)
        np.savez(path, net=zero, slot=zero, audio=zero)
    audio_path = game / "audio_signals.json"
    audio = json.loads(audio_path.read_text())
    audio["whistles"] = []
    audio["stoppages"] = []
    audio_path.write_text(json.dumps(audio))
    report = R.run_suite(suite, games, results, candidate_source="combined")
    current = report["games"][0]
    assert current["status"] == "ok"
    assert current["metrics"]["goals_found"] == 0
    assert current["metrics"]["event_count"] == 0
    assert [c["id"] for c in current["metrics"]["missed_goals"]] == ["g1"]


@pytest.mark.parametrize("source", ["whistles", "combined"])
def test_cli_whistle_source_does_not_read_unfinished_game_sheet(monkeypatch, tmp_path, capsys, source):
    suite, games, game, results = make_whistle_suite(tmp_path, monkeypatch)
    sheet = game / "game_sheet.json"
    sheet.write_text("unfinished user work; not valid JSON")
    assert R.main(["--suite", str(suite), "--games-root", str(games), "--results", str(results),
                   "--candidate-source", source]) == 0
    current = R.previous_results(results)["synthetic"]
    assert current["candidate_source"] == source
    assert current["settings"]["audio_weight"] == 0
    assert current["metrics"]["correct_clip_matches"] == 1
    assert f"[{source}]" in capsys.readouterr().out
    assert sheet.read_text() == "unfinished user work; not valid JSON"
