import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3/scripts"))
import combined_candidates as C


def event(name, start, end, cam=1, score=1):
    return {"id": name, "start_s": start, "end_s": end,
            "primary_cam": cam, "score": score, "confidence": 0.5}


def test_combination_preserves_same_camera_coverage_without_mutating_inputs():
    flow = [event("f1", 10, 23), event("f2", 10.5, 23.5), event("f3", 70, 83)]
    whistles = [event("w1", 11, 21)]
    before = copy.deepcopy((flow, whistles))
    combined = C.combine_events(flow, whistles)
    assert len(combined) == 2
    assert (flow, whistles) == before
    for row in flow + whistles:
        assert any(e["primary_cam"] == row["primary_cam"]
                   and e["start_s"] <= row["start_s"] and e["end_s"] >= row["end_s"]
                   for e in combined)
    assert {p["event_id"] for p in combined[0]["sources"]} == {"f1", "f2", "w1"}


def test_combination_keeps_distinct_whistle_stoppages_even_when_windows_overlap():
    whistles = [event("w1", 10, 20), event("w2", 10.5, 20.5)]
    combined = C.combine_events([event("f1", 10, 23)], whistles)
    assert len(combined) == 2
    assert all(sum(p["source"] == "whistles" for p in e["sources"]) == 1 for e in combined)


def test_combination_removes_identical_whistle_clips_but_keeps_both_source_ids():
    whistles = [event("first", 10, 20, score=2), event("repeated", 10, 20)]
    combined = C.combine_events([], whistles)
    assert len(combined) == 1
    assert combined[0]["score"] == 2
    assert {p["event_id"] for p in combined[0]["sources"]} == {"first", "repeated"}
    assert C.combine_events([], list(reversed(whistles))) == combined


def test_identical_whistle_intervals_keep_different_cameras_and_do_not_mutate_inputs():
    whistles = [event("a", 10, 20), event("b", 10, 20, cam=2), event("c", 10, 20)]
    before = copy.deepcopy(whistles)
    combined = C.combine_events([], whistles)
    assert len(combined) == 2
    assert whistles == before
    assert {row["primary_cam"] for row in combined} == {1, 2}
    assert {p["event_id"] for row in combined for p in row["sources"]} == {"a", "b", "c"}


def test_identical_whistle_cuts_keep_provenance_and_original_pool_files(tmp_path):
    header = "start_s,end_s,score,primary_cam,confidence\n"
    paths = (tmp_path / "events.csv", tmp_path / "whistle_events.csv")
    paths[0].write_text(header)
    paths[1].write_text(header + "10,20,2,1,0.5\n10,20,1,1,0.4\n")
    before = {p.name: p.read_bytes() for p in paths}
    hashes = C.write_combined_events(tmp_path, max_window_s=20)
    report = json.loads((tmp_path / "combined_candidates.json").read_text())
    assert report["removed_count"] == 1
    assert report["settings"]["identical_whistle_clips"] is True
    assert report["candidates"][0]["sources"] == [
        {"source": "whistles", "event_id": "event:1"},
        {"source": "whistles", "event_id": "event:2"},
    ]
    assert len(C._read_pool(tmp_path / "combined_events.csv")) == 1
    for p in paths:
        assert p.read_bytes() == before[p.name]
        assert hashes[p.name] == hashlib.sha256(before[p.name]).hexdigest()


def test_combination_keeps_different_cameras_and_distinct_nearby_moments():
    flow = [event("a", 10, 23), event("b", 13, 26), event("c", 10, 23, cam=2)]
    assert len(C.combine_events(flow, [])) == 3


def test_combination_keeps_camera_handover_even_when_windows_overlap():
    flow = [event("earlier_attack", 10, 23, cam=1)]
    whistles = [event("later_attack", 16, 26, cam=2)]
    combined = C.combine_events(flow, whistles)
    assert len(combined) == 2
    assert {(row["primary_cam"], row["start_s"], row["end_s"]) for row in combined} == {
        (1, 10, 23), (2, 16, 26),
    }


def test_combination_does_not_chain_moment_groups():
    flow = [event("a", 10, 23, score=3), event("b", 10.8, 23.8, score=2),
            event("c", 11.6, 24.6)]
    assert len(C.combine_events(flow, [])) == 2


def test_combination_bounds_expanded_windows():
    flow = [event("a", 10, 30), event("b", 10.5, 30.5)]
    assert len(C.combine_events(flow, [])) == 2


def test_combination_is_deterministic_and_accepts_empty_pools():
    flow = [event("b", 10.5, 23.5), event("a", 10, 23)]
    assert C.combine_events(flow, []) == C.combine_events(list(reversed(flow)), [])
    assert C.combine_events([], []) == []


def test_combination_preserves_dense_interval_coverage():
    flow = [event(f"f{i}", i / 4, i / 4 + 13, cam=i % 2 + 1, score=i % 7)
            for i in range(100)]
    combined = C.combine_events(flow, [])
    assert len(combined) < len(flow)
    for row in flow:
        assert any(e["primary_cam"] == row["primary_cam"]
                   and e["start_s"] <= row["start_s"] and e["end_s"] >= row["end_s"]
                   for e in combined)
    assert all(e["end_s"] - e["start_s"] <= 20 for e in combined)


@pytest.mark.parametrize("field,value", [("score", float("nan")), ("end_s", float("inf")),
                                         ("start_s", -1), ("primary_cam", 3), ("end_s", 10),
                                         ("score", None), ("id", None)])
def test_combination_rejects_invalid_events(field, value):
    row = event("bad", 10, 23)
    row[field] = value
    with pytest.raises(ValueError):
        C.combine_events([row], [])
