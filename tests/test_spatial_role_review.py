import copy
import json
import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3/scripts"))
import spatial_role_review as R


def detection(label, x=10, confidence=.9, box=None):
    return {"class": label, "confidence": confidence, "xyxy": box or [x, 10, x + 10, 30]}


def frame(t, *detections, available=True):
    return {"nominal_detection_s": t,
            "evidence": {"status": "observed" if available else "unavailable", "detections": list(detections)}}


def test_goalie_to_player_label_change_remains_ambiguous_not_a_confirmed_skater():
    result = R.review_roles([frame(10, detection("goalie")), frame(11, detection("player", x=11))])
    assert result["mixed_role_components"] == 1
    assert len(result["ambiguous_detection_ids"]) == 2
    assert result["can_filter"] is result["can_reassign_camera"] is False
    assert result["components"][0]["physical_identity_verified"] is False


def test_adjacent_skater_does_not_inherit_the_goalie_role():
    result = R.review_roles([
        frame(10, detection("goalie"), detection("player", x=50)),
        frame(11, detection("goalie", x=11), detection("player", x=51)),
    ])
    assert result["mixed_role_components"] == 0
    assert len(result["components"]) == 2


def test_large_gap_does_not_connect_role_hints():
    result = R.review_roles([frame(10, detection("goalie")), frame(20, detection("player"))])
    assert result["mixed_role_components"] == 0
    assert result["temporal_links"] == 0


def test_unavailable_sample_breaks_association_even_with_a_large_gap_setting():
    result = R.review_roles([frame(10, detection("goalie")), frame(11, available=False),
                            frame(12, detection("player"))], max_gap_s=5)
    assert result["mixed_role_components"] == 0
    assert result["unavailable_samples"] == 1


def test_overlapping_same_frame_roles_remain_uncertain_without_temporal_support():
    result = R.review_roles([frame(10, detection("goalie"), detection("player"))])
    assert result["mixed_role_components"] == 1
    assert result["temporal_status"] == "insufficient_samples"


def test_partial_occlusion_can_flag_role_ambiguity_but_not_confirm_identity():
    result = R.review_roles([frame(10, detection("goalie", box=[10, 10, 30, 50])),
                            frame(11, detection("player", box=[12, 10, 28, 25]))])
    assert result["mixed_role_components"] == 1
    assert result["components"][0]["physical_identity_verified"] is False


def test_deterministic_role_components_do_not_mutate_raw_observations():
    data = [frame(10, detection("goalie"), detection("player", x=50)),
            frame(11, detection("player", x=11), detection("player", x=51))]
    before = copy.deepcopy(data)
    reversed_boxes = copy.deepcopy(data)
    for f in reversed_boxes:
        f["evidence"]["detections"].reverse()
    assert R.review_roles(data) == R.review_roles(reversed_boxes)
    assert data == before


def test_low_confidence_role_hint_stays_explicit_and_never_authorizes_a_choice():
    result = R.review_roles([frame(10, detection("goalie", confidence=.18)), frame(11, detection("player"))])
    assert result["mixed_role_components"] == 1
    assert result["components"][0]["minimum_box_confidence"] == .18
    assert result["can_reassign_camera"] is False


def test_single_player_sample_has_insufficient_temporal_evidence_not_verified_role():
    result = R.review_roles([frame(10, detection("player"))])
    assert result["temporal_status"] == "insufficient_samples"
    assert result["components"][0]["physical_identity_verified"] is False
    assert result["can_filter"] is False


def test_persistent_wrong_player_label_cannot_be_certified_by_temporal_consistency():
    result = R.review_roles([frame(t, detection("player")) for t in range(10, 18)])
    assert result["temporal_links"] == 7
    assert result["mixed_role_components"] == 0
    assert result["components"][0]["role_verified"] is False
    assert result["can_filter"] is result["can_reassign_camera"] is False


@pytest.mark.parametrize("times", [[10, 10], [11, 10], [10, float("nan")]])
def test_invalid_time_order_is_rejected(times):
    with pytest.raises(ValueError):
        R.review_roles([frame(t, detection("player")) for t in times])


@pytest.mark.parametrize("gap", [0, -1, float("inf")])
def test_invalid_gap_is_rejected(gap):
    with pytest.raises(ValueError):
        R.review_roles([], max_gap_s=gap)


def role_inputs(tmp_path):
    game = tmp_path / "game"
    game.mkdir()
    raw = game / "raw.MP4"
    raw.write_bytes(b"synthetic; never decode")
    roi = game / "rois.json"
    roi.write_text(json.dumps({"camera_1": {"net": [0, 0, 90, 90], "slot": [0, 0, 90, 90]}}))
    cache = tmp_path / "frames.npz"
    np.savez(cache, frames=np.zeros((2, 100, 100, 3), dtype=np.uint8), times=np.array([10., 11.]),
             identity=json.dumps({"file": str(raw), "size": raw.stat().st_size,
                                  "mtime_ns": raw.stat().st_mtime_ns}))
    predictions = tmp_path / "results" / "predictions.json"
    values = iter([[detection("goalie")], [detection("player")]])
    R.P.analyze_cache(cache, roi, "camera_1", predictions, {}, lambda image: next(values))
    output = tmp_path / "results" / "roles.json"
    args = ["--observations", str(predictions), "--frame-cache", str(cache),
            "--rois", str(roi), "--output", str(output)]
    return args, predictions, cache, roi, raw, output


def test_role_cli_uses_existing_predictions_and_preserves_every_input(tmp_path):
    args, predictions, cache, roi, raw, output = role_inputs(tmp_path)
    before = {p: p.read_bytes() for p in (predictions, cache, roi, raw)}
    assert R.main(args) == 0
    result = json.loads(output.read_text())
    assert result["mixed_role_components"] == 1
    assert result["input_identity"]["observations_sha256"] == R.P.sha256(predictions)
    assert result["can_filter"] is result["can_reassign_camera"] is False
    assert all(p.read_bytes() == value for p, value in before.items())
    saved = output.read_bytes()
    assert R.main(args) == 2
    assert output.read_bytes() == saved


def test_role_cli_rejects_invalid_predictions_without_creating_role_output(tmp_path, capsys):
    args, predictions, cache, roi, raw, output = role_inputs(tmp_path)
    data = json.loads(predictions.read_text())
    data["observations"][0]["evidence"]["near_players"] = 50
    predictions.write_text(json.dumps(data))
    assert R.main(args) == 2
    assert "invalid cached spatial observation" in capsys.readouterr().err
    assert not output.exists()


def test_role_cli_refuses_game_folder_output(tmp_path):
    args, predictions, cache, roi, raw, output = role_inputs(tmp_path)
    args[-1] = str(roi.parent / "roles.json")
    assert R.main(args) == 2
    assert not (roi.parent / "roles.json").exists()
