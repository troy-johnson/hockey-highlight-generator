import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3/scripts"))
import spatial_camera_evidence as S


ROIS = {"net": [40, 40, 20, 20], "slot": [40, 20, 20, 20]}


def box(label, x, foot_y, confidence=0.8):
    return {"class": label, "confidence": confidence, "xyxy": [x - 2, foot_y - 8, x + 2, foot_y]}


def test_player_box_bottom_distinguishes_inside_and_outside_roi_positions():
    result = S.spatial_evidence([box("player", 50, 50), box("player", 50, 15)], ROIS, (100, 100))
    assert result["near_players"] == 1
    assert result["outside_roi_players"] == 1
    assert result["observation"] == "near_player_detected"
    assert result["can_filter"] is False


def test_goalie_and_referee_do_not_count_as_nearby_skaters():
    result = S.spatial_evidence([box("goalie", 50, 50), box("referee", 50, 45),
                                 box("player", 50, 15)], ROIS, (100, 100))
    assert result["near_players"] == 0
    assert result["near_goalies"] == 1
    assert result["observation"] == "goalie_without_detected_near_players"
    assert result["can_filter"] is False


def test_missing_model_output_is_not_an_empty_net_label():
    missing = S.spatial_evidence(None, ROIS, (100, 100))
    empty = S.spatial_evidence([], ROIS, (100, 100))
    assert missing["status"] == "unavailable"
    assert empty["status"] == "observed"
    assert empty["observation"] == "no_near_player_detected"
    assert empty["absence_verified"] is False
    assert missing["can_filter"] is empty["can_filter"] is False


def test_low_confidence_player_remains_explicit_uncertain_evidence():
    result = S.spatial_evidence([box("player", 50, 50, 0.2)], ROIS, (100, 100))
    assert result["near_players"] == 0
    assert result["low_confidence_near_players"] == 1
    assert result["absence_verified"] is False


def test_a_goalie_mislabeled_as_player_never_authorizes_camera_reassignment():
    # The predictor can assign the player class to a goalie. Geometry cannot correct that class.
    mislabeled = S.spatial_evidence([box("player", 50, 50, 0.9)], ROIS, (100, 100))
    other = S.spatial_evidence([], ROIS, (100, 100))
    paired = S.paired_evidence(10, mislabeled, other)
    assert mislabeled["near_players"] == 1
    assert mislabeled["can_filter"] is False
    assert paired["can_reassign_camera"] is False


def test_half_open_regions_do_not_include_the_upper_box_bottom_boundary():
    result = S.spatial_evidence([box("player", 60, 50), box("player", 50, 60)], ROIS, (100, 100))
    assert result["near_players"] == 0
    assert result["outside_roi_players"] == 2


def test_overlapping_rois_do_not_count_one_player_twice():
    result = S.spatial_evidence([box("player", 50, 40)],
                                {"net": [40, 30, 20, 30], "slot": [40, 20, 20, 30]}, (100, 100))
    assert result["near_players"] == 1


def test_player_just_outside_roi_is_not_proof_of_distant_or_absent_action():
    result = S.spatial_evidence([box("player", 65, 50)], ROIS, (100, 100))
    assert result["near_players"] == 0
    assert result["outside_roi_players"] == 1
    assert result["absence_verified"] is False
    assert result["can_filter"] is False
    assert "distant_players" not in result


def test_camera_handover_preserves_both_time_specific_observations():
    near = S.spatial_evidence([box("player", 50, 50)], ROIS, (100, 100))
    goalie = S.spatial_evidence([box("goalie", 50, 50)], ROIS, (100, 100))
    earlier = S.paired_evidence(10, near, goalie)
    later = S.paired_evidence(15, goalie, near)
    assert earlier["positive_player_cameras"] == ["cam1"]
    assert later["positive_player_cameras"] == ["cam2"]
    assert earlier["can_reassign_camera"] is later["can_reassign_camera"] is False
    assert "switch_s" not in earlier and "switch_s" not in later


def test_both_cameras_and_missing_camera_never_silently_authorize_a_choice():
    near = S.spatial_evidence([box("player", 50, 50)], ROIS, (100, 100))
    both = S.paired_evidence(10, near, near)
    missing = S.paired_evidence(10, near, S.spatial_evidence(None, ROIS, (100, 100)))
    assert both["positive_player_cameras"] == ["cam1", "cam2"]
    assert missing["status"] == "incomplete"
    assert missing["can_reassign_camera"] is False


@pytest.mark.parametrize("field,value", [("confidence", float("nan")), ("confidence", 2),
                                         ("xyxy", [1, 2, 1, 4]), ("xyxy", [1, 2, 3, float("inf")])])
def test_invalid_model_box_is_rejected(field, value):
    invalid = box("player", 50, 50)
    invalid[field] = value
    with pytest.raises(ValueError):
        S.spatial_evidence([invalid], ROIS, (100, 100))


def test_wrong_coordinate_scale_is_rejected():
    with pytest.raises(ValueError, match="ROI"):
        S.spatial_evidence([], ROIS, (20, 20))
