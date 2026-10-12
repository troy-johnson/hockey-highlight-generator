"""Use synthetic motion to reject matches that share only the rink background."""
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3" / "scripts"))
from answer_key_matching import foreground, match_motion, align_background


def rink():
    rng = np.random.default_rng(12)
    background = rng.integers(110, 200, (180, 320), dtype=np.uint8)
    return cv2.GaussianBlur(background, (3, 3), 0)


def players(background, positions):
    result = background.copy()
    for x, y in positions:
        cv2.rectangle(result, (x, y), (x + 10, y + 18), 20, -1)
    return result


def test_alignment_recovers_crop_without_using_player_positions():
    background = rink()
    transform = np.array([[1.25, 0, -35], [0, 1.25, -25], [0, 0, 1]], dtype=float)
    edited = cv2.warpPerspective(background, transform, (320, 180))
    found = align_background(background, edited)
    assert found is not None
    corners = np.float32([[[100, 60], [200, 100]]])
    assert np.max(np.abs(cv2.perspectiveTransform(corners, found["matrix"])
                         - cv2.perspectiveTransform(corners, transform))) < 2


def test_foreground_rejects_static_background_and_tracks_players():
    background = rink()
    assert np.count_nonzero(foreground(background, background)) == 0
    current = players(background, [(50, 60), (180, 100)])
    assert np.linalg.norm(foreground(current, background)) > 0


def test_matching_finds_motion_not_shared_static_features():
    background = rink()
    frames = [players(background, [(40 + i * 20, 60), (240 - i * 10, 100)]) for i in range(8)]
    result = match_motion(frames[3], background, np.stack(frames), np.arange(8), background,
                          np.eye(3), min_margin=0.04)
    assert result["status"] == "matched"
    assert result["raw_s"] == 3
    assert result["score"] == pytest.approx(1, abs=1e-6)


def test_identical_motion_at_two_times_requires_review():
    background = rink()
    same = players(background, [(50, 60), (180, 100)])
    frames = np.stack([same, background, same])
    result = match_motion(same, background, frames, np.array([0, 5, 10]), background, np.eye(3))
    assert result["status"] == "ambiguous"


def test_empty_and_nonfinite_evidence_do_not_produce_a_reference():
    background = rink()
    result = match_motion(background, background, np.stack([background]), np.array([1.0]),
                          background, np.eye(3))
    assert result["status"] == "no_motion"
    with pytest.raises(ValueError):
        match_motion(background, background, np.stack([background]), np.array([np.nan]),
                     background, np.eye(3))
