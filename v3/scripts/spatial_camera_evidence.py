"""Spatial player observations only. Missing detections never authorize filtering."""
from __future__ import annotations

import math

MIN_CONFIDENCE = 0.25


def spatial_evidence(detections: list[dict] | None, rois: dict,
                     frame_size: tuple[int, int], min_confidence: float = MIN_CONFIDENCE) -> dict:
    """Locate box-bottom centers in existing ROIs without classifying an attack."""
    width, height = frame_size
    if (not all(math.isfinite(v) and v > 0 for v in frame_size)
            or not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1):
        raise ValueError("invalid spatial frame size or confidence")
    regions = {}
    for name in ("net", "slot"):
        try:
            x, y, w, h = (float(v) for v in rois[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid spatial ROI") from exc
        if (not all(math.isfinite(v) for v in (x, y, w, h)) or x < 0 or y < 0
                or w <= 0 or h <= 0 or x + w > width or y + h > height):
            raise ValueError("spatial ROI does not fit the source frame")
        regions[name] = (x, y, x + w, y + h)
    result = {"status": "unavailable" if detections is None else "observed",
              "observation": "unavailable", "near_players": 0, "outside_roi_players": 0,
              "near_goalies": 0, "near_referees": 0, "low_confidence_near_players": 0,
              "detections": [], "absence_verified": False, "can_filter": False,
              "min_confidence": min_confidence}
    if detections is None:
        return result
    for detection in detections:
        try:
            label = detection["class"]
            confidence = float(detection["confidence"])
            x0, y0, x1, y1 = (float(v) for v in detection["xyxy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid spatial detection") from exc
        if (not isinstance(label, str) or not label
                or not all(math.isfinite(v) for v in (confidence, x0, y0, x1, y1))
                or not 0 <= confidence <= 1 or x1 <= x0 or y1 <= y0
                or x0 < 0 or y0 < 0 or x1 > width or y1 > height):
            raise ValueError("invalid spatial detection box or confidence")
        bottom_x, bottom_y = (x0 + x1) / 2, y1
        near = [name for name, (xa, ya, xb, yb) in regions.items()
                if xa <= bottom_x < xb and ya <= bottom_y < yb]
        confident = confidence >= min_confidence
        result["detections"].append({"class": label, "confidence": confidence,
                                     "xyxy": [x0, y0, x1, y1], "bottom_center_xy": [bottom_x, bottom_y],
                                     "near_regions": near, "confident": confident})
        if label == "player":
            if confident:
                result["near_players" if near else "outside_roi_players"] += 1
            elif near:
                result["low_confidence_near_players"] += 1
        elif confident and near and label in ("goalie", "referee"):
            result["near_goalies" if label == "goalie" else "near_referees"] += 1
    if result["near_players"]:
        result["observation"] = "near_player_detected"
    elif result["near_goalies"]:
        result["observation"] = "goalie_without_detected_near_players"
    else:
        result["observation"] = "no_near_player_detected"
    return result


def paired_evidence(nominal_detection_s: float, cam1: dict, cam2: dict) -> dict:
    """Keep camera observations at this sample time; do not infer a switch or cut."""
    if not math.isfinite(nominal_detection_s) or nominal_detection_s < 0:
        raise ValueError("invalid spatial sample time")
    cameras = {"cam1": cam1, "cam2": cam2}
    return {"nominal_detection_s": nominal_detection_s,
            "status": "observed" if all(c["status"] == "observed" for c in cameras.values()) else "incomplete",
            "positive_player_cameras": [name for name, c in cameras.items()
                                        if c["status"] == "observed" and c["near_players"] > 0],
            "cameras": cameras, "can_reassign_camera": False}
