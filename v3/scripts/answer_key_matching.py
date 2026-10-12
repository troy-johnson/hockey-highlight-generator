"""Compare published frames with raw motion. Results require label verification."""
from __future__ import annotations

import cv2
import numpy as np


def align_background(raw: np.ndarray, edited: np.ndarray) -> dict | None:
    """Estimate the raw-to-edit crop from static features, not camera timing."""
    sift = cv2.SIFT_create(nfeatures=2500)
    first, d1 = sift.detectAndCompute(raw, None)
    second, d2 = sift.detectAndCompute(edited, None)
    if d1 is None or d2 is None or len(d2) < 2:
        return None
    pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(good) < 12:
        return None
    src = np.float32([first[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([second[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return None
    count = int(mask.sum())
    if count < 12 or count / len(good) < 0.4 or abs(matrix[2, 2]) < 1e-9:
        return None
    matrix = matrix / matrix[2, 2]
    if np.linalg.det(matrix[:2, :2]) <= 0:
        return None
    return {"matrix": matrix, "inliers": count, "inlier_fraction": count / len(good)}


def foreground(image: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Remove the static rink and global brightness changes before comparison."""
    if image.ndim != 2 or image.shape != background.shape:
        raise ValueError("foreground requires equal grayscale image shapes")
    diff = background.astype(np.float32) - image.astype(np.float32)
    diff -= np.median(diff)
    return cv2.GaussianBlur(np.maximum(diff - 12.0, 0.0), (3, 3), 0)


def match_motion(edited: np.ndarray, edited_background: np.ndarray,
                 raw_frames: np.ndarray, times: np.ndarray, raw_background: np.ndarray,
                 transform: np.ndarray, *, min_score=0.55, min_margin=0.05,
                 separation_s=2.0) -> dict:
    """Return a motion match or an explicit review state. Never assign goal labels."""
    if (raw_frames.ndim != 3 or raw_frames.shape[1:] != raw_background.shape
            or len(raw_frames) != len(times) or not len(times)
            or not np.isfinite(times).all() or np.any(times < 0)
            or edited.shape != edited_background.shape
            or transform.shape != (3, 3) or not np.isfinite(transform).all()):
        raise ValueError("invalid motion comparison inputs")
    height, width = edited.shape
    valid = cv2.warpPerspective(np.ones(raw_background.shape, np.uint8), transform, (width, height)) > 0
    valid[:round(height * 0.15)] = False
    valid[round(height * 0.95):] = False
    query = foreground(edited, edited_background)
    query[~valid] = 0
    query_norm = np.linalg.norm(query)
    if query_norm < 10 or np.count_nonzero(query > 1) < 10:
        return {"status": "no_motion"}
    scores = np.zeros(len(times), np.float32)
    for i, image in enumerate(raw_frames):
        vector = cv2.warpPerspective(foreground(image, raw_background), transform, (width, height))
        vector[~valid] = 0
        norm = np.linalg.norm(vector)
        if norm > 0:
            scores[i] = np.sum(vector * query) / (norm * query_norm)
    best = int(np.argmax(scores))
    outside = np.abs(times - times[best]) > separation_s
    second = float(scores[outside].max()) if outside.any() else 0.0
    score = float(scores[best])
    margin = score - second
    status = "matched" if score >= min_score and margin >= min_margin else "ambiguous"
    return {"status": status, "raw_s": float(times[best]), "score": score,
            "margin": margin, "second_score": second}
