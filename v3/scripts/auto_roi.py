# v3/scripts/auto_roi.py
"""
Automatic net and slot ROIs from the goal frame (hhg-3hk.3, spec 002 §5.4).

The cameras are fixed, so a median of frames spread over the first Recording
is a clean background with the net plainly visible. HockeyAI (YOLOv8) finds
the goal frame on it (100% of frames on the Ghost Pirates test cameras), and
the net and slot ROIs are derived from the goal box by geometry fitted to the
hand-drawn ROIs. No per-rink setup and no interactive picker.

Writes rois.json (the format roi_picker.py writes), rois_auto.json (goal box
and confidence per camera) and rois_preview.png. Exit code 0 on success;
non-zero when a camera has no goal box or the model is unavailable, so
run_detect.sh falls back to the interactive picker.

HockeyAI needs the optional ML stack (requirements-ml.txt). Its weights are
MIT; the Ultralytics runtime is AGPL-3.0, which is fine for local use.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

WIDTH, HEIGHT = 1280, 720          # default analysis size; the real height follows the source aspect ratio
N_BACKGROUND_FRAMES = 15
MIN_CONFIDENCE = 0.5               # below this the ROIs are written but flagged for review
WEIGHTS_URL = "https://huggingface.co/SimulaMet-HOST/HockeyAI/resolve/main/HockeyAI_model_weight.pt"
WEIGHTS_PATH = Path(os.environ.get(
    "HHG_HOCKEYAI_WEIGHTS",
    Path.home() / ".cache" / "hockey-highlight-generator" / "HockeyAI_model_weight.pt"))

# ROI geometry relative to the goal box (x0, y0, x1, y1), with w, h its size.
# Fitted to the hand-drawn Ghost Pirates ROIs: the net ROI covers the goal
# mouth and the ice just in front of it; the slot ROI covers the area beyond.
_NET = (-0.16, 0.33, 1.32, 1.00)    # (x offset, y offset, width, height) in units of w, h
_SLOT = (-0.16, -0.17, 1.32, 0.58)


def rois_from_goal_box(box: tuple[float, float, float, float], width: int = WIDTH, height: int = HEIGHT) -> dict:
    """Net and slot ROIs [x, y, w, h] (ints, clamped to the frame) from a goal box."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    out = {}
    for key, (dx, dy, fw, fh) in (("net", _NET), ("slot", _SLOT)):
        x, y = x0 + dx * bw, y0 + dy * bh
        xa, ya = max(0, round(x)), max(0, round(y))
        xb, yb = min(width, round(x + fw * bw)), min(height, round(y + fh * bh))
        out[key] = [xa, ya, max(1, xb - xa), max(1, yb - ya)]
    return out


def pick_goal_box(dets: list[tuple[str, float, tuple]]) -> tuple[tuple, float] | None:
    """The goal box to use: highest confidence x area among 'goal' detections."""
    goals = [(c, b) for cls, c, b in dets if cls == "goal"]
    if not goals:
        return None
    c, b = max(goals, key=lambda g: g[0] * (g[1][2] - g[1][0]) * (g[1][3] - g[1][1]))
    return b, c


def _source_dims(path: str) -> tuple[int, int]:
    """(width, height) of the first video stream. JSON output: ffprobe 9 also lists
    stream groups, which made the plain-text output span several lines."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "json", path], capture_output=True, text=True).stdout
    stream = json.loads(out)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def analysis_size(path: str, width: int = WIDTH) -> tuple[int, int]:
    """The frame size detection analyses (same rule as signals._ffmpeg_gray_frames):
    `width` wide, height from the source aspect ratio, rounded to an even number."""
    ow, oh = _source_dims(path)
    h = int(round(oh * (width / ow)))
    return width, h + (h % 2)


def _duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    return float(out) if out else 0.0


def median_background(paths: list[str], width: int = WIDTH) -> np.ndarray:
    """Median grayscale frame, at the analysis size detection uses, over frames spread across the Recording."""
    if not paths:
        raise ValueError("No chapter files for this camera")
    width, height = analysis_size(paths[0], width)
    durations = [_duration(p) for p in paths]
    total = sum(durations)
    frames = []
    for t in np.linspace(0.05, 0.95, N_BACKGROUND_FRAMES) * total:
        for p, d in zip(paths, durations):
            if t <= d:
                break
            t -= d
        raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t:.1f}", "-i", p, "-frames:v", "1",
                              "-vf", f"scale={width}:{height},format=gray", "-f", "rawvideo", "pipe:1"],
                             capture_output=True).stdout
        if len(raw) == width * height:
            frames.append(np.frombuffer(raw, np.uint8).reshape(height, width))
    if not frames:
        raise RuntimeError(f"No frames decoded from {paths[0]}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO  # optional ML stack (requirements-ml.txt)
        if not WEIGHTS_PATH.exists():
            # Download to a temporary name and rename only after the weights load,
            # so an interrupted download never leaves a broken cache behind.
            WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            # The temporary name must end in '.pt': Ultralytics loads a checkpoint only from a '.pt' path.
            part = WEIGHTS_PATH.with_name(WEIGHTS_PATH.stem + ".part.pt")
            print(f"[auto_roi] Downloading HockeyAI weights to {WEIGHTS_PATH}", flush=True)
            try:
                urllib.request.urlretrieve(WEIGHTS_URL, part)
                YOLO(str(part))                     # validate before it becomes the cached copy
            except BaseException:
                part.unlink(missing_ok=True)
                raise
            part.replace(WEIGHTS_PATH)
            _MODEL = YOLO(str(WEIGHTS_PATH))
        else:
            _MODEL = YOLO(str(WEIGHTS_PATH))
    return _MODEL


def detect_goal_box(image: np.ndarray) -> tuple[tuple, float] | None:
    """Goal box (x0, y0, x1, y1) in image pixels and its confidence, or None."""
    import cv2
    model = _model()
    r = model.predict(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), imgsz=1280, conf=0.15, verbose=False)[0]
    dets = [(model.names[int(b.cls)], float(b.conf), tuple(float(v) for v in b.xyxy[0].tolist())) for b in r.boxes]
    return pick_goal_box(dets)


def _preview(backgrounds: dict, rois: dict, out: Path) -> None:
    import cv2
    tiles = []
    for cam, key in (("cam1", "camera_1"), ("cam2", "camera_2")):
        im = cv2.cvtColor(backgrounds[cam], cv2.COLOR_GRAY2BGR)
        for name, col in (("net", (0, 0, 255)), ("slot", (0, 255, 0))):
            x, y, w, h = rois[key][name]
            cv2.rectangle(im, (x, y), (x + w, y + h), col, 3)
        tiles.append(cv2.resize(im, (640, round(640 * im.shape[0] / im.shape[1]))))
    height = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    cv2.imwrite(str(out), cv2.hconcat(tiles))


def main(game_folder: str) -> int:
    root = Path(game_folder)
    chapters = json.loads((root / "chapters.json").read_text())
    rois, info, backgrounds = {}, {}, {}
    for cam, key in (("cam1", "camera_1"), ("cam2", "camera_2")):
        first_recording = (chapters.get("recordings", {}).get(cam) or [chapters[cam]])[0]
        try:
            backgrounds[cam] = median_background(first_recording)
            found = detect_goal_box(backgrounds[cam])
        except Exception as exc:  # no ML stack, no network, unreadable file
            print(f"[auto_roi] {cam}: automatic ROIs unavailable ({exc})", flush=True)
            return 2
        if found is None:
            print(f"[auto_roi] {cam}: no goal frame found on the background", flush=True)
            return 3
        box, conf = found
        h, w = backgrounds[cam].shape
        rois[key] = rois_from_goal_box(box, w, h)
        info[key] = {"goal_box": [round(v, 1) for v in box], "confidence": round(conf, 2),
                     "flag": conf < MIN_CONFIDENCE}
        if conf < MIN_CONFIDENCE:
            print(f"[WARN] {cam}: goal frame confidence {conf:.2f}; check rois_preview.png", flush=True)
    (root / "rois.json").write_text(json.dumps(rois, indent=2))
    (root / "rois_auto.json").write_text(json.dumps(info, indent=2))
    _preview(backgrounds, rois, root / "rois_preview.png")
    print(f"[auto_roi] rois.json written (goal confidence cam1 {info['camera_1']['confidence']}, "
          f"cam2 {info['camera_2']['confidence']})", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: auto_roi.py <game_folder>")
    sys.exit(main(sys.argv[1]))
