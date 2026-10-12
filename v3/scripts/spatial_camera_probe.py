"""Opt-in spatial diagnostics from source-bound frame caches; never edit candidates."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from zipfile import BadZipFile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from spatial_camera_evidence import spatial_evidence, MIN_CONFIDENCE
import spatial_camera_evidence


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze_cache(cache: Path, roi_file: Path, camera: str, output: Path,
                  producer: dict, predict, *, stride: int = 1, scope: str = "evaluation_only") -> dict:
    """Bind observations to raw identity, frame bytes, model settings, and ROI geometry."""
    if stride < 1 or not isinstance(stride, int) or camera not in ("camera_1", "camera_2"):
        raise ValueError("invalid spatial stride or camera")
    if scope not in ("evaluation_only", "source_selected"):
        raise ValueError("invalid spatial diagnostic scope")
    with np.load(cache, allow_pickle=False) as data:
        frames, times = data["frames"], data["times"]
        identity = json.loads(str(data["identity"].item()))
    source = Path(identity["file"])
    if (output.resolve() == cache.resolve()
            or any(parent.resolve() == output.resolve() or parent.resolve() in output.resolve().parents
                   for parent in (source.parent, roi_file.parent))):
        raise ValueError("spatial output must be separate from raw sources and Game Folder inputs")
    stat = source.stat()
    if stat.st_size != identity["size"] or stat.st_mtime_ns != identity["mtime_ns"]:
        raise ValueError("spatial frame cache raw source identity changed")
    if (frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8
            or times.ndim != 1 or not len(times) or len(times) != len(frames)
            or not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("invalid spatial frame cache")
    rois = json.loads(roi_file.read_text())[camera]
    frame_size = (frames.shape[2], frames.shape[1])
    spatial_evidence(None, rois, frame_size)
    input_identity = {"frame_cache_sha256": sha256(cache), "source": identity,
                      "roi_sha256": sha256(roi_file), "camera": camera,
                      "frame_size": list(frame_size), "producer": producer,
                      "stride": stride, "scope": scope, "min_confidence": MIN_CONFIDENCE,
                      "code_sha256": {p.name: sha256(p) for p in
                                      (Path(__file__), Path(spatial_camera_evidence.__file__))}}
    selected = list(range(0, len(times), stride))
    if output.exists():
        result = json.loads(output.read_text())
        if (result["input_identity"] != input_identity or len(result["observations"]) != len(selected)
                or result["status"] != "diagnostic_only" or result["can_filter"] is not False
                or result["can_reassign_camera"] is not False):
            raise ValueError("spatial output identity changed; use a new output path")
        for observation, index in zip(result["observations"], selected):
            if (observation["nominal_detection_s"] != float(times[index])
                    or observation["evidence"] != spatial_evidence(
                        None if observation["evidence"]["status"] == "unavailable"
                        else observation["evidence"]["detections"], rois, frame_size)):
                raise ValueError("invalid cached spatial observation")
        return result
    started = time.perf_counter()
    observations = [{"nominal_detection_s": float(times[index]),
                     "evidence": spatial_evidence(predict(frames[index]), rois, frame_size)}
                    for index in selected]
    result = {"schema_version": 1, "status": "diagnostic_only", "input_identity": input_identity,
              "observations": observations, "runtime_s": time.perf_counter() - started,
              "can_filter": False, "can_reassign_camera": False,
              "limits": ["A detected player is not an attack, goal, or Play Type label.",
                         "The model can label a goalie as a player.",
                         "A box-bottom center is not a verified foot or ice-contact point.",
                         "A player outside the ROI can still be beside the net.",
                         "No detected player does not establish absence.",
                         "Source-selected scope does not establish whole-game or held-out precision."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-cache", type=Path, required=True)
    parser.add_argument("--rois", type=Path, required=True)
    parser.add_argument("--camera", choices=("camera_1", "camera_2"), required=True)
    parser.add_argument("--weights", type=Path, required=True, help="Existing local HockeyAI .pt file; never downloaded")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--scope", choices=("evaluation_only", "source_selected"), default="evaluation_only")
    args = parser.parse_args(argv)
    if not args.weights.is_file() or args.weights.suffix != ".pt":
        parser.error("spatial model unavailable; provide existing local .pt weights")
    try:
        os.environ["YOLO_AUTOINSTALL"] = "false"
        from ultralytics import YOLO
        model = None
        producer = {"weights_sha256": sha256(args.weights), "ultralytics": importlib.metadata.version("ultralytics"),
                    "torch": importlib.metadata.version("torch"), "imgsz": 1280,
                    "prediction_confidence": 0.15, "device": "cpu", "augment": False}

        def predict(frame):
            nonlocal model
            if model is None:
                model = YOLO(str(args.weights))
                if not {"player", "goalie", "referee"}.issubset(set(model.names.values())):
                    raise ValueError("spatial model does not distinguish players, goalies, and referees")
            prediction = model.predict(frame, imgsz=1280, conf=0.15, device="cpu", augment=False,
                                       save=False, verbose=False)[0]
            return [{"class": model.names[int(box.cls)], "confidence": float(box.conf),
                     "xyxy": [float(v) for v in box.xyxy[0].tolist()]} for box in prediction.boxes]

        result = analyze_cache(args.frame_cache, args.rois, args.camera, args.output, producer, predict,
                               stride=args.stride, scope=args.scope)
    except (ImportError, OSError, ValueError, KeyError, EOFError, RuntimeError, BadZipFile) as exc:
        print(f"spatial evidence unavailable: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "sample_count": len(result["observations"]),
                      "can_filter": False, "can_reassign_camera": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
