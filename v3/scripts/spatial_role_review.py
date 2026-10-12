"""Flag conflicting model roles in associated boxes, not confirmed actor identities."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(__file__))
import spatial_camera_probe as P

SETTINGS = {"max_gap_s": 1.5, "minimum_iou": 0.3, "minimum_containment": 0.8,
            "minimum_confidence": 0.15}


def _overlap(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return overlap / (area_a + area_b - overlap), overlap / min(area_a, area_b)


def review_roles(observations: list[dict], *, max_gap_s: float = SETTINGS["max_gap_s"]) -> dict:
    """Keep role conflicts explicit across nearby samples and overlapping boxes."""
    if not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("invalid spatial role maximum gap")
    nodes, parents, previous = [], [], []
    previous_time = None
    temporal_links = same_frame_links = available = unavailable = 0

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def join(a, b):
        ra, rb = root(a), root(b)
        parents[max(ra, rb)] = min(ra, rb)

    for frame_index, observation in enumerate(observations):
        t = float(observation["nominal_detection_s"])
        if not math.isfinite(t) or t < 0 or (previous_time is not None and t <= previous_time):
            raise ValueError("spatial role samples must have finite increasing times")
        evidence = observation["evidence"]
        if evidence["status"] != "observed":
            previous = []
            previous_time = t
            unavailable += 1
            continue
        available += 1
        boxes = []
        for detection in evidence["detections"]:
            label = detection["class"]
            confidence = float(detection["confidence"])
            box = [float(v) for v in detection["xyxy"]]
            if (len(box) != 4 or not all(math.isfinite(v) for v in box + [confidence])
                    or not 0 <= confidence <= 1 or box[2] <= box[0] or box[3] <= box[1]):
                raise ValueError("invalid spatial role box")
            if label in ("player", "goalie") and confidence >= SETTINGS["minimum_confidence"]:
                boxes.append({"class": label, "confidence": confidence, "xyxy": box})
        boxes.sort(key=lambda d: (*d["xyxy"], d["class"], -d["confidence"]))
        current = []
        for index, box in enumerate(boxes):
            node = len(nodes)
            current.append(node)
            parents.append(node)
            nodes.append({"id": f"sample:{frame_index}:box:{index}", "sample_index": frame_index,
                          "nominal_detection_s": t, **box})
        for offset, a in enumerate(current):
            for b in current[offset + 1:]:
                if nodes[a]["class"] == nodes[b]["class"]:
                    continue
                iou, containment = _overlap(nodes[a]["xyxy"], nodes[b]["xyxy"])
                if iou >= 0.5 or containment >= SETTINGS["minimum_containment"]:
                    join(a, b)
                    same_frame_links += 1
        if previous_time is not None and t - previous_time <= max_gap_s:
            choices = []
            for a in previous:
                for b in current:
                    iou, containment = _overlap(nodes[a]["xyxy"], nodes[b]["xyxy"])
                    if iou >= SETTINGS["minimum_iou"] or containment >= SETTINGS["minimum_containment"]:
                        choices.append((-max(iou, .9 * containment), a, b))
            used_previous, used_current = set(), set()
            for _, a, b in sorted(choices):
                if a in used_previous or b in used_current:
                    continue
                join(a, b)
                used_previous.add(a)
                used_current.add(b)
                temporal_links += 1
        previous, previous_time = current, t
    grouped = {}
    for i, node in enumerate(nodes):
        grouped.setdefault(root(i), []).append(node)
    components = []
    ambiguous = []
    for members in grouped.values():
        roles = sorted({m["class"] for m in members})
        mixed = roles == ["goalie", "player"]
        if mixed:
            ambiguous.extend(m["id"] for m in members)
        components.append({"id": members[0]["id"], "model_roles": roles, "mixed_role": mixed,
                           "nominal_span_s": [members[0]["nominal_detection_s"], members[-1]["nominal_detection_s"]],
                           "minimum_box_confidence": min(m["confidence"] for m in members),
                           "physical_identity_verified": False, "role_verified": False,
                           "model_boxes": members})
    return {"status": "diagnostic_only", "settings": {**SETTINGS, "max_gap_s": max_gap_s},
            "temporal_status": "insufficient_samples" if available < 2 else
                               "associations_observed" if temporal_links else "no_association",
            "available_samples": available, "unavailable_samples": unavailable,
            "temporal_links": temporal_links, "same_frame_links": same_frame_links,
            "mixed_role_components": sum(c["mixed_role"] for c in components),
            "ambiguous_detection_ids": ambiguous, "components": components,
            "can_filter": False, "can_reassign_camera": False,
            "limits": ["Box associations are not confirmed actor identities or goalkeeper tracks.",
                       "Crossing or overlapping players can create a mixed-role component.",
                       "No mixed role does not establish a correct role.",
                       "Missing samples break associations; no role is inferred across that gap."]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path, required=True)
    parser.add_argument("--rois", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.observations.read_text())
        identity = data["input_identity"]
        source = Path(identity["source"]["file"])
        if (args.output.resolve() in (args.observations.resolve(), args.frame_cache.resolve())
                or any(p.resolve() == args.output.resolve() or p.resolve() in args.output.resolve().parents
                       for p in (source.parent, args.rois.parent))):
            raise ValueError("role output must remain separate from inputs and Game Folders")

        def forbidden(frame):
            raise ValueError("role review requires existing valid predictions; it never runs inference")

        validated = P.analyze_cache(args.frame_cache, args.rois, identity["camera"], args.observations,
                                    identity["producer"], forbidden, stride=identity["stride"], scope=identity["scope"])
        result = review_roles(validated["observations"])
        result["input_identity"] = {"observations_sha256": P.sha256(args.observations),
                                     "source_prediction_identity": identity, "code_sha256": P.sha256(Path(__file__))}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, KeyError, EOFError, P.BadZipFile) as exc:
        print(f"spatial role review unavailable: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "mixed_role_components": result["mixed_role_components"],
                      "temporal_status": result["temporal_status"], "can_filter": False,
                      "can_reassign_camera": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
