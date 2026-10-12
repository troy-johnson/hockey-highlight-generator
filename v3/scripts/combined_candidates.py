"""Combine cached candidate pools without reference labels or camera reassignment."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

SETTINGS = {"same_moment_s": 1.0, "minimum_overlap": 0.5, "identical_whistle_clips": True}


def _checked(row: dict, source: str) -> dict:
    try:
        item = {k: float(row[k]) for k in ("start_s", "end_s", "score", "confidence")}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid combined candidate fields") from exc
    if (not all(math.isfinite(v) for v in item.values())
            or item["start_s"] < 0 or item["end_s"] <= item["start_s"]
            or row.get("primary_cam") not in (1, 2) or not isinstance(row.get("id"), str)):
        raise ValueError("invalid combined candidate interval, camera, or score")
    item.update(primary_cam=row["primary_cam"],
                moment_s=(item["start_s"] + item["end_s"]) / 2,
                sources=[{"source": source, "event_id": row["id"]}])
    return item


def combine_events(flow: list[dict], whistles: list[dict], max_window_s: float = 20.0) -> list[dict]:
    """Keep camera coverage and distinct whistle clips; combine identical source cuts."""
    if not math.isfinite(max_window_s) or max_window_s <= 0:
        raise ValueError("combined candidate maximum window must be positive and finite")
    order = lambda e: (-e["score"], e["start_s"], e["end_s"], e["primary_cam"], e["sources"][0]["event_id"])
    kept = []
    identical = {}
    for candidate in sorted((_checked(row, "whistles") for row in whistles), key=order):
        identity = (candidate["primary_cam"], candidate["start_s"], candidate["end_s"])
        if identity in identical:
            identical[identity]["sources"].extend(candidate["sources"])
        else:
            identical[identity] = candidate
            kept.append(candidate)
    for candidate in sorted((_checked(row, "flow") for row in flow), key=order):
        choices = []
        for index, existing in enumerate(kept):
            if (existing["primary_cam"] != candidate["primary_cam"]
                    or abs(existing["moment_s"] - candidate["moment_s"]) > SETTINGS["same_moment_s"]):
                continue
            start = min(existing["start_s"], candidate["start_s"])
            end = max(existing["end_s"], candidate["end_s"])
            overlap = min(existing["end_s"], candidate["end_s"]) - max(existing["start_s"], candidate["start_s"])
            shorter = min(existing["end_s"] - existing["start_s"], candidate["end_s"] - candidate["start_s"])
            contained = existing["start_s"] <= candidate["start_s"] and candidate["end_s"] <= existing["end_s"]
            if overlap / shorter < SETTINGS["minimum_overlap"] or (not contained and end - start > max_window_s):
                continue
            expansion = end - start - (existing["end_s"] - existing["start_s"])
            choices.append((expansion, index, start, end))
        if not choices:
            kept.append(candidate)
            continue
        _, index, start, end = min(choices)
        existing = kept[index]
        existing.update(start_s=start, end_s=end)
        existing["sources"].extend(candidate["sources"])
        # Keep the original moment so nearby candidates cannot form an unbounded chain.
    return kept


def _read_pool(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return [{**row, "id": f"event:{i}", "primary_cam": int(row["primary_cam"])}
                for i, row in enumerate(csv.DictReader(stream), 1)]


def write_combined_events(output: Path, max_window_s: float) -> dict:
    flow = _read_pool(output / "events.csv")
    whistles = _read_pool(output / "whistle_events.csv")
    candidates = combine_events(flow, whistles, max_window_s)
    with (output / "combined_events.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["start_s", "end_s", "score", "primary_cam", "confidence"])
        for row in candidates:
            writer.writerow([f"{row['start_s']:.3f}", f"{row['end_s']:.3f}", f"{row['score']:.4f}",
                             row["primary_cam"], f"{row['confidence']:.4f}"])
    with (output / "combined_candidates.json").open("x") as stream:
        json.dump({"settings": {**SETTINGS, "max_window_s": max_window_s},
                   "flow_count": len(flow), "whistle_count": len(whistles),
                   "removed_count": len(flow) + len(whistles) - len(candidates),
                   "candidates": candidates}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    paths = [Path(__file__), output / "events.csv", output / "whistle_events.csv"]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
