"""Evaluate existing whistle candidates without extracting signals or reading a Game Sheet."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from zipfile import BadZipFile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import selection as S
import selection_inputs as I


def write_whistle_events(game: Path, output: Path, settings: dict, timeline_offset: float) -> dict:
    flags = []
    audio = I.read_json(game / "audio_signals.json", flags)
    structure = I.read_json(game / "coverage.json", flags)
    sync = I.read_json(game / "sync_info.json", flags)
    offset = sync.get("offset_s")
    if not I.finite_number(offset):
        raise ValueError("whistle candidates require a finite sync offset")
    if not math.isclose(offset, timeline_offset, abs_tol=0.001):
        raise ValueError("whistle candidate sync offset does not match the answer-key manifests")
    for name, value in (("audio_signals.json", audio), ("coverage.json", structure)):
        if (value.get("timeline") != S.TIMELINE
                or not I.finite_number(value.get("sync_offset_s"))
                or not math.isclose(value["sync_offset_s"], offset, abs_tol=0.001)):
            raise ValueError(f"{name}: whistle candidate timeline does not match sync_info.json")
    audio = I.checked_audio(audio, flags)
    structure = I.checked_structure(structure, flags)
    flows, activities = {}, {}
    for cam in ("cam1", "cam2"):
        try:
            flows[cam], activities[cam], _ = I.cached_camera(
                game, cam, settings["fps"], settings["width"], settings["audio_weight"] > 0, flags)
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
            raise ValueError(f"{cam}: cached whistle inputs invalid ({exc})") from exc
        if not np.isfinite(activities[cam]).any():
            flags.append(f"{cam}: valid cached audio activity unavailable")
        if not structure["coverage"].get(cam, {}).get("spans"):
            flags.append(f"{cam}: coverage unavailable")
    if flags:
        raise ValueError("whistle candidate inputs unavailable: " + "; ".join(flags))
    candidates = S.build_candidates(audio, [], flows, activities, structure["coverage"], settings["fps"], flags)
    rows = []
    for candidate in candidates:
        ranked = sorted(((S.weighted_score(e["features"]), cam, e)
                         for cam, e in candidate["cameras"].items()), key=lambda item: (item[0], item[1]),
                        reverse=True)
        score, cam, evidence = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        confidence = abs(score - second) / (score + second + 1e-6)
        moment = evidence["t"]
        rows.append((max(0.0, moment - S.SETTINGS["pre_roll_s"]),
                     moment + S.SETTINGS["post_roll_s"], score, int(cam[3:]), confidence))
        candidate["evaluated_camera"] = cam
        candidate["evaluated_score"] = score
        candidate["evaluated_confidence"] = confidence
    with (output / "whistle_events.csv").open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_s", "end_s", "score", "primary_cam", "confidence"])
        for start, end, score, cam, confidence in sorted(rows, key=lambda row: -row[2]):
            writer.writerow([f"{start:.3f}", f"{end:.3f}", f"{score:.4f}", cam, f"{confidence:.4f}"])
    with (output / "whistle_candidates.json").open("x") as f:
        json.dump({"candidates": candidates, "flags": flags}, f, indent=2, allow_nan=False)
        f.write("\n")
    hashes = {name: hashlib.sha256((game / name).read_bytes()).hexdigest()
              for name in ("audio_signals.json", "coverage.json", "sync_info.json")}
    hashes.update({Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                   for path in (__file__, S.__file__, I.__file__)})
    return hashes
