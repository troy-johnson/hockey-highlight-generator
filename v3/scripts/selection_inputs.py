"""Read Selection inputs and existing caches. Never extract signals."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def read_json(path: Path, flags: list[str]) -> dict:
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        return value
    except (OSError, ValueError) as exc:
        flags.append(f"{path.name}: missing or unreadable ({exc})")
        return {}


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def input_list(value, name: str, flags: list[str]) -> list:
    if isinstance(value, list):
        return value
    flags.append(f"{name}: invalid list")
    return []


def checked_flags(value, name: str, flags: list[str]) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        flags.append(f"{name}: invalid flags list")
        value = [value]
    return [str(v) for v in value]


def checked_sheet(sheet: dict, flags: list[str]) -> dict:
    result = dict(sheet)
    result["flags"] = checked_flags(sheet.get("flags"), "scoresheet", flags)
    if not isinstance(sheet.get("teams", {}), dict):
        flags.append("scoresheet teams: invalid object")
        result["teams"] = {}
    goals = sheet.get("goals", {})
    if not isinstance(goals, dict):
        flags.append("scoresheet goals: invalid object; team unavailable")
        # Preserve claims without guessing which team scored them.
        result["goals"] = {"unknown": goals if isinstance(goals, list) else [goals]}
    return result


def checked_audio(audio: dict, flags: list[str]) -> dict:
    """Keep indices stable so stoppage references cannot move to another row."""
    result = dict(audio)
    for name, keys in (("whistles", ("t",)), ("stoppages", ("start", "end"))):
        rows = input_list(audio.get(name, []), f"audio {name}", flags)
        result[name] = []
        for i, row in enumerate(rows):
            valid = isinstance(row, dict) and all(finite_number(row.get(k)) for k in keys)
            if valid and name == "stoppages":
                valid = row["end"] >= row["start"]
            if not valid:
                flags.append(f"audio {name}[{i}]: invalid time; ignored")
            result[name].append(row if valid else {})
    return result


def checked_structure(structure: dict, flags: list[str]) -> dict:
    """Reject unreadable boundaries instead of assigning them invented times."""
    result = dict(structure)
    result["flags"] = checked_flags(structure.get("flags"), "coverage", flags)
    if not isinstance(structure.get("league_timing", {}), dict):
        flags.append("coverage league_timing: invalid object")
        result["league_timing"] = {}
    game = structure.get("game", {})
    if not isinstance(game, dict):
        flags.append("coverage game: invalid object")
        game = {"start_uncertain": True, "end_uncertain": True}
    result["game"] = game
    periods = []
    for row in input_list(structure.get("periods", []), "coverage periods", flags):
        if (not isinstance(row, dict) or type(row.get("n")) is not int or row["n"] < 1
                or not all(finite_number(row.get(k)) for k in ("start", "end"))
                or row["end"] <= row["start"]):
            flags.append("coverage period: invalid number or bounds")
            continue
        row = dict(row)
        if row.get("break_before") is not None and not isinstance(row["break_before"], dict):
            flags.append(f"coverage period {row['n']}: invalid break")
            row["break_before"] = {"uncertain": True}
        periods.append(row)
    result["periods"] = periods
    cameras = structure.get("coverage", {})
    if not isinstance(cameras, dict):
        flags.append("coverage cameras: invalid object")
        cameras = {}
    result["coverage"] = {}
    for cam, value in cameras.items():
        if not isinstance(value, dict):
            flags.append(f"{cam}: invalid coverage")
            continue
        spans = []
        for span in input_list(value.get("spans", []), f"{cam} spans", flags):
            if (isinstance(span, list) and len(span) == 2 and all(finite_number(v) for v in span)
                    and span[1] > span[0]):
                spans.append(span)
            else:
                flags.append(f"{cam}: invalid coverage span")
        camera_periods = []
        for p in input_list(value.get("periods", []), f"{cam} periods", flags):
            if (isinstance(p, dict) and type(p.get("n")) is int and p["n"] > 0
                    and p.get("defender") in ("A", "B", None)):
                camera_periods.append(p)
            else:
                flags.append(f"{cam}: invalid period or defender")
        result["coverage"][cam] = dict(value, spans=spans, periods=camera_periods)
    return result


def place(parts: list[tuple[float, np.ndarray]], rate: float) -> np.ndarray:
    """Place Recording arrays on the detection timeline; leave gaps as NaN."""
    n = max((round(max(0, t) * rate) + len(a) for t, a in parts), default=0)
    out = np.full(n, np.nan, np.float32)
    for t, a in parts:
        start = round(max(0, t) * rate)
        out[start:start + len(a)] = a
    return out


def cached_camera(root: Path, cam: str, fps: int, width: int, flow_audio: bool,
                  flags: list[str]) -> tuple[dict, np.ndarray, list[dict]]:
    import audio_signals as A
    import coverage as C
    from signals import _manifest_inputs, _recording_blocks, load_rois, signal_cache_key

    manifest = root / f"{cam}_concat.txt"
    try:
        lines = manifest.read_text().splitlines()
        blocks = _recording_blocks(str(manifest))
        groups = [(0.0, lines)] if len(blocks) < 2 else [
            (start, ["ffconcat version 1.0"] + ([f"# seek {-start:.3f}"] if start < 0 else []) + fl)
            for start, fl in blocks]
        paths, _ = _manifest_inputs(lines, str(root))
        durations = {p: C._duration(p) for p in paths}
        layout = C.manifest_layout(lines, durations, str(root))
        if any(d <= 0 for d in durations.values()):
            flags.append(f"{cam}: chapter duration unavailable")
    except (OSError, ValueError) as exc:
        flags.append(f"{cam}: manifest unavailable ({exc})")
        return {}, np.zeros(0, np.float32), []
    try:
        rois = load_rois(str(root / "rois.json"))[f"camera_{cam[3:]}"]
    except (OSError, ValueError, KeyError):
        rois = None
        flags.append(f"{cam}: ROIs unavailable; no cached flow")
    parts = {"net": [], "slot": [], "flux": []}
    music_id = (read_json(root / "music_spans.json", []).get("model") or {}).get("sha256")
    for start, ls in groups:
        files, seek = _manifest_inputs(ls, str(root))
        if rois is not None:
            try:
                key = signal_cache_key(files, seek, rois, fps, width, flow_audio)
                with np.load(root / ".recap_cache" / "signals" / f"signals_{key}.npz") as z:
                    net, slot = z["net"].copy(), z["slot"].copy()
                    if net.ndim != 1 or slot.shape != net.shape:
                        raise ValueError("invalid flow arrays")
                    parts["net"].append((start, net))
                    parts["slot"].append((start, slot))
            except (OSError, ValueError, KeyError) as exc:
                flags.append(f"{cam}: flow cache unavailable at {max(start, 0):.3f}s ({exc})")
        loaded = False
        for identity in dict.fromkeys((music_id, A.YAMNET_SHA256, None)):
            try:
                key = A._feature_key(files, seek, identity)
                with np.load(root / ".recap_cache" / "audio" / f"audio_{key}.npz") as z:
                    flux = z["flux"].copy()
                    if flux.ndim != 1:
                        raise ValueError("invalid flux array")
                    parts["flux"].append((start, flux))
                loaded = True
                break
            except (OSError, ValueError, KeyError):
                continue
        if not loaded:
            flags.append(f"{cam}: audio cache unavailable at {max(start, 0):.3f}s")
    flow = {k: place(parts[k], fps) for k in ("net", "slot")}
    activity = A.activity_rank(place(parts["flux"], A.RATE))
    return flow, activity, layout
