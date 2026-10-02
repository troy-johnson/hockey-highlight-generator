"""Read Selection inputs and existing caches. Never extract signals."""
from __future__ import annotations

import json
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
