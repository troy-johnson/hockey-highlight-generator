# v3/scripts/discover.py
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recordings import group_recordings, is_black_recording, read_camera_serial  # noqa: E402

# Old pipeline or editor outputs that can sit next to the camera files
# (for example Resolve renders named cam1.mp4). They are not camera footage.
_OLD_OUTPUT_VIDEO = re.compile(r"^(cam\d+|recap.*|highlights?.*|.*_overlay)\.mp4$", re.IGNORECASE)


def is_old_output_video(name: str) -> bool:
    return bool(_OLD_OUTPUT_VIDEO.match(name))


def _mp4s(folder: Path) -> list[str]:
    return sorted(str(p) for p in folder.iterdir()
                  if p.is_file() and p.suffix.upper() == ".MP4"
                  and not p.name.startswith(".") and not is_old_output_video(p.name))


def _cameras_from_folders(root: Path, game_folder: str, strict: bool) -> dict[str, list[str]]:
    cams: dict[str, list[str]] = {}
    for cam in ("cam1", "cam2"):
        subfolder = root / cam
        if not subfolder.is_dir():
            if strict:
                raise ValueError(f"[ERROR] {cam}/ subfolder not found in {game_folder}")
            continue
        files = _mp4s(subfolder)
        if not files:
            if strict:
                raise ValueError(f"[ERROR] No .MP4 files found in {cam}/")
            continue
        cams[cam] = files
    return cams


def _size(paths: list[str]) -> int:
    return sum(os.path.getsize(p) for p in paths)


def _cameras_from_flat_folder(root: Path, strict: bool, excluded: list[dict]
                              ) -> tuple[dict[str, list[str]], dict[str, str]]:
    files = _mp4s(root)
    if not files:
        raise ValueError("[ERROR] No .MP4 files found: use cam1/ and cam2/ subfolders or a flat folder of GoPro files")
    by_serial: dict[str, list[str]] = {}
    for p in files:
        by_serial.setdefault(read_camera_serial(p) or "unknown", []).append(p)
    # Files without a camera serial next to real GoPro files are not camera footage.
    if "unknown" in by_serial and len(by_serial) > 1:
        for p in by_serial.pop("unknown"):
            excluded.append({"path": p, "reason": "no camera serial", "cam": None})
    if not strict:
        if len(by_serial) > 2:
            keep = sorted(by_serial, key=lambda s: _size(by_serial[s]), reverse=True)[:2]
            for s in [s for s in by_serial if s not in keep]:
                for p in by_serial.pop(s):
                    excluded.append({"path": p, "reason": f"extra camera {s}", "cam": None})
    elif len(by_serial) != 2:
        raise ValueError(f"[ERROR] Expected 2 cameras in the Game Folder, found {len(by_serial)} "
                         f"(camera serials: {sorted(by_serial)})")
    serials = sorted(by_serial)
    cams = {f"cam{i + 1}": by_serial[s] for i, s in enumerate(serials)}
    return cams, {f"cam{i + 1}": s for i, s in enumerate(serials)}


def discover(game_folder: str, strict: bool = True) -> dict:
    """
    Find the two cameras' chapter files, group them into Recordings, and drop
    black Recordings (lens covered).

    Layouts: cam1/ and cam2/ subfolders, or a flat folder where cameras are
    told apart by the GoPro serial in each file (hhg-38a.13).

    strict=True (default, hockeydetect): exactly two usable cameras or ValueError.
    strict=False (hockeyrecap): keep what is usable; a missing or all-black
    camera goes into "missing" and "flags". ValueError only when no camera is usable.

    Returns and writes chapters.json:
      cam1 / cam2      kept chapter paths in order (used by existing tools)
      recordings       {cam: [[chapter paths of one Recording], ...]} (kept only)
      serials          {cam: camera serial or None}
      excluded         [{"path": first chapter, "reason": "black", "cam": cam}]
      missing / flags  cameras with no usable Recording, and why (strict=False)
    """
    root = Path(game_folder)
    if not root.is_dir():
        raise ValueError(f"[ERROR] Game folder not found: {game_folder}")

    excluded: list[dict] = []
    if (root / "cam1").is_dir() or (root / "cam2").is_dir():
        cams = _cameras_from_folders(root, game_folder, strict)
        serials = {cam: None for cam in cams}
    else:
        cams, serials = _cameras_from_flat_folder(root, strict, excluded)

    result: dict = {"cam1": [], "cam2": [], "recordings": {}, "serials": serials,
                    "excluded": excluded, "missing": [], "flags": []}
    for cam in ("cam1", "cam2"):
        if cam not in cams:
            result["missing"].append(cam)
            result["flags"].append(f"{cam}: no footage found")
    for cam, files in cams.items():
        kept = []
        for rec in group_recordings(files):
            if is_black_recording(rec):
                excluded.append({"path": rec[0], "reason": "black", "cam": cam})
            else:
                kept.append(rec)
        if not kept:
            if strict:
                raise ValueError(f"[ERROR] {cam} has no usable Recording (all black)")
            result["missing"].append(cam)
            result["flags"].append(f"{cam}: no usable Recording (all black, lens covered?)")
            continue
        result["recordings"][cam] = kept
        result[cam] = [p for rec in kept for p in rec]

    if not result["recordings"]:
        raise ValueError("[ERROR] No usable camera: " + "; ".join(result["flags"]))
    (root / "chapters.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--allow-missing-camera"]
    if len(args) != 1:
        sys.exit("Usage: discover.py <game_folder> [--allow-missing-camera]")
    try:
        chapters = discover(args[0], strict="--allow-missing-camera" not in sys.argv)
        for cam, recs in chapters["recordings"].items():
            print(f"[discover] {cam}: {len(chapters[cam])} chapters in {len(recs)} Recording(s)"
                  + (f", serial {chapters['serials'][cam]}" if chapters["serials"].get(cam) else ""))
        for e in chapters["excluded"]:
            print(f"[WARN] Skipped {e['reason']} file or Recording starting at {e['path']}")
        for f in chapters["flags"]:
            print(f"[FLAG] {f}")
        print("[discover] Wrote chapters.json")
    except ValueError as e:
        sys.exit(str(e))
