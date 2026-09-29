# v3/scripts/discover.py
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recordings import group_recordings, is_black_recording, read_camera_serial  # noqa: E402


def _mp4s(folder: Path) -> list[str]:
    return sorted(str(p) for p in folder.iterdir() if p.is_file() and p.suffix.upper() == ".MP4")


def _cameras_from_folders(root: Path, game_folder: str) -> dict[str, list[str]]:
    cams: dict[str, list[str]] = {}
    for cam in ("cam1", "cam2"):
        subfolder = root / cam
        if not subfolder.is_dir():
            raise ValueError(f"[ERROR] {cam}/ subfolder not found in {game_folder}")
        files = _mp4s(subfolder)
        if not files:
            raise ValueError(f"[ERROR] No .MP4 files found in {cam}/")
        cams[cam] = files
    return cams


def _cameras_from_flat_folder(root: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    files = _mp4s(root)
    if not files:
        raise ValueError("[ERROR] No .MP4 files found: use cam1/ and cam2/ subfolders or a flat folder of GoPro files")
    by_serial: dict[str, list[str]] = {}
    for p in files:
        by_serial.setdefault(read_camera_serial(p) or "unknown", []).append(p)
    if len(by_serial) != 2:
        raise ValueError(f"[ERROR] Expected 2 cameras in the Game Folder, found {len(by_serial)} "
                         f"(camera serials: {sorted(by_serial)})")
    serials = sorted(by_serial)
    return ({"cam1": by_serial[serials[0]], "cam2": by_serial[serials[1]]},
            {"cam1": serials[0], "cam2": serials[1]})


def discover(game_folder: str) -> dict:
    """
    Find the two cameras' chapter files, group them into Recordings, and drop
    black Recordings (lens covered).

    Layouts: cam1/ and cam2/ subfolders, or a flat folder where cameras are
    told apart by the GoPro serial in each file (hhg-38a.13).

    Returns and writes chapters.json:
      cam1 / cam2      kept chapter paths in order (used by existing tools)
      recordings       {cam: [[chapter paths of one Recording], ...]} (kept only)
      serials          {cam: camera serial or None}
      excluded         [{"path": first chapter, "reason": "black"}]
    Raises ValueError with a descriptive message on any validation failure.
    """
    root = Path(game_folder)
    if not root.is_dir():
        raise ValueError(f"[ERROR] Game folder not found: {game_folder}")

    if (root / "cam1").is_dir() or (root / "cam2").is_dir():
        cams = _cameras_from_folders(root, game_folder)
        serials = {cam: None for cam in cams}
    else:
        cams, serials = _cameras_from_flat_folder(root)

    result: dict = {"cam1": [], "cam2": [], "recordings": {}, "serials": serials, "excluded": []}
    for cam, files in cams.items():
        kept = []
        for rec in group_recordings(files):
            if is_black_recording(rec):
                result["excluded"].append({"path": rec[0], "reason": "black"})
            else:
                kept.append(rec)
        if not kept:
            raise ValueError(f"[ERROR] {cam} has no usable Recording (all black)")
        result["recordings"][cam] = kept
        result[cam] = [p for rec in kept for p in rec]

    (root / "chapters.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: discover.py <game_folder>")
    try:
        chapters = discover(sys.argv[1])
        for cam in ("cam1", "cam2"):
            recs = chapters["recordings"][cam]
            print(f"[discover] {cam}: {len(chapters[cam])} chapters in {len(recs)} Recording(s)"
                  + (f", serial {chapters['serials'][cam]}" if chapters["serials"][cam] else ""))
        for e in chapters["excluded"]:
            print(f"[WARN] Skipped {e['reason']} Recording starting at {e['path']}")
        print("[discover] Wrote chapters.json")
    except ValueError as e:
        sys.exit(str(e))
