# v3/scripts/discover.py
from __future__ import annotations

import json
import sys
from pathlib import Path


def discover(game_folder: str) -> dict[str, list[str]]:
    """
    Scan game_folder/cam1/ and game_folder/cam2/ for MP4 chapter files.
    Returns {"cam1": [...sorted paths...], "cam2": [...sorted paths...]}.
    Writes chapters.json to game_folder as a side effect.
    Raises SystemExit with a descriptive message on any validation failure.
    """
    root = Path(game_folder)
    result: dict[str, list[str]] = {}

    if not (root / "cam1").is_dir() and not (root / "cam2").is_dir():
        sys.exit("[ERROR] Both cam1/ and cam2/ are required")

    for cam in ("cam1", "cam2"):
        subfolder = root / cam
        if not subfolder.is_dir():
            sys.exit(f"[ERROR] {cam}/ subfolder not found in {game_folder}")
        chapters = sorted(
            str(p) for p in subfolder.iterdir()
            if p.suffix.upper() == ".MP4"
        )
        if not chapters:
            sys.exit(f"[ERROR] No .MP4 files found in {cam}/")
        result[cam] = chapters

    (root / "chapters.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: discover.py <game_folder>")
    chapters = discover(sys.argv[1])
    print(f"[discover] cam1: {len(chapters['cam1'])} chapters, "
          f"cam2: {len(chapters['cam2'])} chapters")
    print(f"[discover] Wrote chapters.json")
