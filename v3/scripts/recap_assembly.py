"""Plan and render a plain, silent Recap from selected Game Sheet goals."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v2" / "scripts"))

FPS = 30
MAX_FRAMES = 240 * FPS


def output_name(root: Path, options: dict, sheet: dict) -> str:
    from recap_options import parse_folder_name
    _, inferred_date = parse_folder_name(root.name)
    date = options.get("date") or sheet.get("date") or inferred_date
    if not date or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(date)):
        raise ValueError("Recap needs a date in YYYY-MM-DD format")
    teams = sheet.get("teams") or {}
    home, away = teams.get("home"), teams.get("away")
    if not isinstance(home, str) or not isinstance(away, str) or not home.strip() or not away.strip():
        raise ValueError("Recap needs both Game Sheet team names")
    focus = options.get("focus_team")
    if focus and focus.casefold() == away.casefold():
        home, away = away, home
    def safe(value):
        return re.sub(r"[^\w.-]+", "-", value.strip()).strip(".-") or "Team"
    return f"{date}_{safe(home)}-vs-{safe(away)}_Recap.mp4"


def source_parts(layout: list[dict], start: float, end: float) -> list[dict]:
    parts = []
    cursor = start
    for block in sorted(layout, key=lambda b: b["start"]):
        file_start = block["start"] - block["seek"]
        for path, duration in block["files"]:
            a, b = max(cursor, file_start, block["start"]), min(end, file_start + duration, block["end"])
            if b > a:
                if a - cursor > .002:
                    raise ValueError("clip crosses a Recording gap")
                parts.append({"file": str(path), "seek_s": a - file_start, "duration_s": b - a})
                cursor = b
            file_start += duration
    if end - cursor > .002 or not parts:
        raise ValueError("clip has incomplete chapter coverage")
    return parts


def plan_recap(selection: dict, layouts: dict, speed: float = 1.1) -> dict:
    if not math.isfinite(speed) or not .25 <= speed <= 4:
        raise ValueError("live_play_speed must be between 0.25 and 4")
    goals = selection.get("goals") or []
    before, after = (6., 3.) if len(goals) <= 6 else (5., 2.) if len(goals) <= 12 else (4., 0.)
    clips, flags = [], []
    for goal in goals:
        chosen = goal.get("chosen")
        if not chosen:
            flags.append(f"{goal['id']}: no clip found; omitted from plain Recap")
            continue
        cam, moment = chosen["primary_cam"], float(chosen["detection_s"])
        # Selection locates action, not puck crossing. Keep three seconds of follow-through.
        start = max(float(chosen["start_s"]), moment - before)
        end = min(float(chosen["end_s"]), moment + 3.0 + after)
        if not all(math.isfinite(t) for t in (start, end, moment)) or not start <= moment < end:
            raise ValueError(f"{goal['id']}: invalid selected clip bounds")
        clips.append({"goal_id": goal["id"], "camera": cam, "moment_s": moment,
                      "start_s": start, "end_s": end})
    clips.sort(key=lambda c: c["moment_s"])
    if not clips:
        raise ValueError("no selected goals to render")
    frames = sum(math.ceil((c["end_s"] - c["start_s"]) / speed * FPS) for c in clips)
    if frames > MAX_FRAMES:
        # Keep every chosen goal. Reduce context equally instead of dropping goals.
        if len(clips) > MAX_FRAMES // 2:
            raise ValueError("too many goals for the four-minute cap")
        ratio = (MAX_FRAMES - len(clips)) / frames
        for c in clips:
            c["start_s"] = c["moment_s"] - (c["moment_s"] - c["start_s"]) * ratio
            c["end_s"] = c["moment_s"] + (c["end_s"] - c["moment_s"]) * ratio
        flags.append("goal context shortened to fit the four-minute cap")
    for c in clips:
        c["frames"] = math.ceil((c["end_s"] - c["start_s"]) / speed * FPS)
        c["parts"] = source_parts(layouts.get(c["camera"], []), c["start_s"], c["end_s"])
    return {"schema_version": 1, "speed": speed, "fps": FPS, "audio": "silent",
            "duration_s": sum(c["frames"] for c in clips) / FPS, "clips": clips,
            "input_flags": selection.get("flags", []),
            "flags": flags + ["plain Recap: no graphics, replays, or audio mix"]}


def load_layouts(root: Path, cameras: set[str]) -> dict:
    import coverage as C
    from signals import _manifest_inputs
    layouts = {}
    for cam in sorted(cameras):
        if not re.fullmatch(r"cam\d+", cam):
            raise ValueError("invalid selected camera")
        lines = (root / f"{cam}_concat.txt").read_text().splitlines()
        paths, _ = _manifest_inputs(lines, str(root))
        durations = {p: C._duration(p) for p in paths}
        if any(not math.isfinite(d) or d <= 0 for d in durations.values()):
            raise ValueError(f"{cam}: chapter duration unavailable")
        layouts[cam] = C.manifest_layout(lines, durations, str(root))
    return layouts


def render(plan: dict, output: Path) -> None:
    """Render to local scratch space and replace the output only after validation."""
    with tempfile.TemporaryDirectory(prefix="hockey-recap-") as scratch:
        work = Path(scratch)
        clips = []
        for i, clip in enumerate(plan["clips"]):
            argv = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
            filters = []
            for j, part in enumerate(clip["parts"]):
                argv += ["-ss", str(part["seek_s"]), "-t", str(part["duration_s"]), "-i", part["file"]]
                filters.append(f"[{j}:v:0]setpts=PTS-STARTPTS,scale=1920:1080:force_original_aspect_ratio=decrease,"
                               f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS}[v{j}]")
            joined = "".join(f"[v{j}]" for j in range(len(clip["parts"])))
            filters.append(f"{joined}concat=n={len(clip['parts'])}:v=1:a=0,setpts=PTS/{plan['speed']},"
                           f"fps={FPS},tpad=stop_mode=clone:stop_duration=1[out]")
            dest = work / f"clip-{i:04d}.mp4"
            argv += ["-filter_complex", ";".join(filters), "-map", "[out]", "-an", "-frames:v", str(clip["frames"]),
                     "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", str(dest)]
            subprocess.run(argv, check=True)
            clips.append(dest)
            print(f"[assembly] rendered {i + 1}/{len(plan['clips'])}", flush=True)
        manifest = work / "clips.txt"
        manifest.write_text("".join(f"file '{p.name}'\n" for p in clips))
        assembled = work / "recap.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                        "-i", str(manifest), "-c", "copy", "-movflags", "+faststart", str(assembled)], check=True)
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_format", "-show_streams",
                                                   "-of", "json", str(assembled)]))
        duration = float(probe["format"]["duration"])
        if not 0 < duration <= 240 or abs(duration - plan["duration_s"]) > .1:
            raise ValueError(f"unexpected Recap duration: {duration}")
        import shutil
        temporary = output.with_suffix(".mp4.tmp")
        shutil.copyfile(assembled, temporary)
        temporary.replace(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game_folder", type=Path)
    parser.add_argument("--options", default="{}")
    args = parser.parse_args(argv)
    root = args.game_folder.resolve()
    try:
        options = json.loads(args.options)
        selection = json.loads((root / "selection.json").read_text())
        sheet = json.loads((root / "game_sheet.json").read_text())
        name = output_name(root, options, sheet)
        cameras = {g["chosen"]["primary_cam"] for g in selection["goals"] if g.get("chosen")}
        plan = plan_recap(selection, load_layouts(root, cameras), float(options.get("live_play_speed", 1.1)))
        plan["output"] = name
        render(plan, root / name)
        temporary = root / "recap_assembly.json.tmp"
        temporary.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
        temporary.replace(root / "recap_assembly.json")
        for flag in plan["flags"]:
            print(f"[assembly] flag: {flag}")
        print(f"[assembly] {name}: {plan['duration_s']:.2f}s, {len(plan['clips'])} goals")
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(f"[ERROR] assembly: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
