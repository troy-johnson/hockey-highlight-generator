# v3/scripts/gopro_meta.py
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def timecode_to_seconds(tc: str, fps: int = 60) -> float:
    """Convert HH:MM:SS:FF timecode string to float seconds."""
    hh, mm, ss, ff = (int(x) for x in tc.split(":"))
    return hh * 3600.0 + mm * 60.0 + ss + ff / fps


def _creation_time_to_seconds(iso_str: str) -> float:
    """Parse ISO 8601 creation_time to seconds-since-midnight (UTC).

    Note: discards the date component. Cross-midnight captures would produce
    an incorrect offset, but hockey games do not span midnight in practice.
    """
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return dt.hour * 3600.0 + dt.minute * 60.0 + dt.second + dt.microsecond / 1_000_000


def _parse_fps(streams: list[dict]) -> int:
    """Extract frame rate from ffprobe stream list; defaults to 60 if unavailable."""
    for stream in streams:
        frac = stream.get("r_frame_rate", "")
        if "/" in frac:
            try:
                num, den = frac.split("/")
                return round(int(num) / int(den))
            except (ValueError, ZeroDivisionError):
                pass
    return 60


def extract_chapter_time(chapter_path: str) -> tuple[float, str, str | None, float]:
    """
    Extract start time and duration from a chapter file using ffprobe.

    Returns (seconds_since_midnight, source, raw_timecode_or_None, duration_s).
    source is "timecode" or "creation_time".
    raw_timecode is the original HH:MM:SS:FF string when source == "timecode", else None.
    Raises ValueError if no usable metadata is found.
    """
    raw = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream_tags=timecode",
            "-show_entries", "stream=r_frame_rate",
            "-show_entries", "format_tags=timecode,creation_time",
            "-show_entries", "format=duration",
            "-of", "json",
            chapter_path,
        ]
    )
    data = json.loads(raw)

    streams = data.get("streams", [])
    fps = _parse_fps(streams)
    duration_s = float(data.get("format", {}).get("duration") or 0.0)

    tc = (
        data.get("format", {}).get("tags", {}).get("timecode")
        or next(
            (s.get("tags", {}).get("timecode")
             for s in streams
             if s.get("tags", {}).get("timecode")),
            None,
        )
    )
    if tc:
        try:
            return timecode_to_seconds(tc, fps), "timecode", tc, duration_s
        except Exception:
            print(
                f"[WARN] Timecode '{tc}' in {chapter_path} is unparseable — "
                "falling back to creation_time",
                flush=True,
            )

    ct = data.get("format", {}).get("tags", {}).get("creation_time")
    if ct:
        try:
            return _creation_time_to_seconds(ct), "creation_time", None, duration_s
        except Exception:
            pass

    raise ValueError(f"[ERROR] No usable timecode or creation_time metadata in {chapter_path}")


def _check_chapter_continuity(
    chapter_metas: list[tuple[float, str, str | None, float]],
    cam_name: str,
) -> list[str]:
    """
    Check for timestamp gaps or overlaps > 5s between consecutive chapters.
    Returns warning strings; also prints each one.
    """
    warnings: list[str] = []
    for i in range(1, len(chapter_metas)):
        prev_start, _, _, prev_dur = chapter_metas[i - 1]
        curr_start, _, _, _ = chapter_metas[i]
        if prev_dur <= 0:
            continue
        gap = curr_start - (prev_start + prev_dur)
        if abs(gap) > 5.0:
            direction = "gap" if gap > 0 else "overlap"
            msg = (
                f"[WARN] {cam_name} chapter {i + 1}: "
                f"timestamp {direction} of {abs(gap):.1f}s detected"
            )
            print(msg, flush=True)
            warnings.append(msg)
    return warnings


def compute_sync(cam1_chapters: list[str], cam2_chapters: list[str]) -> dict:
    """
    Compute camera alignment offset from chapter metadata.
    Probes all chapters for continuity checks.
    Returns sync_info dict. Raises ValueError on implausible offset (> 60s).
    """
    cam1_meta = [extract_chapter_time(p) for p in cam1_chapters]
    cam2_meta = [extract_chapter_time(p) for p in cam2_chapters]

    cam1_s, cam1_src, cam1_tc, _ = cam1_meta[0]
    cam2_s, cam2_src, cam2_tc, _ = cam2_meta[0]

    raw_offset = cam2_s - cam1_s

    if abs(raw_offset) < 0.1:
        raw_offset = 0.0

    if abs(raw_offset) > 60.0:
        raise ValueError(
            f"[ERROR] Sync offset {raw_offset:.1f}s is implausibly large — "
            "likely a metadata error. Check GoPro timecode sync."
        )

    if raw_offset > 0:
        cam1_detect_offset_s = raw_offset
        cam2_detect_offset_s = 0.0
    elif raw_offset < 0:
        cam1_detect_offset_s = 0.0
        cam2_detect_offset_s = abs(raw_offset)
    else:
        cam1_detect_offset_s = 0.0
        cam2_detect_offset_s = 0.0

    sources = {cam1_src, cam2_src}
    if sources == {"timecode"}:
        sync_method = "timecode"
    elif sources == {"creation_time"}:
        sync_method = "creation_time"
    else:
        sync_method = "mixed"
        print(
            f"[WARN] Sync sources differ (cam1={cam1_src}, cam2={cam2_src}). "
            "Accuracy may be reduced.",
            flush=True,
        )

    warnings = _check_chapter_continuity(cam1_meta, "cam1")
    warnings += _check_chapter_continuity(cam2_meta, "cam2")

    return {
        "cam1_start_timecode": cam1_tc,
        "cam2_start_timecode": cam2_tc,
        "offset_s": raw_offset,
        "cam1_detect_offset_s": cam1_detect_offset_s,
        "cam2_detect_offset_s": cam2_detect_offset_s,
        "sync_method": sync_method,
        "warnings": warnings,
    }


def write_concat_manifest(chapters: list[str], detect_offset_s: float, out_path: str) -> None:
    """Write ffmpeg concat manifest, adding inpoint on first file if detect_offset_s > 0."""
    lines = ["ffconcat version 1.0"]
    for i, ch in enumerate(chapters):
        lines.append(f"file '{ch}'")
        if i == 0 and detect_offset_s > 0.0:
            lines.append(f"inpoint {detect_offset_s:.3f}")
    Path(out_path).write_text("\n".join(lines) + "\n")


def main(game_folder: str) -> None:
    chapters_path = Path(game_folder) / "chapters.json"
    if not chapters_path.exists():
        raise ValueError(f"[ERROR] chapters.json not found in {game_folder}. Run discover.py first.")

    chapters = json.loads(chapters_path.read_text())

    print("[gopro_meta] Extracting timecodes...", flush=True)
    sync = compute_sync(chapters["cam1"], chapters["cam2"])

    sync_path = Path(game_folder) / "sync_info.json"
    sync_path.write_text(json.dumps(sync, indent=2))
    print(
        f"[gopro_meta] sync_info.json written (offset={sync['offset_s']:.3f}s, "
        f"method={sync['sync_method']})",
        flush=True,
    )

    for cam, key in (("cam1", "cam1_detect_offset_s"), ("cam2", "cam2_detect_offset_s")):
        out = str(Path(game_folder) / f"{cam}_concat.txt")
        write_concat_manifest(chapters[cam], sync[key], out)
        print(
            f"[gopro_meta] {cam}_concat.txt written ({len(chapters[cam])} chapters, "
            f"offset={sync[key]:.3f}s)",
            flush=True,
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: gopro_meta.py <game_folder>")
    try:
        main(sys.argv[1])
    except ValueError as e:
        sys.exit(str(e))
