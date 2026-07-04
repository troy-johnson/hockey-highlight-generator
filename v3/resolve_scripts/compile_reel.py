#!/usr/bin/env python3
"""Build a pre-cut V3 highlight reel in DaVinci Resolve."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from dataclasses import dataclass
from pathlib import Path

# Reel length cap in seconds. Edit before running if a shorter or longer cap is needed.
_MAX_REEL_S = 900.0


def score_to_color(score: float) -> str:
    if score > 1.60:
        return "Red"
    if score > 1.30:
        return "Orange"
    if score > 1.00:
        return "Yellow"
    return "Blue"


@dataclass(frozen=True)
class Event:
    start_s: float
    end_s: float
    score: float
    primary_cam: str
    confidence: float
    color: str

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def load_events(events_csv: str | Path) -> list[Event]:
    events: list[Event] = []

    with open(events_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            score = float(row["score"])
            events.append(
                Event(
                    start_s=float(row["start_s"]),
                    end_s=float(row["end_s"]),
                    score=score,
                    primary_cam=row["primary_cam"],
                    confidence=float(row["confidence"]),
                    color=score_to_color(score),
                )
            )

    return events


def select_events(events: list[Event], max_reel_s: float = _MAX_REEL_S) -> list[Event]:
    """Return events in chronological order, capped at max_reel_s total duration.

    Drops lowest-priority / lowest-score events first: Blue → Yellow → Orange.
    Red events are never dropped — they are always included regardless of cap.
    Note: if only Red events remain and their total duration exceeds max_reel_s,
    the cap is still exceeded. This is intentional.
    """
    selected = sorted(events, key=lambda event: event.start_s)
    total_duration = sum(event.duration_s for event in selected)

    if total_duration <= max_reel_s:
        return selected

    remaining = list(selected)
    remaining_duration = total_duration

    for color in ("Blue", "Yellow", "Orange"):
        tier_events = sorted(
            (event for event in remaining if event.color == color),
            key=lambda event: event.score,
        )
        for event in tier_events:
            if remaining_duration <= max_reel_s:
                break
            remaining.remove(event)
            remaining_duration -= event.duration_s

    return sorted(remaining, key=lambda event: event.start_s)


def calc_source_frames(
    event: Event,
    detect_offset_s: float,
    timeline_fps: int,
    preroll_s: float = 3.0,
    postroll_s: float = 2.0,
) -> tuple[int, int]:
    src_in_s = max(0.0, detect_offset_s + event.start_s - preroll_s)
    src_out_s = detect_offset_s + event.end_s + postroll_s

    return round(src_in_s * timeline_fps), round(src_out_s * timeline_fps)


def _get_resolve():
    try:
        dvr_script = importlib.import_module("DaVinciResolveScript")
    except ModuleNotFoundError as exc:
        raise RuntimeError("DaVinci Resolve scripting module not available") from exc

    return dvr_script.scriptapp("Resolve")


def _tc_to_frames(tc: str, fps: int) -> int:
    """Convert HH:MM:SS:FF timecode string to frame count."""
    h, m, s, f = (int(x) for x in tc.split(":"))
    return (h * 3600 + m * 60 + s) * fps + f


def _find_chapter_for_frame(items: list, frame_num: int, fps: int) -> tuple:
    """Return (chapter_item, local_frame, chapter_start_frame, chapter_dur_frames).

    Maps an absolute concat-stream frame number to a specific chapter item.
    chapter_start_frame is the absolute frame where the returned chapter begins.
    """
    cumulative = 0
    for item in items:
        try:
            dur_tc = item.GetClipProperty("Duration")
            dur_frames = _tc_to_frames(dur_tc, fps)
        except Exception:
            dur_frames = round(60 * fps)  # assume 60s if unavailable
        if frame_num < cumulative + dur_frames:
            return item, frame_num - cumulative, cumulative, dur_frames
        cumulative += dur_frames
    # Past all chapters — clamp to last chapter
    last_dur = round(60 * fps)
    try:
        last_dur = _tc_to_frames(items[-1].GetClipProperty("Duration"), fps)
    except Exception:
        pass
    last_start = cumulative - last_dur
    return items[-1], max(0, frame_num - last_start), last_start, last_dur


def _place_clips_dual_track(
    media_pool,
    cam1_items: list,
    cam2_items: list,
    events: list[Event],
    sync_info: dict,
    timeline_fps: int,
    stinger_end: int,
) -> None:
    """Fallback: place clips from individual camera items on Track 1 (cam1) / Track 2 (cam2)."""
    record_frame = stinger_end
    for event in events:
        if event.primary_cam == "cam1":
            items = cam1_items
            cam_offset = sync_info["cam1_detect_offset_s"]
            track_idx = 1
        else:
            items = cam2_items
            cam_offset = sync_info["cam2_detect_offset_s"]
            track_idx = 2

        if not items:
            continue

        src_in, src_out = calc_source_frames(event, cam_offset, timeline_fps)
        chapter_item, local_in, ch_start, ch_dur = _find_chapter_for_frame(
            items, src_in, timeline_fps
        )
        # Clamp src_out to the end of the chapter that owns src_in so that
        # cross-chapter events don't produce out-of-range local frame numbers.
        local_out = min(src_out, ch_start + ch_dur) - ch_start

        result = media_pool.AppendToTimeline([{
            "mediaPoolItem": chapter_item,
            "startFrame": local_in,
            "endFrame": local_out,
            "trackIndex": track_idx,
            "recordFrame": record_frame,
            "mediaType": 1,
        }])
        if result:
            record_frame += local_out - local_in

    total_s = (record_frame - stinger_end) / timeline_fps
    print(
        f"[compile_reel] Placed {len(events)} clips (dual-track fallback, no angle switching). "
        f"Total: {total_s:.0f}s ({total_s / 60:.1f} min)"
    )


def _resolve_assemble(resolve, game_folder: str, events: list[Event], sync_info: dict) -> None:
    for cam in ("cam1", "cam2"):
        cam_dir = Path(game_folder) / cam
        if not cam_dir.is_dir():
            print(f"Missing required folder: {cam_dir}")
            raise SystemExit(1)

    chapters_path = Path(game_folder) / "chapters.json"
    if not chapters_path.is_file():
        print(f"Missing required file: {chapters_path}")
        raise SystemExit(1)

    chapters = json.loads(chapters_path.read_text(encoding="utf-8"))

    pm = resolve.GetProjectManager()
    if pm is None:
        print("[compile_reel] No project manager available. Open a project first.")
        raise SystemExit(1)

    proj = pm.GetCurrentProject()
    if proj is None:
        print("[compile_reel] No active project. Open your template project first.")
        raise SystemExit(1)

    tl = proj.GetCurrentTimeline()
    if tl is None:
        print("[compile_reel] No active timeline. Activate a timeline first.")
        raise SystemExit(1)

    folder_name = Path(game_folder).name
    media_pool = proj.GetMediaPool()
    root_folder = media_pool.GetRootFolder()
    game_bin = media_pool.AddSubFolder(root_folder, f"Game Footage — {folder_name}")
    if game_bin is None:
        game_bin = root_folder
    media_pool.SetCurrentFolder(game_bin)

    all_paths = chapters["cam1"] + chapters["cam2"]
    imported = media_pool.ImportMedia(all_paths)
    if not imported:
        print("[compile_reel] ImportMedia returned nothing. Check file paths.")
        raise SystemExit(1)

    cam1_set = set(chapters["cam1"])
    cam2_set = set(chapters["cam2"])
    cam1_items = [i for i in imported if i.GetClipProperty("File Path") in cam1_set]
    cam2_items = [i for i in imported if i.GetClipProperty("File Path") in cam2_set]

    sync_method = sync_info.get("sync_method", "timecode")
    mc_sync_type = "timecode" if sync_method == "timecode" else "audio"
    if sync_method != "timecode":
        print(
            f"[compile_reel] sync_method is '{sync_method}' — "
            "using audio sync for multicam clip creation"
        )

    multicam_item = None
    try:
        multicam_item = media_pool.CreateMultiCamClip(
            cam1_items + cam2_items,
            {
                "name": f"Multicam — {folder_name}",
                "syncType": mc_sync_type,
                "videoTrackCount": 2,
                "audioTrackCount": 2,
            },
        )
    except Exception:
        pass

    timeline_fps = 60
    try:
        fps_str = proj.GetSetting("timelineFrameRate")
        if fps_str:
            timeline_fps = int(float(fps_str))
    except Exception:
        pass

    existing = tl.GetItemListInTrack("video", 1) or []
    stinger_end = 0
    if existing:
        stinger_end = max(item.GetStart() + item.GetDuration() for item in existing)

    sentinel = Path(game_folder) / "MULTICAM_FALLBACK.txt"
    if multicam_item is None:
        print(
            "[compile_reel] [WARN] multicam creation failed — "
            "falling back to dual-track placement"
        )
        # Sentinel is best-effort: never let a filesystem error on the already
        # degraded path abort the fallback placement itself.
        try:
            sentinel.write_text(
                "Multicam clip creation failed in this run — clips were placed on "
                "dual tracks (cam1=Track 1, cam2=Track 2) with NO angle switching. "
                "Review angle selection manually. Delete this file once handled.\n",
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"[compile_reel] [WARN] could not write fallback sentinel: {exc}")
        _place_clips_dual_track(
            media_pool, cam1_items, cam2_items, events, sync_info, timeline_fps, stinger_end
        )
        return

    # Multicam path succeeded — clear any stale sentinel from a prior fallback run.
    try:
        sentinel.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(f"[compile_reel] [WARN] could not remove stale sentinel: {exc}")

    record_frame = stinger_end
    for event in events:
        cam_offset = sync_info[f"{event.primary_cam}_detect_offset_s"]
        src_in, src_out = calc_source_frames(event, cam_offset, timeline_fps)

        result = media_pool.AppendToTimeline([{
            "mediaPoolItem": multicam_item,
            "startFrame": src_in,
            "endFrame": src_out,
            "trackIndex": 1,
            "recordFrame": record_frame,
            "mediaType": 1,
        }])

        if result:
            placed_item = result[0]
            try:
                cam_angle = 1 if event.primary_cam == "cam1" else 2
                placed_item.SetCurrentVideoItem(cam_angle)
            except Exception:
                pass
            record_frame += src_out - src_in

    total_s = (record_frame - stinger_end) / timeline_fps
    print(
        f"[compile_reel] Placed {len(events)} clips. "
        f"Total: {total_s:.0f}s ({total_s / 60:.1f} min)"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Assemble V3 highlight reel in DaVinci Resolve")
    parser.add_argument(
        "--max_reel_s",
        type=float,
        default=_MAX_REEL_S,
        help=f"Maximum reel duration in seconds (default: {_MAX_REEL_S})",
    )
    args, _ = parser.parse_known_args(argv)

    try:
        resolve = _get_resolve()
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(1) from exc

    game_folder = resolve.Fusion().RequestDir("Select game folder")
    if not game_folder:
        print("No folder selected")
        raise SystemExit(1)

    events_csv = Path(game_folder) / "events.csv"
    if not events_csv.is_file():
        print(f"Missing required file: {events_csv}")
        raise SystemExit(1)

    sync_info_json = Path(game_folder) / "sync_info.json"
    if not sync_info_json.is_file():
        print(f"Missing required file: {sync_info_json}")
        raise SystemExit(1)

    try:
        with open(sync_info_json, "r", encoding="utf-8") as f:
            sync_info = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Failed to parse sync_info.json: {exc}")
        raise SystemExit(1) from exc

    for field in ("cam1_detect_offset_s", "cam2_detect_offset_s"):
        if field not in sync_info:
            print(f"Missing required sync field: {field}")
            raise SystemExit(1)
        if not isinstance(sync_info[field], (int, float)):
            print(f"Invalid sync field type: {field}")
            raise SystemExit(1)

    events = load_events(events_csv)
    selected_events = select_events(events, max_reel_s=args.max_reel_s)
    _resolve_assemble(resolve, str(game_folder), selected_events, sync_info)
