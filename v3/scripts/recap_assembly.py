"""Plan and render a plain, silent Recap from selected Game Sheet goals."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "v2" / "scripts"))

FPS = 30
MAX_FRAMES = 240 * FPS
GOAL_ACTION_S = 3.0  # Selection locates action, not puck crossing. Keep this much after the moment.
MIN_BUILD_UP_S = 2.0
ZOOM_MIN, ZOOM_MAX = 1.15, 1.3  # punch-in framing from the 4K source (spec 002 §5.7)
ZOOM_TIGHT_MAX = 2.5  # replay punch-in never crops tighter than 0.4 of the frame
FISHEYE_CROP = 0.06  # fraction cropped from each edge to hide fisheye distortion
MAX_LEVEL_DEG = 6.0  # horizon leveling is clamped to a gentle correction
MAX_LINE_DEG = 8.0  # a "boards" line may deviate at most this much from horizontal
SLOW_MO = 0.5  # replay playback speed
# 0.5x at 30 fps out advances 1/60 s of source per frame, so footage at or above
# 60 fps plays natively smooth slow motion and never needs interpolation.
NATIVE_SLOW_FPS = FPS / SLOW_MO
MIN_REPLAY_S = 0.8
# The whistle detection leads the puck crossing: on the validated game the scoring
# frame sits a median 0.68 s after the detection (offsets -1.5 .. +2.1 s), so replay
# windows center on detection + 0.75 s instead of the detection itself (11/14 covered
# vs 10/14 with no bias).
REPLAY_WINDOW_BIAS = 0.75
# Replay budget tiers by goal count (spec 002 §5.8): shot kinds and their source seconds.
REPLAY_SHOTS = {1: (("wide", 2.0), ("tight", 2.0)),
                2: (("tight", 2.5),),
                3: (("tight", 2.0),)}
# Spec 002 §5.8: the Recap is about three minutes. After the goals, chosen plays
# and their replay blocks, the fill restores cut goals first and then adds filler
# plays from the Next Best list.
FILL_TARGET_S = 180.0
# User policy (2026-10-02): a Next Best play preempts the next cut goal in the
# fill only when its score beats the goal's interest by this margin; goals fill
# first otherwise.
FILL_PLAY_ADVANTAGE = 0.25
# Only the best non-goal play gets a short replay (spec 002 §5.6); one tight shot.
BEST_PLAY_REPLAY_S = 2.0


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


def _num(value, name: str, goal_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{goal_id}: invalid {name}")
    return float(value)


def _containing_block(layout: list[dict], t: float) -> dict | None:
    return next((b for b in layout if b["start"] <= t < b["end"]), None)


def rife_ready() -> bool:
    """RIFE needs PyTorch (optional requirements-ml.txt stack) and rife-ncnn-vulkan on PATH."""
    try:
        import torch  # noqa: F401  lazy: only needed for replay interpolation below 120 fps
    except Exception:
        return False
    return shutil.which("rife-ncnn-vulkan") is not None


def probe_fps(path: str) -> float | None:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                   "-show_entries", "stream=avg_frame_rate", "-of", "json", str(path)])
    rate = ((json.loads(out).get("streams") or [{}])[0].get("avg_frame_rate") or "")
    try:
        num, den = (float(v) for v in rate.split("/"))
        return num / den if den else None
    except (TypeError, ValueError):
        return None


def _frame_horizon_deg(image) -> float | None:
    """Boards-line angle in one frame, or None when no long near-horizontal line is found."""
    import cv2
    import numpy as np
    gray = cv2.cvtColor(cv2.resize(image, (320, 180)), cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(cv2.Canny(gray, 50, 150), 1, np.pi / 180,
                            threshold=50, minLineLength=80, maxLineGap=10)
    if lines is None:
        return None
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = float(x2 - x1), float(y2 - y1)
        # Boards run level: accept only gentle slopes, and demand several lines that
        # agree, so a jersey stripe or the glass top edge cannot fake a horizon.
        if math.hypot(dx, dy) < 80 or abs(dx) < 1 or abs(dy / dx) > math.tan(math.radians(MAX_LINE_DEG)):
            continue
        angles.append(math.degrees(math.atan(dy / dx)))
    if len(angles) < 3:
        return None
    angles.sort()
    if angles[-1] - angles[0] > 2.5:
        return None
    return float(np.median(angles))


def horizon_angle(path, duration: float | None = None) -> float:
    """Median boards-line angle in degrees over sampled frames; 0 when unknown.

    Positive means the horizon slopes down to the right.
    """
    try:
        import cv2
        import numpy as np
        duration = duration or _stream_duration(str(path))
        if not duration:
            return 0.0
        cap = cv2.VideoCapture(str(path))
        samples = []
        try:
            for i in range(5):
                cap.set(cv2.CAP_PROP_POS_MSEC, duration * 1000 * (i + .5) / 5)
                ok, frame = cap.read()
                if ok:
                    angle = _frame_horizon_deg(frame)
                    if angle is not None:
                        samples.append(angle)
        finally:
            cap.release()
        if not samples:
            return 0.0
        return float(np.clip(np.median(samples), -MAX_LEVEL_DEG, MAX_LEVEL_DEG))
    except Exception:
        return 0.0


def _first_file(layout: list[dict]) -> str | None:
    for block in sorted(layout or [], key=lambda b: b["start"]):
        for path, _ in block.get("files", []):
            return path
    return None


def load_rois(root: Path) -> dict:
    """Goal-frame boxes per camera, in 1280x720 analysis coordinates (rois.json, else rois_auto.json)."""
    rois = {}
    for name in ("rois.json", "rois_auto.json"):
        path = root / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for key, val in data.items():
            cam = "cam" + key.removeprefix("camera_") if key.startswith("camera_") else key
            if not re.fullmatch(r"cam\d+", cam) or not isinstance(val, dict):
                continue
            if name == "rois.json":
                if val.get("net") and cam not in rois:
                    rois[cam] = val
            elif val.get("goal_box") and len(val["goal_box"]) == 4 and cam not in rois:
                x0, y0, x1, y1 = val["goal_box"]
                rois[cam] = {"net": [x0, y0, x1 - x0, y1 - y0]}
    return rois


def _tight_zoom(cam: str, rois: dict | None, angle_deg: float = 0.0) -> tuple[dict, bool]:
    """Static punch-in crop on the goal frame box, as fractions of the fisheye-cropped frame."""
    roi = (rois or {}).get(cam, {}).get("net")
    if not roi or len(roi) != 4:
        return {"type": "static", "z": ZOOM_MAX}, False
    x, y, w, h = (float(v) for v in roi)
    keep = 1 - 2 * FISHEYE_CROP
    # Box plus context, keeping the frame aspect; fractions re-based to the cropped frame.
    # The window never gets tighter than 1/ZOOM_TIGHT_MAX of the cropped frame.
    wf = hf = max(min(1.0, max(1.5 * w / 1280, 1.5 * h / 720)) / keep,
                  1 / ZOOM_TIGHT_MAX)
    if wf >= 1:
        return {"type": "roi", "wf": 1.0, "hf": 1.0, "cx": .5, "cy": .5}, True
    cx = min(max(((x + w / 2) / 1280 - FISHEYE_CROP) / keep, wf / 2), 1 - wf / 2)
    cy = min(max(((y + h / 2) / 720 - FISHEYE_CROP) / keep, hf / 2), 1 - hf / 2)
    if angle_deg:
        # The levelling rotation in _pre_filter moves the box: rotate the crop center the
        # same visual amount (positive angle = horizon slopes down to the right = content
        # turns counterclockwise, opposite in y-down crop coordinates). The rotation
        # happens in pixel space, so the fraction offsets carry the frame aspect first.
        angle = math.radians(angle_deg)
        dx, dy = cx - .5, cy - .5
        cx = .5 + dx * math.cos(angle) + dy * math.sin(angle) * (720 / 1280)
        cy = .5 - dx * math.sin(angle) * (1280 / 720) + dy * math.cos(angle)
        cx = min(max(cx, wf / 2), 1 - wf / 2)
        cy = min(max(cy, hf / 2), 1 - hf / 2)
    return {"type": "roi", "wf": wf, "hf": hf, "cx": cx, "cy": cy}, True


def _trim_context(clips: list[dict], side: str, floor_s: float, over_s: float) -> float:
    """Shorten build-up or celebration proportionally to free room, never below the floor."""
    room = []
    for c in clips:
        d = (c["moment_s"] - c["start_s"]) if side == "before" else (c["end_s"] - c["moment_s"])
        room.append(max(0.0, d - min(floor_s, d)))
    take = min(over_s, sum(room))
    if take <= 0:
        return 0.0
    ratio = take / sum(room)
    for c, r in zip(clips, room):
        if side == "before":
            d = c["moment_s"] - c["start_s"]
            c["start_s"] = c["moment_s"] - (d - r * ratio)
        else:
            d = c["end_s"] - c["moment_s"]
            c["end_s"] = c["moment_s"] + (d - r * ratio)
    return take


def plan_recap(selection: dict, layouts: dict, speed: float = 1.1, rois: dict | None = None,
               source_fps: dict | None = None, horizon: dict | None = None) -> dict:
    if not math.isfinite(speed) or not .25 <= speed <= 4:
        raise ValueError("live_play_speed must be between 0.25 and 4")
    goals = selection.get("goals") or []
    chosen = [g for g in goals if g.get("chosen") and not g.get("cut")]
    # The tier follows the goals chosen before the fill: goals the fill later
    # restores from the cut list keep the same tier-based budgets.
    before, after = (6., 3.) if len(chosen) <= 6 else (5., 2.) if len(chosen) <= 12 else (4., 0.)
    tier = 1 if len(chosen) <= 6 else 2 if len(chosen) <= 12 else 3
    clips, flags, replays = [], [], []

    def goal_clip(goal):
        """One live goal clip from its chosen window, or None with a flag."""
        choice = goal.get("chosen")
        if not choice:
            flags.append(f"{goal['id']}: no clip found; omitted from plain Recap")
            return None
        cam = choice["primary_cam"]
        moment = _num(choice["detection_s"], "detection_s", goal["id"])
        window = (_num(choice["start_s"], "start_s", goal["id"]),
                  _num(choice["end_s"], "end_s", goal["id"]))
        if not window[0] <= moment < window[1]:
            raise ValueError(f"{goal['id']}: selected moment outside its search window")
        block = _containing_block(layouts.get(cam, []), moment)
        if block is None:
            flags.append(f"{goal['id']}: moment outside Recording coverage; omitted from plain Recap")
            return None
        start, end = moment - before, moment + GOAL_ACTION_S + after
        trimmed = []
        if start < block["start"]:
            start, trimmed = block["start"], ["start"]
        if end > block["end"]:
            end = block["end"]
            trimmed.append("end")
        if trimmed:
            flags.append(f"{goal['id']}: clip {' and '.join(trimmed)} trimmed to Recording coverage")
        if not start <= moment < end:
            flags.append(f"{goal['id']}: no room for goal action in Recording coverage; omitted")
            return None
        return {"goal_id": goal["id"], "kind": "goal", "camera": cam, "moment_s": moment,
                "start_s": start, "end_s": end, "speed": speed,
                "zoom": {"type": "push", "z0": ZOOM_MIN, "z1": ZOOM_MAX},
                "angle_deg": float((horizon or {}).get(cam) or 0.0),
                "best_goal": bool(goal.get("best_goal"))}

    # Goals cut by the interest policy are a reserve, not a deletion: when the
    # Recap falls short of about three minutes they come back before any play.
    cut_goals = []
    for goal in goals:
        if goal.get("chosen") and goal.get("cut"):
            # Spec 002 §5.8 goal cutting: selection never cuts a protected goal,
            # so any `cut` arriving here is by definition a cuttable goal.
            cut_goals.append(goal)
            continue
        clip = goal_clip(goal)
        if clip:
            clips.append(clip)
    clips.sort(key=lambda c: c["moment_s"])
    if not clips:
        raise ValueError("no selected goals to render")

    def replay_shot(clip, kind, src_d, order):
        """Build one replay shot, or flag and return None when it cannot be cut."""
        layout = layouts.get(clip["camera"], [])
        block = _containing_block(layout, clip["moment_s"]) or {}
        # The bias centers on the scoring frame: the whistle detection leads the
        # puck crossing, but a non-goal play has no such offset to correct for.
        center = clip["moment_s"] + (REPLAY_WINDOW_BIAS if clip.get("kind") == "goal" else 0.0)
        a = max(center - src_d / 2, block.get("start", clip["moment_s"]))
        b = min(center + src_d / 2, block.get("end", clip["moment_s"]))
        if b - a < MIN_REPLAY_S:
            flags.append(f"{clip['goal_id']}: replay {kind} shot has no room in Recording"
                         " coverage; omitted")
            return None
        try:
            parts = source_parts(layout, a, b)
        except ValueError as exc:
            flags.append(f"{clip['goal_id']}: replay {kind} shot {exc}; omitted")
            return None
        fps = (source_fps or {}).get(clip["camera"])
        if fps is not None and fps >= NATIVE_SLOW_FPS:
            slowmo = "native"
        elif fps is None:
            # An unprobed rate is unknown, not native: interpolate or duplicate, and say so.
            if rife_ready():
                slowmo = "rife"
            else:
                slowmo = "duplicated"
                flags.append(f"{clip['goal_id']}: frame rate unknown; replay slow motion"
                             " without interpolation")
        elif rife_ready():
            slowmo = "rife"
        else:
            slowmo = "duplicated"
            flags.append(f"{clip['goal_id']}: RIFE unavailable; replay slow motion without"
                         " interpolation")
        if kind == "tight":
            zoom, found = _tight_zoom(clip["camera"], rois, clip.get("angle_deg") or 0.0)
            if not found:
                flags.append(f"{clip['goal_id']}: no goal-frame box; replay uses a centered"
                             " punch-in")
        else:
            zoom = {"type": "static", "z": ZOOM_MIN}
        return {"goal_id": clip["goal_id"], "kind": kind, "camera": clip["camera"],
                "order": order, "speed": SLOW_MO, "slowmo": slowmo, "source_fps": fps,
                "angle_deg": clip["angle_deg"], "start_s": a, "end_s": b,
                "parts": parts, "frames": math.ceil((b - a) / SLOW_MO * FPS),
                "zoom": zoom}

    # Replay blocks (spec 002 §5.7): from the scoring camera, slow motion from native 120 fps.
    # Spec 002 §5.8: the 2-3 best goals always get the full treatment, i.e. the
    # tier-1 replay block, whatever the game's budget tier.
    replay_budget = {c["goal_id"]: REPLAY_SHOTS[1] if c.get("best_goal") else REPLAY_SHOTS[tier]
                     for c in clips}
    for order, clip in enumerate(clips):
        for kind, src_d in replay_budget[clip["goal_id"]]:
            shot = replay_shot(clip, kind, src_d, order)
            if shot:
                if clip.get("best_goal"):
                    shot["best_goal"] = True
                replays.append(shot)

    # Non-goal plays (spec 002 §5.6/§5.8): chosen plays always render; plays from
    # the Next Best list top the Recap up to about three minutes at play speed.
    plays_by_id = {}
    for play in selection.get("plays") or []:
        if isinstance(play, dict) and play.get("id"):
            plays_by_id[play["id"]] = play
    play_ids_added: set[str] = set()

    def build_play_clip(play, is_chosen):
        play_id = play.get("id")
        if play_id in play_ids_added:
            return None
        cam = play.get("primary_cam")
        moment = _num(play.get("detection_s"), "detection_s", str(play_id))
        start = _num(play.get("start_s"), "start_s", str(play_id))
        end = _num(play.get("end_s"), "end_s", str(play_id))
        if not start <= moment < end:
            flags.append(f"{play_id}: invalid play window; omitted")
            return None
        if _containing_block(layouts.get(cam, []), moment) is None:
            flags.append(f"{play_id}: play outside Recording coverage; omitted")
            return None
        play_ids_added.add(play_id)
        return {"goal_id": play_id, "kind": "play", "play_kind": play.get("kind"),
                "camera": cam, "moment_s": moment, "start_s": start, "end_s": end,
                "speed": speed, "play_score": float(play.get("score") or 0.0),
                "chosen": bool(is_chosen),
                # Spec 002 §5.7: goals get the slow push-in; other live clips
                # get the same framing as a fixed punch-in crop.
                "zoom": {"type": "static", "z": ZOOM_MIN},
                "angle_deg": float((horizon or {}).get(cam) or 0.0)}

    play_clips, filler_ids = [], []
    for play in plays_by_id.values():
        if play.get("chosen"):
            clip = build_play_clip(play, True)
            if clip:
                play_clips.append(clip)

    goal_frames = sum(math.ceil((c["end_s"] - c["start_s"]) / speed * FPS) for c in clips)
    total_frames = goal_frames + sum(r["frames"] for r in replays) + sum(
        math.ceil((c["end_s"] - c["start_s"]) / speed * FPS) for c in play_clips)
    fill_target_frames = FILL_TARGET_S * FPS
    restored_ids: list[str] = []
    cut_pool = sorted(cut_goals, key=lambda g: -(g.get("interest") or 0.0))
    play_pool = list(selection.get("next_best") or [])
    play_i = 0
    # Fill policy: restore cut goals first; a Next Best play preempts the next
    # cut goal only when its score beats the goal's interest by a wide margin.
    while total_frames < fill_target_frames and (cut_pool or play_i < len(play_pool)):
        take_goal = bool(cut_pool)
        if take_goal and play_i < len(play_pool):
            peek = plays_by_id.get(play_pool[play_i].get("id")) if isinstance(play_pool[play_i], dict) else None
            if peek and not peek.get("chosen") and \
                    (float(peek.get("score") or 0.0) - (cut_pool[0].get("interest") or 0.0)) >= FILL_PLAY_ADVANTAGE:
                take_goal = False
        if take_goal:
            goal = cut_pool.pop(0)
            clip = goal_clip(goal)
            if clip is None:
                continue
            shots = [s for s in (replay_shot(clip, kind, src_d, len(clips))
                                 for kind, src_d in (REPLAY_SHOTS[1] if clip.get("best_goal")
                                                     else REPLAY_SHOTS[tier])) if s]
            needed = math.ceil((clip["end_s"] - clip["start_s"]) / speed * FPS) + sum(s["frames"] for s in shots)
            if total_frames + needed > MAX_FRAMES:
                continue
            clips.append(clip)
            replays.extend(shots)
            total_frames += needed
            restored_ids.append(goal["id"])
            flags.append(f"{goal['id']}: restored from the cut list to fill the Recap")
            continue
        nb = play_pool[play_i]
        play_i += 1
        play = plays_by_id.get(nb.get("id")) if isinstance(nb, dict) else None
        if not play or play.get("chosen"):
            continue
        clip = build_play_clip(play, False)
        if clip is None:
            continue
        filler_frames = math.ceil((clip["end_s"] - clip["start_s"]) / speed * FPS)
        if total_frames + filler_frames > MAX_FRAMES:
            continue
        play_clips.append(clip)
        filler_ids.append(clip["goal_id"])
        total_frames += filler_frames

    # Only the best play gets a short replay: one tight shot, slow motion (spec 002 §5.6).
    best_play_id = None
    if play_clips:
        best_play = max(play_clips, key=lambda c: (c["play_score"], -c["moment_s"]))
        shot = replay_shot(best_play, "tight", BEST_PLAY_REPLAY_S, 0)
        if shot:
            replays.append(shot)
            best_play_id = best_play["goal_id"]

    if play_clips or filler_ids:
        flags.append(f"including {len(play_clips)} non-goal play(s); {len(filler_ids)} filler play(s)"
                     + (f"; best-play replay on {best_play_id}" if best_play_id else ""))

    # Cut goals that the fill never restored are dropped after all.
    for goal in cut_goals:
        if goal["id"] not in restored_ids:
            flags.append(f"{goal['id']}: cut by interest policy; omitted from Recap")

    clips = sorted(clips + play_clips, key=lambda c: c["moment_s"])
    kept = []
    for c in clips:
        try:
            c["parts"] = source_parts(layouts.get(c["camera"], []), c["start_s"], c["end_s"])
        except ValueError as exc:
            flags.append(f"{c['goal_id']}: {exc}; omitted from plain Recap")
            continue
        c["frames"] = math.ceil((c["end_s"] - c["start_s"]) / speed * FPS)
        kept.append(c)
    if not kept:
        raise ValueError("no selected goals to render")
    clips = kept
    frames = sum(math.ceil((c["end_s"] - c["start_s"]) / speed * FPS) for c in clips)
    total = frames + sum(r["frames"] for r in replays)
    if total > MAX_FRAMES:
        # Plays go first (spec 002 §5.8 has plays behind goals in priority):
        # they exist only because there was room. Filler plays are the purest
        # padding, so they go before chosen ones, lowest score first.
        filler_dropped = []
        for c in sorted((c for c in clips if c.get("kind") == "play" and not c.get("chosen")),
                        key=lambda c: (c["play_score"], -c["moment_s"])):
            if total <= MAX_FRAMES:
                break
            clips.remove(c)
            total -= c["frames"]
            frames -= c["frames"]
            filler_dropped.append(c["goal_id"])
            # A dropped play cannot keep its replay (only the best play has one):
            # stop the stages below from counting it, or a goal replay goes too.
            for r in [r for r in replays if r["goal_id"] == c["goal_id"]]:
                replays.remove(r)
                total -= r["frames"]
        if filler_dropped:
            flags.append(f"dropped {len(filler_dropped)} filler play(s) to fit the four-minute"
                         f" cap ({', '.join(filler_dropped)})")
    if total > MAX_FRAMES:
        chosen_dropped = []
        for c in sorted((c for c in clips if c.get("kind") == "play" and c.get("chosen")),
                        key=lambda c: (c["play_score"], -c["moment_s"])):
            if total <= MAX_FRAMES:
                break
            clips.remove(c)
            total -= c["frames"]
            frames -= c["frames"]
            chosen_dropped.append(c["goal_id"])
            for r in [r for r in replays if r["goal_id"] == c["goal_id"]]:
                replays.remove(r)
                total -= r["frames"]
        if chosen_dropped:
            flags.append(f"dropped {len(chosen_dropped)} chosen play(s) to fit the four-minute"
                         f" cap ({', '.join(chosen_dropped)})")
    if total > MAX_FRAMES:
        # Shorten the replay blocks next (spec 002 §5.8): wide shots, then tight ones,
        # latest goals first. Dead goals are already gone, so no replay is dropped for
        # a goal that will not render.
        dropped, dropped_ids = 0, []
        # The best goals keep the full treatment as long as anything else can go:
        # their shots rank behind every other replay in the drop order.
        for r in sorted(replays, key=lambda r: (r.get("best_goal", False),
                                                r["kind"] == "tight", -r["order"])):
            if total <= MAX_FRAMES:
                break
            replays.remove(r)
            total -= r["frames"]
            dropped += 1
            if r["goal_id"] not in dropped_ids:
                dropped_ids.append(r["goal_id"])
        if dropped:
            flags.append(f"dropped {dropped} replay shot(s) to fit the four-minute cap "
                         f"({', '.join(dropped_ids)})")
    goal_clip_count = sum(1 for c in clips if c.get("kind") == "goal")
    goal_frames = sum(c["frames"] for c in clips if c.get("kind") == "goal")
    if goal_frames > MAX_FRAMES:
        # Keep every chosen goal and the goal action. Trim build-up first, then
        # celebration. The count and the overflow come from goal clips alone:
        # plays never buy extra goal context at the cap.
        # One extra frame per clip: frame counts are rounded up after trimming.
        over_s = (goal_frames - MAX_FRAMES + goal_clip_count) * speed / FPS
        goal_entries = [c for c in clips if c.get("kind") == "goal"]
        over_s -= _trim_context(goal_entries, "before", MIN_BUILD_UP_S, over_s)
        over_s -= _trim_context(goal_entries, "after", GOAL_ACTION_S, over_s)
        for c in clips:
            # Trimmed windows sit inside the resolved ones; the manifests stay valid.
            c["parts"] = source_parts(layouts.get(c["camera"], []), c["start_s"], c["end_s"])
            c["frames"] = math.ceil((c["end_s"] - c["start_s"]) / speed * FPS)
        frames = sum(c["frames"] for c in clips)
        total = frames + sum(r["frames"] for r in replays)
        goal_frames = sum(c["frames"] for c in clips if c.get("kind") == "goal")
        if goal_frames > MAX_FRAMES:
            raise ValueError("too many goals for the four-minute cap")
        flags.append("goal context shortened to fit the four-minute cap")
    if total > MAX_FRAMES:
        raise ValueError("too many goals for the four-minute cap")
    live_ids = {c["goal_id"] for c in clips}
    replays = [r for r in replays if r["goal_id"] in live_ids]
    duration = (sum(c["frames"] for c in clips) + sum(r["frames"] for r in replays)) / FPS
    return {"schema_version": 3, "speed": speed, "fps": FPS, "audio": "silent",
            "duration_s": duration, "clips": clips, "replays": replays,
            "input_flags": selection.get("flags", []),
            "flags": flags + ["plain Recap: no graphics or audio mix"]}


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


def _stream_duration(path: str) -> float | None:
    """Decodable video duration; chapter format duration can be longer than the stream."""
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                   "-show_entries", "stream=duration,nb_frames,r_frame_rate",
                                   "-of", "json", path])
    stream = (json.loads(out).get("streams") or [{}])[0]
    try:
        duration = float(stream.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0
    if duration <= 0 and stream.get("nb_frames"):
        try:
            num, den = (float(v) for v in str(stream["r_frame_rate"]).split("/"))
            duration = float(stream["nb_frames"]) / (num / den)
        except (TypeError, ValueError, ZeroDivisionError):
            duration = 0
    return duration if duration > 0 else None


def verify_sources(plan: dict) -> None:
    """Drop entries whose chapters decode fewer frames than planned; frozen padding hides that."""
    def check(entries):
        kept, dropped = [], []
        for entry in entries:
            short = None
            for part in entry["parts"]:
                duration = _stream_duration(part["file"])
                if duration is not None and part["seek_s"] + part["duration_s"] - duration > 2 / FPS:
                    short = part
                    break
            if short:
                dropped.append((entry, short))
            else:
                kept.append(entry)
        return kept, dropped

    kept, dropped = check(plan["clips"])
    for entry, short in dropped:
        plan["flags"].append(f"{entry['goal_id']}: chapter video shorter than its manifest duration"
                             f" ({Path(short['file']).name}); omitted from plain Recap")
    r_kept, r_dropped = check(plan.get("replays", []))
    for entry, short in r_dropped:
        plan["flags"].append(f"{entry['goal_id']}: replay {entry['kind']} shot chapter video shorter"
                             f" than its manifest duration ({Path(short['file']).name}); omitted")
    live_ids = {c["goal_id"] for c in kept}
    replays = [r for r in r_kept if r["goal_id"] in live_ids]
    if not kept:
        raise ValueError("no selected goals to render")
    plan["clips"], plan["replays"] = kept, replays
    plan["duration_s"] = sum(e["frames"] for e in kept + replays) / FPS


def _zoompan_filter(zoom: dict, m: int, fps: int, angle_deg: float = 0.0) -> str:
    """Build a zoompan filter. zoompan re-evaluates every frame, so push-ins actually move."""
    x_expr = "iw/2-(iw/zoom/2)"
    y_expr = "ih/2-(ih/zoom/2)"
    if zoom["type"] == "push":
        z_expr = (f"{zoom['z0']:.4f}+({zoom['z1']:.4f}-{zoom['z0']:.4f})*min(on/{m},1)")
    elif zoom["type"] == "roi":
        z_expr = f"{1.0 / zoom['wf']:.6f}"
        x_expr = f"iw*{zoom['cx']:.6f}-(iw/zoom/2)"
        y_expr = f"ih*{zoom['cy']:.6f}-(ih/zoom/2)"
    else:
        z_expr = f"{zoom['z']:.4f}"
    if angle_deg:
        # Levelled frames must not show the rotation's black wedges: keep the zoom at
        # or above the inscribed-rectangle factor of the rotated fisheye-cropped frame.
        angle = math.radians(angle_deg)
        floor = ((1 - 2 * FISHEYE_CROP)
                 * (math.cos(angle) + max(1280 / 720, 720 / 1280) * math.sin(angle)))
        z_expr = f"max({z_expr},{floor:.4f})"
    return (f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}'"
            f":d=1:s=1920x1080:fps={fps}")


def _pre_filter(entry: dict) -> str:
    # Negative angle: ffmpeg rotate is clockwise, and a horizon sloping down to the
    # right (positive degrees) needs a counterclockwise correction.
    angle = math.radians(-float(entry.get("angle_deg") or 0.0))
    rot = f"rotate={angle:.8f}:c=black," if abs(angle) > 1e-9 else ""
    keep = 1 - 2 * FISHEYE_CROP
    fish = (f"crop=w='trunc(iw*{keep:.4f}/2)*2':h='trunc(ih*{keep:.4f}/2)*2':"
            f"x='(iw-out_w)/2':y='(ih-out_h)/2'")
    return f"{rot}{fish}"


def _encode(entry: dict, dest: Path, speed: float, frames: int | None,
            fps_floor: int | None = None) -> None:
    argv = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    filters = []
    # Intermediate rate to feed setpts; slow motion duplicates first so the
    # half-speed output keeps 30 fresh frames per second.
    intermediate_fps = max(FPS, math.ceil(FPS / speed))
    if fps_floor is not None:
        # Authoritative: the RIFE pass consumes this piece at its native rate,
        # never thinned to 30 and never padded up with duplicates.
        intermediate_fps = fps_floor
    # The RIFE pass consumes this piece: never thin native frames to 30 first.
    tail_fps = fps_floor if fps_floor is not None else FPS
    pre = _pre_filter(entry)
    for j, part in enumerate(entry["parts"]):
        argv += ["-ss", str(part["seek_s"]), "-t", str(part["duration_s"]), "-i", part["file"]]
        filters.append(f"[{j}:v:0]fps={intermediate_fps},setpts=PTS-STARTPTS,{pre}[v{j}]")
    joined = "".join(f"[v{j}]" for j in range(len(entry["parts"])))
    m = int(math.ceil((entry["end_s"] - entry["start_s"]) * intermediate_fps)) + 1
    # settb: concat leaves a timebase that sends zoompan into an endless frame loop.
    # zoompan crops at 4K (the spec punches in from the source, not from a 1080p
    # intermediate) and writes the framed 1080p output itself.
    filters.append(f"{joined}concat=n={len(entry['parts'])}:v=1:a=0,"
                   f"settb=1/{intermediate_fps},setpts=PTS-STARTPTS,"
                   f"scale=3840:2160:force_original_aspect_ratio=increase,crop=3840:2160,"
                   f"{_zoompan_filter(entry['zoom'], m, intermediate_fps, entry.get('angle_deg') or 0.0)},"
                   f"setsar=1,setpts=PTS/{speed},fps={tail_fps},"
                   f"tpad=stop_mode=clone:stop=2[out]")
    argv += ["-filter_complex", ";".join(filters), "-map", "[out]", "-an"]
    if frames is not None:
        argv += ["-frames:v", str(frames)]
    argv += ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
             "-color_primaries", "bt709", "-colorspace", "bt709", "-color_trc", "bt709",
             str(dest)]
    subprocess.run(argv, check=True)


def _rife_slowmo(source: Path, dest: Path, work: Path, entry: dict,
                 fps_floor: int | None = None) -> None:
    """Interpolate to at least 60 effective fps with rife-ncnn-vulkan, then slow to SLOW_MO.

    Only sub-60 fps footage reaches this path: at 0.5x and 30 fps output each output
    frame advances 1/60 s, so 60 fps sources are already native-quality slow motion.
    """
    # The floor is the native rate itself: duplicating a 24 fps source to 30 first
    # would feed the interpolator stuttered input.
    fps = fps_floor or FPS
    passes = math.ceil(math.log2((FPS / SLOW_MO) / max(fps, 1e-6)))
    inp, out = work / "rife-in", work / "rife-out"
    inp.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                    "-vf", f"fps={fps}", str(inp / "%08d.png")], check=True)
    for _ in range(passes):
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rife-ncnn-vulkan", "-i", str(inp), "-o", str(out)], check=True)
        inp, out = out, inp
    shutil.rmtree(out, ignore_errors=True)
    rate = fps * 2 ** passes
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(rate),
                    "-i", str(inp / "%08d.png"),
                    "-vf", f"setpts=PTS/{SLOW_MO},fps={FPS},tpad=stop_mode=clone:stop=2",
                    "-frames:v", str(entry["frames"]), "-c:v", "libx264", "-preset", "fast",
                    "-crf", "20", "-pix_fmt", "yuv420p",
                    "-color_primaries", "bt709", "-colorspace", "bt709", "-color_trc", "bt709",
                    str(dest)], check=True)


def render(plan: dict, output: Path) -> None:
    """Render to local scratch space and replace the output only after validation."""
    with tempfile.TemporaryDirectory(prefix="hockey-recap-") as scratch:
        work = Path(scratch)
        sequence = []
        for clip in plan["clips"]:
            sequence.append(clip)
            sequence.extend(r for r in plan.get("replays", []) if r["goal_id"] == clip["goal_id"])
        pieces = []
        for i, entry in enumerate(sequence):
            dest = work / f"clip-{i:04d}.mp4"
            if entry.get("slowmo") == "rife":
                if rife_ready():
                    plain = work / f"plain-{i:04d}.mp4"
                    keep_fps = math.ceil(entry.get("source_fps") or FPS)
                    _encode(entry, plain, speed=1.0, frames=None, fps_floor=keep_fps)
                    _rife_slowmo(plain, dest, work / f"rife-{i:04d}", entry, keep_fps)
                else:
                    # The planned output says RIFE but the interpolator is gone at
                    # render time: fall back out loud, never silently.
                    entry["slowmo"] = "duplicated"
                    note = f"{entry['goal_id']}: RIFE unavailable at render time; replay slow motion without interpolation"
                    if note not in plan["flags"]:
                        plan["flags"].append(note)
                    _encode(entry, dest, speed=entry["speed"], frames=entry["frames"])
            else:
                _encode(entry, dest, speed=entry["speed"], frames=entry["frames"])
            pieces.append(dest)
            print(f"[assembly] rendered {i + 1}/{len(sequence)}", flush=True)
        manifest = work / "clips.txt"
        manifest.write_text("".join(f"file '{p.name}'\n" for p in pieces))
        assembled = work / "recap.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                        "-i", str(manifest), "-c", "copy", "-movflags", "+faststart", str(assembled)], check=True)
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_format", "-show_streams",
                                                   "-of", "json", str(assembled)]))
        duration = float(probe["format"]["duration"])
        expected = sum(e["frames"] for e in sequence) / FPS
        if not 0 < duration <= 240 or abs(duration - expected) > .1:
            raise ValueError(f"unexpected Recap duration: {duration}")
        temporary = output.with_suffix(".mp4.tmp")
        try:
            shutil.copyfile(assembled, temporary)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)


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
        cameras.update(play["primary_cam"] for play in selection.get("plays") or []
                       if isinstance(play, dict) and play.get("primary_cam"))
        layouts = load_layouts(root, cameras)
        rois = load_rois(root)
        source_fps, horizon, probe_flags = {}, {}, []
        for cam in sorted(cameras):
            path = _first_file(layouts.get(cam) or [])
            if not path:
                continue
            try:
                source_fps[cam] = probe_fps(path)
            except Exception:
                probe_flags.append(f"{cam}: could not probe frame rate; replay slow motion defaulted")
            angle = horizon_angle(path)
            if angle:
                probe_flags.append(f"{cam}: horizon levelled by {-angle:+.1f} degrees")
            horizon[cam] = angle
        plan = plan_recap(selection, layouts, float(options.get("live_play_speed", 1.1)),
                          rois=rois, source_fps=source_fps, horizon=horizon)
        plan["flags"] = probe_flags + plan["flags"]
        plan["output"] = name
        verify_sources(plan)
        render(plan, root / name)
        for stale in sorted(root.glob("*_Recap.mp4")):
            if stale.name != name and not stale.name.startswith("._"):
                plan["flags"].append(f"older Recap kept: {stale.name}")
        temporary = root / "recap_assembly.json.tmp"
        temporary.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
        temporary.replace(root / "recap_assembly.json")
        for flag in plan["flags"]:
            print(f"[assembly] flag: {flag}")
        play_count = sum(1 for c in plan["clips"] if c.get("kind") == "play")
        print(f"[assembly] {name}: {plan['duration_s']:.2f}s, "
              f"{len(plan['clips']) - play_count} goals, {play_count} non-goal plays")
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(f"[ERROR] assembly: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
