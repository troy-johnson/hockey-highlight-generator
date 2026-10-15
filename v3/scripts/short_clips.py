"""Plan and render a Short and up to three single-goal clips (spec 002 section 5.11)."""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import json
import subprocess
import tempfile
import hashlib
import copy

sys.path.insert(0, os.path.dirname(__file__))

from recap_assembly import FPS, source_parts, load_play_reviews


def frame_play(motion: list[list[float]]) -> dict:
    """Measure action fit and smooth centers using preceding and following samples."""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    if not motion:
        return {"layout": "1:1", "tracking": [[0., .5, .5]], "fit": 0.,
                "flags": ["no measured motion; using centered 1:1 window"]}
    data = np.asarray(motion, dtype=float)
    if data.ndim != 2 or data.shape[1] != 5 or not np.isfinite(data).all():
        raise ValueError("invalid motion samples")
    if (np.diff(data[:, 0]) < 0).any() or (data[:, 3:5] < data[:, 1:3]).any():
        raise ValueError("motion samples must be ordered boxes")
    centers = gaussian_filter1d((data[:, 1:3] + data[:, 3:5]) / 2, sigma=3, axis=0, mode="nearest")
    layout, fit = "1:1", 0.
    for name, aspect in (("9:16", 9/16), ("4:5", 4/5), ("1:1", 1.)):
        width = aspect / (16/9)
        cx = np.clip(centers[:, 0], width / 2, 1 - width / 2)
        fits = (data[:, 1] >= cx - width / 2) & (data[:, 3] <= cx + width / 2)
        fit = float(np.mean(fits))
        layout = name
        if fit >= .85:
            break
    centers[:, 0] = cx
    centers[:, 1] = .5
    return {"layout": layout, "fit": fit, "tracking":
            [[float(t), float(x), float(y)] for t, (x, y) in zip(data[:, 0], centers)], "flags": []}


def short_sequence(plan: dict) -> list[dict]:
    """Return the shared video, audio, and graphics order."""
    sequence = list(plan["opening"])
    for clip in plan["clips"]:
        sequence.append(clip)
        sequence.extend(r for r in plan["replays"] if r["goal_id"] == clip["goal_id"])
    return sequence + [plan["final"]]


def single_goal_plan(plan: dict, goal_id: str) -> dict:
    """Use the same framing, Bed, and FINAL for a separate goal output."""
    result = copy.deepcopy(plan)
    result["opening"] = []
    result["clips"] = [c for c in result["clips"] if c["goal_id"] == goal_id]
    result["replays"] = [c for c in result["replays"] if c["goal_id"] == goal_id]
    if len(result["clips"]) != 1 or result["clips"][0]["kind"] != "goal":
        raise ValueError("single-goal output needs a selected goal")
    result["bed_offset_s"] = 0.
    result["duration_s"] = sum(e["frames"] for e in short_sequence(result)) / FPS
    return result


def measure_motion(entry: dict, cache: Path) -> list[list[float]]:
    """Cache measured optical-flow spread for the selected source cuts."""
    import cv2
    import numpy as np
    identities = []
    for part in entry["parts"]:
        stat = Path(part["file"]).stat()
        identities.append([part, stat.st_size, stat.st_mtime_ns])
    algorithm = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    key = hashlib.sha256(json.dumps([algorithm, identities], sort_keys=True).encode()).hexdigest()
    cache = Path(cache)
    path = cache / f"{key}.json"
    if path.exists():
        try:
            saved = json.loads(path.read_text())
            frame_play(saved)
            return saved
        except (OSError, ValueError, TypeError):
            pass
    samples, elapsed, previous = [], 0., None
    for part in entry["parts"]:
        decoder = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(part["seek_s"]), "-i", part["file"],
            "-t", str(part["duration_s"]), "-an", "-vf", "fps=5,scale=320:180,format=gray",
            "-f", "rawvideo", "pipe:1"], stdout=subprocess.PIPE)
        assert decoder.stdout is not None
        try:
            while True:
                raw = decoder.stdout.read(320 * 180)
                if len(raw) != 320 * 180:
                    break
                image = np.frombuffer(raw, np.uint8).reshape(180, 320)
                box = [0., 0., 1., 1.]
                if previous is not None:
                    flow = cv2.calcOpticalFlowFarneback(previous, image, None, .5, 3, 15, 3, 5, 1.2, 0)
                    magnitude = np.linalg.norm(flow, axis=2)
                    ys, xs = np.nonzero(magnitude > max(.5, float(np.percentile(magnitude, 90))))
                    if len(xs) >= 20:
                        x0, x1 = np.quantile(xs / 320, [.05, .95])
                        y0, y1 = np.quantile(ys / 180, [.05, .95])
                        box = [float(x0), float(y0), float(x1), float(y1)]
                samples.append([round(elapsed, 6), *box])
                previous = image
                elapsed += .2
        finally:
            decoder.stdout.close()
            rc = decoder.wait()
        if rc:
            raise RuntimeError("Short motion decode failed")
        previous = None
    cache.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(samples))
    tmp.replace(path)
    return samples


def plan_short(selection: dict, layouts: dict, *, rois=None, sheet=None, bar_s=2., focus_side=None, coverage=None) -> dict:
    """Choose the five highest-interest covered plays, then restore game order."""
    from recap_graphics import goal_data, final_data
    if not math.isfinite(bar_s) or not .5 <= bar_s <= 8:
        raise ValueError("Short Bed bar duration must be between 0.5 and 8 seconds")
    sheet = sheet or {}
    bar_frames = round(bar_s * FPS)
    candidates, flags = [], []
    for goal in selection.get("goals") or []:
        choice = goal.get("chosen")
        if not choice:
            flags.append(f"{goal['id']}: uncovered goal excluded from Short")
            continue
        if focus_side and goal["id"].split(":")[0] != focus_side:
            continue
        candidates.append(dict(choice, id=goal["id"], kind="goal", interest=goal.get("interest", 0)))
    for play in selection.get("plays") or []:
        if play.get("review_status") in ("reject", "uncertain") or (
                play.get("kind") == "net_play" and play.get("review_status") != "keep"):
            flags.append(f"{play['id']}: non-goal candidate requires a keep review; excluded from Short")
            continue
        side = play.get("side")
        if play["id"].startswith("penalty:"):
            side = play["id"].split(":")[1]
        if focus_side and not side and play["kind"] == "net_play" and coverage:
            from selection import scoring_camera
            period = next((p["n"] for p in selection.get("periods", [])
                           if p["start"] <= play["detection_s"] < p["end"]), None)
            sides = [s for s in ("home", "away") if period is not None and
                     scoring_camera(coverage, period, s, (selection.get("mapping") or {}).get("defender_teams") or {}) == play["primary_cam"]]
            if len(sides) == 1:
                side = sides[0]
                flags.append(f"{play['id']}: attacking team inferred from period net mapping")
        if focus_side and side and side != focus_side:
            continue
        if focus_side and not side:
            flags.append(f"{play['id']}: Focus Team participation unverified; excluded from Short")
            continue
        candidates.append(dict(play, interest=play.get("interest", play.get("score", 0))))
    ranked = sorted(candidates, key=lambda c: (-c["interest"], c["detection_s"], c["id"]))
    clips, replays = [], []
    for item in ranked:
        cam, moment = item["primary_cam"], item["detection_s"]
        block = next((b for b in layouts.get(cam, []) if b["start"] <= moment < b["end"]), None)
        if block is None:
            flags.append(f"{item['id']}: no Recording coverage; excluded from Short")
            continue
        if block["end"] - block["start"] < 2.:
            flags.append(f"{item['id']}: insufficient Recording coverage; excluded from Short")
            continue
        start, end = max(block["start"], moment - 3), min(block["end"], moment + 3)
        clip = {"goal_id": item["id"], "kind": item["kind"], "camera": cam,
                "moment_s": moment, "start_s": start, "end_s": end, "speed": 1.,
                "frames": round((end - start) * FPS), "parts": source_parts(layouts[cam], start, end),
                "interest": item["interest"], "caption": None, "score": [0, 0]}
        if item["kind"] == "goal" and sheet:
            data = goal_data(sheet, item["id"])
            clip["caption"] = f"GOAL {data['scorer']}"
        elif item["kind"] == "penalty":
            clip["caption"] = item.get("caption") or item.get("label") or "PENALTY"
        goals = selection.get("goals") or []

        def expected(goal):
            return (goal.get("chosen") or {}).get("detection_s", (goal.get("window") or {}).get("expected"))
        if any(g.get("period") is None or g.get("elapsed_s") is None for g in goals):
            # An unknown sheet clock cannot be ordered against the others: use camera time,
            # as recap_graphics does (a goal with no period sorted last counted too early).
            ordered = sorted(goals, key=lambda g: (expected(g) is None,
                             expected(g) if expected(g) is not None else math.inf,
                             int(g["id"].split(":")[-1])))
        else:
            ordered = sorted(goals, key=lambda g: (g["period"], g["elapsed_s"], int(g["id"].split(":")[-1])))
        if item["kind"] == "goal":
            prior = ordered[:next((i + 1 for i, g in enumerate(ordered) if g["id"] == item["id"]), 0)]
        else:
            # Every goal in game order up to the last one before this moment, so a goal with
            # no source moment still counts once a later goal has (the score never goes down).
            prior = ordered[:max((i + 1 for i, g in enumerate(ordered)
                                  if expected(g) is not None and expected(g) <= moment), default=0)]
        for goal in prior:
            side = goal["id"].split(":")[0]
            if side in ("home", "away"):
                clip["score"][side == "away"] += 1
        clip["score_before"] = list(clip["score"])
        if item["kind"] == "goal":
            clip["score_before"][item["id"].startswith("away:")] -= 1
        clip["period"] = item.get("period")
        clips.append(clip)
        if item["kind"] == "goal":
            a, b = max(block["start"], moment - .25), min(block["end"], moment + 1.75)
            replays.append(dict(clip, kind="replay", start_s=a, end_s=b, speed=.5,
                                parts=source_parts(layouts[cam], a, b), frames=round((b - a) * 2 * FPS),
                                layout="9:16", goal_box=(rois or {}).get(cam, {}).get("net")))
    if len(clips) < 3:
        raise ValueError("Short needs at least three covered plays")
    # Budget whole bars before cutting sources. The hook ends on the first bar.
    preferred, replay_bars = math.ceil(6 / bar_s), math.ceil(4 / bar_s)
    pool, selected, best_rank = clips, None, None
    bars = list(range(preferred, 0, -1)) + list(range(preferred + 1, math.ceil(53 / (3 * bar_s)) + 1))
    for count in range(min(5, len(pool)), 2, -1):
        for live_bars in bars:
            eligible = []
            for clip in pool:
                block = next(b for b in layouts[clip["camera"]] if b["start"] <= clip["moment_s"] < b["end"])
                required = max(live_bars * bar_s, replay_bars * bar_s * .5 if clip["kind"] == "goal" else 2.)
                if block["end"] - block["start"] >= required + 1/FPS:
                    eligible.append(clip)
            chosen = eligible[:count]
            goal_count = sum(c["kind"] == "goal" for c in chosen)
            budget = 7 + (count * live_bars + goal_count * replay_bars) * bar_s
            if len(chosen) == count and 30 <= budget <= 60:
                rank = tuple(-pool.index(c) for c in chosen)
                if best_rank is None or rank > best_rank:
                    selected, best_rank = (chosen, live_bars), rank
    if selected is None:
        raise ValueError("Bed bars cannot fit Short duration budget")
    clips, live_bars = selected
    ids = {c["goal_id"] for c in clips}
    replays = [r for r in replays if r["goal_id"] in ids]
    flags.extend(f"{c['goal_id']}: coverage or duration budget excluded play" for c in pool if c["goal_id"] not in ids)
    # Round cumulative cut times, not each bar, to avoid accumulating frame error.
    cursor_bars, previous_frame = 0, 0
    for clip in sorted(clips, key=lambda c: (c["moment_s"], c["goal_id"])):
        entries = [clip] + [r for r in replays if r["goal_id"] == clip["goal_id"]]
        for entry in entries:
            cursor_bars += replay_bars if entry["kind"] == "replay" else live_bars
            next_frame = round(cursor_bars * bar_s * FPS)
            entry["frames"] = next_frame - previous_frame
            previous_frame = next_frame
    for entry in clips + replays:
        duration = entry["frames"] / FPS * entry["speed"]
        block = next(b for b in layouts[entry["camera"]] if b["start"] <= entry["moment_s"] < b["end"])
        if block["end"] - block["start"] < duration:
            raise ValueError(f"{entry['goal_id']}: insufficient coverage for a bar-aligned cut")
        center = entry["moment_s"] + (.75 if entry["kind"] == "replay" else 0)
        entry["start_s"] = max(block["start"], min(center - duration / 2, block["end"] - duration))
        entry["end_s"] = entry["start_s"] + duration
        entry["parts"] = source_parts(layouts[entry["camera"]], entry["start_s"], entry["end_s"])
    hook: dict = dict(clips[0], kind="hook", start_s=clips[0]["moment_s"] - .5,
                      end_s=clips[0]["moment_s"] + 1.5, frames=60)
    block = next(b for b in layouts[hook["camera"]] if b["start"] <= hook["moment_s"] < b["end"])
    hook["start_s"] = max(block["start"], min(hook["start_s"], block["end"] - 2))
    hook["end_s"] = hook["start_s"] + 2
    hook["parts"] = source_parts(layouts[hook["camera"]], hook["start_s"], hook["end_s"])
    singles = [c["goal_id"] for c in clips if c["kind"] == "goal"][:3]
    clips.sort(key=lambda c: (c["moment_s"], c["goal_id"]))
    final = dict(final_data(sheet, flags), kind="final", frames=150, teams=sheet.get("teams") or {})
    frames = hook["frames"] + sum(c["frames"] for c in clips + replays) + final["frames"]
    return {"schema_version": 1, "width": 1080, "height": 1920, "fps": FPS,
            "opening": [hook], "clips": clips, "replays": replays, "final": final,
            "duration_s": frames / FPS, "single_goals": singles, "flags": flags,
             "bar_frames": bar_frames, "bar_s": bar_s, "bed_offset_s": round((-2) % bar_s, 6),
            "teams": sheet.get("teams") or {}}


def _text(frame, text, x, y, scale=1., color=(245, 245, 245)):
    import cv2
    cv2.putText(frame, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def score_at(entry: dict, elapsed_s: float) -> list[int]:
    """Change the live score at the source goal moment, never during a replay."""
    if entry["kind"] in ("goal", "hook") and entry.get("score_before") is not None:
        if entry["start_s"] + elapsed_s * entry["speed"] < entry["moment_s"]:
            return entry["score_before"]
    return entry.get("score", [0, 0])


def replay_mode(source_fps: float | None, interpolation_available: bool) -> str:
    """Use the Recap native half-speed gate and its optional RIFE fallback."""
    from recap_assembly import NATIVE_SLOW_FPS
    if source_fps is not None and source_fps >= NATIVE_SLOW_FPS:
        return "native"
    return "rife" if interpolation_available else "duplicated"


def _overlay(frame, entry, plan, elapsed_s=0.):
    import cv2
    teams = plan.get("teams") or {}
    frame[:180] = (25, 25, 32)
    score = score_at(entry, elapsed_s)
    _text(frame, f"{teams.get('home', 'HOME')}  {score[0]}", 40, 65, 1.1)
    _text(frame, f"{teams.get('away', 'AWAY')}  {score[1]}", 40, 135, 1.1)
    if entry["kind"] == "final":
        _text(frame, "FINAL", 40, 330, 2.)
        table = entry.get("table") or {}
        for column, side in enumerate(("home", "away")):
            x, y = 30 + column * 540, 440
            _text(frame, teams.get(side, side.upper())[:22], x, y, .9)
            y += 55
            _text(frame, "PLAYER", x, y, .7)
            y += 45
            data = table.get(side) or {}
            rows = data.get("skaters", [])
            step = min(65, 1250 / max(1, len(rows) + len(data.get("goalies", []))))
            scale = min(.6, step / 100)
            for row in rows:
                _text(frame, f"#{row.get('num', '')} {row.get('name', '')}"[:25], x, round(y), scale)
                values = [str(row[k]) if row.get(k) is not None else "?" for k in ("g", "a", "pts", "pim")]
                _text(frame, "G/A/PTS/PIM " + "/".join(values), x, round(y + step*.4), scale*.8)
                y += step
            for goalie in data.get("goalies", []):
                _text(frame, f"#{goalie['num']} SAVES {goalie['saves']}", x, round(y), scale)
                y += step
        return
    if entry.get("caption"):
        cv2.rectangle(frame, (0, 1750), (1080, 1920), (25, 25, 32), -1)
        _text(frame, entry["caption"][:48], 40, 1840, 1.)
    if entry["kind"] == "replay":
        _text(frame, "REPLAY", 790, 230, .8)


def _window(image, entry, elapsed):
    import cv2
    import numpy as np
    h, w = image.shape[:2]
    framing = entry.get("framing") or {"layout": "1:1", "tracking": [[0, .5, .5]]}
    aspect = {"9:16": 9/16, "4:5": .8, "1:1": 1.}[framing["layout"]]
    tracking = np.asarray(framing["tracking"])
    x = float(np.interp(elapsed, tracking[:, 0], tracking[:, 1]))
    y = float(np.interp(elapsed, tracking[:, 0], tracking[:, 2]))
    ch = h
    if entry["kind"] == "replay" and entry.get("goal_box"):
        bx, by, bw, bh = entry["goal_box"]
        x, y = (bx + bw / 2) / 1280, (by + bh / 2) / 720
        ch = min(h, max(h * .25, bh / 720 * h * 2.5, bw / 1280 * w * 1.5 / aspect))
    cw = min(w, ch * aspect)
    a = int(max(0, min(x * w - cw / 2, w - cw)))
    b = int(max(0, min(y * h - ch / 2, h - ch)))
    crop = image[b:b + int(ch), a:a + int(cw)]
    target_h = round(1080 / aspect)
    resized = cv2.resize(crop, (1080, target_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((1920, 1080, 3), (12, 12, 18), dtype=np.uint8)
    top = (1920 - target_h) // 2
    canvas[top:top + target_h] = resized
    return canvas


def render_short(plan: dict, target: Path) -> None:
    """Render frame-counted cuts with an offline camera and vertical graphics."""
    import numpy as np
    target = Path(target)
    with tempfile.TemporaryDirectory(prefix="short-", dir=target.parent) as scratch:
        output = Path(scratch) / "video.mp4"
        encoder = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", "1080x1920", "-r", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", str(output)], stdin=subprocess.PIPE)
        assert encoder.stdin is not None
        try:
            for entry in short_sequence(plan):
                count = 0
                if entry["kind"] == "final":
                    frame = np.full((1920, 1080, 3), (12, 12, 18), dtype=np.uint8)
                    _overlay(frame, entry, plan)
                    for _ in range(entry["frames"]):
                        encoder.stdin.write(frame.tobytes())
                    continue
                if entry.get("slowmo") == "rife":
                    from recap_assembly import rife_ready
                    if not rife_ready():
                        entry["slowmo"] = "duplicated"
                        plan["flags"].append(f"{entry['goal_id']}: RIFE unavailable at render time; duplicate replay frames")
                for index, part in enumerate(entry["parts"]):
                    file, seek, duration, speed = part["file"], part["seek_s"], part["duration_s"], entry["speed"]
                    if entry.get("slowmo") == "rife":
                        from recap_assembly import _rife_slowmo, probe_fps
                        native, slow = Path(scratch) / "native.mp4", Path(scratch) / "slow.mp4"
                        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(seek), "-i", file,
                            "-t", str(duration), "-an", "-vf", "scale=1920:-2", "-c:v", "libx264",
                            "-preset", "fast", str(native)], check=True)
                        rate = math.ceil(probe_fps(file) or FPS)
                        _rife_slowmo(native, slow, Path(scratch) / f"rife-{entry['goal_id']}-{index}",
                                     {"frames": round(duration / speed * FPS)}, rate)
                        file, seek, duration, speed = str(slow), 0., duration / speed, 1.
                    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "json", file]))["streams"][0]
                    w, h = probe["width"], probe["height"]
                    # Each part fills its own share of the entry. A fractional seek can make
                    # the fps filter drop one frame, so decode 2 frames more and stop at the share.
                    last_part = index == len(entry["parts"]) - 1
                    part_end = entry["frames"] if last_part else min(entry["frames"],
                                                                   count + round(duration / speed * FPS))
                    decoder = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(seek),
                        "-i", file, "-t", str(duration + 2 * speed / FPS), "-an", "-vf",
                        f"fps={FPS / speed}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], stdout=subprocess.PIPE)
                    assert decoder.stdout is not None
                    try:
                        while count < part_end:
                            raw = decoder.stdout.read(w * h * 3)
                            if len(raw) != w * h * 3:
                                break
                            image = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
                            frame = _window(image, entry, count / FPS * entry["speed"])
                            _overlay(frame, entry, plan, count / FPS)
                            encoder.stdin.write(frame.tobytes())
                            count += 1
                    finally:
                        decoder.stdout.close()
                        decoder.wait()
                if count != entry["frames"]:
                    raise ValueError(f"{entry['goal_id']}: decoded {count}, expected {entry['frames']} frames")
        finally:
            encoder.stdin.close()
            rc = encoder.wait()
        if rc:
            raise RuntimeError("Short video encoding failed")
        output.replace(target)


def prepare_audio(plan: dict, bed: dict | None, spans: list[dict], *, cues=None) -> tuple:
    """Mix one looping Bed with PA-muted rink audio twelve decibels below it."""
    import numpy as np
    from audio_mix import (SR, decode_audio, fit_to_length, loop_bars, bar_seconds,
                           synth_placeholder_bed, mute_windows, _entry_audio, _faded_zero, duck_envelope)
    flags = []
    n = round(plan["duration_s"] * SR)
    origin = float((bed.get("beats") or [0.])[0]) if bed else 0.
    if bed:
        duration = loop_bars(bed["duration_s"] - origin, bar_seconds(bed))
        buffer = decode_audio(bed["file"], origin, duration)
        if len(buffer) < round(duration * SR) - SR / 10:
            raise ValueError("Short Bed is shorter than its measured grid")
        buffer = fit_to_length(buffer, round(duration * SR))
    else:
        buffer = synth_placeholder_bed()
        flags.append("no approved Short Bed; synthetic placeholder music")
    if len(buffer) == 0 or float(np.sqrt(np.mean(buffer ** 2))) < 1e-8:
        raise ValueError("Short Bed is silent")
    offset = round(plan.get("bed_offset_s", 0) * SR)
    music = buffer[(np.arange(n) + offset) % len(buffer)].copy()
    music *= .12 / float(np.sqrt(np.mean(music ** 2)))
    rink = np.zeros_like(music)
    position, muted = 0, 0
    for entry in short_sequence(plan):
        length = round(entry["frames"] / FPS * SR)
        if entry.get("parts"):
            e = copy.deepcopy(entry)
            cursor = e["start_s"]
            for part in e["parts"]:
                part["tl_start"] = cursor
                cursor += part["duration_s"]
            windows = mute_windows(e, spans)
            muted += len(windows)
            try:
                audio = fit_to_length(_entry_audio(e, []), length)
                rms = float(np.sqrt(np.mean(audio ** 2)))
                if rms > 1e-8:
                    audio *= .12 * 10 ** (-12 / 20) / rms
                for a, b in windows:
                    _faded_zero(audio, max(0, round(a / e["speed"] * SR)),
                                min(length, round(b / e["speed"] * SR)), round(.01 * SR))
                rink[position:position + length] = audio
            except RuntimeError as exc:
                flags.append(f"{entry['goal_id']}: rink audio unavailable: {exc}")
        position += length
    events = list(cues or [])
    cursor = 0.
    for entry in short_sequence(plan):
        if entry["kind"] == "goal" or (entry["kind"] == "hook" and entry["goal_id"].startswith(("home:", "away:"))):
            t = cursor + (entry["moment_s"] - entry["start_s"]) / entry["speed"]
            events.append({"t": t, "dur_s": .8})
        cursor += entry["frames"] / FPS
    music *= duck_envelope(n, SR, events)[:, None]
    master = music + rink
    from audio_mix import _place
    for event in events:
        if "buffer" in event:
            _place(master, event["buffer"], event["out_start"])
    return master, {"schema_version": 1, "bed": bed["id"] if bed else None,
                    "placeholder": bed is None, "muted_windows": muted, "rink_under_bed_db": 12.,
                    "flags": flags, "duration_s": plan["duration_s"], "bed_source_start_s": origin,
                    "duck_events": [{k: e[k] for k in ("t", "dur_s") if k in e} for e in events]}


def mix_short(plan: dict, video: Path, output: Path, bed: dict | None, spans: list[dict], *, cues=None) -> dict:
    """Use measured two-pass loudnorm and check the encoded AAC track."""
    from audio_mix import SR, _measure_loudness, _probe_duration
    master, report = prepare_audio(plan, bed, spans, cues=cues)
    with tempfile.TemporaryDirectory(prefix="short-mix-", dir=Path(output).parent) as scratch:
        raw = Path(scratch) / "audio.f32"
        master.astype("float32").tofile(raw)
        measured = _measure_loudness(raw, True, tp=-1.35)
        if not math.isfinite(measured.get("input_i", float("nan"))):
            raise ValueError("Short audio loudness measurement failed")
        af = (f"loudnorm=I=-14:TP=-1.35:LRA=11:measured_i={measured['input_i']}"
              f":measured_tp={measured['input_tp']}:measured_lra={measured['input_lra']}"
              f":measured_thresh={measured['input_thresh']}:offset={measured['target_offset']}:linear=true,"
              "alimiter=limit=0.85:level=false")
        target = Path(scratch) / "mix.mp4"
        i, tp = float("nan"), float("nan")
        for _ in range(3):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-f", "f32le",
                "-ar", str(SR), "-ac", "2", "-i", str(raw), "-map", "0:v", "-map", "1:a",
                "-af", af, "-ar", str(SR), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart", "-shortest", str(target)], check=True)
            final = _measure_loudness(target, False)
            i, tp = final.get("input_i", float("nan")), final.get("input_tp", float("nan"))
            if not math.isfinite(i) or not math.isfinite(tp):
                raise ValueError("encoded Short loudness measurement failed")
            if tp <= -1 and i <= -13.5:
                break
            af += f",volume={min(-1.3-tp, -14.2-i):.3f}dB"
        if abs(_probe_duration(target) - plan["duration_s"]) > .15:
            raise ValueError("Short audio and video duration differ")
        subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(target), "-f", "null", "-"], check=True)
        report["loudnorm"] = {"measured": measured, "output_i": i, "output_tp": tp}
        if not -15.5 <= i <= -13.5 or tp > -1:
            report["flags"].append(f"Short loudness off target: {i:.2f} LUFS / {tp:.2f} dBTP")
        target.replace(output)
    report["output"] = Path(output).name
    return report


def main(argv=None) -> int:
    import argparse
    from recap_assembly import load_layouts, load_rois, output_name, probe_fps
    from cue_library import load_manifest, select_beds, record_rotation, write_json
    from audio_mix import DEFAULT_MANIFEST, bar_seconds
    parser = argparse.ArgumentParser(description="Render a Short and up to three single-goal clips")
    parser.add_argument("game_folder")
    parser.add_argument("--options", default="{}")
    args = parser.parse_args(argv)
    root = Path(args.game_folder).resolve()
    try:
        options = json.loads(args.options)
        selection = load_play_reviews(root, json.loads((root / "selection.json").read_text()))
        selection["plays"] = [p for p in selection.get("plays", [])
                              if p.get("review_status") not in ("reject", "uncertain")]
        sheet = json.loads((root / "game_sheet.json").read_text())
        flags = list(selection.get("flags") or []) + list(selection.get("input_flags") or [])
        manifest_path = Path((options.get("audio") or {}).get("manifest") or DEFAULT_MANIFEST).expanduser()
        key = f"{options.get('date', sheet.get('date', 'unknown'))}|{sheet['teams']['home']}-vs-{sheet['teams']['away']}|Short"
        choices, bed = {}, None
        history = manifest_path.parent / "short_rotation.json"
        try:
            library = load_manifest(manifest_path)
            beds = [dict(b, file=str(manifest_path.parent / b["file"])) for b in library["beds"]]
            choices, notes = select_beds(beds, key, [1], history)
            flags.extend(notes)
            bed = choices.get(1)
        except (OSError, ValueError) as exc:
            flags.append(f"approved Short Bed unavailable: {exc}")
        cameras = ({g["chosen"]["primary_cam"] for g in selection.get("goals", []) if g.get("chosen")}
                   | {p["primary_cam"] for p in selection.get("plays", [])})
        layouts = load_layouts(root, cameras)
        from audio_mix import resolve_focus_side
        focus_name = options.get("focus_team")
        focus_name = ((options.get("teams") or {}).get(focus_name) or {}).get("name", focus_name)
        focus_side, focus_flag = resolve_focus_side("focus", focus_name, sheet)
        if focus_flag:
            flags.append(focus_flag)
        structure = json.loads((root / "coverage.json").read_text()) if (root / "coverage.json").exists() else {}
        plan = plan_short(selection, layouts, rois=load_rois(root), sheet=sheet,
                          bar_s=bar_seconds(bed), focus_side=focus_side, coverage=structure.get("coverage"))
        plan["flags"].extend(flags)
        for entry in plan["opening"] + plan["clips"] + plan["replays"]:
            if entry["kind"] == "replay":
                entry["framing"] = {"layout": "9:16", "tracking": [[0, .5, .5]], "fit": 1., "flags": []}
                if not entry.get("goal_box"):
                    plan["flags"].append(f"{entry['goal_id']}: missing goal ROI; replay uses full-height 9:16")
                from recap_assembly import rife_ready
                rates = [probe_fps(p["file"]) for p in entry["parts"]]
                known_rates = [r for r in rates if r is not None]
                entry["slowmo"] = replay_mode(min(known_rates) if len(known_rates) == len(rates) else None, rife_ready())
                if entry["slowmo"] == "duplicated":
                    plan["flags"].append(f"{entry['goal_id']}: RIFE unavailable; replay duplicates frames below the native 60 fps gate")
            else:
                entry["framing"] = frame_play(measure_motion(entry, root / ".recap_cache" / "short_motion"))
                plan["flags"].extend(entry["framing"]["flags"])
        for side in ("home", "away"):
            if not plan["final"]["table"][side]["goalies"]:
                plan["flags"].append(f"{side}: goalie saves unavailable; omitted from FINAL")
        try:
            spans = json.loads((root / "music_spans.json").read_text()).get("spans", [])
        except (OSError, ValueError):
            spans = []
            plan["flags"].append("PA music spans unavailable; rink music muting unverified")
        base = Path(output_name(root, dict(options, focus_team=focus_name), sheet)).stem
        compilation = f"{base}_short.mp4"
        outputs = [(compilation, plan)]
        for index, goal in enumerate(plan["single_goals"], 1):
            outputs.append((f"{base}_goal_{index}.mp4", single_goal_plan(plan, goal)))
        reports = []
        with tempfile.TemporaryDirectory(prefix="short-output-", dir=root) as scratch:
            for name, output_plan in outputs:
                video = Path(scratch) / "plain.mp4"
                render_short(output_plan, video)
                reports.append(mix_short(output_plan, video, Path(scratch) / name, bed, spans))
            for name, _ in outputs:
                (Path(scratch) / name).replace(root / name)
        plan["outputs"] = [name for name, _ in outputs]
        plan["bed"] = bed
        plan["flags"].extend(f for report in reports for f in report["flags"])
        plan["flags"] = list(dict.fromkeys(plan["flags"]))
        report = {"schema_version": 1, "outputs": reports, "placeholder": bed is None, "flags": plan["flags"]}
        write_json(root / "short_assembly.json", plan)
        write_json(root / "short_audio.json", report)
        if bed:
            try:
                record_rotation(history, key, choices)
            except (OSError, ValueError) as exc:
                plan["flags"].append(f"Short rotation could not be saved: {exc}")
                report["flags"] = plan["flags"]
                write_json(root / "short_audio.json", report)
                write_json(root / "short_assembly.json", plan)
        for flag in plan["flags"]:
            print(f"[short] flag: {flag}")
        print(f"[short] {compilation}: {plan['duration_s']:.1f}s; {len(outputs)-1} single-goal clips")
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[ERROR] Short: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
