import re
import traceback
import subprocess
import time

WIN_RE = re.compile(r"\bwin\s*=\s*([0-9]+(?:\.[0-9]+)?)s\b", re.IGNORECASE)
COLOR_RE = re.compile(r"\[([A-Za-z]+)\]")
CAM_RE = re.compile(r"\bcam(\d+)\b", re.IGNORECASE)

def _log(msg: str):
    print(f"[expand_markers] {msg}")

def _get_resolve():
    try:
        if res is not None:
            return res
    except NameError:
        pass
    try:
        if resolve is not None:
            return resolve
    except NameError:
        pass
    try:
        import DaVinciResolveScript as dvr
        return dvr.scriptapp("Resolve")
    except Exception:
        return None

def _get_project_and_timeline(resolve):
    try:
        pm = resolve.GetProjectManager()
        proj = pm.GetCurrentProject() if pm else None
        tl = proj.GetCurrentTimeline() if proj else None
        return proj, tl
    except Exception:
        return None, None

def _get_timeline_fps(proj) -> float:
    for key in ("timelineFrameRate", "timelineFrameRateFloat", "timelineFrameRatePlayback", "videoFrameRate"):
        try:
            v = proj.GetSetting(key)
            if v:
                return float(v)
        except Exception:
            pass
    return 60.0

def _get_timeline_start_frame(tl) -> int:
    try:
        return int(tl.GetStartFrame())
    except Exception:
        return 0

def _parse_win_seconds(note: str):
    if not note:
        return None
    m = WIN_RE.search(note)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def _parse_color(name: str, note: str) -> str:
    for text in (name, note):
        if text:
            m = COLOR_RE.search(text)
            if m:
                return m.group(1).capitalize()
    return "Blue"

def _parse_cam_angle(name: str, note: str) -> int:
    """Returns 0-based angle index: cam1 -> 0, cam2 -> 1, etc."""
    for text in (name, note):
        if text:
            m = CAM_RE.search(text)
            if m:
                return int(m.group(1)) - 1
    return 0

def _frame_to_tc(frame: int, fps: float) -> str:
    fps_int = int(round(fps))
    f = frame % fps_int
    s = (frame // fps_int) % 60
    mn = (frame // fps_int // 60) % 60
    h = frame // fps_int // 3600
    return f"{h:02d}:{mn:02d}:{s:02d}:{f:02d}"

def _blade_at_frame(tl, frame: int, fps: float):
    tc = _frame_to_tc(frame, fps)
    tl.SetCurrentTimecode(tc)
    time.sleep(0.2)
    script = '''
tell application "DaVinci Resolve" to activate
delay 0.2
tell application "System Events"
    tell process "DaVinci Resolve"
        key code 42 using control down
    end tell
end tell
'''
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if result.returncode != 0:
        _log(f"  osascript error: {result.stderr.strip()}")

def _blade_all(tl, split_frames, fps):
    if not split_frames:
        return
    unique_frames = sorted(set(split_frames))
    _log(f"Blading at {len(unique_frames)} points.")
    _log("Blading starts in 5 seconds — click the EDITOR window now...")
    time.sleep(5)
    for frame in unique_frames:
        tc = _frame_to_tc(frame, fps)
        _log(f"  Blading frame {frame} ({tc})")
        _blade_at_frame(tl, frame, fps)
    time.sleep(0.5)

def _set_angles(tl, ranges: list):
    """
    For each (start_frame, end_frame, angle_index) in ranges, find the clip
    in V1 whose midpoint falls within the range and set its multicam angle.
    Clip start/end are absolute frames (same coordinate space as ranges).
    """
    try:
        clips = tl.GetItemsInTrack("video", 1)
    except Exception as e:
        _log(f"Could not get clips in V1: {e}")
        return 0, 0

    if not clips:
        _log("No clips found in V1.")
        return 0, 0

    set_ok = 0
    set_fail = 0

    for clip in clips.values():
        try:
            clip_start = clip.GetStart()
            clip_end = clip.GetEnd()
            mid = (clip_start + clip_end) / 2
            for (range_start, range_end, angle) in ranges:
                if range_start <= mid < range_end:
                    ok = clip.SetCurrentClipAngle(angle)
                    if ok:
                        set_ok += 1
                    else:
                        set_fail += 1
                    break
        except Exception as e:
            _log(f"  Angle set exception on clip: {e}")
            set_fail += 1

    return set_ok, set_fail

def main():
    resolve = _get_resolve()
    if resolve is None:
        _log("Resolve scripting API not available.")
        return

    proj, tl = _get_project_and_timeline(resolve)
    if proj is None:
        _log("No current project open.")
        return
    if tl is None:
        _log("No active timeline selected. Click your editing timeline, then run again.")
        return

    fps = _get_timeline_fps(proj)
    start_frame = _get_timeline_start_frame(tl)
    _log("Active timeline: " + tl.GetName())
    _log("Timeline FPS (best guess): " + str(fps))
    _log("Timeline start frame offset: " + str(start_frame))

    markers = tl.GetMarkers() or {}
    if not markers:
        _log("No markers found on the active timeline.")
        return

    frames = sorted(int(k) for k in markers.keys())

    expanded = 0
    skipped = 0
    failed = 0
    split_frames = []
    ranges = []

    can_delete = hasattr(tl, "DeleteMarkerAtFrame")

    # ── Phase 1: Expand markers into ranges ──────────────────────────────────
    for frame in frames:
        m = markers.get(frame) or {}
        name = (m.get("name", "") or "").strip()
        note = m.get("note", "") or ""

        if name.upper() == "REPLAY":
            skipped += 1
            continue

        win_s = _parse_win_seconds(note)
        if not win_s or win_s <= 0:
            skipped += 1
            continue

        color = _parse_color(name, note)
        angle = _parse_cam_angle(name, note)
        dur_frames = max(1, int(round(win_s * fps)))
        end_frame = frame + dur_frames

        # Absolute frames for blading and angle-matching
        abs_frame = int(frame) + start_frame
        abs_end = int(end_frame) + start_frame

        try:
            # Marker API uses relative frames
            if can_delete:
                tl.DeleteMarkerAtFrame(int(frame))
            ok = tl.AddMarker(int(frame), color, name, note, int(dur_frames))
            if ok:
                expanded += 1
                split_frames.append(abs_frame)
                split_frames.append(abs_end)
                ranges.append((abs_frame, abs_end, angle))
            else:
                failed += 1
        except Exception:
            failed += 1

    _log(f"Markers expanded: {expanded} | Skipped: {skipped} | Failed: {failed}")

    if not ranges:
        _log("No ranges to process. Done.")
        return

    # ── Phase 2: Blade at range boundaries ───────────────────────────────────
    _log("Starting blade phase...")
    _blade_all(tl, split_frames, fps)
    _log("Blading complete.")

    # ── Phase 3: Set multicam angles ─────────────────────────────────────────
    _log("Setting multicam angles...")
    set_ok, set_fail = _set_angles(tl, ranges)
    _log(f"Angles set: {set_ok} | Failed: {set_fail}")
    _log("All done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _log("ERROR: " + str(e))
        try:
            _log(traceback.format_exc())
        except Exception:
            pass
        
        
        
        
        
        
        
import time
pm = res.GetProjectManager()
proj = pm.GetCurrentProject()
tl = proj.GetCurrentTimeline()

fusion.SetCurrentTimecode("00:52:12:55")
time.sleep(0.3)
print("After: " + str(tl.GetCurrentTimecode()))

print("Before: " + str(tl.GetCurrentTimecode()))
app.SetCurrentTimecode("00:52:12:55")
time.sleep(0.3)
print("After: " + str(tl.GetCurrentTimecode()))

print("Current TC before: " + str(tl.GetCurrentTimecode()))
tl.SetCurrentTimecode("00:52:12:55")
time.sleep(0.3)
print("Current TC after: " + str(tl.GetCurrentTimecode()))