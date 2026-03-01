# import_markers_resolve.py
# Resolve menu script:
# - Looks for markers.fcpxml in the project folder you choose
# - Imports it as a temp timeline
# - Copies markers onto the currently active timeline
# - Optionally clears existing markers (off by default)
# - Optionally de-dupes markers (on by default)
# - Logs to Workspace > Console

import os
import time
import traceback

# ====== SETTINGS ======
DEDUPE = True               # avoid re-adding markers already present at same frame+color+name+note+duration
CLEAR_DEST_FIRST = False    # if True, deletes all markers on active timeline before copying
# ======================

def _log(msg):
    print(f"[import_markers] {msg}", flush=True)

def _get_resolve():
    try:
        import DaVinciResolveScript as dvr
        return dvr.scriptapp("Resolve")
    except Exception:
        return None

def _pick_folder(resolve):
    fusion = resolve.Fusion()
    if fusion is None:
        return None
    try:
        return fusion.RequestDir("Pick the video project folder (contains markers.fcpxml)")
    except Exception:
        return None

def _find_fcpxml(folder):
    p = os.path.join(folder, "markers.fcpxml")
    return p if os.path.isfile(p) else None

def _get_current_project_and_timeline(resolve):
    pm = resolve.GetProjectManager()
    if not pm:
        return None, None
    proj = pm.GetCurrentProject()
    if not proj:
        return None, None
    tl = proj.GetCurrentTimeline()
    return proj, tl

def _timeline_list(proj):
    out = []
    try:
        count = proj.GetTimelineCount()
        for i in range(1, count + 1):
            tl = proj.GetTimelineByIndex(i)
            if tl:
                out.append(tl)
    except Exception:
        pass
    return out

def _timeline_names_set(proj):
    return set([tl.GetName() for tl in _timeline_list(proj) if tl and hasattr(tl, "GetName")])

def _import_fcpxml(media_pool, fcpxml_path):
    # returns either a timeline object or True/False depending on build
    return media_pool.ImportTimelineFromFile(fcpxml_path)

def _get_newly_created_timeline(proj, names_before, maybe_tl_obj=None):
    # Best case: API returned a timeline object
    if maybe_tl_obj is not None and hasattr(maybe_tl_obj, "GetName"):
        return maybe_tl_obj

    # Otherwise: detect which timeline name is new
    after = _timeline_list(proj)
    for tl in after:
        try:
            name = tl.GetName()
        except Exception:
            continue
        if name not in names_before:
            return tl

    # Fallback: last timeline
    try:
        count = proj.GetTimelineCount()
        return proj.GetTimelineByIndex(count)
    except Exception:
        return None

def _delete_timeline(media_pool, tl):
    try:
        return bool(media_pool.DeleteTimelines([tl]))
    except Exception:
        return False

def _markers_signature_dict(tl):
    """
    Build a set of signatures for existing markers to support de-dupe.
    signature = (frameId, color, name, note, duration)
    """
    sigs = set()
    m = tl.GetMarkers() or {}
    for frameId, d in m.items():
        sigs.add((
            int(frameId),
            d.get("color", "Blue"),
            d.get("name", ""),
            d.get("note", ""),
            int(d.get("duration", 1) or 1)
        ))
    return sigs

def _clear_all_markers(tl):
    # Resolve exposes DeleteMarkerAtFrame(frameId) in many builds.
    # We'll attempt it; if missing, we won't clear.
    markers = tl.GetMarkers() or {}
    if not markers:
        return 0
    deleted = 0
    if not hasattr(tl, "DeleteMarkerAtFrame"):
        return 0
    for frameId in list(markers.keys()):
        try:
            ok = tl.DeleteMarkerAtFrame(int(frameId))
            if ok:
                deleted += 1
        except Exception:
            pass
    return deleted

def _copy_markers(src_tl, dst_tl, dedupe=True):
    src = src_tl.GetMarkers() or {}
    if not src:
        return 0, 0

    existing = _markers_signature_dict(dst_tl) if dedupe else set()
    copied = 0
    skipped = 0

    for frameId, m in src.items():
        frameId = int(frameId)
        color = m.get("color", "Blue")
        name = m.get("name", "")
        note = m.get("note", "")
        duration = int(m.get("duration", 1) or 1)

        sig = (frameId, color, name, note, duration)
        if dedupe and sig in existing:
            skipped += 1
            continue

        try:
            ok = dst_tl.AddMarker(frameId, color, name, note, duration)
        except Exception:
            ok = False

        if ok:
            copied += 1
            if dedupe:
                existing.add(sig)
        else:
            skipped += 1

    return copied, skipped

def main():
    resolve = _get_resolve()
    if resolve is None:
        _log("Resolve scripting API not available. Run from Workspace > Scripts.")
        return

    proj, active_tl = _get_current_project_and_timeline(resolve)
    if proj is None:
        _log("No current project open.")
        return
    if active_tl is None:
        _log("No active timeline selected. Click your editing timeline first, then run again.")
        return

    _log(f"Active timeline: {active_tl.GetName()}")

    folder = _pick_folder(resolve)
    if not folder:
        _log("No folder selected. Aborting.")
        return

    fcpxml = _find_fcpxml(folder)
    if not fcpxml:
        _log(f"markers.fcpxml not found in: {folder}")
        _log("Run hockeydetect first (it should create markers.fcpxml).")
        return

    _log(f"Using: {fcpxml}")

    media_pool = proj.GetMediaPool()
    if media_pool is None:
        _log("Could not access Media Pool.")
        return

    # Snapshot timeline names BEFORE import so we can identify the newly created one
    names_before = _timeline_names_set(proj)

    tmp_name = f"__marker_import__{int(time.time())}"
    _log("Importing FCPXML as temporary timeline...")

    res = _import_fcpxml(media_pool, fcpxml)

    tmp_tl = _get_newly_created_timeline(proj, names_before, maybe_tl_obj=res)
    if tmp_tl is None:
        _log("Import failed (no timeline created).")
        return

    # Try renaming to a known temp name
    try:
        tmp_tl.SetName(tmp_name)
    except Exception:
        pass

    # Guard: make sure we didn't accidentally target the active timeline
    if tmp_tl.GetName() == active_tl.GetName():
        _log("ERROR: temp timeline appears to be the active timeline. Aborting to avoid self-copy.")
        return

    _log(f"Temp timeline: {tmp_tl.GetName()}")

    # Optionally clear destination markers
    if CLEAR_DEST_FIRST:
        _log("Clearing existing markers on active timeline...")
        deleted = _clear_all_markers(active_tl)
        _log(f"Deleted markers: {deleted} (0 may mean API doesn't support DeleteMarkerAtFrame)")

    _log("Copying markers to active timeline...")
    copied, skipped = _copy_markers(tmp_tl, active_tl, dedupe=DEDUPE)
    _log(f"Copied: {copied} | Skipped: {skipped}")

    _log("Deleting temp timeline...")
    deleted = _delete_timeline(media_pool, tmp_tl)
    _log(f"Temp timeline deleted: {deleted}")

    _log("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _log("ERROR:")
        _log(str(e))
        _log(traceback.format_exc())