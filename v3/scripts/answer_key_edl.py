"""Read published EDL cuts. Leave camera, raw time, and goal labels unassigned."""
from __future__ import annotations

import re


_ROW = re.compile(
    r"^(\d+)\s+(\S+)\s+(\S+)\s+(C|D)\s+(?:(\d+)\s+)?"
    r"(\d{2}:\d{2}:\d{2}:\d{2})\s+(\d{2}:\d{2}:\d{2}:\d{2})\s+"
    r"(\d{2}:\d{2}:\d{2}:\d{2})\s+(\d{2}:\d{2}:\d{2}:\d{2})\s*$"
)
_GAME = re.compile(r"^Game v\s*1$", re.IGNORECASE)


def timecode_frames(value: str, fps: int) -> int:
    if isinstance(fps, bool) or fps not in (24, 25, 30, 50, 60):
        raise ValueError("use an explicit integer non-drop frame rate")
    if not re.fullmatch(r"\d{2}:\d{2}:\d{2}:\d{2}", value):
        raise ValueError(f"invalid non-drop timecode: {value}")
    h, m, s, f = map(int, value.split(":"))
    if m >= 60 or s >= 60 or f >= fps:
        raise ValueError(f"invalid timecode frame or clock field: {value}")
    return ((h * 60 + m) * 60 + s) * fps + f


def parse_edl(text: str, *, fps: int, record_origin: str = "01:00:00:00") -> dict:
    """Read Resolve cuts and incoming dissolves with their associated clip names."""
    origin = timecode_frames(record_origin, fps)
    blocks, rows, names, effects, event_id = [], [], {}, {}, None
    title = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("TITLE:"):
            title = line.partition(":")[2].strip()
        elif line.startswith("FCM:"):
            if line != "FCM: NON-DROP FRAME":
                raise ValueError("drop-frame EDLs are not supported")
        elif line.startswith("M2"):
            match = re.fullmatch(r"M2\s+(\S+)\s+(\d+(?:\.\d+)?)\s+(\d{2}:\d{2}:\d{2}:\d{2})", line)
            if match is None or not rows or float(match[2]) <= 0:
                raise ValueError("unsupported or orphan EDL speed effect")
            timecode_frames(match[3], fps)
            effects[(match[1], match[3])] = {"source_rate": float(match[2]), "source_anchor": match[3]}
        elif line.startswith(("* FROM CLIP NAME:", "* TO CLIP NAME:")):
            direction = "to" if line.startswith("* TO") else "from"
            names[direction] = line.partition(":")[2].strip()
        elif re.match(r"^\d+\s", line):
            match = _ROW.fullmatch(line)
            if match is None:
                raise ValueError(f"unsupported EDL event: {line}")
            values = match.groups()
            if event_id is not None and values[0] != event_id:
                blocks.append((rows, names, effects))
                rows, names, effects = [], {}, {}
            event_id = values[0]
            rows.append(values)
    if rows:
        blocks.append((rows, names, effects))
    clips, audio = [], []
    for rows, names, effects in blocks:
        for eid, reel, track, edit, transition, si, so, ri, ro in rows:
            sf, ef, start, end = (timecode_frames(v, fps) for v in (si, so, ri, ro))
            if ef < sf or end < start:
                raise ValueError(f"event {eid}: reversed interval")
            if end == start:
                continue
            if start < origin:
                raise ValueError(f"event {eid}: record time precedes recap origin")
            name = names.get("to" if edit == "D" else "from")
            if not name:
                raise ValueError(f"event {eid}: missing source clip name")
            row = {"id": f"edl:{eid}", "track": track, "source_name": name,
                   "source_start_s": sf / fps, "source_end_s": ef / fps,
                   "recap_start_s": (start - origin) / fps, "recap_end_s": (end - origin) / fps,
                   "transition_frames": int(transition or 0), "speed_effect": effects.get((reel, si))}
            if track == "V" and reel != "BL" and _GAME.fullmatch(name):
                if ef == sf:
                    raise ValueError(f"event {eid}: still source cannot supply a footage mapping")
                clips.append(dict(row, camera=None, goal=None))
            elif track.startswith("A") and reel != "BL":
                audio.append(row)
    if not clips:
        raise ValueError("EDL has no recognized Game v1 video cuts")
    return {"title": title, "fps": fps, "record_origin": record_origin,
            "clips": clips, "audio_cues": audio}
