from __future__ import annotations

import csv
import re
from dataclasses import dataclass

FPS = 60

WIN_RE = re.compile(r"\bwin\s*=\s*([0-9]+(?:\.[0-9]+)?)s\b", re.IGNORECASE)

def tc_to_frames(tc: str, fps: int = FPS) -> int:
    # "HH:MM:SS:FF" -> frames
    hh, mm, ss, ff = [int(x) for x in tc.split(":")]
    return (((hh * 60 + mm) * 60 + ss) * fps) + ff

def frames_to_tc(frames: int, fps: int = FPS) -> str:
    if frames < 0:
        frames = 0
    ff = frames % fps
    total_seconds = frames // fps
    ss = total_seconds % 60
    total_minutes = total_seconds // 60
    mm = total_minutes % 60
    hh = total_minutes // 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"

def sniff_dialect(path: str) -> csv.Dialect:
    with open(path, "r", newline="") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";"])
    except Exception:
        class Tab(csv.Dialect):
            delimiter = "\t"
            quotechar = '"'
            doublequote = True
            skipinitialspace = False
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL
        return Tab()

def _parse_win_seconds(note: str) -> float | None:
    if not note:
        return None
    m = WIN_RE.search(note)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

@dataclass
class EdlEvent:
    in_frame: int
    out_frame: int
    label: str

def read_markers_csv_as_events(markers_csv: str, fps: int = FPS) -> list[EdlEvent]:
    dialect = sniff_dialect(markers_csv)
    with open(markers_csv, "r", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        rows = list(reader)

    if not rows:
        return []

    # normalize header keys
    norm = []
    for r in rows:
        nr = {}
        for k, v in r.items():
            if k is None:
                continue
            nr[k.strip()] = (v.strip() if isinstance(v, str) else v)
        norm.append(nr)

    required = ["Frame", "Name", "Note", "Color", "Duration"]
    missing = [c for c in required if c not in norm[0]]
    if missing:
        raise ValueError(f"markers.csv missing columns: {missing}. Found: {list(norm[0].keys())}")

    out: list[EdlEvent] = []
    for r in norm:
        try:
            in_frame = int(float(r["Frame"]))
        except Exception:
            continue

        name = (r.get("Name") or "").strip()
        note = (r.get("Note") or "").strip()
        color = (r.get("Color") or "").strip()

        # REPLAY markers should stay point-ish (1 frame)
        if name.upper() == "REPLAY":
            out_frame = in_frame + 1
        else:
            win_s = _parse_win_seconds(note)
            if win_s is None or win_s <= 0:
                out_frame = in_frame + 1
            else:
                dur_frames = int(round(win_s * fps))
                dur_frames = max(1, dur_frames)
                out_frame = in_frame + dur_frames

        # Build a compact label
        parts = []
        if color:
            parts.append(f"[{color}]")
        if name:
            parts.append(name)
        if note:
            parts.append(note)

        label = " ".join(parts).replace("\n", " ").strip()
        if len(label) > 120:
            label = label[:117] + "..."

        out.append(EdlEvent(in_frame=in_frame, out_frame=out_frame, label=label))

    out.sort(key=lambda e: e.in_frame)
    return out

def write_edl(events: list[EdlEvent], out_path: str, fps: int = FPS, tc_start: str = "01:00:00:00") -> None:
    offset_frames = tc_to_frames(tc_start, fps=fps)
    lines: list[str] = []
    lines.append("TITLE: HOCKEY_WINDOWS")
    lines.append("FCM: NON-DROP FRAME")
    lines.append("")

    for i, ev in enumerate(events, start=1):
        tc_in = frames_to_tc(ev.in_frame + offset_frames, fps=fps)
        tc_out = frames_to_tc(ev.out_frame + offset_frames, fps=fps)
        idx = f"{i:03d}"

        # EDL event line: <num>  AX       V     C        <src in> <src out> <rec in> <rec out>
        # Using same TC for source/record keeps it simple; we mainly want Resolve to create ranged events.
        lines.append(f"{idx}  AX       V     C        {tc_in} {tc_out} {tc_in} {tc_out}")
        lines.append(f"* LOC: {tc_in}  NAME: {ev.label}")
        lines.append(f"* COMMENT: {ev.label}")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    import argparse, os, sys

    ap = argparse.ArgumentParser()
    ap.add_argument("markers_csv", help="Path to markers.csv")
    ap.add_argument("--out", required=True, help="Output .edl path")
    ap.add_argument("--fps", type=int, default=FPS)
    args = ap.parse_args()

    markers_csv = os.path.abspath(args.markers_csv)
    if not os.path.isfile(markers_csv):
        print(f"ERROR: not found: {markers_csv}", file=sys.stderr)
        sys.exit(1)

    events = read_markers_csv_as_events(markers_csv, fps=args.fps)
    write_edl(events, os.path.abspath(args.out), fps=args.fps)
    print(f"[edl] Wrote: {os.path.abspath(args.out)}")
    print(f"[edl] Events: {len(events)} | FPS: {args.fps}")