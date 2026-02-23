from __future__ import annotations

import csv
from dataclasses import dataclass

FPS = 60

def frame_to_tc(frame: int, fps: int = FPS) -> str:
    if frame < 0:
        frame = 0
    ff = frame % fps
    total_seconds = frame // fps
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

@dataclass
class Marker:
    frame: int
    label: str

def read_markers_csv(markers_csv: str) -> list[Marker]:
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

    if "Frame" not in norm[0]:
        raise ValueError(f"markers.csv must contain a 'Frame' column. Found: {list(norm[0].keys())}")

    out: list[Marker] = []
    for r in norm:
        try:
            frame = int(float(r["Frame"]))
        except Exception:
            continue
        name = (r.get("Name") or "").strip()
        note = (r.get("Note") or "").strip()
        color = (r.get("Color") or "").strip()

        # Put everything in the label so the EDL is self-contained
        parts = [p for p in [name, note] if p]
        label = " | ".join(parts) if parts else "AI"
        if color:
            label = f"[{color}] {label}"

        # EDL lines shouldn't be huge; keep it reasonable
        label = label.replace("\n", " ").strip()
        if len(label) > 120:
            label = label[:117] + "..."

        out.append(Marker(frame=frame, label=label))

    out.sort(key=lambda m: m.frame)
    return out

def write_marker_edl(markers: list[Marker], out_path: str, fps: int = FPS) -> None:
    lines = []
    lines.append("TITLE: HOCKEY_MARKERS")
    lines.append("FCM: NON-DROP FRAME")
    lines.append("")

    for i, m in enumerate(markers, start=1):
        tc_in = frame_to_tc(m.frame, fps=fps)
        tc_out = frame_to_tc(m.frame + 1, fps=fps)  # 1-frame duration
        idx = f"{i:03d}"

        # Conservative event line Resolve tends to accept
        lines.append(f"{idx}  AX       V     C        {tc_in} {tc_out} {tc_in} {tc_out}")

        # Marker-as-comment (some Resolve builds parse this)
        # Keep both forms to maximize compatibility.
        lines.append(f"* LOC: {tc_in}  NAME: {m.label}")
        lines.append(f"* COMMENT: {m.label}")
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

    markers = read_markers_csv(markers_csv)
    write_marker_edl(markers, os.path.abspath(args.out), fps=args.fps)
    print(f"[edl] Wrote: {os.path.abspath(args.out)}")
    print(f"[edl] Markers: {len(markers)} | FPS: {args.fps}")