# v2/scripts/timeline_convert.py
#
# Generate an FCPXML 1.4 file from markers.csv for import into Final Cut Pro
# or DaVinci Resolve.
#
# Copied from v1/scripts/timeline_convert.py and completed with:
#   - escape() / t_rational() helpers that were missing in the V1 file
#   - main() function and argparse CLI block so the script is runnable
#
# Usage:
#   python timeline_convert.py markers.csv --fps 60 --out markers.fcpxml
#                                           --project-name "AI Markers"

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from math import gcd


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def escape(text: str) -> str:
    """Minimal XML attribute escaping."""
    return (
        text
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def t_rational(frames: int, fps: int) -> str:
    """
    Convert a frame count to an FCPXML rational time string.
    e.g. t_rational(120, 60) -> "2s"
         t_rational(90, 60)  -> "3/2s"
    """
    if frames == 0:
        return "0s"
    frames = int(frames)
    fps = int(fps)
    g = gcd(frames, fps)
    num = frames // g
    den = fps // g
    if den == 1:
        return f"{num}s"
    return f"{num}/{den}s"


# ---------------------------------------------------------------------------
# FCPXML builder (same logic as V1 — do not change)
# ---------------------------------------------------------------------------

def build_fcpxml(markers: list[dict], fps: int, project_name: str) -> str:
    if not markers:
        raise ValueError("No markers to write.")

    last_frame = markers[-1]["frame"]
    tail = fps * 5  # 5 s padding
    total_frames = last_frame + tail

    frame_duration = f"1/{fps}s"

    xml = []
    xml.append('<?xml version="1.0" encoding="UTF-8"?>')
    xml.append('<fcpxml version="1.4">')
    xml.append('  <resources>')
    xml.append(
        f'    <format id="r1" name="FFVideoFormat1080p{fps}" frameDuration="{frame_duration}" '
        f'width="1920" height="1080" colorSpace="1-1-1 (Rec. 709)"/>'
    )
    xml.append('  </resources>')
    xml.append('  <library>')
    xml.append('    <event name="markers">')
    xml.append(f'      <project name="{escape(project_name)}">')
    xml.append(
        f'        <sequence format="r1" duration="{t_rational(total_frames, fps)}" '
        f'tcStart="0s" tcFormat="NDF">'
    )
    xml.append('          <spine>')
    xml.append(
        f'            <gap name="Markers" offset="0s" start="0s" '
        f'duration="{t_rational(total_frames, fps)}">'
    )

    for m in markers:
        note = (m.get("note") or "").strip()
        color_tag = (m.get("color_in") or "Blue").strip()
        prefix = f"[{color_tag}]"
        note2 = f"{prefix} {note}".strip() if note else prefix

        xml.append(
            f'              <marker start="{t_rational(m["frame"], fps)}" '
            f'duration="{t_rational(m["duration_frames"], fps)}" '
            f'value="{escape(m["name"])}" note="{escape(note2)}"/>'
        )

    xml.append('            </gap>')
    xml.append('          </spine>')
    xml.append('        </sequence>')
    xml.append('      </project>')
    xml.append('    </event>')
    xml.append('  </library>')
    xml.append('</fcpxml>')
    return "\n".join(xml) + "\n"


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------

WIN_RE = re.compile(r"\bwin\s*=\s*([0-9]+(?:\.[0-9]+)?)s\b", re.IGNORECASE)


def _sniff_dialect(path: str) -> csv.Dialect:
    with open(path, "r", newline="") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";"])
    except Exception:
        class _Comma(csv.Dialect):
            delimiter = ","
            quotechar = '"'
            doublequote = True
            skipinitialspace = True
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL
        return _Comma()


def read_markers_csv(markers_csv: str, fps: int) -> list[dict]:
    """
    Read markers.csv and return a list of marker dicts suitable for build_fcpxml.

    markers.csv columns: Frame, Name, Note, Color, Duration
    Output dict keys:    frame, name, note, color_in, duration_frames
    """
    dialect = _sniff_dialect(markers_csv)
    with open(markers_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, dialect=dialect)
        raw_rows = list(reader)

    if not raw_rows:
        return []

    # Normalize header whitespace
    rows = []
    for r in raw_rows:
        rows.append({k.strip(): (v.strip() if isinstance(v, str) else v)
                     for k, v in r.items() if k is not None})

    required = ["Frame", "Name", "Note", "Color", "Duration"]
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"markers.csv missing columns: {missing}. Found: {list(rows[0].keys())}")

    out: list[dict] = []
    for r in rows:
        try:
            frame = int(float(r["Frame"]))
        except (ValueError, KeyError):
            continue

        name = (r.get("Name") or "").strip()
        note = (r.get("Note") or "").strip()
        color = (r.get("Color") or "Blue").strip()

        # Duration column is in frames (written by detect_events.py)
        try:
            dur_frames = max(1, int(float(r["Duration"])))
        except (ValueError, KeyError):
            # Fall back: try to parse win=Xs from note
            m = WIN_RE.search(note)
            dur_frames = max(1, int(round(float(m.group(1)) * fps))) if m else 1

        out.append({
            "frame": frame,
            "name": name,
            "note": note,
            "color_in": color,
            "duration_frames": dur_frames,
        })

    out.sort(key=lambda x: x["frame"])
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert markers.csv to FCPXML for DaVinci Resolve / Final Cut Pro"
    )
    ap.add_argument("markers_csv", help="Path to markers.csv")
    ap.add_argument("--fps", type=int, default=60, help="Timeline fps (default 60)")
    ap.add_argument("--out", required=True, help="Output .fcpxml path")
    ap.add_argument("--project-name", default="AI Markers",
                    help="FCPXML project name (default: 'AI Markers')")
    args = ap.parse_args()

    markers_csv = os.path.abspath(args.markers_csv)
    if not os.path.isfile(markers_csv):
        print(f"ERROR: not found: {markers_csv}", file=sys.stderr)
        sys.exit(1)

    markers = read_markers_csv(markers_csv, args.fps)
    if not markers:
        print("ERROR: no valid markers found in CSV.", file=sys.stderr)
        sys.exit(1)

    fcpxml = build_fcpxml(markers, fps=args.fps, project_name=args.project_name)

    out_path = os.path.abspath(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(fcpxml)

    print(f"[timeline_convert] Wrote: {out_path}")
    print(f"[timeline_convert] Markers: {len(markers)} | FPS: {args.fps}")


if __name__ == "__main__":
    main()
