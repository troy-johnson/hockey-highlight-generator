# v3/scripts/recap_check.py
"""
Recording check behind `hockeyrecap check <game_folder>` (hhg-3r5.62).

A quick pass over the camera files, before the long run. It reads only file
metadata (ffprobe) and a small number of sampled frames, and reports:

  - a missing camera,
  - a covered lens (the full Recording or a part of it is black),
  - a camera that started late or stopped early, compared to the other camera,
  - gaps between Recordings and between chapters.

It writes recording_check.json in the Game Folder and changes nothing else.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from discover import _cameras_from_flat_folder, _cameras_from_folders  # noqa: E402
from recordings import BLACK_MEAN_LUMA, group_recordings, sample_lumas  # noqa: E402

REPORT_FILE = "recording_check.json"
SAMPLE_EVERY_S = 120.0      # one sampled frame per 2 minutes of footage
MIN_SAMPLES = 9             # the same count discover.py uses for a black Recording
MAX_SAMPLES = 30            # per Recording, so a check takes a few minutes
GAP_S = 5.0                 # the chapter continuity limit of gopro_meta.py
EARLY_S = 120.0             # a camera that ends this much before the other "stopped early"


# ---------------------------------------------------------------------------
# Measurements (ffprobe / ffmpeg)
# ---------------------------------------------------------------------------

def _chapter_meta(path: str) -> dict:
    from gopro_meta import extract_chapter_time
    try:
        start, source, _tc, duration = extract_chapter_time(path)
        return {"path": path, "start": start, "clock": source, "duration": duration}
    except Exception:  # no clock: keep the duration only
        from recordings import _duration
        return {"path": path, "start": None, "clock": None, "duration": _duration(path)}


def _sample_count(total_s: float) -> int:
    return int(min(MAX_SAMPLES, max(MIN_SAMPLES, round(total_s / SAMPLE_EVERY_S))))


def measure_recording(recording: list[str]) -> dict:
    chapters = [_chapter_meta(p) for p in recording]
    durations = [c["duration"] for c in chapters]
    total = sum(durations)
    fractions = np.linspace(0.02, 0.98, _sample_count(total))
    samples = sample_lumas(recording, fractions, durations) if total > 0 else []
    return {"chapters": chapters, "duration": total, "start": chapters[0]["start"],
            "samples": [[round(t, 1), round(l, 1)] for t, l in samples]}


def measure_game(game_folder: str | Path) -> dict:
    """Cameras -> list of measured Recordings, plus files that are not used."""
    root = Path(game_folder)
    excluded: list[dict] = []
    if (root / "cam1").is_dir() or (root / "cam2").is_dir():
        cams, serials = _cameras_from_folders(root, str(root), strict=False), {}
    else:
        try:
            cams, serials = _cameras_from_flat_folder(root, strict=False, excluded=excluded)
        except ValueError:
            cams, serials = {}, {}
    jobs = [(cam, rec) for cam, files in cams.items() for rec in group_recordings(files)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        measured = list(pool.map(lambda j: measure_recording(j[1]), jobs))
    cameras: dict[str, list[dict]] = {cam: [] for cam in cams}
    for (cam, _), m in zip(jobs, measured):
        cameras[cam].append(m)
    return {"cameras": cameras, "serials": serials, "excluded": excluded}


# ---------------------------------------------------------------------------
# Analysis (pure: tests call it with made-up measurements)
# ---------------------------------------------------------------------------

def _clock(s: float | None) -> str:
    if s is None:
        return "?"
    s = int(round(s)) % 86400
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _mmss(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def dark_spans(samples: list[list[float]], duration: float) -> list[tuple[float, float]]:
    """Merge consecutive dark samples into (from, to) spans in Recording seconds."""
    spans: list[tuple[float, float]] = []
    run_start = None
    prev_t = 0.0
    for i, (t, luma) in enumerate(samples):
        dark = luma < BLACK_MEAN_LUMA
        if dark and run_start is None:
            run_start = prev_t if i else 0.0
        if not dark and run_start is not None:
            spans.append((run_start, t))
            run_start = None
        prev_t = t
    if run_start is not None:
        spans.append((run_start, duration))
    return spans


def analyze(measured: dict) -> dict:
    """Findings from measure_game() output."""
    findings: list[dict] = []

    def add(kind, cam, text):
        findings.append({"kind": kind, "camera": cam, "text": text})

    cams = measured.get("cameras", {})
    for e in measured.get("excluded", []):
        add("ignored_file", None, f"{Path(e['path']).name} not used ({e['reason']})")
    if len(cams) < 2:
        add("missing_camera", None, f"found {len(cams)} camera(s); a two-camera game needs 2")

    summary: dict[str, dict] = {}
    for cam, recs in cams.items():
        usable = []
        for i, r in enumerate(recs, 1):
            name = Path(r["chapters"][0]["path"]).name
            samples = r.get("samples", [])
            spans = dark_spans(samples, r["duration"])
            r["dark_spans"] = [[round(a, 1), round(b, 1)] for a, b in spans]
            if samples and all(l < BLACK_MEAN_LUMA for _, l in samples):
                r["covered"] = "full"
                add("covered_lens", cam, f"Recording {i} ({name}, {_mmss(r['duration'])}) is black: lens covered")
            else:
                r["covered"] = "partial" if spans else None
                usable.append(r)
                for a, b in spans:
                    add("partly_covered", cam, f"Recording {i} ({name}) is black from {_mmss(a)} to {_mmss(b)}")
            if not samples:
                add("unreadable", cam, f"Recording {i} ({name}): no frame could be read")
            for j in range(1, len(r["chapters"])):
                p, c = r["chapters"][j - 1], r["chapters"][j]
                if p["start"] is not None and c["start"] is not None and p["duration"] > 0:
                    gap = c["start"] - (p["start"] + p["duration"])
                    if abs(gap) > GAP_S:
                        add("chapter_gap", cam, f"Recording {i} chapter {j + 1}: clock "
                            f"{'gap' if gap > 0 else 'overlap'} of {abs(gap):.0f} s")
            if r["start"] is None:
                add("no_clock", cam, f"Recording {i} ({name}): no timecode or creation time")
        timed = sorted((r for r in usable if r["start"] is not None), key=lambda r: r["start"])
        for a, b in zip(timed, timed[1:]):
            gap = b["start"] - (a["start"] + a["duration"])
            if gap > GAP_S:
                add("recording_gap", cam, f"no footage for {_mmss(gap)} between "
                    f"{_clock(a['start'] + a['duration'])} and {_clock(b['start'])} (camera stopped and restarted)")
        summary[cam] = {
            "recordings": len(recs), "usable_recordings": len(usable),
            "footage_s": round(sum(r["duration"] for r in usable), 1),
            "start": timed[0]["start"] if timed else None,
            "end": max(r["start"] + r["duration"] for r in timed) if timed else None,
        }

    timed_cams = {c: s for c, s in summary.items() if s["start"] is not None}
    if len(timed_cams) >= 2:
        first = min(s["start"] for s in timed_cams.values())
        last = max(s["end"] for s in timed_cams.values())
        for cam, s in timed_cams.items():
            if s["start"] - first > EARLY_S:
                add("started_late", cam, f"started {_mmss(s['start'] - first)} after the other camera "
                    f"({_clock(s['start'])})")
            if last - s["end"] > EARLY_S:
                add("stopped_early", cam, f"stopped {_mmss(last - s['end'])} before the other camera "
                    f"({_clock(s['end'])}): battery, card full, or stopped by hand?")
    for cam, s in summary.items():
        s["start_clock"], s["end_clock"] = _clock(s["start"]), _clock(s["end"])
    return {"cameras": summary, "findings": findings, "recordings": cams,
            "serials": measured.get("serials", {}), "ok": not any(f["kind"] != "ignored_file" for f in findings)}


def check_game(game_folder: str | Path) -> dict:
    report = analyze(measure_game(game_folder))
    (Path(game_folder) / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n")
    return report


def format_report(report: dict) -> list[str]:
    lines = []
    for cam, s in report["cameras"].items():
        lines.append(f"{cam}: {s['usable_recordings']}/{s['recordings']} Recording(s) with footage, "
                     f"{_mmss(s['footage_s'])} of footage, {s['start_clock']} to {s['end_clock']}")
    if report["ok"]:
        lines.append("No problems found.")
    for f in report["findings"]:
        lines.append(f"[FLAG] {f['camera'] + ': ' if f['camera'] else ''}{f['text']}")
    return lines


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: recap_check.py <game_folder>")
    rep = check_game(sys.argv[1])
    print("\n".join(format_report(rep)))
