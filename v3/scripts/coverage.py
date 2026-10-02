#!/usr/bin/env python3
"""
Coverage and period structure (hhg-3r5.30, spec 002 §5.3).

Finds the game start, the breaks between periods, the period starts, the game
end, and each camera's coverage of each period. Reads the outputs and caches
of earlier stages. It decodes the audio again only when the audio cache is
missing (for example after an audio run with --no-cache).

Inputs (Game Folder):
  audio_signals.json            whistles (detection timeline)
  .recap_cache/audio/*.npz      audio activity per camera (audio stage)
  .recap_cache/signals/*.npz    net + slot flow per Recording (detection stage)
  cam1_concat.txt, cam2_concat.txt, chapters.json, rois.json

Output: coverage.json. All times are on the detection timeline (cam1 concat
time; cam2 moved by the sync offset), the same as audio_signals.json and
markers.csv. Each period start also gives the chapter file and the time in
that file for each camera.

Method (approved in hhg-3r5.30):
  * Break candidate: a run of seconds where the loudest camera's audio
    activity rank is below 0.15 (gaps up to 5 s joined, 25 s or more). Score =
    half quiet length (full at 60 s), half net flow (median flow in the run
    against the game median, at the camera with the most flow; low is good).
  * Breaks: the set of (periods - 1) candidates with the best score, with
    balanced period lengths, at least 40 % of the mean period length each,
    and whistles in each period. League timing (when the League file has it)
    adds to the score (running clock) or sets a minimum period (stop clock).
  * Game start: the first faceoff after the first game whistle. Referees
    blow the whistle just before the first faceoff, so it is the first
    second within 30 s after that whistle where the 3-s mean activity is 0.6
    or more.
  * Game end: the start of the first quiet run (activity below 0.3, 40 s or
    more, or up to the end of the last Recording) after the last break plus
    0.6 x the mean length of the earlier periods.
  * Goalie net swap (optional ML stack): evidence only. A confident "same
    goalie" result flags the break; it does not choose breaks.
  * Coverage: each camera's Recording spans inside each period. The
    end-switch rule gives the defending team at each net per period.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "v2", "scripts"))

TIMELINE = "detection (cam1 concat time; cam2 moved by the sync offset)"

SETTINGS = {
    "break": {"quiet_below": 0.15, "merge_s": 5, "min_s": 25, "full_quiet_s": 60,
              "flow_good": 0.4, "flow_bad": 0.8, "max_candidates": 20},
    "plan": {"default_periods": 3, "min_period_frac": 0.4, "imbalance_weight": 1.0,
             "min_period_whistles": 3, "min_period_whistle_frac": 0.25,
             "running_timing_weight": 0.3, "timing_sigma_s": 180},
    "uncertain": {"min_score": 0.7, "min_quiet_s": 40, "alt_ratio": 0.9, "length_ratio": 1.5,
                  "candidates": 5},
    "start": {"whistles_after": 3, "within_s": 300, "faceoff_window_s": 30,
              "active": 0.6, "mean_s": 3},
    "end": {"quiet_below": 0.3, "merge_s": 5, "min_s": 40, "after_frac": 0.6, "small_margin_s": 50,
            "other_min_s": 20},
    "coverage": {"gap_flag_s": 30},
    "goalie": {"window_s": 150, "step_s": 10, "edge_s": 10, "min_samples": 3, "same_samples": 5,
               "swapped_dist": 12.0, "same_dist": 6.0, "min_chroma": 8.0, "min_color_frac": 0.05},
}


def mmss(t: float | None) -> str:
    if t is None:
        return "?"
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{int(round(t % 60)):02d}" if round(t % 60) < 60 else f"{int(t // 60) + 1}:00"


# ---------------------------------------------------------------------------
# Pure logic: per-second series and runs
# ---------------------------------------------------------------------------

def per_second(x: np.ndarray | None, rate: float) -> np.ndarray:
    """Mean of each whole second; NaN stays NaN (a second with no data is NaN)."""
    if x is None or not len(x):
        return np.zeros(0, np.float32)
    r = int(round(rate))
    n = len(x) // r
    if n == 0:
        return np.zeros(0, np.float32)
    a = np.asarray(x[:n * r], np.float64).reshape(n, r)
    ok = np.isfinite(a)
    cnt = ok.sum(axis=1)
    s = np.where(ok, a, 0.0).sum(axis=1)
    out = np.full(n, np.nan, np.float32)
    out[cnt > 0] = (s[cnt > 0] / cnt[cnt > 0]).astype(np.float32)
    return out


def max_over(series: list[np.ndarray]) -> np.ndarray:
    """Per-second maximum over cameras; NaN where no camera has data."""
    series = [s for s in series if s is not None and len(s)]
    if not series:
        return np.zeros(0, np.float32)
    n = max(len(s) for s in series)
    stack = np.full((len(series), n), -np.inf)
    for i, s in enumerate(series):
        stack[i, :len(s)] = np.where(np.isfinite(s), s, -np.inf)
    out = stack.max(axis=0)
    return np.where(np.isfinite(out), out, np.nan).astype(np.float32)


def quiet_runs(series: np.ndarray, below: float, min_s: float, merge_s: float,
               t_from: float = 0, t_to: float | None = None) -> list[tuple[int, int]]:
    """
    Runs of seconds [s, e) where the series is below the threshold. Runs with
    gaps up to merge_s are joined. NaN (no audio) is not quiet. Runs are cut
    to [t_from, t_to).
    """
    n = len(series)
    lo, hi = max(0, int(math.ceil(t_from))), n if t_to is None else min(n, int(t_to))
    runs: list[list[int]] = []
    s = None
    for t in range(lo, hi):
        q = bool(np.isfinite(series[t]) and series[t] < below)
        if q and s is None:
            s = t
        elif not q and s is not None:
            runs.append([s, t])
            s = None
    if s is not None:
        runs.append([s, hi])
    merged: list[list[int]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= merge_s:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if b - a >= min_s]


def flow_ratio(flows: dict[str, np.ndarray], s: int, e: int) -> float | None:
    """
    Largest, over the cameras with flow in [s, e), of the median flow in the
    run divided by that camera's median flow over the whole Recording. Empty
    nets in a break give about 0.1-0.4; play gives 0.7 or more.
    """
    vals = []
    for f in flows.values():
        if f is None or s >= len(f):
            continue
        seg = f[s:e]
        seg = seg[np.isfinite(seg) & (seg > 0)]
        allv = f[np.isfinite(f) & (f > 0)]
        if len(seg) < max(3, (e - s) // 3) or not len(allv):
            continue
        vals.append(float(np.median(seg)) / max(float(np.median(allv)), 1e-9))
    return max(vals) if vals else None


def candidate_score(quiet_s: float, ratio: float | None, cfg: dict = SETTINGS["break"]) -> float:
    q = min(quiet_s / cfg["full_quiet_s"], 1.0)
    if ratio is None:
        f = 0.5
    else:
        f = float(np.clip((cfg["flow_bad"] - ratio) / (cfg["flow_bad"] - cfg["flow_good"]), 0.0, 1.0))
    return round(0.5 * q + 0.5 * f, 3)


def break_candidates(act_max: np.ndarray, flows: dict[str, np.ndarray], t_from: float, t_to: float,
                     cfg: dict = SETTINGS["break"]) -> list[dict]:
    out = []
    for s, e in quiet_runs(act_max, cfg["quiet_below"], cfg["min_s"], cfg["merge_s"], t_from, t_to):
        r = flow_ratio(flows, s, e)
        out.append({"start": float(s), "end": float(e), "quiet_s": float(e - s),
                    "flow_ratio": None if r is None else round(r, 2), "score": candidate_score(e - s, r, cfg)})
    out.sort(key=lambda c: -c["score"])
    return sorted(out[:cfg["max_candidates"]], key=lambda c: c["start"])


# ---------------------------------------------------------------------------
# Pure logic: game start, faceoffs, game end
# ---------------------------------------------------------------------------

def first_game_whistle(whistles: list[dict], cfg: dict = SETTINGS["start"]) -> float | None:
    """First whistle followed by at least N whistles within the window (warm-up has none)."""
    ts = sorted(float(w["t"]) for w in whistles)
    for i, t in enumerate(ts):
        if sum(1 for u in ts[i + 1:] if u - t <= cfg["within_s"]) >= cfg["whistles_after"]:
            return t
    return None


def faceoff_after(act_max: np.ndarray, t: float, window_s: float, cfg: dict = SETTINGS["start"]) -> float | None:
    """First second in [t, t + window) where the mean activity over the next mean_s seconds is >= active."""
    k = cfg["mean_s"]
    for s in range(max(0, int(math.floor(t))), min(len(act_max) - k + 1, int(t + window_s) + 1)):
        seg = act_max[s:s + k]
        if np.all(np.isfinite(seg)) and float(np.mean(seg)) >= cfg["active"]:
            return float(max(s, t))
    return None


def game_start(whistles: list[dict], act_max: np.ndarray, cfg: dict = SETTINGS["start"]) -> dict:
    w1 = first_game_whistle(whistles, cfg)
    if w1 is None:
        return {"t": None, "rule": "no game whistles", "whistle": None}
    f = faceoff_after(act_max, w1, cfg["faceoff_window_s"], cfg)
    if f is None:
        return {"t": round(w1, 1), "rule": "first game whistle (no faceoff activity found after it)", "whistle": w1}
    return {"t": round(f, 1), "rule": "first faceoff after the first game whistle", "whistle": round(w1, 2)}


def game_end(act_max: np.ndarray, after: float, signal_end: float, whistles: list[dict] | None = None,
             cfg: dict = SETTINGS["end"]) -> dict:
    """
    End = start of the first quiet run (activity below quiet_below, min_s or
    more, or reaching the end of the signal) at or after `after`.
    """
    runs = quiet_runs(act_max, cfg["quiet_below"], 1, cfg["merge_s"], after, signal_end)
    end_s = int(signal_end)
    pick = None
    for s, e in runs:
        if e - s >= cfg["min_s"] or e >= end_s - 1:
            pick = (s, e)
            break
    cands = [{"t": float(s), "why": f"quiet {e - s} s"} for s, e in runs
             if e - s >= cfg["other_min_s"] and (pick is None or s != pick[0])]
    last_w = None
    if whistles:
        before = [float(w["t"]) for w in whistles if after - 600 <= float(w["t"]) <= (pick[0] if pick else signal_end)]
        if before:
            last_w = max(before)
            cands.append({"t": round(last_w, 1), "why": "last whistle"})
    if pick is None:
        return {"t": round(float(signal_end), 1), "rule": "end of the last Recording (no quiet run found)",
                "uncertain": True, "quiet_s": None, "candidates": sorted(cands, key=lambda c: c["t"])}
    s, e = pick
    reaches_end = e >= end_s - 1
    uncertain = (e - s) < cfg["min_s"] or (not reaches_end and (e - s) < cfg["small_margin_s"])
    rule = "first long quiet after the last period" + (" (quiet reaches the end of the Recording)" if reaches_end else "")
    cands.append({"t": round(float(signal_end), 1), "why": "end of the last Recording"})
    return {"t": float(s), "rule": rule, "uncertain": bool(uncertain), "quiet_s": float(e - s),
            "candidates": sorted(cands, key=lambda c: c["t"])}


# ---------------------------------------------------------------------------
# Pure logic: period plan
# ---------------------------------------------------------------------------

def league_timing(rules: dict | None) -> dict:
    """Period timing from the League rules; keys: periods, period_minutes, clock, break_minutes."""
    rules = rules or {}
    out = {"periods": SETTINGS["plan"]["default_periods"], "period_s": None, "clock": None, "break_s": 60.0,
           "source": "default (no League timing)"}
    try:
        if rules.get("periods"):
            out["periods"] = max(1, int(rules["periods"]))
            out["source"] = "League"
        if rules.get("period_minutes"):
            out["period_s"] = float(rules["period_minutes"]) * 60.0
            if out["source"] != "League":
                out["source"] = "League (period count assumed)"
        if rules.get("clock") in ("running", "stop"):
            out["clock"] = rules["clock"]
        if rules.get("break_minutes") is not None:
            out["break_s"] = float(rules["break_minutes"]) * 60.0
    except (TypeError, ValueError):
        pass
    return out


def _evaluate(combo: tuple[dict, ...], start: float, act_max: np.ndarray, signal_end: float,
              whistles: list[dict], timing: dict, cfg: dict = SETTINGS["plan"]) -> dict | None:
    """Total score and period layout for one set of breaks, or None when it is not a valid layout."""
    starts = [start]
    ends = []
    for c in combo:
        if c["start"] <= starts[-1]:
            return None
        ends.append(c["start"])
        f = faceoff_after(act_max, c["end"], 60.0)
        starts.append(f if f is not None else c["end"])
    prev = [e - s for s, e in zip(starts, ends)]
    mean_prev = float(np.mean(prev)) if prev else 600.0 / SETTINGS["end"]["after_frac"]
    end = game_end(act_max, starts[-1] + SETTINGS["end"]["after_frac"] * mean_prev, signal_end, whistles)
    ends.append(end["t"])
    lengths = [e - s for s, e in zip(starts, ends)]
    if any(L <= 0 for L in lengths):
        return None
    mean_len = float(np.mean(lengths))
    if min(lengths) < cfg["min_period_frac"] * mean_len:
        return None
    if timing["clock"] == "stop" and timing["period_s"] and min(lengths) < timing["period_s"]:
        return None
    ws = [float(w["t"]) for w in whistles]
    counts = [sum(1 for t in ws if s <= t < e) for s, e in zip(starts, ends)]
    need = max(cfg["min_period_whistles"], cfg["min_period_whistle_frac"] * sum(counts) / len(counts))
    if min(counts) < need:
        return None
    scores = [c["score"] for c in combo]
    base = float(np.mean(scores)) if scores else 0.0
    if timing["clock"] == "running" and timing["period_s"] and combo:
        tim = []
        for k, c in enumerate(combo, 1):
            exp = start + k * timing["period_s"] + (k - 1) * timing["break_s"]
            tim.append(math.exp(-((c["start"] - exp) / cfg["timing_sigma_s"]) ** 2))
        w = cfg["running_timing_weight"]
        base = (1 - w) * base + w * float(np.mean(tim))
    weight = cfg["imbalance_weight"] * (0.5 if timing["clock"] == "stop" else 1.0)
    total = base - weight * (max(lengths) - min(lengths)) / mean_len
    return {"total": round(total, 4), "starts": starts, "ends": ends, "lengths": lengths, "end": end,
            "whistles": counts}


def plan_periods(cands: list[dict], start: float, act_max: np.ndarray, signal_end: float,
                 whistles: list[dict], timing: dict, cfg: dict = SETTINGS) -> dict:
    """Choose the breaks; mark each break uncertain or not, with its other candidates."""
    flags: list[str] = []
    n_breaks = max(0, timing["periods"] - 1)
    best = None
    used_n = n_breaks
    evals: list[tuple[tuple[dict, ...], dict]] = []
    for n in range(n_breaks, -1, -1):
        evals = []
        for combo in itertools.combinations(cands, n):
            ev = _evaluate(combo, start, act_max, signal_end, whistles, timing, cfg["plan"])
            if ev is not None:
                evals.append((combo, ev))
        if evals:
            used_n = n
            best = max(evals, key=lambda x: x[1]["total"])
            break
    if best is None:
        end = game_end(act_max, start + 600, signal_end, whistles)
        return {"periods": [{"n": 1, "start": start, "end": end["t"], "break_before": None}], "end": end,
                "flags": ["no period structure found; the whole game is one period"]}
    if used_n < n_breaks:
        flags.append(f"found {used_n} of {n_breaks} breaks; check the period starts")
    combo, ev = best
    ucfg = cfg["uncertain"]
    lengths = ev["lengths"]
    unbalanced = len(lengths) > 1 and max(lengths) / max(min(lengths), 1e-9) > ucfg["length_ratio"]
    periods = []
    for i in range(len(ev["starts"])):
        brk = None
        if i > 0:
            c = combo[i - 1]
            reasons = []
            if c["score"] < ucfg["min_score"]:
                reasons.append(f"weak quiet (score {c['score']:.2f})")
            if c["quiet_s"] < ucfg["min_quiet_s"]:
                reasons.append(f"short quiet ({c['quiet_s']:.0f} s)")
            if unbalanced:
                reasons.append("period lengths differ a lot")
            alts = []
            chosen = {b["start"] for b in combo}
            for oc, oev in evals:
                if oc[i - 1]["start"] not in chosen:
                    alts.append((oc[i - 1], oev["total"], oev["starts"][i]))
            seen: dict[float, tuple] = {}
            for a, tot, pstart in alts:
                if a["start"] not in seen or tot > seen[a["start"]][1]:
                    seen[a["start"]] = (a, tot, pstart)
            ranked = sorted(seen.values(), key=lambda x: -x[1])
            margin = (1.0 - ucfg["alt_ratio"]) * max(abs(ev["total"]), 0.05)
            if ranked and ranked[0][1] >= ev["total"] - margin:
                reasons.append(f"another break at {mmss(ranked[0][0]['start'])} scores close")
            brk = dict(c, uncertain=bool(reasons), reasons=reasons,
                       candidates=[dict(a, period_start=round(p, 1), plan_score=round(t, 3))
                                   for a, t, p in ranked[:ucfg["candidates"]]])
        periods.append({"n": i + 1, "start": round(ev["starts"][i], 1), "end": round(ev["ends"][i], 1),
                        "length_s": round(lengths[i], 1), "whistles": ev["whistles"][i], "break_before": brk})
    return {"periods": periods, "end": ev["end"], "plan_score": ev["total"], "flags": flags}


# ---------------------------------------------------------------------------
# Pure logic: Recording spans, chapter times, coverage
# ---------------------------------------------------------------------------

def manifest_layout(lines: list[str], durations: dict[str, float], base: str = "/") -> list[dict]:
    """
    Recording blocks of one camera on the detection timeline:
    [{"start", "end", "seek", "files": [(path, duration)]}]. 'seek' is the
    time into the block's first file at the block start.
    """
    from signals import _manifest_inputs
    blocks: list[tuple[float | None, list[str]]] = []
    for line in lines:
        parts = line.strip().split()
        if parts[:2] == ["#", "recording"] and len(parts) == 3:
            blocks.append((float(parts[2]), []))
        elif line.strip().startswith("file "):
            if not blocks:
                blocks.append((None, []))
            blocks[-1][1].append(line.strip())
    out = []
    for start, files in blocks:
        if start is None:
            paths, seek = _manifest_inputs(lines, base)
            t0 = 0.0
        else:
            paths, _ = _manifest_inputs(files, base)
            seek = -start if start < 0 else 0.0
            t0 = max(start, 0.0)
        fl = [(p, float(durations.get(p, 0.0))) for p in paths]
        total = sum(d for _, d in fl)
        out.append({"start": round(t0, 3), "end": round(t0 + total - seek, 3), "seek": seek, "files": fl})
    return out


def chapter_at(layout: list[dict], t: float) -> tuple[str, float] | None:
    """(chapter file name, time in that file) for detection time t, or None when the camera has no video then."""
    for b in layout:
        if b["start"] <= t <= b["end"] and b["files"]:
            c = t - b["start"] + b["seek"]
            for p, d in b["files"]:
                if c < d:
                    return os.path.basename(p), round(c, 2)
                c -= d
            p, d = b["files"][-1]
            return os.path.basename(p), round(d, 2)
    return None


def _overlap(spans: list[tuple[float, float]], a: float, b: float) -> list[tuple[float, float]]:
    return [(max(s, a), min(e, b)) for s, e in spans if min(e, b) > max(s, a)]


def coverage(spans: dict[str, list[tuple[float, float]]], periods: list[dict], n_regular: int,
             cfg: dict = SETTINGS["coverage"]) -> tuple[dict, list[str]]:
    """
    Per camera and period: covered seconds, fraction, gaps, and the defending
    team at the camera's net. Team A is the team whose goalie is at the cam1
    net in period 1; teams switch ends every regular period.
    """
    flags: list[str] = []
    out = {}
    cams = sorted(spans)
    for cam in cams:
        ci = int(cam[3:]) - 1 if cam[3:].isdigit() else 0
        rows = []
        for p in periods:
            a, b = p["start"], p["end"]
            cov = _overlap(spans[cam], a, b)
            covered = sum(e - s for s, e in cov)
            gaps, cur = [], a
            for s, e in cov:
                if s > cur:
                    gaps.append((round(cur, 1), round(s, 1)))
                cur = max(cur, e)
            if cur < b:
                gaps.append((round(cur, 1), round(b, 1)))
            if p["n"] <= n_regular:
                defender = "A" if (p["n"] + ci) % 2 == 1 else "B"
            else:
                defender = None
            rows.append({"n": p["n"], "period_s": round(b - a, 1), "covered_s": round(covered, 1),
                         "fraction": round(covered / (b - a), 3) if b > a else 0.0,
                         "gaps": [list(g) for g in gaps], "defender": defender})
            for g0, g1 in gaps:
                if g1 - g0 >= cfg["gap_flag_s"]:
                    flags.append(f"{cam} has no video for {mmss(g1 - g0)} in period {p['n']} "
                                 f"({mmss(g0)}-{mmss(g1)}); goals at the {cam} net then are not covered")
        out[cam] = {"net": f"{cam} net", "spans": [[round(s, 1), round(e, 1)] for s, e in spans[cam]],
                    "periods": rows}
    return out, flags


# ---------------------------------------------------------------------------
# Goalie net swap (optional ML stack; evidence only)
# ---------------------------------------------------------------------------

def swap_verdict(before: list[tuple[float, float]], after: list[tuple[float, float]],
                 cfg: dict = SETTINGS["goalie"]) -> dict:
    """Compare goalie jersey colors (Lab a, b) before and after a break."""
    nb, na = len(before), len(after)
    if nb < cfg["min_samples"] or na < cfg["min_samples"]:
        return {"result": "unclear", "dist": None, "samples": [nb, na]}
    d = float(np.linalg.norm(np.median(np.array(before), axis=0) - np.median(np.array(after), axis=0)))
    if d >= cfg["swapped_dist"]:
        res = "swapped"
    elif d <= cfg["same_dist"] and nb >= cfg["same_samples"] and na >= cfg["same_samples"]:
        res = "same"
    else:
        res = "unclear"
    return {"result": res, "dist": round(d, 1), "samples": [nb, na]}


def break_swap_flag(per_cam: dict[str, dict]) -> str | None:
    """A flag when the goalie check says confidently that the goalies did not swap nets."""
    res = [v["result"] for v in per_cam.values()]
    if res and "same" in res and "swapped" not in res:
        return "goalie colors look the same before and after"
    return None


def _grab(path: str, t: float) -> np.ndarray | None:
    raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", path,
                          "-frames:v", "1", "-vf", "scale=1280:720", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True).stdout
    if len(raw) < 1280 * 720 * 3:
        return None
    return np.frombuffer(raw[:1280 * 720 * 3], np.uint8).reshape(720, 1280, 3)


def jersey_color(crop: np.ndarray, cfg: dict = SETTINGS["goalie"]) -> tuple[float, float] | None:
    """
    Median Lab (a, b) of the colored pixels in the jersey part of a figure box.
    White and grey pixels (pads, net mesh, ice) are left out.
    """
    import cv2
    h, w = crop.shape[:2]
    c = crop[int(0.15 * h): int(0.6 * h), w // 6: w - w // 6]
    if c.size == 0:
        return None
    lab = cv2.cvtColor(np.ascontiguousarray(c), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    chroma = np.hypot(lab[:, 1] - 128, lab[:, 2] - 128)
    sel = lab[chroma > cfg["min_chroma"]]
    if len(sel) < max(20, cfg["min_color_frac"] * len(lab)):
        return None
    m = np.median(sel[:, 1:], axis=0)
    return float(m[0]), float(m[1])


def net_figure(dets: list[tuple[str, float, list[float]]]) -> list[int] | None:
    """
    Box of the figure in the goal crease: a goalie or player box whose center is
    over the goal box and whose feet are near the goal line. Goalie boxes come first.
    """
    goals = [d for d in dets if d[0] == "goal"]
    if not goals:
        return None
    gx0, gy0, gx1, gy1 = max(goals, key=lambda d: d[1])[2]
    best = None
    for name, conf, (x0, y0, x1, y1) in dets:
        if name not in ("goalie", "player"):
            continue
        cx = (x0 + x1) / 2
        if gx0 - 40 <= cx <= gx1 + 40 and gy0 - 60 <= y1 <= gy1 + 20:
            key = (name == "goalie", conf)
            if best is None or key > best[0]:
                best = (key, [int(x0), int(y0), int(x1), int(y1)])
    return best[1] if best else None


def _goalie_color(model, img) -> tuple[float, float] | None:
    r = model.predict(img, imgsz=1280, conf=0.15, verbose=False)[0]
    dets = [(model.names[int(b.cls)], float(b.conf), [float(v) for v in b.xyxy[0].tolist()]) for b in r.boxes]
    box = net_figure(dets)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    return jersey_color(img[y0:y1, x0:x1])


def goalie_swaps(layouts: dict[str, list[dict]], periods: list[dict], cfg: dict = SETTINGS["goalie"]
                 ) -> tuple[dict, str | None]:
    """Run the goalie check at each break. Returns ({break n: {cam: verdict}}, reason when skipped)."""
    try:
        import auto_roi
        model = auto_roi._model()
    except Exception as exc:  # noqa: BLE001 - optional ML stack
        return {}, f"goalie swap check skipped ({str(exc).splitlines()[0][:120] or type(exc).__name__})"
    out = {}
    for p in periods:
        brk = p.get("break_before")
        if not brk:
            continue
        res = {}
        for cam, layout in layouts.items():
            sides = []
            for a, b in ((brk["start"] - cfg["window_s"], brk["start"] - cfg["edge_s"]),
                         (brk["end"] + cfg["edge_s"], brk["end"] + cfg["window_s"])):
                vals = []
                for t in np.arange(a, b, cfg["step_s"]):
                    ch = chapter_at(layout, float(t))
                    if ch is None:
                        continue
                    path = next((f for blk in layout for f, _ in blk["files"] if os.path.basename(f) == ch[0]), None)
                    img = _grab(path, ch[1]) if path else None
                    c = _goalie_color(model, img) if img is not None else None
                    if c is not None:
                        vals.append(c)
                sides.append(vals)
            res[cam] = swap_verdict(sides[0], sides[1], cfg)
        out[p["n"]] = res
    return out, None


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    return float(out) if out else 0.0


def _fps_width(root: Path) -> tuple[int, int]:
    try:
        d = json.loads((root / "recap_options.json").read_text()).get("detection", {}) or {}
    except (OSError, json.JSONDecodeError, AttributeError):
        d = {}
    return int(d.get("fps", 12) or 12), int(d.get("width", 1280) or 1280)


def analyse(root: Path, league_rules: dict | None = None, goalie: bool = True, fps: int = 12,
            width: int = 1280, flow_audio: bool = False) -> dict:
    import audio_signals as A
    flags: list[str] = []
    sig = json.loads((root / "audio_signals.json").read_text())
    whistles = sig.get("whistles", [])
    cams = [c for c in ("cam1", "cam2") if (root / f"{c}_concat.txt").exists()]
    cache_dir = str(root / ".recap_cache" / "audio")
    act, flows, layouts, spans = {}, {}, {}, {}
    for cam in cams:
        manifest = str(root / f"{cam}_concat.txt")
        try:
            feats = A.camera_features(manifest, None, cache_dir)
            act[cam] = per_second(A.activity_rank(feats["flux"]), A.RATE)
        except Exception as exc:  # noqa: BLE001 - a bad camera is a flag
            flags.append(f"{cam}: audio activity could not be read ({exc})")
        try:
            f = A._cached_flow(root, cam, manifest, fps, width, flow_audio)
            if f is not None:
                fs = per_second(f.astype(np.float64), fps)
                fs[fs <= 0] = np.nan
                flows[cam] = fs
            else:
                flags.append(f"{cam}: no cached flow; breaks use audio only")
        except Exception as exc:  # noqa: BLE001
            flags.append(f"{cam}: cached flow could not be read ({exc}); breaks use audio only")
        lines = Path(manifest).read_text().splitlines()
        from signals import _manifest_inputs
        paths, _ = _manifest_inputs(lines, str(root))
        durations = {p: _duration(p) for p in paths}
        for p_, d_ in durations.items():
            if d_ <= 0:
                flags.append(f"{cam}: could not read the length of {os.path.basename(p_)}; coverage may be short")
        layouts[cam] = manifest_layout(lines, durations, str(root))
        spans[cam] = [(b["start"], b["end"]) for b in layouts[cam]]
    if not act:
        raise RuntimeError("no camera audio activity; run the audio stage first")
    act_max = max_over(list(act.values()))
    signal_end = float(len(act_max))

    timing = league_timing(league_rules)
    start = game_start(whistles, act_max)
    if start["t"] is None:
        flags.append("no game whistles found; the game start is the start of the Recording")
        start["t"] = 0.0
    if timing["source"].startswith("default"):
        flags.append(f"no League period timing; assumed {timing['periods']} periods")
    elif timing["source"] != "League":
        flags.append(f"no period count in the League rules; assumed {timing['periods']} periods")
    cands = break_candidates(act_max, flows, start["t"] + 120, signal_end)
    plan = plan_periods(cands, start["t"], act_max, signal_end, whistles, timing)
    flags += plan["flags"]
    periods = plan["periods"]
    end = plan["end"]
    for p in periods:
        p["chapter"] = {cam: chapter_at(layouts[cam], p["start"]) for cam in cams}

    swap_skip = None
    if goalie:
        swaps, swap_skip = goalie_swaps(layouts, periods)
        for p in periods:
            brk = p.get("break_before")
            if brk is None:
                continue
            brk["goalie_swap"] = swaps.get(p["n"])
            # Not all Leagues switch ends before overtime.
            why = break_swap_flag(swaps.get(p["n"]) or {}) if p["n"] <= timing["periods"] else None
            if why:
                brk["reasons"].append(why)
                brk["uncertain"] = True
    else:
        swap_skip = "goalie swap check turned off (--no-goalie)"
    if swap_skip:
        flags.append(swap_skip)
    for p in periods:
        brk = p.get("break_before")
        if brk and brk["uncertain"]:
            flags.append(f"period {p['n']} start at {mmss(p['start'])} is uncertain: {'; '.join(brk['reasons'])}")
    if end.get("uncertain"):
        flags.append(f"game end at {mmss(end['t'])} is uncertain ({end['rule']})")

    cov, cov_flags = coverage(spans, periods, timing["periods"])
    flags += cov_flags
    return {
        "timeline": TIMELINE,
        "sync_offset_s": sig.get("sync_offset_s"),
        "game": {"start": start["t"], "start_rule": start["rule"], "first_whistle": start.get("whistle"),
                 "end": end["t"], "end_rule": end["rule"], "end_uncertain": bool(end.get("uncertain")),
                 "end_quiet_s": end.get("quiet_s"), "end_candidates": end.get("candidates", []),
                 "start_chapter": {cam: chapter_at(layouts[cam], start["t"]) for cam in cams},
                 "end_chapter": {cam: chapter_at(layouts[cam], end["t"]) for cam in cams}},
        "league_timing": timing,
        "periods": periods,
        "coverage": cov,
        "break_candidates": cands,
        "settings": SETTINGS,
        "flags": flags,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Period starts, game start and end, and camera coverage.")
    ap.add_argument("game_folder")
    ap.add_argument("--league", default="{}", help="League rules as JSON (periods, period_minutes, clock, ...)")
    ap.add_argument("--no-goalie", action="store_true", help="skip the goalie net swap check")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--flow_audio", action="store_true", help="detection ran with --audio_weight > 0")
    args = ap.parse_args(argv)
    root = Path(args.game_folder)
    fps, width = _fps_width(root)
    try:
        rules = json.loads(args.league or "{}")
    except json.JSONDecodeError:
        rules = {}
    try:
        out = analyse(root, rules, goalie=not args.no_goalie, fps=args.fps or fps, width=args.width or width,
                      flow_audio=args.flow_audio)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", flush=True)
        return 1
    (root / "coverage.json").write_text(json.dumps(out, indent=2) + "\n")
    g = out["game"]
    starts = ", ".join(mmss(p["start"]) for p in out["periods"])
    print(f"[coverage] {len(out['periods'])} periods (starts {starts}), game {mmss(g['start'])}-{mmss(g['end'])}",
          flush=True)
    for f in out["flags"]:
        print(f"[coverage] flag: {f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
