#!/usr/bin/env python3
"""Locate Game Sheet goals with the approved V1 rule (hhg-3r5.31).

All times use the detection timeline. Read flow and audio caches only. Missing
evidence gives flags, not signal extraction. Preserve every sheet goal.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "v2", "scripts"))

WEIGHTS = {"flow": 0.15, "whistle": 0.25, "quiet": 0.25, "clock": 0.25, "center_faceoff": 0.10}
SETTINGS = {"minimum_score": 0.55, "minimum_margin": 0.08, "clock_sigma_s": 90.0,
            "clock_slop_s": 15.0, "uncertain_slop_s": 120.0, "order_slop_s": 3.0,
             "moment_before_whistle_s": [0.25, 3.0], "pre_roll_s": 6.0, "post_roll_s": 4.0,
             "whistle_seed_offset_s": 1.2, "same_moment_s": 1.0,
            "quiet_low_activity": 0.35, "quiet_high_activity": 0.6}
TIMELINE = "detection (cam1 concat time; cam2 moved by the sync offset)"


def elapsed_time(clock_s, period_s, direction: str | None) -> float | None:
    if direction not in ("remaining", "elapsed") or clock_s is None or period_s is None:
        return None
    try:
        t, length = float(clock_s), float(period_s)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(t) or not math.isfinite(length) or not 0 <= t <= length or length <= 0:
        return None
    return length - t if direction == "remaining" else t


def boundary_uncertain(period: dict) -> bool:
    return bool((period.get("break_before") or {}).get("uncertain")
                or period.get("start_uncertain") or period.get("end_uncertain"))


def selection_periods(structure: dict) -> list[dict]:
    """Include uncertainty at both ends without changing Coverage's input."""
    periods = sorted((dict(p) for p in structure.get("periods") or []), key=lambda p: p["n"])
    for before, after in zip(periods, periods[1:]):
        if (after.get("break_before") or {}).get("uncertain"):
            before["end_uncertain"] = True
    if periods:
        game = structure.get("game") or {}
        fallback = game.get("start_rule") in (
            "no game whistles", "first game whistle (no faceoff activity found after it)")
        periods[0]["start_uncertain"] = bool(periods[0].get("start_uncertain")
                                             or game.get("start_uncertain") or fallback)
        periods[-1]["end_uncertain"] = bool(periods[-1].get("end_uncertain") or game.get("end_uncertain"))
    return periods


def clock_window(period: dict, elapsed: float | None, period_s: float | None,
                  mode: str | None) -> dict:
    start, end = float(period["start"]), float(period["end"])
    uncertain = boundary_uncertain(period)
    pad = SETTINGS["uncertain_slop_s"] if uncertain else SETTINGS["clock_slop_s"]
    if elapsed is None or period_s is None or mode not in ("stop", "running"):
        return {"start": start - pad, "end": end + pad, "expected": None, "sigma": None}
    if mode == "running":
        expected = start + elapsed
        lo, hi, sigma = expected - pad, expected + pad, pad
    else:
        # Stop time bounds order. The stretched period is only a weak prior.
        lo = start + elapsed - pad
        hi = end - (period_s - elapsed) + pad
        expected = start + (end - start) * elapsed / period_s
        sigma = max(SETTINGS["clock_sigma_s"], (end - start - period_s) * 0.25, pad)
    return {"start": lo, "end": hi, "expected": expected, "sigma": sigma}


def weighted_score(features: dict) -> float:
    """Missing features contribute zero. Never redistribute their weights."""
    return sum(w * float(np.clip(features.get(k) or 0.0, 0, 1)) for k, w in WEIGHTS.items())


def scoring_camera(coverage: dict, period: int, side: str, mapping: dict) -> str | None:
    if side not in ("home", "away"):
        return None
    opposite = {"home": "away", "away": "home"}[side]
    cams = [cam for cam, c in coverage.items()
            if any(p["n"] == period and mapping.get(p.get("defender")) == opposite
                   for p in c.get("periods", []))]
    return cams[0] if len(cams) == 1 else None


def covered(spans: list, t: float) -> bool:
    return any(float(a) <= t < float(b) for a, b in spans)


def segment_mean(a: np.ndarray, rate: float, lo: float, hi: float) -> float | None:
    if lo < 0 or hi * rate > len(a):
        return None
    values = a[round(lo * rate):round(hi * rate)]
    if not len(values) or not np.all(np.isfinite(values)):
        return None
    return float(np.mean(values))


def prepare_goals(sheet: dict, rules: dict) -> list[dict]:
    from scoresheet import parse_time
    length = rules.get("period_minutes")
    try:
        length = float(length) * 60 if length is not None else None
        if length is not None and (not math.isfinite(length) or length <= 0):
            length = None
    except (TypeError, ValueError):
        length = None
    direction = rules.get("time_direction")
    out = []
    for side in ("home", "away", "unknown"):
        previous = {}
        rows = (sheet.get("goals") or {}).get(side) or []
        if not isinstance(rows, list):
            rows = [rows]
        for i, claim in enumerate(rows):
            row = claim if isinstance(claim, dict) else {}
            per = str(row.get("per", ""))
            try:
                n = int(per) if per.isdecimal() else None
            except ValueError:
                n = None
            clock = row.get("time_s")
            if clock is None:
                clock = parse_time(str(row.get("time", "")))
            elapsed = elapsed_time(clock, length, direction)
            reasons = row.get("reasons") or []
            flags = [str(r) for r in reasons] if isinstance(reasons, list) else [str(reasons)]
            if not isinstance(claim, dict):
                flags.append("unreadable sheet goal row")
            if side == "unknown":
                flags.append("sheet team unavailable")
            if row.get("status") == "review":
                flags.append("sheet claim needs review")
            if n is None:
                flags.append("unsupported or unreadable period")
            if elapsed is None:
                flags.append("clock rules or sheet clock unavailable")
            if n in previous and elapsed is not None and elapsed < previous[n]:
                flags.append("sheet clocks disagree with row order")
            if elapsed is not None:
                previous[n] = elapsed
            out.append({"id": f"{side}:{i + 1}", "side": side, "team": (sheet.get("teams") or {}).get(side),
                         "sheet": claim, "period": n, "elapsed_s": elapsed, "period_s": length, "flags": flags})
    return sorted(out, key=lambda g: (g["period"] or 999, g["elapsed_s"] if g["elapsed_s"] is not None else math.inf,
                                      g["side"], int(g["id"].split(":")[1])))


def moment_and_flow(flow: dict, fps: int, t: float, whistle: bool) -> tuple[float, float | None]:
    net, slot = flow.get("net", np.zeros(0)), flow.get("slot", np.zeros(0))
    if not len(net) or len(net) != len(slot):
        return t, None
    near, far = SETTINGS["moment_before_whistle_s"]
    w = t + SETTINGS["whistle_seed_offset_s"]
    lo, hi = (w - far, w - near) if whistle else (t - 1.0, t + 1.0)
    a, b = max(0, round(lo * fps)), min(len(net), round(hi * fps))
    if b <= a:
        return t, None
    strength = 0.7 * net + 0.3 * slot
    window = strength[a:b]
    valid = np.isfinite(window)
    all_values = strength[np.isfinite(strength)]
    if not valid.any() or not len(all_values):
        return t, None
    peak = int(np.argmax(np.where(valid, window, -np.inf)))
    value = float(window[peak])
    # Clip against the 95th percentile; tiny game medians do not inflate scores.
    scale = max(float(np.percentile(all_values, 95)), 1e-6)
    return (a + peak) / fps, float(np.clip(value / scale, 0, 1))


def repeated_whistle(whistle: dict, audio: dict) -> bool:
    """Group a repeated whistle only when its stoppage membership agrees."""
    index = whistle.get("stoppage")
    stops = audio.get("stoppages") or []
    if whistle.get("role") != "in_stoppage" or type(index) is not int or not 0 <= index < len(stops):
        return False
    stop = stops[index]
    if not isinstance(stop, dict):
        return False
    try:
        t, start, end = float(whistle["t"]), float(stop["start"]), float(stop["end"])
        members = [float(v) for v in stop.get("whistles") or []]
        if not all(math.isfinite(v) for v in (t, start, end)) or not start < t <= end:
            return False
        if not any(abs(v - t) <= 0.01 for v in members) or not any(abs(v - start) <= 0.01 for v in members):
            return False
        # Keep the candidate if the initiating whistle is absent from the input.
        return any(w.get("role") == "stoppage" and w.get("stoppage") == index
                   and abs(float(w["t"]) - start) <= 0.01 for w in audio.get("whistles") or [])
    except (KeyError, TypeError, ValueError):
        return False


def build_candidates(audio: dict, events: list[dict], flows: dict, activities: dict,
                     coverage: dict, fps: int, flags: list[str] | None = None) -> list[dict]:
    from selection_inputs import checked_audio
    audio = checked_audio(audio, flags if flags is not None else [])
    seeds: list[tuple[float, float | None, str]] = [(float(w["t"]) - SETTINGS["whistle_seed_offset_s"], float(w["t"]), f"whistle:{i}")
             for i, w in enumerate(audio.get("whistles") or [])
             if "t" in w and not repeated_whistle(w, audio)]
    for i, e in enumerate(events):
        try:
            cam = "cam" + str(e["primary_cam"]).removeprefix("cam")
            a, b = float(e["start_s"]), float(e["end_s"])
            net = flows.get(cam, {}).get("net", np.zeros(0))
            lo, hi = max(0, round(a * fps)), min(len(net), round(b * fps))
            if hi <= lo or not np.isfinite(net[lo:hi]).any():
                continue
            t = (lo + int(np.nanargmax(net[lo:hi]))) / fps
            if any(abs(t - s) < 3 and abs(t - moment_and_flow(flows.get(cam, {}), fps, s, w is not None)[0]) < 1
                   for s, w, _ in seeds):
                continue
            seeds.append((t, None, f"flow:{i}"))
        except (KeyError, ValueError, TypeError):
            continue
    out = []
    for t, whistle_t, cid in sorted(seeds):
        per_cam = {}
        for cam, cov in coverage.items():
            moment, flow_score = moment_and_flow(flows.get(cam, {}), fps, t, whistle_t is not None)
            if not covered(cov.get("spans", []), moment):
                continue
            activity = activities.get(cam, np.zeros(0))
            base = whistle_t if whistle_t is not None else t
            late = [segment_mean(activity, 100, base + a, base + b) for a, b in ((5, 20), (10, 25))]
            finite = [v for v in late if v is not None]
            # The existing audio detector treats activity <= 0.35 as quiet.
            low, high = SETTINGS["quiet_low_activity"], SETTINGS["quiet_high_activity"]
            quiet = float(np.clip((high - min(finite)) / (high - low), 0, 1)) if finite else None
            stops = [s for s in audio.get("stoppages") or [] if "start" in s and abs(float(s["start"]) - base) <= 3]
            duration = max((float(s["end"]) - float(s["start"]) for s in stops), default=None)
            if quiet is not None and duration is not None:
                quiet *= min(1.0, duration / 20.0)
            evidence_flags = []
            net = flows.get(cam, {}).get("net", np.zeros(0))
            index = round(moment * fps)
            net_motion = bool(0 <= index < len(net) and np.isfinite(net[index]) and net[index] > 0)
            if not net_motion:
                evidence_flags.append("scoring-net motion unavailable or zero")
            if flow_score is None:
                evidence_flags.append("flow evidence unavailable")
            if quiet is None:
                evidence_flags.append("quiet evidence unavailable")
            per_cam[cam] = {"t": moment, "features": {"flow": flow_score,
                             "whistle": 1.0 if whistle_t is not None else 0.0,
                             "quiet": quiet, "center_faceoff": None},
                             "whistle_t": whistle_t, "stoppage_s": duration,
                             "net_motion": net_motion, "flags": evidence_flags}
        if per_cam:
            out.append({"id": cid, "t": t, "cameras": per_cam})
    return out


def edge_score(goal: dict, candidate: dict, period: dict, coverage: dict, mapping: dict,
               rules: dict) -> dict | None:
    cam = scoring_camera(coverage, period["n"], goal["side"], mapping)
    evidence = candidate["cameras"].get(cam)
    if evidence is None or goal["elapsed_s"] is None or rules.get("clock") not in ("stop", "running"):
        return None
    if not evidence.get("net_motion"):
        return None
    window = clock_window(period, goal["elapsed_s"], goal["period_s"], rules.get("clock"))
    t = evidence["t"]
    if not window["start"] <= t <= window["end"]:
        return None
    clock = (math.exp(-0.5 * ((t - window["expected"]) / window["sigma"]) ** 2)
             if window["expected"] is not None else None)
    # Audio restart is uncertain. Penalize an impossible clock tail softly.
    duration = evidence.get("stoppage_s")
    if rules.get("clock") == "stop" and duration is not None and clock is not None:
        remaining = goal["period_s"] - goal["elapsed_s"]
        base = evidence["whistle_t"] if evidence.get("whistle_t") is not None else t
        available = period["end"] - (base + duration)
        slop = SETTINGS["uncertain_slop_s"] if boundary_uncertain(period) else SETTINGS["clock_slop_s"]
        clock *= math.exp(-0.5 * (max(0, remaining - available) / slop) ** 2)
    features = dict(evidence["features"], clock=clock)
    return {"camera": cam, "t": t, "score": weighted_score(features), "features": features,
            "window": window, "flags": evidence["flags"]}


def order_penalty(before: dict, goal: dict, before_candidate: dict, candidate: dict,
                  before_edge: dict, edge: dict, mode: str | None) -> float:
    """A stoppage can explain wall time, but it cannot prove the clock stopped."""
    if mode != "stop" or before.get("period") != goal.get("period"):
        return 0.0
    duration = (before_candidate.get("cameras", {}).get(before_edge.get("camera"), {}).get("stoppage_s"))
    clock = edge.get("features", {}).get("clock")
    if duration is None or clock is None or before["elapsed_s"] is None or goal["elapsed_s"] is None:
        return 0.0
    elapsed = goal["elapsed_s"] - before["elapsed_s"]
    available = edge.get("t", candidate["t"]) - before_edge.get("t", before_candidate["t"]) - duration
    factor = math.exp(-0.5 * (max(0, elapsed - available) / SETTINGS["clock_slop_s"]) ** 2)
    return WEIGHTS["clock"] * clock * (1 - factor)


def ordered_match(goals: list[dict], candidates: list[dict], edges: dict,
                  mode: str | None, forbidden: tuple | None = None) -> tuple[float, dict]:
    """Best increasing match, with an unmatched option and no candidate reuse."""
    latest = {}
    for (_, j), edge in edges.items():
        latest[j] = max(latest.get(j, -math.inf), edge.get("t", candidates[j]["t"]))
    # Keep used candidates only while their later camera moments can recur.
    states = {(-1, -1, frozenset()): (0.0, {})}
    for i, goal in enumerate(goals):
        next_states = dict(states)
        for (last, last_goal, used), (total, path) in states.items():
            for j in range(len(candidates)):
                edge = edges.get((i, j))
                if (j in used or edge is None or edge["score"] < SETTINGS["minimum_score"]
                        or forbidden == (i, j)):
                    continue
                penalty = 0.0
                if last >= 0:
                    delta = edge.get("t", candidates[j]["t"]) - edges[last_goal, last].get("t", candidates[last]["t"])
                    if delta < 2:
                        continue
                    before = goals[last_goal]
                    if (mode == "stop" and before.get("period") == goal.get("period")
                            and before["elapsed_s"] is not None and goal["elapsed_s"] is not None):
                        if delta + SETTINGS["order_slop_s"] < goal["elapsed_s"] - before["elapsed_s"]:
                            continue
                    penalty = order_penalty(before, goal, candidates[last], candidates[j],
                                            edges[last_goal, last], edge, mode)
                score = total + edge["score"] - penalty - SETTINGS["minimum_score"]
                active = frozenset(k for k in used | {j}
                                   if latest[k] >= edge.get("t", candidates[j]["t"]) + 2)
                key = (j, i, active)
                if key not in next_states or score > next_states[key][0]:
                    next_states[key] = (score, dict(path, **{str(i): j}))
        states = next_states
    score, path = max(states.values(), key=lambda v: v[0])
    return score, {int(i): j for i, j in path.items()}


def match_goals(goals: list[dict], candidates: list[dict], periods: list[dict], coverage: dict,
                 mapping: dict, rules: dict) -> dict:
    by_n = {p["n"]: p for p in periods}
    edges = {(i, j): edge for i, g in enumerate(goals) for j, c in enumerate(candidates)
             if g["period"] in by_n
             and (edge := edge_score(g, c, by_n[g["period"]], coverage, mapping, rules)) is not None}
    score, path = ordered_match(goals, candidates, edges, rules.get("clock"))
    results = {}
    def path_edge(i, j, matched):
        edge = dict(edges[i, j])
        previous = [k for k in matched if k < i]
        penalty = 0.0
        if previous:
            k = max(previous)
            penalty = order_penalty(goals[k], goals[i], candidates[matched[k]], candidates[j],
                                    edges[k, matched[k]], edge, rules.get("clock"))
        edge["features"] = dict(edge["features"])
        if edge["features"]["clock"] is not None:
            edge["features"]["clock"] -= penalty / WEIGHTS["clock"]
        edge["score"] -= penalty
        edge["order_penalty"] = penalty
        return edge

    for i, j in path.items():
        chosen = edges[i, j]
        # Compare distinct plays, not different seeds for the same camera moment.
        alternatives = {key: edge for key, edge in edges.items()
                        if not (key[0] == i and edge["camera"] == chosen["camera"]
                                and abs(edge["t"] - chosen["t"]) <= SETTINGS["same_moment_s"])}
        alternate, alternate_path = ordered_match(goals, candidates, alternatives, rules.get("clock"))
        runner = alternate_path.get(i)
        margin = score - alternate
        results[goals[i]["id"]] = {"edge": path_edge(i, j, path), "candidate": candidates[j], "margin": margin,
                                  "runner_up": ({"candidate_id": candidates[runner]["id"], **path_edge(i, runner, alternate_path)}
                                                if runner is not None else None)}
    return {"score": score, "results": results}


def select_goals(sheet: dict, structure: dict, audio: dict, events: list[dict], flows: dict,
                 activities: dict, layouts: dict, rules: dict, options: dict | None = None,
                 fps: int = 12, input_flags: list[str] | None = None) -> dict:
    import coverage as C
    from selection_inputs import checked_structure, checked_sheet
    options = options or {}
    settings = dict(SETTINGS)
    flags = list(input_flags or [])
    sheet = checked_sheet(sheet, flags)
    structure = checked_structure(structure, flags)
    upstream_flags = {"scoresheet": sheet["flags"], "coverage": structure["flags"]}
    flags += ["center faceoff evidence unavailable; contribution is zero"]
    if "minimum_margin" in options:
        try:
            margin = float(options["minimum_margin"])
            if not math.isfinite(margin) or not 0 <= margin <= 1:
                raise ValueError("must be between zero and one")
            settings["minimum_margin"] = margin
        except (TypeError, ValueError):
            flags.append("invalid minimum_margin; using the default")
    if rules.get("clock") not in ("stop", "running"):
        flags.append("League clock mode unavailable")
    goals = prepare_goals(sheet, rules)
    coverage = structure.get("coverage") or {}
    periods = selection_periods(structure)
    candidates = build_candidates(audio, events, flows, activities, coverage, fps, flags)
    expected_periods = rules.get("periods", (structure.get("league_timing") or {}).get("periods"))
    if isinstance(expected_periods, str) and expected_periods.isdecimal():
        try:
            expected_periods = int(expected_periods)
        except ValueError:
            pass
    numbers = [p["n"] for p in periods]
    numbering_uncertain = (len(set(numbers)) != len(numbers)
                           or numbers != list(range(1, len(numbers) + 1)))
    if expected_periods is not None:
        invalid_count = type(expected_periods) is not int or expected_periods < 1
        if invalid_count:
            flags.append("invalid League periods; expected a positive integer")
        numbering_uncertain |= invalid_count or len(periods) != expected_periods
    if not goals:
        flags.append("no readable Game Sheet goals; video-only fallback is not part of V1")
    explicit = options.get("defender_teams")
    mappings = [explicit] if explicit is not None else [{"A": "home", "B": "away"}, {"A": "away", "B": "home"}]
    mappings = [m for m in mappings if isinstance(m, dict) and set(m) == {"A", "B"}
                and sorted(str(v) for v in m.values()) == ["away", "home"]]
    if explicit is not None and not mappings:
        flags.append("invalid defender_teams; expected A/B mapped to home/away")
    plans = []
    for mapping in mappings:
        plan = match_goals(goals, candidates, [] if numbering_uncertain else periods, coverage, mapping, rules)
        plans.append({"mapping": mapping, **plan})
    plans.sort(key=lambda p: -p["score"])
    mapping_margin = plans[0]["score"] - plans[1]["score"] if len(plans) > 1 else None
    ambiguous_mapping = not plans or (mapping_margin is not None and mapping_margin < settings["minimum_margin"])
    mapping = None if ambiguous_mapping else plans[0]["mapping"]
    if ambiguous_mapping:
        flags.append("defender-to-sheet-team mapping unavailable or ambiguous")
    results = plans[0]["results"] if plans else {}
    output_goals = []
    for g in goals:
        item: dict = dict(g, chosen=None, score=None, margin=None, runner_up=None, status="no clip found")
        item["flags"] = list(g["flags"])
        if numbering_uncertain:
            item["flags"].append("period numbering uncertain; check the Coverage period plan")
        period = next((p for p in periods if p["n"] == g["period"]), None)
        if period is None:
            item["flags"].append("period coverage unavailable")
        elif boundary_uncertain(period):
            item["flags"].append("period boundary uncertain; clock window widened")
        match = results.get(g["id"])
        if not match and period is not None and mapping is not None:
            cam = scoring_camera(coverage, period["n"], g["side"], mapping)
            window = clock_window(period, g["elapsed_s"], g["period_s"], rules.get("clock"))
            nearby = [c["cameras"][cam] for c in candidates if cam in c["cameras"]
                      and window["start"] <= c["cameras"][cam]["t"] <= window["end"]]
            if nearby and not any(e.get("net_motion") for e in nearby):
                item["flags"].append("scoring-net motion unavailable or zero")
        if match:
            edge = match["edge"]
            item.update(score=round(edge["score"], 4), margin=round(match["margin"], 4),
                        runner_up=match["runner_up"], features=edge["features"], window=edge["window"],
                        best_candidate={"candidate_id": match["candidate"]["id"], **edge})
            item["flags"].extend(edge["flags"])
            chapters = {cam: (C.chapter_at(layout, edge["t"])
                             if covered(coverage.get(cam, {}).get("spans", []), edge["t"]) else None)
                        for cam, layout in layouts.items()}
            if ambiguous_mapping:
                item["flags"].append("defender-to-sheet-team mapping ambiguous")
            elif match["margin"] < settings["minimum_margin"]:
                item["flags"].append("goal moment ambiguous")
            elif chapters.get(edge["camera"]) is None:
                item["flags"].append("scoring-camera chapter unavailable")
            elif any("row order" in f for f in item["flags"]):
                item["flags"].append("sheet clock order needs review")
            else:
                t = edge["t"]
                spans = coverage[edge["camera"]].get("spans", [])
                a, b = next((a, b) for a, b in spans if a <= t < b)
                item["chosen"] = {"candidate_id": match["candidate"]["id"], "detection_s": round(t, 3),
                                  "primary_cam": edge["camera"], "chapters": chapters,
                                  "start_s": round(max(a, t - SETTINGS["pre_roll_s"]), 3),
                                  "end_s": round(min(b, t + SETTINGS["post_roll_s"]), 3)}
                item["status"] = "selected"
                for cam in coverage:
                    if chapters.get(cam) is None:
                        item["flags"].append(f"{cam}: no coverage at goal moment")
        if item["chosen"] is None:
            item["flags"].append("no clip found")
        flags.extend(f"{g['id']}: {flag}" for flag in item["flags"])
        output_goals.append(item)
    return {"schema_version": 1, "timeline": TIMELINE, "rules": rules, "settings": settings,
            "weights": WEIGHTS, "mapping": {"defender_teams": mapping, "source": "explicit" if explicit is not None else "inferred",
                                            "margin": mapping_margin},
            "periods": periods, "goals": output_goals, "candidate_count": len(candidates),
            "input_flags": upstream_flags, "flags": list(dict.fromkeys(flags))}


def analyse(root: Path, rules: dict, options: dict | None = None, fps: int = 12,
            width: int = 1280, flow_audio: bool = False) -> dict:
    from selection_inputs import read_json, cached_camera, checked_structure
    flags = []
    sheet = read_json(root / "game_sheet.json", flags)
    structure = checked_structure(read_json(root / "coverage.json", flags), flags)
    audio = read_json(root / "audio_signals.json", flags)
    events = []
    try:
        with (root / "events.csv").open() as f:
            events = list(csv.DictReader(f))
    except OSError:
        flags.append("events.csv unavailable; candidates use whistles only")
    flows, activities, layouts = {}, {}, {}
    for cam in structure.get("coverage") or {}:
        flows[cam], activities[cam], layouts[cam] = cached_camera(root, cam, fps, width, flow_audio, flags)
    return select_goals(sheet, structure, audio, events, flows, activities, layouts, rules, options, fps, flags)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("game_folder", type=Path)
    ap.add_argument("--league", default="{}", help="League rules, including time_direction, as JSON")
    ap.add_argument("--options", default="{}", help="Selection options as JSON (defender_teams, minimum_margin)")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--flow_audio", action="store_true")
    args = ap.parse_args()
    result = analyse(args.game_folder.resolve(), json.loads(args.league), json.loads(args.options),
                     args.fps, args.width, args.flow_audio)
    output = args.game_folder / "selection.json"
    tmp = output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(output)
    selected = sum(g["chosen"] is not None for g in result["goals"])
    print(f"[selection] {selected}/{len(result['goals'])} goals selected, {len(result['goals']) - selected} no clip found")
    for flag in result["flags"]:
        print(f"[selection] flag: {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
