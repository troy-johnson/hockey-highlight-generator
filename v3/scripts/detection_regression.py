"""Check detection against read-only answer keys without changing Game Folders."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from uuid import uuid4
from zipfile import BadZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "v2" / "scripts"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

import detect_events
import signals
from recap_options import DEFAULTS, parse_set

RULES = {
    "version": 1,
    "interval": "start_s <= reference_s < end_s; no tolerance",
    "goal_recall": "reference point in any event, regardless of primary camera",
    "clip_match": "one-to-one maximum matching; contained point and same primary camera",
    "incomplete_key": "unmatched events with no reference point are unjudged, not incorrect",
}
METRICS = ("goals_found", "goal_recall", "missed_goal_count", "correct_clip_matches",
           "incorrect_clip_matches", "unjudged_events", "event_count", "non_goals_found",
           "non_goal_recall", "missed_non_goal_count", "goal_clip_matches", "non_goal_clip_matches")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def number(value, name) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name}: expected a finite number")
    n = float(value)
    if not math.isfinite(n):
        raise ValueError(f"{name}: expected a finite number")
    return n


def load_key(path: Path) -> dict:
    key = json.loads(path.read_text())
    if key.get("schema_version") != 1 or key.get("timeline_kind") != "detection":
        raise ValueError("answer key requires schema_version=1 and timeline_kind=detection")
    if not isinstance(key.get("clips_complete"), bool):
        raise ValueError("answer key requires an explicit clips_complete boolean")
    if not key.get("timeline"):
        raise ValueError("answer key requires its camera timeline mapping")
    ids = set()
    for clip in key["clips"]:
        cid = clip["id"]
        if not isinstance(cid, str) or not cid or cid in ids:
            raise ValueError("answer key clip IDs must be unique nonempty strings")
        ids.add(cid)
        clip["reference_s"] = number(clip["reference_s"], "reference_s")
        if clip["reference_s"] < 0 or clip["camera"] not in ("cam1", "cam2"):
            raise ValueError(f"{cid}: invalid reference time or camera")
        if not isinstance(clip.get("goal"), bool):
            raise ValueError(f"{cid}: goal must be explicitly true or false")
    return key


def read_events(path: Path) -> list[dict]:
    events = []
    with path.open(newline="") as f:
        for i, row in enumerate(csv.DictReader(f), 1):
            start, end = (number(row[k], k) for k in ("start_s", "end_s"))
            cam = row["primary_cam"]
            if start < 0 or end <= start or cam not in ("1", "2"):
                raise ValueError(f"event {i}: invalid interval or primary camera")
            events.append({"id": f"event:{i}", "start_s": start, "end_s": end,
                           "camera": f"cam{cam}"})
    return events


def contains(event: dict, clip: dict) -> bool:
    return event["start_s"] <= clip["reference_s"] < event["end_s"]


def match_clips(clips: list[dict], events: list[dict]) -> dict[int, int]:
    """Find a deterministic maximum-cardinality matching, not a greedy assignment."""
    edges = [sorted((i for i, event in enumerate(events)
                     if event["camera"] == clip["camera"] and contains(event, clip)),
                    key=lambda i: (abs((events[i]["start_s"] + events[i]["end_s"]) / 2
                                       - clip["reference_s"]), i)) for clip in clips]
    owners = {}

    def assign(ci, visited):
        for ei in edges[ci]:
            if ei in visited:
                continue
            visited.add(ei)
            if ei not in owners or assign(owners[ei], visited):
                owners[ei] = ci
                return True
        return False

    for ci in range(len(clips)):
        assign(ci, set())
    return {ci: ei for ei, ci in owners.items()}


def evaluate(key: dict, events: list[dict]) -> dict:
    clips = key["clips"]
    goals = [c for c in clips if c["goal"]]
    non_goals = [c for c in clips if not c["goal"]]
    missed = [c for c in goals if not any(contains(e, c) for e in events)]
    missed_non_goals = [c for c in non_goals if not any(contains(e, c) for e in events)]
    matched = match_clips(clips, events)
    used = set(matched.values())
    incorrect, unjudged = [], []
    for ei, event in enumerate(events):
        if ei in used:
            continue
        refs = [c for c in clips if contains(event, c)]
        if refs:
            reason = "duplicate" if any(c["camera"] == event["camera"] for c in refs) else "wrong_camera"
        elif key["clips_complete"]:
            reason = "no_reference"
        else:
            unjudged.append(event["id"])
            continue
        incorrect.append({"event_id": event["id"], "reason": reason,
                          "reference_ids": [c["id"] for c in refs]})
    found = len(goals) - len(missed)
    non_goals_found = len(non_goals) - len(missed_non_goals)
    return {
        "goals_total": len(goals), "goals_found": found,
        "goal_recall": found / len(goals) if goals else None,
        "missed_goal_count": len(missed), "missed_goals": missed,
        "correct_clip_matches": len(matched), "incorrect_clip_matches": len(incorrect),
        "goal_clip_matches": sum(clips[ci]["goal"] for ci in matched),
        "non_goal_clip_matches": sum(not clips[ci]["goal"] for ci in matched),
        "non_goals_total": len(non_goals), "non_goals_found": non_goals_found,
        "non_goal_recall": non_goals_found / len(non_goals) if non_goals else None,
        "missed_non_goal_count": len(missed_non_goals), "missed_non_goals": missed_non_goals,
        "matches": [{"clip_id": clips[ci]["id"], "event_id": events[ei]["id"]}
                    for ci, ei in sorted(matched.items())],
        "incorrect_matches": incorrect, "unjudged_events": len(unjudged),
        "unjudged_event_ids": unjudged, "event_count": len(events),
        "reference_clips": len(clips), "clips_complete": key["clips_complete"],
        "unmatched_clip_ids": [c["id"] for i, c in enumerate(clips) if i not in matched],
    }


def compare(current: dict, previous: dict | None) -> dict:
    if previous is None:
        return {"status": "first_result"}
    if current["comparison_key"] != previous.get("comparison_key"):
        return {"status": "incomparable", "reason": "answer key, rules, or input timeline changed"}
    a, b = current["metrics"], previous["metrics"]
    delta = {k: a[k] - b[k] if a.get(k) is not None and b.get(k) is not None else None for k in METRICS}
    am, bm = ({c["id"] for c in m["missed_goals"]} for m in (a, b))
    ac, bc = ({c["clip_id"] for c in m["matches"]} for m in (a, b))
    runtime = (current["runtime_s"] - previous["runtime_s"]
               if current["runtime_kind"] == previous.get("runtime_kind") else None)
    return {"status": "compared", "previous_run": previous["run_id"], "delta": delta,
            "previous_candidate_source": previous.get("candidate_source", "flow"),
            "runtime_delta_s": runtime, "newly_missed_goals": sorted(am - bm),
            "recovered_goals": sorted(bm - am), "lost_clip_matches": sorted(bc - ac),
            "new_clip_matches": sorted(ac - bc)}


def previous_results(results: Path) -> dict:
    previous = {}
    for path in sorted(results.glob("*/result.json"), reverse=True):
        saved = json.loads(path.read_text())
        if saved.get("schema_version") != 1:
            raise ValueError(f"unsupported saved result: {path}")
        for game in saved["games"]:
            if game.get("status") == "ok":
                previous.setdefault(game["game_id"], game)
    return previous


def manifest_parts(path: Path) -> list[dict]:
    """Use the extractor's Recording and seek rules for cache identity and placement."""
    blocks = signals._recording_blocks(str(path))
    lines = path.read_text().splitlines()
    if len(blocks) < 2:
        blocks = [(0.0, lines)]
    else:
        blocks = [(start, ["ffconcat version 1.0"]
                   + ([f"# seek {-start:.3f}"] if start < 0 else []) + files)
                  for start, files in blocks]
    parts = []
    for start, lines in blocks:
        files, seek = signals._manifest_inputs(lines, str(path.parent))
        if not files:
            raise ValueError(f"empty manifest: {path}")
        parts.append({"start_s": start, "seek_s": seek, "files": files})
    return parts


def validate_signal_arrays(arrays) -> tuple:
    error = "invalid signal arrays: expected three nonempty finite 1-D arrays with equal lengths"
    try:
        validated = tuple(np.asarray(a, dtype=np.float32) for a in arrays)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if (len(validated) != 3
            or any(a.ndim != 1 or not len(a) or not np.isfinite(a).all() for a in validated)
            or len({len(a) for a in validated}) != 1):
        raise ValueError(error)
    return validated


def load_camera_cache(parts: list[dict], rois: dict, settings: dict,
                      cache_dirs: list[Path], *, processing_cache: Path | None = None
                      ) -> tuple[tuple | None, list[str]]:
    placed, misses = [], []
    for part in parts:
        key = signals.signal_cache_key(part["files"], part["seek_s"], rois,
                                       settings["fps"], settings["width"], settings["audio_weight"] > 0)
        arrays = None
        for folder in cache_dirs:
            try:
                with np.load(folder / f"signals_{key}.npz", allow_pickle=False) as z:
                    arrays = validate_signal_arrays(z[k] for k in ("net", "slot", "audio"))
                break
            except (OSError, ValueError, KeyError, EOFError, TypeError, BadZipFile):
                continue
        if arrays is None:
            misses.append(key)
            if processing_cache is not None:
                processing_cache.mkdir(parents=True, exist_ok=True)
                # Process only this missing Recording, not the cached Recordings beside it.
                with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", dir=processing_cache,
                                                 delete=False) as manifest:
                    manifest.write("ffconcat version 1.0\n")
                    manifest.write(f"# seek {part['seek_s']:.3f}\n")
                    for filename in part["files"]:
                        manifest.write(f"file '{filename}'\n")
                try:
                    arrays = signals.extract_signals(manifest.name, rois, settings["fps"], settings["width"],
                                                     with_audio=settings["audio_weight"] > 0,
                                                     cache_dir=None)
                finally:
                    Path(manifest.name).unlink()
                arrays = validate_signal_arrays(arrays)
                with tempfile.TemporaryDirectory(dir=processing_cache) as staging:
                    fresh = Path(staging) / f"signals_{key}.npz"
                    np.savez(fresh, net=arrays[0], slot=arrays[1], audio=arrays[2])
                    os.replace(fresh, processing_cache / fresh.name)
        if arrays is not None:
            placed.append((int(round(max(part["start_s"], 0) * settings["fps"])), arrays))
    if len(placed) != len(parts):
        return None, misses
    n = max(i + len(a[0]) for i, a in placed)
    output = tuple(np.zeros(n, dtype=np.float32) for _ in range(3))
    for i, arrays in placed:
        for out, array in zip(output, arrays):
            out[i:i + len(array)] = array
    return output, misses


def detection_args(game: Path, output: Path, settings: dict) -> list[str]:
    args = [str(game / "cam1_concat.txt"), str(game / "cam2_concat.txt"), str(game / "rois.json")]
    for key, value in settings.items():
        flag = "timeline_fps" if key == "marker_fps" else key
        if isinstance(value, bool):
            if value:
                args.append(f"--{flag}")
        else:
            args.extend((f"--{flag}", str(value)))
    return args + ["--out_csv", str(output / "events.csv"),
                   "--out_markers", str(output / "markers.csv")]


def run_game(entry: dict, key: dict, game: Path, output: Path, cache: Path,
              settings: dict, allow_processing: bool, candidate_source: str) -> dict:
    started = time.perf_counter()
    parts = {f"cam{n}": manifest_parts(game / f"cam{n}_concat.txt") for n in (1, 2)}
    timeline = {cam: [{**p, "files": [Path(f).name for f in p["files"]]} for p in ps]
                for cam, ps in parts.items()}
    if timeline != key["timeline"]:
        raise ValueError("camera timeline differs from the answer key; remap the key before evaluation")
    rois = signals.load_rois(str(game / "rois.json"))
    cached, misses = [], []
    def extract(cam1, cam2, rois, fps, width, verbose, with_audio=True, cache_dir=None):
        for n in (1, 2):
            arrays, absent = load_camera_cache(parts[f"cam{n}"], rois[f"camera_{n}"], settings,
                                              [game / ".recap_cache" / "signals", cache],
                                              processing_cache=cache if allow_processing else None)
            cached.append(arrays)
            misses.extend(absent)
        if any(a is None for a in cached):
            raise ValueError(f"{len(misses)} missing or invalid signal caches; processing requires approval and --allow-processing")
        return tuple(cached)

    with (output / "detection.log").open("w") as log, contextlib.redirect_stdout(log):
        detect_events.main(detection_args(game, output, settings), signal_extractor=extract)
    evaluated_file = "events.csv"
    candidate_inputs = {}
    if candidate_source in ("whistles", "combined"):
        from whistle_candidates import write_whistle_events
        origins = {cam: max(ps[0]["start_s"], 0.0) - ps[0]["seek_s"] for cam, ps in parts.items()}
        candidate_inputs = write_whistle_events(game, output, settings, origins["cam2"] - origins["cam1"])
        evaluated_file = "whistle_events.csv"
    if candidate_source == "combined":
        from combined_candidates import write_combined_events
        candidate_inputs.update(write_combined_events(output, settings["max_win_s"]))
        evaluated_file = "combined_events.csv"
    metrics = evaluate(key, read_events(output / evaluated_file))
    input_identity = {cam: [signals._file_identity(f) for p in ps for f in p["files"]]
                      for cam, ps in parts.items()}
    return {"game_id": entry["id"], "status": "ok", "metrics": metrics,
            "candidate_source": candidate_source, "evaluated_events_file": evaluated_file,
            "candidate_input_hashes": candidate_inputs,
            "runtime_s": round(time.perf_counter() - started, 6),
            "runtime_kind": "processing" if misses else "cached_signals",
            "missing_cache_keys": misses, "settings": settings,
            "answer_key_hash": digest(key),
            "comparison_key": digest({"key": key, "rules": RULES, "inputs": input_identity})}


def validate_paths(results: Path, suite: Path, entries: list[dict], games_root: Path) -> None:
    protected = [suite.parent]
    protected.extend((suite.parent / e["answer_key"]).resolve().parent for e in entries)
    protected.extend((games_root / e["folder"]).resolve() for e in entries)
    for path in protected:
        if results == path or path in results.parents or results in path.parents:
            raise ValueError(f"results must be separate from answer keys and Game Folders: {path}")


def run_suite(suite: Path, games_root: Path, results: Path, *, allow_processing=False,
              detection_overrides: dict | None = None, candidate_source="flow") -> dict:
    suite, games_root, results = (p.resolve() for p in (suite, games_root, results))
    config = json.loads(suite.read_text())
    if candidate_source not in ("flow", "whistles", "combined"):
        raise ValueError("candidate_source must be flow, whistles, or combined")
    if config.get("schema_version") != 1 or not config.get("games"):
        raise ValueError("suite requires schema_version=1 and at least one answer-key game")
    entries = config["games"]
    ids = [e["id"] for e in entries]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", i) for i in ids):
        raise ValueError("suite game IDs must be unique safe directory names")
    validate_paths(results, suite, entries, games_root)
    previous = previous_results(results)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]
    run_dir = results / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    settings = {**DEFAULTS["detection"], **config.get("detection", {}), **(detection_overrides or {})}
    allowed = set(DEFAULTS["detection"]) | {"flow_weight", "w_attack", "w_rebound", "w_drop", "preserve_peaks"}
    if set(settings) - allowed:
        raise ValueError(f"unsupported detection settings: {sorted(set(settings) - allowed)}")
    report = {"schema_version": 1, "run_id": run_id, "rules": RULES,
              "detector_hash": hashlib.sha256(Path(detect_events.__file__).read_bytes()).hexdigest(),
              "signals_hash": signals._code_hash(), "games": []}
    for entry in entries:
        output = run_dir / entry["id"]
        output.mkdir()
        try:
            key = load_key(suite.parent / entry["answer_key"])
            result = run_game(entry, key, games_root / entry["folder"], output,
                               results / "cache" / entry["id"], settings, allow_processing, candidate_source)
            result["run_id"] = run_id
            result["comparison"] = compare(result, previous.get(entry["id"]))
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            result = {"game_id": entry["id"], "status": "blocked", "error": str(exc), "run_id": run_id}
        report["games"].append(result)
    # Each result is immutable. A failed game does not replace its last successful result.
    with (run_dir / "result.json").open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    return report


def print_report(report: dict) -> None:
    for game in report["games"]:
        if game["status"] != "ok":
            print(f"{game['game_id']}: BLOCKED: {game['error']}")
            continue
        m = game["metrics"]
        recall = f"{m['goal_recall']:.1%}" if m["goal_recall"] is not None else "N/A"
        print(f"{game['game_id']} [{game['candidate_source']}]: goals {m['goals_found']}/{m['goals_total']} ({recall}); "
              f"clips correct {m['correct_clip_matches']}/{m['reference_clips']}, "
              f"incorrect {m['incorrect_clip_matches']}, unjudged {m['unjudged_events']}; "
              f"runtime {game['runtime_s']:.3f}s ({game['runtime_kind']})")
        print("  Missed goals: " + (", ".join(f"{g['id']} at {g['reference_s']:.3f}s ({g['camera']})"
                                                for g in m["missed_goals"]) or "none"))
        if m.get("non_goals_total", 0):
            print(f"  Known non-goals: {m['non_goals_found']}/{m['non_goals_total']}; "
                  f"correct-camera matches {m['non_goal_clip_matches']}; coverage, not classification accuracy.")
            print("  Missed non-goals: " + (", ".join(c["id"] for c in m["missed_non_goals"]) or "none"))
        comparison = game["comparison"]
        if comparison["status"] == "compared":
            print("  Change: " + json.dumps(comparison, sort_keys=True))
        else:
            print("  Change: " + comparison["status"] + (": " + comparison.get("reason", "") if comparison.get("reason") else ""))
        if not m["clips_complete"]:
            print("  Clip key is incomplete; unjudged events are not false positives.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "tests" / "answer_keys" / "suite.json")
    parser.add_argument("--games-root", type=Path, default=os.environ.get("HHG_GAME_ROOT"))
    parser.add_argument("--results", type=Path, default=ROOT / ".regression-results")
    parser.add_argument("--candidate-source", choices=("flow", "whistles", "combined"), default="flow",
                        help="evaluate flow, cached whistles, or combined proposals; always keep the original pools")
    parser.add_argument("--set", action="append", default=[], metavar="detection.KEY=VALUE",
                        help="override detection settings for this run only; never change keys or Game Folder options")
    parser.add_argument("--allow-processing", action="store_true",
                        help="allow footage extraction on cache misses; obtain approval first")
    args = parser.parse_args(argv)
    if args.games_root is None:
        parser.error("provide --games-root or HHG_GAME_ROOT")
    try:
        overrides = parse_set(args.set)
        if set(overrides) - {"detection"} or not isinstance(overrides.get("detection", {}), dict):
            raise ValueError("--set only accepts detection.KEY=VALUE")
        report = run_suite(args.suite, args.games_root, args.results, allow_processing=args.allow_processing,
                           detection_overrides=overrides.get("detection", {}), candidate_source=args.candidate_source)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print_report(report)
    print(f"Saved: {args.results / report['run_id'] / 'result.json'}")
    return 0 if all(g["status"] == "ok" for g in report["games"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
