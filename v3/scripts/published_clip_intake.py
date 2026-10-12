"""Generate review drafts from published edits. Never promote proposals to answer keys."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import subprocess
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from answer_key_edl import parse_edl
from answer_key_frames import probe, read_frame, sample_recording
from answer_key_matching import align_background, match_motion
from detection_regression import manifest_parts

ROOT = Path(__file__).resolve().parents[2]


def separate_output(path: Path, inputs: list[Path]) -> None:
    """Keep generated files outside source folders and the answer-key directory."""
    output = path.resolve()
    for source in [ROOT / "tests" / "answer_keys", *inputs]:
        source = source.resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(f"output overlaps a protected source: {source}")


def propose_clip(query: np.ndarray, recordings: list[dict], *, camera_margin=0.05) -> dict:
    alternatives = []
    for recording in recordings:
        background = recording["background"]
        alignment = align_background(background, query)
        if alignment is None:
            continue
        matrix = alignment["matrix"]
        edited_background = cv2.warpPerspective(background, matrix, (query.shape[1], query.shape[0]))
        matched = match_motion(query, edited_background, recording["frames"], recording["times"],
                               background, matrix)
        if "raw_s" not in matched:
            continue
        alternatives.append({**matched, "camera": recording["camera"],
                             "recording": recording["index"], "inliers": alignment["inliers"],
                             "inlier_fraction": alignment["inlier_fraction"],
                             "transform": matrix.tolist(),
                             "detection_s": max(recording["start_s"], 0.0)
                                 + matched["raw_s"] - recording["seek_s"]})
    if not alternatives:
        return {"status": "unmatched", "alternatives": []}
    alternatives.sort(key=lambda row: row["score"], reverse=True)
    best = alternatives[0]
    other = max((p["score"] for p in alternatives[1:]), default=0.0)
    status = "proposed" if (best["status"] == "matched" and best["score"] - other >= camera_margin
                            and best["detection_s"] >= 0) else "needs_review"
    return {"status": status, "best": best, "alternative_margin": best["score"] - other,
            "alternatives": alternatives}


def published_references(path: Path, recap: Path, parsed: dict) -> list[dict]:
    """Attach independently verified published labels without assigning raw camera time."""
    evidence = json.loads(path.read_text())
    expected = hashlib.sha256(recap.read_bytes()).hexdigest()
    if evidence.get("recap_sha256") != expected:
        raise ValueError("published references do not match this Recap")
    references, ids = [], set()
    for ref in evidence["references"]:
        t = ref["recap_s"]
        if (not isinstance(ref.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:_-]*", ref["id"])
                or ref["id"] in ids
                or not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(t)
                or not isinstance(ref.get("goal"), bool) or not ref.get("evidence")):
            raise ValueError("published reference requires a unique ID, finite time, goal label, and evidence")
        cuts = [c for c in parsed["clips"] if c["recap_start_s"] <= t < c["recap_end_s"]]
        if len(cuts) != 1:
            raise ValueError(f"{ref['id']}: published reference must fall inside one recognized video cut")
        ids.add(ref["id"])
        references.append({**cuts[0], "edl_id": cuts[0]["id"], "id": ref["id"],
                           "goal": ref["goal"], "label_evidence": ref["evidence"], "recap_sample_s": t})
    if not references:
        raise ValueError("published references are empty")
    return references


def refine_clip(query: np.ndarray, proposal: dict, recordings: list[dict], cache: Path) -> dict:
    """Check coarse candidates with fully decoded frames. Keep the result as a draft."""
    refined = []
    for candidate in proposal.get("alternatives", []):
        recording = next(r for r in recordings if r["camera"] == candidate["camera"]
                         and r["index"] == candidate["recording"])
        start = max(0.0, candidate["raw_s"] - 2.0)
        pieces, timestamps, origin = [], [], 0.0
        for filename in recording["files"]:
            path = Path(filename)
            duration = probe(path)["duration_s"]
            lo, hi = max(start, origin), min(start + 4.0, origin + duration)
            if hi > lo and hi - lo >= 1 / 12:
                frames, times, _ = sample_recording([path], cache, fps=12.0, width=320,
                                                    keyframes_only=False, start_s=lo - origin,
                                                    duration_s=hi - lo)
                pieces.append(frames)
                timestamps.append(times + origin)
            origin += duration
            if origin >= start + 4.0:
                break
        if not pieces:
            continue
        frames, times = np.concatenate(pieces), np.concatenate(timestamps)
        matrix = np.asarray(candidate["transform"])
        background = recording["background"]
        edited_background = cv2.warpPerspective(background, matrix, (query.shape[1], query.shape[0]))
        result = match_motion(query, edited_background, frames, times, background, matrix,
                              min_score=0.8, min_margin=0.03, separation_s=0.5)
        if "raw_s" in result:
            refined.append({**candidate, **result, "detection_s": max(recording["start_s"], 0.0)
                             + result["raw_s"] - recording["seek_s"]})
    if not refined:
        return {"status": "unmatched", "alternatives": []}
    refined.sort(key=lambda row: row["score"], reverse=True)
    best = refined[0]
    margin = best["score"] - max((r["score"] for r in refined[1:]), default=0.0)
    status = "refined_proposal" if (best["status"] == "matched" and margin >= 0.1
                                    and best["detection_s"] >= 0) else "needs_review"
    return {"status": status, "best": best, "alternative_margin": margin, "alternatives": refined}


def atomic_json(path: Path, value: dict, *, replace=False) -> None:
    """Publish a complete JSON file. Replace only this run's validated checkpoint."""
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def intake_identity(edl: Path, recap: Path, references: Path | None, parts: dict,
                    fps: int, limit: int | None, refine: bool) -> dict:
    code = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            for name in ("published_clip_intake.py", "answer_key_edl.py",
                         "answer_key_frames.py", "answer_key_matching.py")}
    sources = {cam: [{**p, "files": [[str(Path(f).resolve()), Path(f).stat().st_size,
                                      Path(f).stat().st_mtime_ns] for f in p["files"]]} for p in rows]
               for cam, rows in parts.items()}
    return {"edl_sha256": hashlib.sha256(edl.read_bytes()).hexdigest(),
            "recap_sha256": hashlib.sha256(recap.read_bytes()).hexdigest(),
            "references_sha256": hashlib.sha256(references.read_bytes()).hexdigest() if references else None,
            "code_hashes": code, "raw_inputs": sources, "fps": fps, "limit": limit, "refine": refine,
            "libraries": {"opencv": cv2.__version__, "numpy": np.__version__}}


def run_intake(edl: Path, recap: Path, game: Path, output: Path, cache: Path, *, fps=60,
               limit: int | None = None, allow_processing=False, references: Path | None = None,
               refine=False, resume=False) -> dict:
    if not allow_processing:
        raise ValueError("footage matching requires explicit --allow-processing approval")
    parts = {cam: manifest_parts(game / f"{cam}_concat.txt") for cam in ("cam1", "cam2")}
    source_folders = [game, edl.parent, recap.parent]
    source_folders += [Path(f).resolve().parent for rows in parts.values() for p in rows for f in p["files"]]
    separate_output(output, source_folders)
    if references:
        separate_output(output, [references.parent])
    separate_output(cache, [game, edl.parent, ROOT / "tests" / "answer_keys",
                            *source_folders[3:]])
    parsed = parse_edl(edl.read_text(), fps=fps)
    selected = published_references(references, recap, parsed) if references else parsed["clips"]
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    identity = intake_identity(edl, recap, references, parts, fps, limit, refine)
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if resume:
        metadata = output / "inputs.json"
        if not metadata.is_file() or json.loads(metadata.read_text()).get("signature") != signature:
            raise ValueError("cannot resume: input or code identity differs from the saved run")
        if (output / "draft.json").exists():
            return json.loads((output / "draft.json").read_text())
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "inputs.json", {"schema_version": 1, "signature": signature, "identity": identity})
    started = time.perf_counter()
    saved = {}
    checkpoint_paths = {}
    for clip in selected[:limit]:
        checkpoint = output / "checkpoints" / (clip["id"].replace(":", "_") + ".json")
        checkpoint_paths[clip["id"]] = checkpoint
        if checkpoint.exists():
            value = json.loads(checkpoint.read_text())
            if (value.get("signature") != signature or value.get("stage") not in ("coarse_done", "complete")
                    or value.get("proposal", {}).get("id") != clip["id"]):
                raise ValueError(f"checkpoint identity is invalid: {checkpoint}")
            saved[clip["id"]] = value
    recordings = []
    pending = any(saved.get(clip["id"], {}).get("stage") != "complete" for clip in selected[:limit])
    if pending:
        for cam, camera_parts in parts.items():
            for i, part in enumerate(camera_parts):
                frames, times, cached = sample_recording([Path(p) for p in part["files"]], cache)
                indices = np.linspace(0, len(frames) - 1, min(201, len(frames))).astype(int)
                background = np.median(frames[indices], axis=0).astype(np.uint8)
                recordings.append({**part, "camera": cam, "index": i, "frames": frames,
                                   "times": times, "background": background, "cache": str(cached)})
                background_path = output / f"{cam}_recording{i}_background.png"
                if not background_path.exists():
                    cv2.imwrite(str(background_path), background)
    proposals = []
    for clip in selected[:limit]:
        checkpoint = checkpoint_paths[clip["id"]]
        stored = saved.get(clip["id"])
        if stored and stored["stage"] == "complete":
            proposals.append(stored["proposal"])
            print(f"{clip['id']}: reused completed checkpoint", flush=True)
            continue
        t = clip.get("recap_sample_s", (clip["recap_start_s"] + clip["recap_end_s"]) / 2)
        query = read_frame(recap, t, width=320)
        name = clip["id"].replace(":", "_") + ".png"
        if stored:
            proposal = stored["proposal"]
        else:
            cv2.imwrite(str(output / name), query)
            proposal = {**clip, "recap_sample_s": t, "preview": name,
                        "motion": propose_clip(query, recordings)}
            stored = {"signature": signature, "stage": "coarse_done", "proposal": proposal,
                      "frame_caches": [r["cache"] for r in recordings]}
            atomic_json(checkpoint, stored)
            print(f"{clip['id']}: saved coarse proposal", flush=True)
        if refine:
            proposal["refinement"] = refine_clip(query, proposal["motion"], recordings, cache)
        stored = {**stored, "stage": "complete", "proposal": proposal}
        atomic_json(checkpoint, stored, replace=True)
        saved[clip["id"]] = stored
        proposals.append(proposal)
        status = proposal.get("refinement", proposal["motion"])["status"]
        print(f"{clip['id']}: {status}", flush=True)
    if intake_identity(edl, recap, references, parts, fps, limit, refine) != identity:
        raise ValueError("input or code identity changed during matching; report not accepted")
    report = {"schema_version": 1, "status": "review_draft", "game": str(game.resolve()),
              "published_cut_count": len(parsed["clips"]), "clips": proposals,
              "audio_cues": parsed["audio_cues"], "fps": fps, "runtime_s": time.perf_counter() - started,
              "resumed": resume, "runtime_scope": "current invocation only; earlier processing is not included",
              "limits": ["Camera and time matches are proposals, not verified answer-key labels.",
                         "Coarse samples use keyframes at one sample per second; timestamps are approximate.",
                         "Static alignment alone does not verify raw time. Motion must also match.",
                         "Compound Game v1 cuts can include logos, credits, or multiple camera views.",
                         "Goal and Play Type labels require independent verification."],
              "source": {"edl": str(edl.resolve()), "recap": str(recap.resolve()),
                          **{k: identity[k] for k in ("edl_sha256", "recap_sha256", "references_sha256", "code_hashes")}},
              "timeline": {cam: [{**p, "files": [Path(f).name for f in p["files"]]} for p in rows]
                           for cam, rows in parts.items()},
              "frame_caches": sorted({p for row in saved.values() for p in row["frame_caches"]}),
              "input_signature": signature}
    atomic_json(output / "draft.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("edl", "recap", "game", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path, default=ROOT / ".regression-results" / "frame_cache")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--references", type=Path, help="independently verified Recap references, bound to its SHA-256")
    parser.add_argument("--refine", action="store_true", help="fully decode local candidates after coarse matching")
    parser.add_argument("--resume", action="store_true", help="resume this output directory only if input and code identities match")
    parser.add_argument("--allow-processing", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_intake(args.edl, args.recap, args.game, args.output, args.frame_cache,
                            fps=args.fps, limit=args.limit, allow_processing=args.allow_processing,
                            references=args.references, refine=args.refine, resume=args.resume)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"Blocked: {exc}", file=sys.stderr)
        return 2
    print(f"Saved {len(report['clips'])} review proposals: {args.output / 'draft.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
