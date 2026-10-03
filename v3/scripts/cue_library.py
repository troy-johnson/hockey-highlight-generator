#!/usr/bin/env python3
"""Author and validate the hand-approved schema-1 Cue Library."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from statistics import median
from typing import Callable
from urllib.parse import urlparse


CUE_IDS = (
    "game_start", "horn", "close_win", "close_loss", "close_neutral",
    "sting_penalty", "sting_power_play", "sting_fight", "sting_comic_call",
    "sting_neutral", "sfx_goal_card", "sfx_penalty_card", "sfx_period_wipe", "sfx_final_card",
)
SOURCES = ("YouTube Audio Library", "Pixabay", "Freesound CC0")


def _text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: expected a nonempty string")
    return value


def _number(value, field: str, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (positive and value <= 0)):
        raise ValueError(f"{field}: expected a finite {'positive ' if positive else ''}number")
    return float(value)


def _local_file(root: Path, value, field: str) -> Path:
    name = Path(_text(value, field))
    path = (root / name).resolve()
    if name.is_absolute() or ".." in name.parts or not path.is_relative_to(root.resolve()):
        raise ValueError(f"{field}: expected a relative path inside the Cue Library")
    if not path.is_file():
        raise ValueError(f"{field}: missing file {name}")
    return path


def file_sha256(path: Path) -> str:
    """Identify the approved media bytes."""
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def _provenance(entry: dict, root: Path, audio: Path) -> None:
    if entry.get("source") not in SOURCES:
        raise ValueError(f"source: expected one of {SOURCES}")
    if entry.get("content_id_safe") is not True:
        raise ValueError("content_id_safe: must be true after checking the source")
    if entry.get("approved") is not True:
        raise ValueError("approved: must be true after hand approval")
    proof = entry.get("provenance")
    if not isinstance(proof, dict) or not proof:
        raise ValueError("provenance: source, license proof, hash, and approval are required")
    url = urlparse(_text(proof.get("url"), "provenance.url"))
    if url.scheme not in ("https", "http") or not url.netloc:
        raise ValueError("provenance.url: expected an HTTP source URL")
    license_name = _text(proof.get("license"), "provenance.license")
    if entry["source"] == "Freesound CC0" and license_name != "CC0":
        raise ValueError("provenance.license: Freesound requires CC0")
    _local_file(root, proof.get("proof"), "provenance.proof")
    _text(proof.get("approved_by"), "provenance.approved_by")
    try:
        date.fromisoformat(_text(proof.get("approved_at"), "provenance.approved_at"))
    except ValueError as exc:
        raise ValueError("provenance.approved_at: expected an ISO date") from exc
    if proof.get("sha256") != file_sha256(audio):
        raise ValueError("provenance.sha256: media differs from the approved file")


def _grid(entry: dict, required: bool) -> None:
    meter = entry.get("beats_per_bar", 4)
    if type(meter) is not int or meter <= 0:
        raise ValueError("beats_per_bar: expected a positive integer")
    for field in ("bpm", "bar_s"):
        if field in entry:
            _number(entry[field], field, positive=True)
    if "beats" in entry:
        beats = entry["beats"]
        if not isinstance(beats, list) or len(beats) < 2:
            raise ValueError("beats: expected at least two beat times")
        times = [_number(t, "beats") for t in beats]
        if times[0] < 0 or times[-1] > entry["duration_s"] or any(
                b <= a for a, b in zip(times, times[1:])):
            raise ValueError("beats: times must increase inside the media duration")
    if required and not any(field in entry for field in ("bpm", "bar_s", "beats")):
        raise ValueError("beats: a Bed needs a beat grid, bpm, or bar_s")


def validate_manifest(data, root: Path, require_grids: bool = True) -> None:
    """Reject invalid authoring data, unsafe sources, and changed media."""
    if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1):
        raise ValueError("schema_version: expected 1")
    seen: set[str] = set()
    for kind in ("beds", "cues"):
        entries = data.get(kind)
        if not isinstance(entries, list):
            raise ValueError(f"{kind}: expected a list")
        for item in entries:
            if not isinstance(item, dict):
                raise ValueError(f"{kind}: expected entry objects")
            cid = _text(item.get("id"), "id")
            if cid in seen:
                raise ValueError(f"id: duplicate {cid}")
            seen.add(cid)
            if kind == "cues" and cid not in CUE_IDS:
                raise ValueError(f"cue id: unknown {cid}")
            _number(item.get("duration_s"), "duration_s", positive=True)
            if cid.startswith("sting_") and kind == "cues" and not 2 <= item["duration_s"] <= 4:
                raise ValueError(f"duration_s: {cid} must be 2-4 s")
            audio = _local_file(root, item.get("file"), "file")
            _provenance(item, root, audio)
            if kind == "beds":
                _grid(item, require_grids)
            for field in ("lufs", "gain_db"):
                if field in item:
                    _number(item[field], field)
            tags = item.get("tags", [])
            if not isinstance(tags, list) or any(not isinstance(t, str) for t in tags):
                raise ValueError("tags: expected a list of strings")


def load_manifest(path: Path | str, require_grids: bool = True) -> dict:
    """Load a validated Cue Library with relative media paths intact."""
    path = Path(path).expanduser()
    data = json.loads(path.read_text())
    validate_manifest(data, path.parent, require_grids)
    return data


def library_flags(data: dict) -> list[str]:
    """Report library gaps without inventing approved audio."""
    flags = []
    if len(data["beds"]) < 6:
        flags.append("fewer than 6 Beds: three-period games may need to reuse tracks")
    present = {cue["id"] for cue in data["cues"]}
    flags.extend(f"cue {cid}: no approved audio" for cid in CUE_IDS if cid not in present)
    return flags


def write_json(path: Path, data: dict) -> None:
    """Replace JSON only after the complete file is ready."""
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def derive_grid(duration_s: float, *, beats=None, bpm=None, bar_s=None,
                beats_per_bar: int = 4) -> dict:
    """Derive mixer timing from detected beats or a manually checked tempo."""
    duration = _number(duration_s, "duration_s", positive=True)
    grid = {"duration_s": duration, "beats_per_bar": beats_per_bar}
    _grid(grid, required=False)
    if beats is not None:
        grid["beats"] = list(beats)
        _grid(grid, required=True)
        interval = median(b - a for a, b in zip(grid["beats"], grid["beats"][1:]))
        tempo = 60 / interval
    else:
        tempo = (60 * beats_per_bar / _number(bar_s, "bar_s", positive=True)
                 if bpm is None and bar_s is not None else _number(bpm, "bpm", positive=True))
        interval = 60 / tempo
        count = math.ceil(duration / interval)
        if count > 1_000_000:
            raise ValueError("bpm: grid exceeds one million beats")
        grid["beats"] = [i * interval for i in range(count)]
    grid["bpm"] = tempo
    grid["bar_s"] = (interval * beats_per_bar if bar_s is None
                     else _number(bar_s, "bar_s", positive=True))
    _grid(grid, required=True)
    del grid["duration_s"]
    return grid


def detect_beats(audio: Path) -> list[float]:
    """Run optional beat_this inference without loading it during validation."""
    try:
        from beat_this.inference import File2Beats  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ValueError("beat_this is unavailable; install requirements-beats.txt or use --bpm") from exc
    beats, _downbeats = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)(str(audio))
    return [float(t) for t in beats]


def update_grids(path: Path, *, detector: Callable | None = None, bpm=None,
                 beats_per_bar: int | None = None, bar_s=None, track_id: str | None = None) -> dict:
    """Derive Bed grids and keep all provenance fields intact."""
    path = path.expanduser()
    data = load_manifest(path, require_grids=False)
    tracks = [bed for bed in data["beds"] if track_id is None or bed["id"] == track_id]
    if track_id is not None and not tracks:
        raise ValueError(f"id: no Bed named {track_id}")
    for bed in tracks:
        beats = None if bpm is not None else (detector or detect_beats)(path.parent / bed["file"])
        meter = bed.get("beats_per_bar", 4) if beats_per_bar is None else beats_per_bar
        bed.update(derive_grid(bed["duration_s"], beats=beats, bpm=bpm,
                              beats_per_bar=meter, bar_s=bar_s))
    validate_manifest(data, path.parent)
    write_json(path, data)
    return data


def _history(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 1, "games": []}
    try:
        data = json.loads(path.read_text())
        if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
                or data["schema_version"] != 1 or not isinstance(data.get("games"), list)):
            raise ValueError("expected schema 1 and games list")
        seen = set()
        for game in data["games"]:
            if not isinstance(game, dict):
                raise ValueError("expected game object")
            key = _text(game.get("key"), "key")
            if key in seen:
                raise ValueError("duplicate game key")
            seen.add(key)
            choices = game.get("beds")
            if not isinstance(choices, dict):
                raise ValueError("expected period-to-Bed mapping")
            for period, cid in choices.items():
                if not period.isdigit() or int(period) < 1 or str(int(period)) != period:
                    raise ValueError("expected positive period number")
                _text(cid, "id")
        return data
    except (OSError, ValueError) as exc:
        raise ValueError(f"rotation history {path}: {exc}") from exc


def select_beds(beds: list[dict], game_key: str, periods: list[int],
                history_path: Path) -> tuple[dict[int, dict], list[str]]:
    """Keep reruns stable and avoid the previous successful game's Beds."""
    history = _history(history_path)
    games = history["games"]
    saved = next((g for g in games if g["key"] == game_key), None)
    position = games.index(saved) if saved else len(games)
    previous = set(games[position - 1]["beds"].values()) if position else set()
    available = {bed["id"]: bed for bed in beds}
    if not available:
        return {}, ["rotation: no usable Beds"]
    order = sorted(available, key=lambda cid: hashlib.sha256(f"{game_key}|{cid}".encode()).hexdigest())
    pool = [cid for cid in order if cid not in previous] + [cid for cid in order if cid in previous]
    choices: dict[int, dict] = {}
    flags = []
    used: set[str] = set()
    reserved = {saved["beds"].get(str(p)) for p in periods} & available.keys() if saved else set()
    for period in sorted(set(periods)):
        cid = saved["beds"].get(str(period)) if saved else None
        if cid not in available:
            if cid is not None:
                flags.append(f"rotation: saved Bed {cid} is unavailable; selected a replacement")
            cid = next((item for item in pool if item not in used and item not in reserved),
                       next((item for item in pool if item not in used), pool[0]))
        cid = str(cid)
        choices[period] = available[cid]
        if (cid in previous and not saved) or cid in used:
            flags.append(f"rotation: insufficient Beds; period {period} must reuse {cid}")
        used.add(cid)
    return choices, flags


def record_rotation(history_path: Path, game_key: str, choices: dict[int, dict]) -> None:
    """Record used Beds after a successful mix; keep reruns in their original position."""
    history = _history(history_path)
    game = {"key": game_key, "beds": {str(p): bed["id"] for p, bed in choices.items()}}
    for i, existing in enumerate(history["games"]):
        if existing["key"] == game_key:
            history["games"][i] = game
            break
    else:
        history["games"].append(game)
    write_json(history_path, history)


def add_track(path: Path, kind: str, metadata: Path, *, detector: Callable | None = None) -> dict:
    """Register hand-approved metadata with the measured duration and media hash."""
    data = load_manifest(path)
    item = json.loads(metadata.read_text())
    if not isinstance(item, dict) or not isinstance(item.get("provenance"), dict):
        raise ValueError("provenance: expected approved track metadata")
    if item.get("approved") is not True:
        raise ValueError("approved: hand approval is required before adding a track")
    audio = _local_file(path.parent, item.get("file"), "file")
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=noprint_wrappers=1:nokey=1", str(audio)],
                           capture_output=True, text=True)
    if probe.returncode:
        raise ValueError(f"file: ffprobe could not read {audio}: {probe.stderr.strip()}")
    item["duration_s"] = _number(float(probe.stdout.strip()), "duration_s", positive=True)
    item["provenance"]["sha256"] = file_sha256(audio)
    _provenance(item, path.parent, audio)
    if kind == "beds":
        beats = item.get("beats")
        if not any(key in item for key in ("beats", "bpm", "bar_s")):
            beats = (detector or detect_beats)(audio)
        item.update(derive_grid(item["duration_s"], beats=beats, bpm=item.get("bpm"),
                                bar_s=item.get("bar_s"), beats_per_bar=item.get("beats_per_bar", 4)))
    data[kind].append(item)
    validate_manifest(data, path.parent)
    write_json(path, data)
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="create an empty Cue Library")
    init.add_argument("directory", type=Path)
    validate = commands.add_parser("validate", help="check schema, provenance, and library gaps")
    validate.add_argument("manifest", type=Path)
    add = commands.add_parser("add", help="register approved metadata without copying or downloading audio")
    add.add_argument("manifest", type=Path)
    add.add_argument("kind", choices=("beds", "cues"))
    add.add_argument("metadata", type=Path)
    grid = commands.add_parser("grid", help="derive Bed grids with beat_this or a checked BPM")
    grid.add_argument("manifest", type=Path)
    grid.add_argument("--id", dest="track_id")
    grid.add_argument("--bpm", type=float)
    grid.add_argument("--bar-s", type=float)
    grid.add_argument("--beats-per-bar", type=int,
                      help="override the stored meter (default: keep each Bed's meter, or 4 if absent)")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            root = args.directory.expanduser()
            path = root / "manifest.json"
            if path.exists():
                raise ValueError(f"manifest already exists: {path}")
            for folder in ("beds", "cues", "provenance"):
                (root / folder).mkdir(parents=True, exist_ok=True)
            data = {"schema_version": 1, "beds": [], "cues": []}
            write_json(path, data)
        elif args.command == "validate":
            data = load_manifest(args.manifest)
        elif args.command == "add":
            data = add_track(args.manifest.expanduser(), args.kind, args.metadata.expanduser())
        else:
            data = update_grids(args.manifest, bpm=args.bpm, bar_s=args.bar_s,
                                beats_per_bar=args.beats_per_bar, track_id=args.track_id)
        for message in library_flags(data):
            print(f"[cues] flag: {message}")
        print(f"[cues] valid schema-1 library: {len(data['beds'])} Beds, {len(data['cues'])} cues")
        return 0
    except (OSError, ValueError) as exc:
        print(f"[ERROR] cues: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
