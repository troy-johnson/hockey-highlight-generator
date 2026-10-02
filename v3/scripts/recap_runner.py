# v3/scripts/recap_runner.py
"""
Stage runner behind `hockeyrecap run | status | rerun --from` (hhg-3r5.28).

Each stage runs an existing script as a subprocess. Its full output goes to
recap.log in the Game Folder. recap_status.json keeps, per stage, the state,
the fingerprint of its inputs and settings, times and flags.

A stage is skipped when it finished before, its fingerprint is unchanged and
its outputs exist. A stage that was running when a run stopped runs again, so
a stopped run resumes at that stage. `rerun --from <stage>` runs that stage and
every later stage again.

Failure policy (spec 002 §4): the run stops only when there is no footage or
no usable camera. Every other problem is a flag, and stages that need the
missing output are skipped with a flag.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
V2 = REPO / "v2" / "scripts"
sys.path.insert(0, str(HERE))

STATUS_FILE = "recap_status.json"
LOG_FILE = "recap.log"
CACHE_DIR = ".recap_cache"
PREVIOUS_DIR = "recap_previous"      # copies of outputs from older tools, kept before the first overwrite

OK_STATES = ("done", "flagged")


# ---------------------------------------------------------------------------
# Context and results
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    state: str                      # done | flagged | skipped | failed
    flags: list[str] = field(default_factory=list)
    message: str = ""
    fatal: bool = False             # stop the run (no footage or no usable camera)


class Reporter:
    """Receives progress events. The CLI shows them with rich; tests record them."""
    def stage_start(self, name: str, title: str) -> None: ...
    def stage_progress(self, name: str, fraction: float | None, text: str = "") -> None: ...
    def stage_end(self, name: str, record: dict, reused: bool) -> None: ...
    def line(self, text: str) -> None: ...


@dataclass
class Context:
    game_folder: Path
    options: dict
    reporter: Reporter
    use_cache: bool = True
    python: str = sys.executable
    log_path: Path | None = None
    current_stage: str = ""

    @property
    def cache_dir(self) -> Path:
        return self.game_folder / CACHE_DIR

    def log(self, text: str) -> None:
        with open(self.log_path or (self.game_folder / LOG_FILE), "a") as f:
            for line in text.rstrip("\n").splitlines() or [""]:
                f.write(f"{_now()} [{self.current_stage or 'runner'}] {line}\n")

    def run_cmd(self, argv: list[str], on_line: Callable[[str], None] | None = None) -> tuple[int, list[str]]:
        """Run a command, log every output line, return (exit code, output lines)."""
        self.log("$ " + " ".join(_quote(a) for a in argv))
        env = dict(os.environ, PYTHONUNBUFFERED="1", HOCKEY_UNATTENDED="1")
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                bufsize=1, cwd=str(self.game_folder), env=env, stdin=subprocess.DEVNULL)
        lines: list[str] = []
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                lines.append(line)
                self.log(line)
                if on_line:
                    on_line(line)
            rc = proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        self.log(f"exit code {rc}")
        return rc, lines


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _quote(a: str) -> str:
    return a if re.fullmatch(r"[\w./:=+-]+", a) else "'" + a.replace("'", "'\\''") + "'"


def _last_error(lines: list[str]) -> str:
    for line in reversed(lines):
        if line.strip():
            return line.strip()[:200]
    return "no output"


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def file_identity(path: Path | str) -> list:
    p = Path(path)
    try:
        st = p.stat()
        return [p.name, st.st_size, int(st.st_mtime)]
    except OSError:
        return [p.name, None, None]


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _file_hash(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def _chapters(ctx: Context) -> dict:
    try:
        return json.loads((ctx.game_folder / "chapters.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def usable_cameras(ctx: Context) -> list[str]:
    return sorted((_chapters(ctx).get("recordings") or {}).keys())


def camera_files(folder: Path) -> list[Path]:
    """Candidate camera files, with the same rule as discover.py."""
    from discover import _mp4s
    files = [Path(p) for p in _mp4s(folder)]
    for sub in ("cam1", "cam2"):
        if (folder / sub).is_dir():
            files += [Path(p) for p in _mp4s(folder / sub)]
    return sorted(files)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    name: str
    title: str
    fingerprint: Callable[[Context], object]
    outputs: Callable[[Context], list[str]]
    run: Callable[[Context], StageResult]
    needs: tuple[str, ...] = ()
    needs_two_cameras: bool = False


# discovery -----------------------------------------------------------------

def _fp_discovery(ctx):
    return {"files": [file_identity(p) + [str(p.parent.name)] for p in camera_files(ctx.game_folder)],
            "scripts": [_file_hash(HERE / n) for n in ("discover.py", "recordings.py")]}


def _run_discovery(ctx):
    rc, lines = ctx.run_cmd([ctx.python, str(HERE / "discover.py"), str(ctx.game_folder), "--allow-missing-camera"])
    if rc != 0:
        return StageResult("failed", [_last_error(lines)], _last_error(lines), fatal=True)
    ch = _chapters(ctx)
    flags = list(ch.get("flags", []))
    for e in ch.get("excluded", []):
        if e.get("reason") == "black":
            flags.append(f"{e.get('cam')}: skipped black Recording starting at {Path(e['path']).name} (lens covered)")
        else:
            flags.append(f"ignored {Path(e['path']).name} ({e.get('reason')})")
    cams = usable_cameras(ctx)
    msg = ", ".join(f"{c}: {len(ch['recordings'][c])} Recording(s), {len(ch.get(c, []))} chapters" for c in cams)
    return StageResult("flagged" if flags else "done", flags, msg)


# sync ----------------------------------------------------------------------

def _fp_sync(ctx):
    return {"chapters": _file_hash(ctx.game_folder / "chapters.json"),
            "scripts": [_file_hash(HERE / n) for n in ("gopro_meta.py", "audio_sync.py")]}


def _run_sync(ctx):
    rc, lines = ctx.run_cmd([ctx.python, str(HERE / "gopro_meta.py"), str(ctx.game_folder)])
    if rc != 0:
        return StageResult("failed", [f"sync failed: {_last_error(lines)}"], _last_error(lines))
    try:
        info = json.loads((ctx.game_folder / "sync_info.json").read_text())
    except (OSError, json.JSONDecodeError):
        info = {}
    flags = [f"sync: {w}" for w in info.get("warnings", [])]
    msg = f"offset {info.get('offset_s', '?')} s ({info.get('sync_method', '?')})"
    return StageResult("flagged" if flags else "done", flags, msg)


# ROIs ----------------------------------------------------------------------

def _roi_mode(ctx) -> str:
    """
    'keep' when rois.json was picked or changed by hand, else 'auto'.
    auto_roi.py writes rois.json and then rois_auto.json, so a rois.json newer
    than rois_auto.json (or without it) comes from roi_picker.py or an edit.
    """
    root = ctx.game_folder
    if ctx.options.get("rois", {}).get("source") == "auto":
        return "auto"
    rois, auto = root / "rois.json", root / "rois_auto.json"
    if not rois.exists():
        return "auto"
    if not auto.exists() or rois.stat().st_mtime_ns > auto.stat().st_mtime_ns:
        return "keep"
    return "auto"


def _fp_rois(ctx):
    mode = _roi_mode(ctx)
    if mode == "keep":
        return {"mode": "keep", "rois": _file_hash(ctx.game_folder / "rois.json")}
    return {"mode": "auto", "recordings": digest((_chapters(ctx).get("recordings") or {})),
            "files": [file_identity(p) for p in camera_files(ctx.game_folder)],
            "script": _file_hash(HERE / "auto_roi.py")}


def _run_rois(ctx):
    if _roi_mode(ctx) == "keep":
        return StageResult("flagged", ["using the existing rois.json (picked or changed by hand); check it, "
                                       "or use --set rois.source=auto"], "kept rois.json")
    rc, lines = ctx.run_cmd([ctx.python, str(HERE / "auto_roi.py"), str(ctx.game_folder)])
    if rc != 0:
        why = {2: "automatic ROIs unavailable (optional ML stack?)", 3: "no goal frame found"}.get(rc, "auto_roi failed")
        return StageResult("failed", [f"ROIs: {why}: {_last_error(lines)}; run roi_picker.py"], why)
    flags = []
    try:
        info = json.loads((ctx.game_folder / "rois_auto.json").read_text())
        for cam, v in info.items():
            if v.get("flag"):
                flags.append(f"ROIs: {cam} goal frame confidence {v.get('confidence')}; check rois_preview.png")
    except (OSError, json.JSONDecodeError):
        pass
    return StageResult("flagged" if flags else "done", flags, "rois.json written")


# detection -----------------------------------------------------------------

DETECTION_FLAGS = ("fps", "width", "thresh_pct", "min_sep_s", "merge_gap_s", "max_win_s", "max_big_win_s",
                   "big_color_min", "replay_offset_s", "cooldown_s", "audio_weight")


def _detection_settings(ctx) -> dict:
    return dict(ctx.options.get("detection", {}))


def _fp_detection(ctx):
    root = ctx.game_folder
    return {"manifests": [_file_hash(root / f"cam{i}_concat.txt") for i in (1, 2)],
            "files": [file_identity(p) for p in camera_files(root)],
            "rois": _file_hash(root / "rois.json"),
            "settings": _detection_settings(ctx),
            "scripts": [_file_hash(V2 / n) for n in ("detect_events.py", "signals.py", "timeline_convert.py", "edl.py")]}


def detection_argv(ctx) -> list[str]:
    s = _detection_settings(ctx)
    argv = [ctx.python, str(V2 / "detect_events.py"), "cam1_concat.txt", "cam2_concat.txt", "rois.json"]
    for k in DETECTION_FLAGS:
        if k in s:
            argv += [f"--{k}", str(s[k])]
    if s.get("replay_markers"):
        argv.append("--replay_markers")
    argv += ["--out_csv", "events.csv", "--out_markers", "markers.csv", "--verbose"]
    if ctx.use_cache:
        argv += ["--signal_cache", str(ctx.cache_dir / "signals")]
    return argv


def _expected_frames(ctx) -> int:
    """Analysis frames for both cameras, for the progress bar (0 when unknown)."""
    from recordings import _duration
    fps = int(_detection_settings(ctx).get("fps", 12))
    total = 0.0
    for cam, recs in (_chapters(ctx).get("recordings") or {}).items():
        for rec in recs:
            for p in rec:
                try:
                    total += _duration(p)
                except Exception:
                    return 0
    return int(total * fps)


_FRAMES = re.compile(r"\[signals\] frames processed: (\d+)(?: \((.+)\))?")
_HIT = re.compile(r"\[signals\] cache hit: (.+) \((\d+) flow values\)")
_DONE = re.compile(r"\[signals\] Flow done: (.+?) in .*\((\d+) frames sampled")


def _run_detection(ctx):
    root = ctx.game_folder
    expected = _expected_frames(ctx)
    counts: dict[str, int] = {}
    hits = [0]

    def on_line(line: str):
        for rx, key_group in ((_FRAMES, 2), (_DONE, 1)):
            m = rx.search(line)
            if m:
                key = os.path.basename(m.group(key_group) or "video")
                counts[key] = max(counts.get(key, 0), int(m.group(1) if rx is _FRAMES else m.group(2)))
        m = _HIT.search(line)
        if m:
            hits[0] += 1
            counts[f"hit:{m.group(1)}:{hits[0]}"] = int(m.group(2))
        if expected:
            done = sum(counts.values())
            ctx.reporter.stage_progress("detection", min(done / expected, 1.0),
                                        f"{done}/{expected} frames" + (f", {hits[0]} Recording(s) from cache" if hits[0] else ""))

    rc, lines = ctx.run_cmd(detection_argv(ctx), on_line)
    if rc != 0:
        return StageResult("failed", [f"detection failed: {_last_error(lines)}"], _last_error(lines))
    fps = str(_detection_settings(ctx).get("marker_fps", 60))
    flags = []
    for argv in ([ctx.python, str(V2 / "timeline_convert.py"), "markers.csv", "--fps", fps, "--out", "markers.fcpxml",
                  "--project-name", "AI Markers"],
                 [ctx.python, str(V2 / "edl.py"), "markers.csv", "--fps", fps, "--out", "markers.edl"]):
        rc2, out = ctx.run_cmd(argv)
        if rc2 != 0:
            flags.append(f"{Path(argv[1]).name} failed: {_last_error(out)}")
    n = 0
    if (root / "markers.csv").exists():
        with open(root / "markers.csv") as f:
            n = max(0, sum(1 for _ in f) - 1)
    msg = f"{n} markers" + (f"; {hits[0]} Recording(s) from signal cache" if hits[0] else "")
    return StageResult("flagged" if flags else "done", flags, msg)


# Scoresheet ----------------------------------------------------------------

def _fp_scoresheet(ctx):
    from scoresheet import find_scoresheet_pdfs, find_scoresheet_photos
    root = str(ctx.game_folder)
    return {"sheets": [file_identity(p) for p in find_scoresheet_pdfs(root) + find_scoresheet_photos(root)[:1]],
            "scripts": [_file_hash(HERE / n) for n in ("scoresheet.py", "gamesheet_pdf.py")]}


def _run_scoresheet(ctx):
    rc, lines = ctx.run_cmd([ctx.python, str(HERE / "scoresheet.py"), str(ctx.game_folder)])
    if rc == 2:
        return StageResult("flagged", ["no Scoresheet photo or GameSheet PDF; enter the Game Sheet in review"],
                           "no Scoresheet")
    if rc == 4:
        return StageResult("flagged", ["Scoresheet reader needs the optional ML stack (requirements-ml.txt)"],
                           "ML stack missing")
    if rc != 0:
        return StageResult("failed", [f"Scoresheet reader failed: {_last_error(lines)}"], _last_error(lines))
    flags = []
    try:
        sheet = json.loads((ctx.game_folder / "game_sheet.json").read_text())
        rows = [r for side in ("goals", "penalties") for team in ("home", "away") for r in sheet[side][team]]
        review = sum(r.get("status") == "review" for r in rows)
        if review:
            flags.append(f"Game Sheet: {review} row(s) to review")
        flags += [f"Game Sheet: {f}" for f in sheet.get("flags", [])]
        msg = f"{len(rows)} rows from {Path(sheet.get('photo') or 'sheet').name}"
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        msg = "game_sheet.json written"
    return StageResult("flagged" if flags else "done", flags, msg)


# Audio signals ---------------------------------------------------------------

def _music_stack_present() -> bool:
    import importlib.util
    return importlib.util.find_spec("ai_edge_litert") is not None


def _music_model_identity():
    """
    Identity of the YAMNet file audio_signals.py will use ($HHG_YAMNET_MODEL or
    the user cache). When the ML stack is present but the file is not, return
    a new value each run, so a run after a failed download tries again.
    """
    if not _music_stack_present():
        return None
    env = os.environ.get("HHG_YAMNET_MODEL")
    p = Path(env).expanduser() if env else Path.home() / ".cache" / "hockey-highlight-generator" / "yamnet.tflite"
    if p.is_file():
        return {"path": str(p), "file": file_identity(p)}
    return {"path": str(p), "missing": time.time()}


def _fp_audio(ctx):
    root = ctx.game_folder
    s = _detection_settings(ctx)
    return {"manifests": [_file_hash(root / f"cam{i}_concat.txt") for i in (1, 2)],
            "files": [file_identity(p) for p in camera_files(root)],
            "flow": {"fps": s.get("fps", 12), "width": s.get("width", 1280), "rois": _file_hash(root / "rois.json"),
                     "audio": float(s.get("audio_weight", 0) or 0) > 0},
            "music_stack": _music_stack_present(),
            "music_model": _music_model_identity(),
            "scripts": [_file_hash(HERE / "audio_signals.py"), _file_hash(V2 / "signals.py")]}


def audio_argv(ctx) -> list[str]:
    s = _detection_settings(ctx)
    argv = [ctx.python, str(HERE / "audio_signals.py"), str(ctx.game_folder),
            "--fps", str(s.get("fps", 12)), "--width", str(s.get("width", 1280))]
    if float(s.get("audio_weight", 0) or 0) > 0:
        argv.append("--flow_audio")
    if not ctx.use_cache:
        argv.append("--no-cache")
    return argv


_AUDIO_PROGRESS = re.compile(r"\[audio\] decoded (\d+)/(\d+) s")
_AUDIO_SUMMARY = re.compile(r"\[audio\] (\d+ whistles, .*)")
_AUDIO_FLAG = re.compile(r"\[audio\] flag: (.+)")


def _run_audio(ctx):
    summary, flags = [""], []

    def on_line(line: str):
        m = _AUDIO_PROGRESS.search(line)
        if m and int(m.group(2)):
            done, total = int(m.group(1)), int(m.group(2))
            ctx.reporter.stage_progress("audio", min(done / total, 1.0), f"{done}/{total} s of audio")
        m = _AUDIO_SUMMARY.search(line)
        if m:
            summary[0] = m.group(1)
        m = _AUDIO_FLAG.search(line)
        if m:
            flags.append(m.group(1))

    rc, lines = ctx.run_cmd(audio_argv(ctx), on_line)
    if rc != 0:
        return StageResult("failed", [f"audio signals failed: {_last_error(lines)}"], _last_error(lines))
    return StageResult("flagged" if flags else "done", flags, summary[0] or "audio_signals.json written")


def _goalie_stack_present() -> bool:
    import importlib.util
    return importlib.util.find_spec("ultralytics") is not None


def league_timing_rules(ctx) -> dict:
    """The League keys that the coverage stage reads (period timing)."""
    rules = ctx.options.get("league_rules") or {}
    return {k: rules[k] for k in ("periods", "period_minutes", "clock", "break_minutes") if k in rules}


def _fp_coverage(ctx):
    root = ctx.game_folder
    s = _detection_settings(ctx)
    return {"audio_signals": _file_hash(root / "audio_signals.json"),
            "manifests": [_file_hash(root / f"cam{i}_concat.txt") for i in (1, 2)],
            "files": [file_identity(p) for p in camera_files(root)],
            "flow": {"fps": s.get("fps", 12), "width": s.get("width", 1280), "rois": _file_hash(root / "rois.json"),
                     "audio": float(s.get("audio_weight", 0) or 0) > 0},
            "league": league_timing_rules(ctx),
            "goalie_stack": _goalie_stack_present(),
            "scripts": [_file_hash(HERE / "coverage.py"), _file_hash(HERE / "audio_signals.py"),
                        _file_hash(HERE / "auto_roi.py"), _file_hash(V2 / "signals.py")]}


def coverage_argv(ctx) -> list[str]:
    s = _detection_settings(ctx)
    argv = [ctx.python, str(HERE / "coverage.py"), str(ctx.game_folder),
            "--fps", str(s.get("fps", 12)), "--width", str(s.get("width", 1280)),
            "--league", json.dumps(league_timing_rules(ctx), sort_keys=True)]
    if float(s.get("audio_weight", 0) or 0) > 0:
        argv.append("--flow_audio")
    return argv


_COVERAGE_SUMMARY = re.compile(r"\[coverage\] (\d+ periods? .*)")
_COVERAGE_FLAG = re.compile(r"\[coverage\] flag: (.+)")


def _run_coverage(ctx):
    summary, flags = [""], []

    def on_line(line: str):
        m = _COVERAGE_FLAG.search(line)
        if m:
            flags.append(m.group(1))
            return
        m = _COVERAGE_SUMMARY.search(line)
        if m:
            summary[0] = m.group(1)

    rc, lines = ctx.run_cmd(coverage_argv(ctx), on_line)
    if rc != 0:
        return StageResult("failed", [f"coverage failed: {_last_error(lines)}"], _last_error(lines))
    return StageResult("flagged" if flags else "done", flags, summary[0] or "coverage.json written")


def selection_rules(ctx) -> dict:
    rules = ctx.options.get("league_rules") or {}
    return {k: rules[k] for k in ("periods", "period_minutes", "clock", "time_direction") if k in rules}


def _fp_selection(ctx):
    root = ctx.game_folder
    s = _detection_settings(ctx)
    return {"inputs": {n: _file_hash(root / n) for n in
                       ("game_sheet.json", "coverage.json", "audio_signals.json", "events.csv", "music_spans.json",
                        "rois.json", "cam1_concat.txt", "cam2_concat.txt")},
            "files": [file_identity(p) for p in camera_files(root)],
            "caches": [file_identity(p) for p in sorted((root / CACHE_DIR).glob("*/*.npz"))
                       if not p.name.startswith("._")],
            "flow": {"fps": s.get("fps", 12), "width": s.get("width", 1280),
                     "audio": float(s.get("audio_weight", 0) or 0) > 0,
                     "hwaccel": os.environ.get("HHG_HWACCEL", "1") != "0"},
            "league": selection_rules(ctx), "options": ctx.options.get("selection") or {},
            "scripts": [_file_hash(HERE / n) for n in
                        ("selection.py", "selection_inputs.py", "audio_signals.py", "coverage.py", "scoresheet.py")]
                       + [_file_hash(V2 / "signals.py")]}


def selection_argv(ctx) -> list[str]:
    s = _detection_settings(ctx)
    argv = [ctx.python, str(HERE / "selection.py"), str(ctx.game_folder),
            "--fps", str(s.get("fps", 12)), "--width", str(s.get("width", 1280)),
            "--league", json.dumps(selection_rules(ctx), sort_keys=True),
            "--options", json.dumps(ctx.options.get("selection") or {}, sort_keys=True)]
    if float(s.get("audio_weight", 0) or 0) > 0:
        argv.append("--flow_audio")
    return argv


def _run_selection(ctx):
    summary, flags = [""], []

    def on_line(line: str):
        if line.startswith("[selection] flag: "):
            flags.append(line.removeprefix("[selection] flag: "))
        elif line.startswith("[selection] "):
            summary[0] = line.removeprefix("[selection] ")

    rc, lines = ctx.run_cmd(selection_argv(ctx), on_line)
    if rc != 0:
        return StageResult("failed", [f"selection failed: {_last_error(lines)}"], _last_error(lines))
    return StageResult("flagged" if flags else "done", flags, summary[0] or "selection.json written")


def assembly_options(ctx):
    options = {k: ctx.options[k] for k in ("date", "focus_team", "live_play_speed") if k in ctx.options}
    team = (ctx.options.get("teams") or {}).get(options.get("focus_team"), {})
    if team.get("name"):
        options["focus_team"] = team["name"]
    return options


def assembly_outputs(ctx):
    from recap_assembly import output_name
    sheet = json.loads((ctx.game_folder / "game_sheet.json").read_text())
    return ["recap_assembly.json", output_name(ctx.game_folder, assembly_options(ctx), sheet)]


def _fp_assembly(ctx):
    return {"inputs": {n: _file_hash(ctx.game_folder / n) for n in
                       ("selection.json", "game_sheet.json", "cam1_concat.txt", "cam2_concat.txt")},
            "files": [file_identity(p) for p in camera_files(ctx.game_folder)],
            "options": assembly_options(ctx),
            "scripts": [_file_hash(HERE / n) for n in ("recap_assembly.py", "coverage.py", "recap_options.py")]
                       + [_file_hash(V2 / "signals.py")]}


def _run_assembly(ctx):
    flags, summary = [], [""]
    def on_line(line):
        if line.startswith("[assembly] flag: "):
            flags.append(line.removeprefix("[assembly] flag: "))
        elif line.startswith("[assembly] "):
            summary[0] = line.removeprefix("[assembly] ")
    argv = [ctx.python, str(HERE / "recap_assembly.py"), str(ctx.game_folder),
            "--options", json.dumps(assembly_options(ctx))]
    rc, lines = ctx.run_cmd(argv, on_line)
    if rc != 0:
        return StageResult("failed", [f"assembly failed: {_last_error(lines)}"], _last_error(lines))
    return StageResult("flagged" if flags else "done", flags, summary[0])


STAGES: list[Stage] = [
    Stage("discovery", "Find cameras and Recordings", _fp_discovery, lambda c: ["chapters.json"], _run_discovery),
    Stage("sync", "Sync cameras", _fp_sync,
          lambda c: ["sync_info.json", "cam1_concat.txt", "cam2_concat.txt"], _run_sync,
          needs=("discovery",), needs_two_cameras=True),
    Stage("rois", "Find net and slot ROIs", _fp_rois, lambda c: ["rois.json"], _run_rois,
          needs=("discovery",), needs_two_cameras=True),
    Stage("detection", "Detect events", _fp_detection,
          lambda c: ["events.csv", "markers.csv", "markers.fcpxml", "markers.edl"], _run_detection,
          needs=("sync", "rois"), needs_two_cameras=True),
    Stage("audio", "Find whistles, stoppages, PA music", _fp_audio,
          lambda c: ["audio_signals.json", "music_spans.json"], _run_audio,
          needs=("sync",), needs_two_cameras=True),
    Stage("coverage", "Find periods, game start and end, coverage", _fp_coverage,
          lambda c: ["coverage.json"], _run_coverage,
          needs=("audio", "detection"), needs_two_cameras=True),
    Stage("scoresheet", "Read Scoresheet / GameSheet", _fp_scoresheet, lambda c: ["game_sheet.json"],
          _run_scoresheet),
    Stage("selection", "Locate Game Sheet goals", _fp_selection, lambda c: ["selection.json"],
          _run_selection, needs=("coverage", "scoresheet", "audio", "detection")),
    Stage("assembly", "Render plain Recap", _fp_assembly, assembly_outputs,
          _run_assembly, needs=("selection",)),
]

STAGE_NAMES = [s.name for s in STAGES]


# ---------------------------------------------------------------------------
# Status file
# ---------------------------------------------------------------------------

def load_status(game_folder: Path) -> dict:
    try:
        return json.loads((Path(game_folder) / STATUS_FILE).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_status(game_folder: Path, status: dict) -> None:
    path = Path(game_folder) / STATUS_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2) + "\n")
    tmp.replace(path)


def cache_size(game_folder: Path) -> int:
    root = Path(game_folder) / CACHE_DIR
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.is_dir() else 0


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def _keep_previous_outputs(ctx: Context, stage: Stage, status: dict) -> None:
    """Copy outputs that an older tool wrote (not this runner) to recap_previous/ before the first overwrite."""
    if stage.name in status.get("stages", {}):
        return
    for name in stage.outputs(ctx):
        src = ctx.game_folder / name
        if src.exists() and not (stage.name == "rois" and _roi_mode(ctx) == "keep"):
            dst_dir = ctx.game_folder / PREVIOUS_DIR
            dst_dir.mkdir(exist_ok=True)
            dst = dst_dir / name
            if dst.exists():
                continue
            shutil.copy2(src, dst)
            ctx.log(f"kept a copy of the previous {name} in {PREVIOUS_DIR}/")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_game(game_folder: str | Path, options: dict, reporter: Reporter | None = None,
             from_stage: str | None = None, use_cache: bool = True,
             stages: list[Stage] | None = None) -> dict:
    """
    Run the stages for one Game Folder and return the status dict.
    status["state"] is "done" (every stage ran or was reused), or "stopped"
    (no footage or no usable camera, or interrupted).
    """
    stages = stages if stages is not None else STAGES
    names = [s.name for s in stages]
    if from_stage is not None and from_stage not in names:
        raise ValueError(f"[ERROR] Unknown stage {from_stage!r}; stages: {', '.join(names)}")
    root = Path(game_folder).resolve()
    reporter = reporter or Reporter()
    ctx = Context(root, options, reporter, use_cache=use_cache, log_path=root / LOG_FILE)

    status = load_status(root)
    status.setdefault("stages", {})
    status.update({"game_folder": str(root), "state": "running", "started": _now(), "finished": None,
                   "from_stage": from_stage, "stage_order": names})
    ctx.log(f"=== hockeyrecap run started{f' (rerun from {from_stage})' if from_stage else ''} ===")
    for f in options.get("_flags", []):
        ctx.log(f"[FLAG] options: {f}")
    save_status(root, status)

    force_from = names.index(from_stage) if from_stage else len(names)
    run_flags: list[str] = [f"options: {f}" for f in options.get("_flags", [])]
    try:
        for i, stage in enumerate(stages):
            ctx.current_stage = stage.name
            prev = status["stages"].get(stage.name, {})
            reporter.stage_start(stage.name, stage.title)

            missing = [n for n in stage.needs if status["stages"].get(n, {}).get("state") not in OK_STATES]
            cams = usable_cameras(ctx)
            if stage.needs_two_cameras and len(cams) < 2 and not missing:
                missing = ["two usable cameras"]
            if missing:
                rec = {"state": "skipped", "fingerprint": None, "started": _now(), "finished": _now(),
                       "duration_s": 0.0, "flags": [f"{stage.name} skipped: needs {', '.join(missing)}"],
                       "message": ""}
                status["stages"][stage.name] = rec
                run_flags += rec["flags"]
                ctx.log(f"skipped: needs {', '.join(missing)}")
                save_status(root, status)
                reporter.stage_end(stage.name, rec, reused=False)
                continue

            try:
                fp = digest(stage.fingerprint(ctx))
                outputs_ok = all((root / o).exists() for o in stage.outputs(ctx))
            except Exception as exc:  # e.g. an unreadable file: run the stage, let it report
                ctx.log(f"fingerprint error: {exc!r}")
                fp, outputs_ok = None, False
            if (i < force_from and prev.get("state") in OK_STATES and prev.get("fingerprint") == fp
                    and outputs_ok):
                prev["reused"] = True
                status["stages"][stage.name] = prev
                run_flags += prev.get("flags", [])
                ctx.log("unchanged since the last run; reused")
                save_status(root, status)
                reporter.stage_end(stage.name, prev, reused=True)
                continue

            rec = {"state": "running", "fingerprint": fp, "started": _now(), "finished": None,
                   "duration_s": None, "flags": [], "message": ""}
            t0 = time.time()
            try:
                _keep_previous_outputs(ctx, stage, status)
                status["stages"][stage.name] = rec
                save_status(root, status)
                result = stage.run(ctx)
            except Exception as exc:  # a bug or an unexpected error in one stage is a flag
                ctx.log(f"error: {exc!r}")
                result = StageResult("failed", [f"{stage.name} error: {exc}"], str(exc))
            status["stages"][stage.name] = rec
            rec.update({"state": result.state, "finished": _now(), "duration_s": round(time.time() - t0, 1),
                        "flags": result.flags, "message": result.message, "reused": False})
            if result.state not in OK_STATES:
                rec["fingerprint"] = None           # run it again next time
            run_flags += result.flags
            for f in result.flags:
                ctx.log(f"[FLAG] {f}")
            save_status(root, status)
            reporter.stage_end(stage.name, rec, reused=False)
            if result.fatal:
                status["state"] = "stopped"
                status["stop_reason"] = result.message
                break
        else:
            status["state"] = "done"
            status.pop("stop_reason", None)
        # stages after a stop are not run: drop stale records so status shows them as pending
        if status["state"] == "stopped":
            stop = names.index(ctx.current_stage)
            for n in names[stop + 1:]:
                status["stages"].pop(n, None)
    except KeyboardInterrupt:
        status["state"] = "stopped"
        status["stop_reason"] = "interrupted"
        cur = status["stages"].get(ctx.current_stage)
        if cur and cur.get("state") == "running":
            cur["state"] = "interrupted"
            cur["fingerprint"] = None
        ctx.log("interrupted")
        save_status(root, status)
        raise
    finally:
        ctx.current_stage = ""
        status["finished"] = _now()
        status["flags"] = run_flags
        status["cache_bytes"] = cache_size(root)
        save_status(root, status)
        ctx.log(f"=== run {status['state']}; {len(run_flags)} flag(s); cache {human_size(status['cache_bytes'])} ===")
    return status


def _ordered_states(records: dict, stages: list[Stage]) -> dict:
    """Stage states in pipeline order; unknown names (old status files) last."""
    order = [s.name for s in stages if s.name in records]
    order += [n for n in records if n not in order]
    return {n: records[n].get("state") for n in order}


def run_batch(folders: list[str], make_options: Callable[[str], dict], reporter_for: Callable[[str], Reporter],
              from_stage: str | None = None, use_cache: bool = True,
              stages: list[Stage] | None = None) -> list[dict]:
    """
    Run several Game Folders one after another (hhg-3r5.58). A game that stops
    or fails does not stop the others. Returns one summary per game.
    """
    summaries = []
    for folder in folders:
        summary = {"game_folder": str(folder), "state": "failed", "flags": [], "stages": {}, "error": None,
                   "cache_bytes": 0}
        try:
            opts = make_options(folder)
            st = run_game(folder, opts, reporter_for(folder), from_stage=from_stage, use_cache=use_cache,
                          stages=stages)
            summary.update(state=st["state"], flags=st.get("flags", []), cache_bytes=st.get("cache_bytes", 0),
                           stages=_ordered_states(st["stages"], stages or STAGES),
                           error=st.get("stop_reason"))
        except KeyboardInterrupt:
            summary.update(state="stopped", error="interrupted")
            summaries.append(summary)
            raise
        except Exception as exc:
            summary["error"] = str(exc)
        summaries.append(summary)
    return summaries
