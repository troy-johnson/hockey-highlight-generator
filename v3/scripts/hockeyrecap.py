# v3/scripts/hockeyrecap.py
"""
hockeyrecap: one command from a Game Folder to the pipeline outputs (spec 002 §4).

  hockeyrecap run <game_folder> [<game_folder> ...]     run, or resume, every stage
  hockeyrecap rerun --from <stage> <game_folder> ...    run <stage> and later stages again
  hockeyrecap status <game_folder>                      show recap_status.json
  hockeyrecap check <game_folder> [...]                 quick Recording check (a few minutes)

Stages: discovery, sync, rois, detection, scoresheet. Options come from four
layers, each over the one before: League, Team Config (~/hockey), the per-game
file recap_options.json, and the flags of this command.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recap_runner as rr  # noqa: E402
from recap_options import OPTIONS_FILE, parse_set, resolve_options  # noqa: E402

try:
    from rich.console import Console
    from rich.markup import escape
    from rich.progress import (BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn,
                               TimeElapsedColumn, TimeRemainingColumn)
    from rich.table import Table
    HAVE_RICH = True
except ImportError:  # plain output without rich
    HAVE_RICH = False

    def escape(text):
        return text

STATE_MARK = {"done": "✓", "flagged": "⚑", "skipped": "–", "failed": "✗", "running": "…",
              "interrupted": "✗", None: " "}
STATE_STYLE = {"done": "green", "flagged": "yellow", "skipped": "dim", "failed": "red", "running": "cyan",
               "interrupted": "red"}


class PlainConsole:
    def print(self, *args, **_kw):
        print(*(getattr(a, "plain", a) for a in args), flush=True)

    def rule(self, text=""):
        print(f"--- {text} ---", flush=True)


def _console():
    return Console(highlight=False) if HAVE_RICH else PlainConsole()


def _strip(text: str) -> str:
    """Drop rich markup for plain output."""
    import re
    return text if HAVE_RICH else re.sub(r"\[/?[a-z ]+\]", "", text)


class RichReporter(rr.Reporter):
    """Stage list with marks, a progress bar with elapsed time and ETA, and flags as they come."""

    def __init__(self, console, game_folder: str):
        self.console = console
        self.game = Path(game_folder).name
        self.progress = None
        self.task = None
        self.t0 = 0.0

    def stage_start(self, name, title):
        self.t0 = time.time()
        if HAVE_RICH:
            self.progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                                     TaskProgressColumn(), TimeElapsedColumn(), TextColumn("ETA"),
                                     TimeRemainingColumn(), TextColumn("{task.fields[info]}"),
                                     console=self.console, transient=True)
            self.progress.start()
            self.task = self.progress.add_task(f"{name}: {title}", total=None, info="")

    def stage_progress(self, name, fraction, text=""):
        if self.progress is not None and self.task is not None:
            if fraction is None:
                self.progress.update(self.task, info=text)
            else:
                self.progress.update(self.task, total=1000, completed=int(fraction * 1000), info=text)

    def _stop(self):
        if self.progress is not None:
            self.progress.stop()
            self.progress = None

    def stage_end(self, name, record, reused):
        self._stop()
        state = record.get("state")
        mark = STATE_MARK.get(state, "?")
        style = STATE_STYLE.get(state, "")
        dur = record.get("duration_s")
        how = "unchanged, reused" if reused else (f"{dur:.0f} s" if isinstance(dur, (int, float)) else "")
        msg = record.get("message") or ""
        self.console.print(_strip(f"[{style}]{mark} {name:<11}[/{style}] {state:<8} {how:<18} {escape(msg)}"))
        if not reused:
            for f in record.get("flags", []):
                self.console.print(_strip(f"    [yellow]⚑ {escape(f)}[/yellow]"))

    def line(self, text):
        self.console.print(text)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _cli_options(args) -> dict:
    cli = parse_set(args.set or [])
    for key in ("league", "opponent", "focus_team", "perspective", "live_play_speed", "start"):
        v = getattr(args, key, None)
        if v is not None:
            cli[key] = v
    return cli


def cmd_run(args, from_stage=None) -> int:
    console = _console()
    folders = [str(Path(f).expanduser()) for f in args.game_folders]
    cli = _cli_options(args)

    def make_options(folder):
        if not Path(folder).is_dir():
            raise ValueError(f"[ERROR] Game Folder not found: {folder}")
        opts = resolve_options(folder, cli, args.config_dir)
        if opts["_created"]:
            console.print(f"Created {OPTIONS_FILE} (inferred values are marked; edit to change them)")
        return opts

    def reporter_for(folder):
        console.rule(_strip(f"[bold]{Path(folder).name}[/bold]"))
        return RichReporter(console, folder)

    try:
        summaries = rr.run_batch(folders, make_options, reporter_for, from_stage=from_stage,
                                 use_cache=not args.no_cache)
    except KeyboardInterrupt:
        console.print("Interrupted. Run the same command again to resume.")
        return 130
    _print_summary(console, summaries)
    return 0 if all(s["state"] == "done" for s in summaries) else 1


def _print_summary(console, summaries):
    console.rule("Summary")
    for s in summaries:
        stages = " ".join(f"{STATE_MARK.get(st, '?')}{n}" for n, st in s["stages"].items())
        state = {"done": "[green]done[/green]", "stopped": "[red]stopped[/red]"}.get(s["state"], f"[red]{s['state']}[/red]")
        console.print(_strip(f"{Path(s['game_folder']).name}: {state}  {stages}  "
                             f"{len(s['flags'])} flag(s), cache {rr.human_size(s['cache_bytes'])}"))
        if s.get("error"):
            console.print(_strip(f"    [red]{escape(s['error'])}[/red]"))
        for f in s["flags"]:
            console.print(f"    ⚑ {escape(f)}")


def cmd_status(args) -> int:
    console = _console()
    rc = 0
    for folder in args.game_folders:
        root = Path(folder).expanduser()
        st = rr.load_status(root)
        console.rule(root.name)
        if not st:
            console.print(f"No {rr.STATUS_FILE}: hockeyrecap has not run in this Game Folder.")
            rc = 1
            continue
        state = st.get("state")
        if state == "running":
            state = "running (or stopped without a clean exit; run again to resume)"
        console.print(f"Run: {state}; started {st.get('started')}, finished {st.get('finished') or '-'}"
                      + (f"; rerun from {st['from_stage']}" if st.get("from_stage") else ""))
        if st.get("stop_reason"):
            console.print(f"Stopped: {escape(st['stop_reason'])}")
        order = st.get("stage_order") or rr.STAGE_NAMES
        if HAVE_RICH:
            table = Table(show_header=True, header_style="bold")
            for col in ("", "stage", "state", "time", "finished", "result"):
                table.add_column(col)
            for n in order:
                r = st["stages"].get(n, {})
                s = r.get("state")
                table.add_row(STATE_MARK.get(s, " "), n, f"[{STATE_STYLE.get(s, '')}]{s or 'pending'}[/]",
                              "reused" if r.get("reused") else (f"{r['duration_s']:.0f} s" if r.get("duration_s") is not None else ""),
                              r.get("finished") or "", escape(r.get("message") or ""))
            console.print(table)
        else:
            for n in order:
                r = st["stages"].get(n, {})
                console.print(f"{STATE_MARK.get(r.get('state'), ' ')} {n:<11} {r.get('state') or 'pending':<8} {r.get('message') or ''}")
        flags = st.get("flags", [])
        console.print(f"{len(flags)} flag(s):" if flags else "No flags.")
        for f in flags:
            console.print(f"  ⚑ {escape(f)}")
        console.print(f"Signal cache: {rr.human_size(rr.cache_size(root))} in {rr.CACHE_DIR}/  "
                      f"Log: {root / rr.LOG_FILE}")
    return rc


def cmd_check(args) -> int:
    from recap_check import REPORT_FILE, check_game, format_report
    console = _console()
    rc = 0
    for folder in args.game_folders:
        root = Path(folder).expanduser()
        console.rule(root.name)
        t0 = time.time()
        if HAVE_RICH:
            with console.status("Checking Recordings (metadata and sampled frames)..."):
                rep = check_game(root)
        else:
            rep = check_game(root)
        for line in format_report(rep):
            console.print(_strip(f"[yellow]{escape(line)}[/yellow]") if line.startswith("[FLAG]") else escape(line))
        console.print(f"Checked in {time.time() - t0:.0f} s; report in {root / REPORT_FILE}")
        rc = rc or (0 if rep["ok"] else 1)
    return rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hockeyrecap", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_run_options(sp):
        sp.add_argument("game_folders", nargs="+", metavar="game_folder")
        sp.add_argument("--no-cache", action="store_true", help="compute the signals again, do not use the cache")
        sp.add_argument("--config-dir", help="League and Team Config folder (default ~/hockey or $HHG_CONFIG_DIR)")
        sp.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="override one option for this run, e.g. --set detection.thresh_pct=93")
        sp.add_argument("--league")
        sp.add_argument("--opponent")
        sp.add_argument("--focus-team", dest="focus_team")
        sp.add_argument("--perspective", choices=("focus", "neutral"))
        sp.add_argument("--live-play-speed", dest="live_play_speed", type=float)
        sp.add_argument("--start", choices=("cold_open", "play"))

    add_run_options(sub.add_parser("run", help="run or resume every stage"))
    rerun = sub.add_parser("rerun", help="run one stage and every later stage again")
    rerun.add_argument("--from", dest="from_stage", required=True, choices=rr.STAGE_NAMES)
    add_run_options(rerun)
    status = sub.add_parser("status", help="show the status of the last run")
    status.add_argument("game_folders", nargs="+", metavar="game_folder")
    check = sub.add_parser("check", help="quick Recording check: covered lens, camera stopped early, gaps")
    check.add_argument("game_folders", nargs="+", metavar="game_folder")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "rerun":
        return cmd_run(args, from_stage=args.from_stage)
    if args.command == "status":
        return cmd_status(args)
    return cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
