# v3/scripts/recap_options.py
"""
Options for one hockeyrecap run, in four layers (spec 002 §4). Each layer
overrides the one before it:

  1. League        <config>/leagues/<id>.json   rules; optional "options" block
  2. Team Config   <config>/teams/<id>.json     name, colors, Roster, own_team;
                                                optional "options" block
  3. Per-game file <Game Folder>/recap_options.json, written on the first run.
                   Each entry is {"value", "inferred", "source"}. An inferred
                   entry is inferred again on each run; set "inferred" to false
                   to keep your own value.
  4. CLI flags     one run only, never written to a file

<config> is ~/hockey, or $HHG_CONFIG_DIR, or --config-dir. It is outside the
repository because it holds the user's teams and Rosters.
"""
from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

OPTIONS_FILE = "recap_options.json"

DEFAULTS: dict = {
    "league": None,
    "opponent": None,
    "date": None,
    "focus_team": None,
    "perspective": "neutral",
    "live_play_speed": 1.10,
    "start": "cold_open",           # cold_open | play (start on play)
    "rois": {"source": "keep_existing"},   # keep_existing | auto (like hockeydetect)
    "detection": {                  # same values as run_detect.sh
        "fps": 12, "width": 1280, "thresh_pct": 95, "min_sep_s": 12, "merge_gap_s": 0,
        "max_win_s": 20, "max_big_win_s": 32, "big_color_min": "Orange",
        "replay_markers": True, "replay_offset_s": 1.0, "cooldown_s": 12, "audio_weight": 0,
        "marker_fps": 60,
    },
}

# Keys of the per-game file (layer 3). Other keys in the file are plain overrides.
GAME_KEYS = ("league", "opponent", "date", "focus_team", "perspective", "live_play_speed", "start")

_FOLDER_NAME = re.compile(r"^(?P<opponent>.+)_(?P<mm>\d{2})(?P<dd>\d{2})(?P<yyyy>\d{4})$")


def config_dir(override: str | None = None) -> Path:
    return Path(override or os.environ.get("HHG_CONFIG_DIR") or Path.home() / "hockey").expanduser()


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_dir(folder: Path) -> dict[str, dict]:
    items = {}
    if folder.is_dir():
        for p in sorted(folder.glob("*.json")):
            try:
                items[p.stem] = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError) as e:
                raise ValueError(f"[ERROR] Cannot read {p}: {e}")
    return items


def load_config(cfg: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """(leagues, teams) from the config folder, keyed by file name without .json."""
    return _load_dir(cfg / "leagues"), _load_dir(cfg / "teams")


def parse_folder_name(name: str) -> tuple[str | None, str | None]:
    """('wild', '2026-03-01') from 'wild_03012026'; (name, None) when there is no date."""
    m = _FOLDER_NAME.match(name)
    if not m:
        return (name or None), None
    return m["opponent"], f"{m['yyyy']}-{m['mm']}-{m['dd']}"


def _match_team(name: str | None, teams: dict[str, dict]) -> str | None:
    if not name:
        return None
    key = name.strip().lower()
    for tid, t in teams.items():
        if key in {tid.lower(), str(t.get("name", "")).lower(), str(t.get("short_name", "")).lower()}:
            return tid
    return None


def infer_game_values(game_folder: str, leagues: dict, teams: dict) -> dict[str, dict]:
    """Inferred layer-3 entries: {key: {"value", "inferred": True, "source"}}."""
    def entry(value, source):
        return {"value": value, "inferred": True, "source": source}

    opponent_name, date = parse_folder_name(Path(game_folder).resolve().name)
    out = {"date": entry(date, "folder name" if date else "not in folder name (opponent_MMDDYYYY)")}
    opp_team = _match_team(opponent_name, teams)
    out["opponent"] = entry(opp_team or opponent_name,
                            "folder name, matched Team Config" if opp_team else "folder name")

    own = [tid for tid, t in teams.items() if t.get("own_team") and tid != opp_team]
    if len(own) == 1:
        out["focus_team"] = entry(own[0], "the only own team in Team Config")
    else:
        out["focus_team"] = entry(None, f"{len(own)} own teams in Team Config; choose one")
    focus = out["focus_team"]["value"]
    out["perspective"] = entry("focus" if focus else "neutral",
                               "Focus Team known" if focus else "no Focus Team")

    league = teams.get(focus, {}).get("league") if focus else None
    if league:
        out["league"] = entry(league, f"Team Config of {focus}")
    elif len(leagues) == 1:
        out["league"] = entry(next(iter(leagues)), "the only League config")
    else:
        out["league"] = entry(None, f"{len(leagues)} League configs; choose one")
    out["live_play_speed"] = entry(DEFAULTS["live_play_speed"], "default")
    out["start"] = entry(DEFAULTS["start"], "default")
    return out


def load_game_file(game_folder: str, leagues: dict, teams: dict) -> tuple[dict, bool]:
    """
    Read the per-game file, create it on the first run, and infer again every
    entry that is still marked inferred. Returns (file content, created).
    """
    path = Path(game_folder) / OPTIONS_FILE
    created = not path.exists()
    data = {} if created else json.loads(path.read_text())
    inferred = infer_game_values(game_folder, leagues, teams)
    changed = created
    for key in GAME_KEYS:
        cur = data.get(key)
        if not isinstance(cur, dict) or "value" not in cur or cur.get("inferred", False) is True:
            if cur != inferred[key]:
                data[key] = inferred[key]
                changed = True
    if "_help" not in data:
        data = {"_help": "Per-game options for hockeyrecap. Entries with \"inferred\": true are "
                         "worked out again on each run. To keep your own value, change \"value\" "
                         "and set \"inferred\" to false. Other keys (for example \"detection\") "
                         "override the defaults.", **data}
        changed = True
    if changed:
        path.write_text(json.dumps(data, indent=2) + "\n")
    return data, created


def parse_set(items: list[str]) -> dict:
    """--set key.path=value pairs into a nested dict. Values are JSON when they parse."""
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"[ERROR] --set needs key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def resolve_options(game_folder: str, cli: dict | None = None, cfg_override: str | None = None) -> dict:
    """
    Effective options for one run. Also returns where things came from:
    options["_layers"] lists the files used, options["_created"] tells whether
    the per-game file was created now.
    """
    cfg = config_dir(cfg_override)
    leagues, teams = load_config(cfg)
    game, created = load_game_file(game_folder, leagues, teams)
    game_values = {k: (v["value"] if isinstance(v, dict) and "value" in v and k in GAME_KEYS else v)
                   for k, v in game.items() if not k.startswith("_")}
    cli = cli or {}

    # League and Focus Team are known only after layers 3 and 4, so find them first.
    league_id = cli.get("league", game_values.get("league"))
    focus_id = cli.get("focus_team", game_values.get("focus_team"))
    league = leagues.get(league_id, {}) if league_id else {}
    focus = teams.get(focus_id, {}) if focus_id else {}

    opts = deep_merge(DEFAULTS, league.get("options", {}))
    opts = deep_merge(opts, focus.get("options", {}))
    opts = deep_merge(opts, game_values)
    opts = deep_merge(opts, cli)
    opts["league_rules"] = {k: v for k, v in league.items() if k != "options"}
    opts["teams"] = teams
    opts["_layers"] = {
        "config_dir": str(cfg),
        "league": f"leagues/{league_id}.json" if league else None,
        "team": f"teams/{focus_id}.json" if focus else None,
        "game": str(Path(game_folder) / OPTIONS_FILE),
        "cli": sorted(cli),
    }
    opts["_created"] = created
    flags = []
    if league_id and not league:
        flags.append(f"League '{league_id}' has no config in {cfg / 'leagues'}")
    if focus_id and not focus:
        flags.append(f"Focus Team '{focus_id}' has no Team Config in {cfg / 'teams'}")
    if not teams:
        flags.append(f"No Team Config in {cfg / 'teams'}; Focus Team and Roster unknown")
    opts["_flags"] = flags
    return opts
