# v3/scripts/team_db.py
"""
Team, Roster, and game data in one SQLite database (ADR-0004).

The database is <config>/hockey.db (see recap_options.config_dir). It is outside
the repository because it holds the user's teams and Rosters. The schema is
versioned with PRAGMA user_version; connect() applies missing migrations, each
in one transaction, and refuses a database newer than this code.

Rules:
- A person is one human across teams and seasons. A Roster entry gives that
  person a number on one team for one Season.
- A game uses the Roster for its Season, or the latest earlier Roster (flagged).
- An unknown name stops the import with the closest Roster matches.
- The number worn in a game wins over the Roster and is flagged when they differ.
- 08 and 8 are the same number; the printed form is kept as a label.
- Every record keeps its source: sheet, override, user, or import.

Renders read a per-game snapshot (game_data.json in the Game Folder) written
by write_snapshot(), so a past render can be repeated after a Roster changes.

  team_db.py init
  team_db.py import-team <teams/x.json> --season 2026-2027
  team_db.py import-events <events.json> --game <id> --date YYYY-MM-DD --teams a,b [--focus a] [--folder DIR]
  team_db.py snapshot <game id> [--out FILE]
  team_db.py flags <game id>
  team_db.py people-flags
  team_db.py merge-people <keep id> <drop id>
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import os
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from recap_options import config_dir

DB_FILE = "hockey.db"
SNAPSHOT_FILE = "game_data.json"
SNAPSHOT_VERSION = 2
SOURCES = ("sheet", "override", "user", "import")
MAX_ASSISTS = 2

MIGRATIONS = [
    """
    CREATE TABLE teams (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        short_name TEXT,
        own_team INTEGER NOT NULL DEFAULT 0,
        colors TEXT NOT NULL DEFAULT '{}',
        kits TEXT NOT NULL DEFAULT '{}',
        assets TEXT NOT NULL DEFAULT '{}',
        extra TEXT NOT NULL DEFAULT '{}',
        source TEXT NOT NULL
    );
    CREATE TABLE people (
        id INTEGER PRIMARY KEY,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        display_name TEXT,
        source TEXT NOT NULL
    );
    CREATE TABLE family_links (
        parent_id INTEGER NOT NULL REFERENCES people(id),
        child_id INTEGER NOT NULL REFERENCES people(id),
        relation TEXT NOT NULL,
        source TEXT NOT NULL,
        PRIMARY KEY (parent_id, child_id, relation)
    );
    CREATE TABLE roster_entries (
        id INTEGER PRIMARY KEY,
        team_id TEXT NOT NULL REFERENCES teams(id),
        season TEXT NOT NULL,
        person_id INTEGER NOT NULL REFERENCES people(id),
        jersey INTEGER,
        position TEXT,
        role TEXT NOT NULL DEFAULT 'player',
        source TEXT NOT NULL,
        UNIQUE (team_id, season, person_id)
    );
    CREATE TABLE games (
        id TEXT PRIMARY KEY,
        date TEXT NOT NULL,
        season TEXT NOT NULL,
        focus_team_id TEXT REFERENCES teams(id),
        folder TEXT,
        source TEXT NOT NULL
    );
    CREATE TABLE game_teams (
        game_id TEXT NOT NULL REFERENCES games(id),
        team_id TEXT NOT NULL REFERENCES teams(id),
        side TEXT CHECK (side IN ('home', 'away')),
        PRIMARY KEY (game_id, team_id)
    );
    CREATE TABLE game_players (
        game_id TEXT NOT NULL REFERENCES games(id),
        team_id TEXT NOT NULL REFERENCES teams(id),
        person_id INTEGER NOT NULL REFERENCES people(id),
        jersey_worn INTEGER,
        note TEXT,
        source TEXT NOT NULL,
        PRIMARY KEY (game_id, person_id)
    );
    CREATE TABLE events (
        id INTEGER PRIMARY KEY,
        game_id TEXT NOT NULL REFERENCES games(id),
        team_id TEXT NOT NULL REFERENCES teams(id),
        kind TEXT NOT NULL CHECK (kind IN ('goal', 'penalty')),
        ordinal INTEGER NOT NULL,
        parent_event_id INTEGER REFERENCES events(id),
        person_id INTEGER REFERENCES people(id),
        period INTEGER,
        clock TEXT,
        scene INTEGER,
        infraction TEXT,
        classification TEXT,
        minutes REAL,
        unassisted INTEGER,
        note TEXT,
        source TEXT NOT NULL
    );
    CREATE TABLE event_assists (
        event_id INTEGER NOT NULL REFERENCES events(id),
        ordinal INTEGER NOT NULL,
        person_id INTEGER NOT NULL REFERENCES people(id),
        PRIMARY KEY (event_id, ordinal)
    );
    """,
    """
    ALTER TABLE roster_entries ADD COLUMN jersey_label TEXT;
    ALTER TABLE game_players ADD COLUMN jersey_label TEXT;
    ALTER TABLE game_teams ADD COLUMN source TEXT;
    ALTER TABLE event_assists ADD COLUMN source TEXT;
    ALTER TABLE events ADD COLUMN match_note TEXT;
    """,
]


def db_path(override: str | None = None) -> Path:
    return config_dir(override) / DB_FILE


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Open the database and apply missing migrations, each in one transaction."""
    path = Path(path) if path else db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > len(MIGRATIONS):
        conn.close()
        raise RuntimeError(f"{path} has schema version {version}; this code knows {len(MIGRATIONS)}. Update the code.")
    for number, sql in enumerate(MIGRATIONS[version:], start=version + 1):
        conn.execute("BEGIN")
        try:
            for statement in (s.strip() for s in sql.split(";")):
                if statement:
                    conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {number}")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    conn.isolation_level = "DEFERRED"  # `with conn:` blocks are transactions from here on
    return conn


# --- value rules ------------------------------------------------------------------------

def season_for(date: str) -> str:
    """Adult league seasons start in August: 2026-10-11 belongs to 2026-2027."""
    if not isinstance(date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise ValueError(f"date must be YYYY-MM-DD, not {date!r}")
    d = dt.date.fromisoformat(date)
    start = d.year if d.month >= 8 else d.year - 1
    return f"{start}-{start + 1}"


def source_of(value: str | None, default: str = "user") -> str:
    """A source must start with sheet, override, user, or import; details may follow."""
    text = " ".join(str(value or default).split())
    if not text.casefold().startswith(SOURCES):
        raise ValueError(f"source {text!r} must start with one of {', '.join(SOURCES)}")
    return text


def clean_name(value) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value)).split())


def name_key(value) -> str:
    """Case, whitespace, and accent differences do not make a different person."""
    text = unicodedata.normalize("NFKD", clean_name(value).casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


def split_name(full: str) -> tuple[str, str]:
    """'John David Mohr' -> ('John David', 'Mohr'). A one-word name is its own last name."""
    first, _, last = clean_name(full).rpartition(" ")
    if not (first or last):
        raise ValueError("empty name")
    return (first or last), last


def parse_jersey(value) -> tuple[int | None, str | None]:
    """Return (number, printed label). 08 and 8 are the same number; the label keeps '08'."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, None
    if isinstance(value, bool):
        raise ValueError(f"jersey {value!r} is not a number")
    if isinstance(value, int):
        return value, None
    text = str(value).strip().lstrip("#")
    if not text.isdigit():
        raise ValueError(f"jersey {value!r} is not a number")
    number = int(text)
    return number, (text if text != str(number) else None)


def parse_minutes(value) -> float | None:
    """2, '2', '2+10' (12), '5:00' (5). None means not recorded."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if re.fullmatch(r"\d+(\.\d+)?(\s*\+\s*\d+(\.\d+)?)*", text):
        return float(sum(float(p) for p in text.split("+")))
    if re.fullmatch(r"\d+:[0-5]\d", text):
        m, s = map(int, text.split(":"))
        return m + s / 60
    raise ValueError(f"penalty minutes {value!r} are not a number, a sum like 2+10, or m:ss")


def _whole(value: float):
    return int(value) if float(value).is_integer() else value


# --- people and rosters -----------------------------------------------------------------

def _new_person(conn, first: str, last: str, source: str, display: str | None = None) -> int:
    return conn.execute("INSERT INTO people (first_name, last_name, display_name, source) VALUES (?, ?, ?, ?)",
                        (clean_name(first), clean_name(last), clean_name(display) if display else None,
                         source)).lastrowid


def _keys(row) -> set[str]:
    keys = {name_key(f"{row['first_name']} {row['last_name']}")}
    if row["display_name"]:
        keys.add(name_key(row["display_name"]))
    return keys


def _team_people(conn, team_id: str) -> list[sqlite3.Row]:
    """Everyone with a Roster entry on this team in any Season, with each Season's number."""
    return conn.execute(
        "SELECT p.id, p.first_name, p.last_name, p.display_name, r.season, r.jersey FROM people p "
        "JOIN roster_entries r ON r.person_id = p.id WHERE r.team_id = ?", (team_id,)).fetchall()


def _roster_season(conn, team_id: str, season: str) -> str | None:
    """The game's Season if that Roster exists, else the latest earlier Season with a Roster."""
    row = conn.execute("SELECT MAX(season) FROM roster_entries WHERE team_id = ? AND season <= ?",
                       (team_id, season)).fetchone()
    return row[0]


def _match_for_import(conn, team_id: str, first: str, last: str, jersey: int | None, used: set[int]) -> int | None:
    """Reuse a person already on this team (any Season) on a reimport; never merge two rows of one file."""
    key = name_key(f"{first} {last}")
    hits = {}
    for r in _team_people(conn, team_id):
        if r["id"] not in used and key in _keys(r):
            hits.setdefault(r["id"], set()).add(r["jersey"])
    if len(hits) > 1 and jersey is not None:
        hits = {pid: nums for pid, nums in hits.items() if jersey in nums}
    if len(hits) == 1:
        return next(iter(hits))
    if hits:
        raise ValueError(f"{team_id}: {first} {last} matches {len(hits)} people; give each row a distinct jersey")
    return None


def _family_member(conn, team_id: str, season: str, member: dict, source: str) -> int:
    """A family link names a Roster player, or a relative off the Roster that this team already linked."""
    jersey, _ = parse_jersey(member.get("jersey"))
    pid = find_person(conn, team_id, season, name=member["name"], jersey=jersey, required=False)
    if pid:
        return pid
    key = name_key(member["name"])
    linked = conn.execute(
        "SELECT p.id, p.first_name, p.last_name, p.display_name FROM people p JOIN family_links f "
        "ON p.id IN (f.parent_id, f.child_id) WHERE (CASE WHEN p.id = f.parent_id THEN f.child_id ELSE f.parent_id END) "
        "IN (SELECT person_id FROM roster_entries WHERE team_id = ?)", (team_id,)).fetchall()
    for r in linked:
        if key in _keys(r):
            return r["id"]
    first, last = split_name(member["name"])
    return _new_person(conn, first, last, source)


def import_team(conn, path: str | Path, season: str, source: str | None = None) -> str:
    """Import one Team Config JSON file and its listed Roster for one Season. All or nothing."""
    path = Path(path)
    cfg = json.loads(path.read_text())
    team_id = path.stem
    source = source_of(source or f"import {path.name}")
    if not re.fullmatch(r"\d{4}-\d{4}", season):
        raise ValueError(f"season must look like 2026-2027, not {season!r}")
    known = {"name", "short_name", "own_team", "colors", "kits", "logo", "lockup", "crest",
             "roster", "listed_roster", "family_links"}
    assets = {k: {"path": cfg[k], "source": source_of(cfg.get(f"{k}_source") or source)}
              for k in ("logo", "lockup", "crest") if cfg.get(k)}
    extra = {k: v for k, v in cfg.items() if k not in known}
    rows = cfg.get("listed_roster")
    if rows is None:  # older files have only the number -> name map
        rows = [{"jersey": num, "name": name} for num, name in (cfg.get("roster") or {}).items()]
    with conn:
        conn.execute(
            "INSERT INTO teams (id, name, short_name, own_team, colors, kits, assets, extra, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET name = excluded.name, "
            "short_name = excluded.short_name, own_team = excluded.own_team, colors = excluded.colors, "
            "kits = excluded.kits, assets = excluded.assets, extra = excluded.extra, source = excluded.source",
            (team_id, clean_name(cfg["name"]), cfg.get("short_name"), int(bool(cfg.get("own_team"))),
             json.dumps(cfg.get("colors") or {}), json.dumps(cfg.get("kits") or {}), json.dumps(assets),
             json.dumps(extra), source))
        used: set[int] = set()
        for i, row in enumerate(rows, start=1):
            if row.get("first_name") and row.get("last_name"):
                first, last = clean_name(row["first_name"]), clean_name(row["last_name"])
            elif row.get("name"):
                first, last = split_name(row["name"])
            else:
                raise ValueError(f"{path.name}: roster row {i} has no name")
            jersey, label = parse_jersey(row.get("jersey"))
            row_source = source_of(row.get("jersey_source") or row.get("source") or source)
            pid = _match_for_import(conn, team_id, first, last, jersey, used)
            if pid is None:
                pid = _new_person(conn, first, last, row_source, row.get("display_name"))
            elif row.get("display_name"):
                conn.execute("UPDATE people SET display_name = ?, source = ? WHERE id = ?",
                             (clean_name(row["display_name"]), row_source, pid))
            used.add(pid)
            conn.execute(
                "INSERT INTO roster_entries (team_id, season, person_id, jersey, jersey_label, position, role, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(team_id, season, person_id) DO UPDATE SET "
                "jersey = excluded.jersey, jersey_label = excluded.jersey_label, position = excluded.position, "
                "role = excluded.role, source = excluded.source",
                (team_id, season, pid, jersey, label, row.get("position"),
                 "goalie" if row.get("position") == "G" else "player", row_source))
        for link in cfg.get("family_links") or []:
            link_source = source_of(link.get("source") or source)
            parent = _family_member(conn, team_id, season, link["parent"], link_source)
            child = _family_member(conn, team_id, season, link["child"], link_source)
            conn.execute("INSERT OR REPLACE INTO family_links VALUES (?, ?, ?, ?)",
                         (parent, child, link["relation"], link_source))
    return team_id


def find_person(conn, team_id: str, season: str, name: str | None = None, jersey=None,
                required: bool = True) -> int | None:
    """Match a name on the Season Roster (or the latest earlier one), using the jersey when names collide.

    An unknown name raises ValueError with the closest Roster names unless required is False.
    """
    used = _roster_season(conn, team_id, season)
    rows = conn.execute(
        "SELECT p.id, p.first_name, p.last_name, p.display_name, r.jersey FROM roster_entries r "
        "JOIN people p ON p.id = r.person_id WHERE r.team_id = ? AND r.season = ?", (team_id, used)).fetchall()
    number, _ = parse_jersey(jersey)
    if name and clean_name(name):
        key = name_key(name)
        hits = [r for r in rows if key in _keys(r)]
        if len(hits) > 1 and number is not None:
            hits = [r for r in hits if r["jersey"] == number]
        if len(hits) == 1:
            return hits[0]["id"]
        if hits:
            raise ValueError(f"{team_id}: {name!r} matches {len(hits)} people; give the jersey")
        if not required:
            return None
        names = {name_key(f"{r['first_name']} {r['last_name']}"): r for r in rows}
        close = [names[k] for k in difflib.get_close_matches(key, names, n=3, cutoff=0.6)]
        if number is not None:
            close += [r for r in rows if r["jersey"] == number and r not in close]
        hint = ", ".join(f"#{r['jersey']} {r['display_name'] or r['first_name'] + ' ' + r['last_name']}"
                         if r["jersey"] is not None else (r["display_name"] or f"{r['first_name']} {r['last_name']}")
                         for r in close)
        raise ValueError(f"{team_id}: {clean_name(name)!r} is not on the {used or season} roster"
                         + (f"; did you mean {hint}?" if hint else "; add the player or fix the spelling"))
    if number is not None:
        hits = [r for r in rows if r["jersey"] == number]
        if len(hits) == 1:
            return hits[0]["id"]
        if hits:
            raise ValueError(f"{team_id}: jersey #{number} is shared by {len(hits)} people; give a name")
        if required:
            raise ValueError(f"{team_id}: #{number} is not on the {used or season} roster; give a name")
    return None


def merge_people(conn, keep: int, drop: int) -> None:
    """Make two person records one (for example the same player on two teams), after review."""
    if keep == drop:
        raise ValueError("keep and drop are the same person")
    with conn:
        for table, keys in (("roster_entries", "team_id, season"), ("game_players", "game_id")):
            clash = conn.execute(f"SELECT COUNT(*) FROM {table} a JOIN {table} b USING ({keys}) "
                                 "WHERE a.person_id = ? AND b.person_id = ?", (keep, drop)).fetchone()[0]
            if clash:
                raise ValueError(f"people {keep} and {drop} both appear in the same {table} entry; not one person")
        for table, col in (("roster_entries", "person_id"), ("game_players", "person_id"), ("events", "person_id"),
                           ("event_assists", "person_id"), ("family_links", "parent_id"),
                           ("family_links", "child_id")):
            conn.execute(f"UPDATE OR IGNORE {table} SET {col} = ? WHERE {col} = ?", (keep, drop))
        conn.execute("DELETE FROM family_links WHERE parent_id = ? OR child_id = ?", (drop, drop))
        conn.execute("DELETE FROM people WHERE id = ?", (drop,))


def people_flags(conn) -> list[str]:
    """Different person records with the same name: possibly one person on two teams."""
    groups: dict[str, list] = {}
    for r in conn.execute("SELECT p.id, p.first_name, p.last_name, p.display_name, "
                          "GROUP_CONCAT(DISTINCT r.team_id) teams FROM people p "
                          "LEFT JOIN roster_entries r ON r.person_id = p.id GROUP BY p.id"):
        groups.setdefault(name_key(f"{r['first_name']} {r['last_name']}"), []).append(r)
    flags = []
    for rows in groups.values():
        team_sets = [set((r["teams"] or "").split(",")) for r in rows]
        if len(rows) > 1 and not any(a & b - {""} for i, a in enumerate(team_sets) for b in team_sets[i + 1:]):
            who = ", ".join(f"id {r['id']} ({r['teams'] or 'no roster'})" for r in rows)
            flags.append(f"{rows[0]['first_name']} {rows[0]['last_name']} has {len(rows)} records: {who}; "
                         "if one person, run merge-people")
    return flags


# --- games ------------------------------------------------------------------------------

def add_game(conn, game_id: str, date: str, team_ids: list[str], focus: str | None = None,
             folder: str | None = None, source: str = "user") -> str:
    season = season_for(date)
    source = source_of(source)
    with conn:
        conn.execute("INSERT INTO games (id, date, season, focus_team_id, folder, source) VALUES (?, ?, ?, ?, ?, ?) "
                     "ON CONFLICT(id) DO UPDATE SET date = excluded.date, season = excluded.season, "
                     "focus_team_id = excluded.focus_team_id, folder = excluded.folder",
                     (game_id, date, season, focus, folder, source))
        for tid in team_ids:
            conn.execute("INSERT INTO game_teams (game_id, team_id, source) VALUES (?, ?, ?) "
                         "ON CONFLICT(game_id, team_id) DO NOTHING", (game_id, tid, source))
    return game_id


def _game(conn, game_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
    if row is None:
        raise KeyError(f"no game {game_id!r}")
    return row


def _game_player(conn, game_id, team_id, season, name, jersey, source, note=None) -> int:
    number, label = parse_jersey(jersey)
    pid = find_person(conn, team_id, season, name=name, jersey=number)
    old = conn.execute("SELECT jersey_worn FROM game_players WHERE game_id = ? AND person_id = ?",
                       (game_id, pid)).fetchone()
    if old and old[0] is not None and number is not None and old[0] != number:
        raise ValueError(f"{team_id}: {clean_name(name or '#' + str(number))} wore #{old[0]} and #{number} "
                         "in this game; fix one event")
    conn.execute("INSERT INTO game_players (game_id, team_id, person_id, jersey_worn, jersey_label, note, source) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(game_id, person_id) DO UPDATE SET "
                 "jersey_worn = COALESCE(excluded.jersey_worn, game_players.jersey_worn), "
                 "jersey_label = COALESCE(excluded.jersey_label, game_players.jersey_label), "
                 "note = COALESCE(excluded.note, game_players.note)",
                 (game_id, team_id, pid, number, label, note, source))
    return pid


def import_events(conn, game_id: str, events: list[dict], team_names: dict[str, str],
                  source: str = "user") -> int:
    """Replace a game's events, all or nothing. team_names maps an event 'team' name to a team id.

    The import file is the record for these events: corrections go into that file, then reimport.
    """
    game = _game(conn, game_id)
    season = game["season"]
    in_game = {r[0] for r in conn.execute("SELECT team_id FROM game_teams WHERE game_id = ?", (game_id,))}
    with conn:
        conn.execute("DELETE FROM event_assists WHERE event_id IN (SELECT id FROM events WHERE game_id = ?)", (game_id,))
        conn.execute("DELETE FROM events WHERE game_id = ? AND parent_event_id IS NOT NULL", (game_id,))
        conn.execute("DELETE FROM events WHERE game_id = ?", (game_id,))
        conn.execute("DELETE FROM game_players WHERE game_id = ?", (game_id,))
        order = {"goal": 0, "penalty": 0}
        for n, e in enumerate(events, start=1):
            kind = str(e.get("event", "")).lower()
            if kind not in order:
                raise ValueError(f"event {n}: kind {e.get('event')!r} is not GOAL or PENALTY")
            team_id = team_names[e["team"]]
            if team_id not in in_game:
                raise ValueError(f"event {n}: {e['team']} is not a team in game {game_id}")
            src = source_of(e.get("source") or source)
            order[kind] += 1
            name = e.get("scorer") if kind == "goal" else e.get("player")
            if kind == "penalty" and not name and e.get("jersey") in (None, ""):
                pid = None  # team or bench penalty: no player
            else:
                pid = _game_player(conn, game_id, team_id, season, name, e.get("jersey"), src, e.get("jersey_note"))
            minutes = parse_minutes(e.get("minutes")) if kind == "penalty" else None
            assists = e.get("assists") or []
            if kind == "goal":
                assist_ids = _assist_ids(conn, game_id, team_id, season, in_game, assists, pid, src, n)
            elif assists:
                raise ValueError(f"event {n}: a penalty has no assists")
            if e.get("additional") and kind != "penalty":
                raise ValueError(f"event {n}: only a penalty can have additional penalties")
            eid = conn.execute(
                "INSERT INTO events (game_id, team_id, kind, ordinal, person_id, period, clock, scene, infraction, "
                "classification, minutes, unassisted, match_note, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (game_id, team_id, kind, order[kind], pid, e.get("period"), e.get("game_clock"), e.get("scene"),
                 e.get("infraction"), e.get("classification"), minutes,
                 int(not assists) if kind == "goal" else None, e.get("roster_match"), src)).lastrowid
            if kind == "goal":
                for i, apid in enumerate(assist_ids, start=1):
                    conn.execute("INSERT INTO event_assists (event_id, ordinal, person_id, source) VALUES (?, ?, ?, ?)",
                                 (eid, i, apid, src))
            for extra in e.get("additional") or []:
                if not extra.get("classification"):
                    raise ValueError(f"event {n}: an additional penalty needs a classification")
                order["penalty"] += 1
                conn.execute(
                    "INSERT INTO events (game_id, team_id, kind, ordinal, parent_event_id, person_id, period, scene, "
                    "classification, minutes, note, source) VALUES (?, ?, 'penalty', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (game_id, team_id, order["penalty"], eid, pid, e.get("period"), e.get("scene"),
                     extra["classification"], parse_minutes(extra.get("minutes")), extra.get("note"), src))
    return sum(order.values())


def _assist_ids(conn, game_id, team_id, season, in_game, assists, scorer, source, n) -> list[int]:
    if len(assists) > MAX_ASSISTS:
        raise ValueError(f"event {n}: a goal has at most {MAX_ASSISTS} assists, not {len(assists)}")
    ids = []
    for a in assists:
        try:
            pid = _game_player(conn, game_id, team_id, season, a.get("name"), a.get("jersey"), source)
        except ValueError:
            for other in in_game - {team_id}:
                if find_person(conn, other, season, name=a.get("name"), jersey=a.get("jersey"), required=False):
                    raise ValueError(f"event {n}: assist {a.get('name')!r} plays for {other}, not {team_id}") from None
            raise
        if pid == scorer:
            raise ValueError(f"event {n}: the scorer cannot assist the same goal")
        if pid in ids:
            raise ValueError(f"event {n}: the same player is listed twice as an assist")
        ids.append(pid)
    return ids


def game_flags(conn, game_id: str) -> list[str]:
    """Review flags: an earlier Roster in use, numbers worn that differ from it, shared or missing numbers."""
    game = _game(conn, game_id)
    season = game["season"]
    flags = []
    rosters = {}
    for (tid,) in conn.execute("SELECT team_id FROM game_teams WHERE game_id = ?", (game_id,)):
        rosters[tid] = _roster_season(conn, tid, season)
        if rosters[tid] is None:
            flags.append(f"{tid}: no roster for {season} or earlier")
        elif rosters[tid] != season:
            flags.append(f"{tid}: no {season} roster; using {rosters[tid]}")
    for r in conn.execute(
            "SELECT gp.team_id, gp.jersey_worn, p.first_name, p.last_name, p.display_name, re.jersey AS listed "
            "FROM game_players gp JOIN people p ON p.id = gp.person_id JOIN games g ON g.id = gp.game_id "
            "LEFT JOIN roster_entries re ON re.person_id = gp.person_id AND re.team_id = gp.team_id "
            "AND re.season = (SELECT MAX(season) FROM roster_entries WHERE team_id = gp.team_id AND season <= g.season) "
            "WHERE gp.game_id = ?", (game_id,)):
        who = r["display_name"] or f"{r['first_name']} {r['last_name']}"
        if r["listed"] is None and r["jersey_worn"] is None:
            flags.append(f"{r['team_id']}: {who} has no number in this game or on the roster")
        elif r["listed"] is not None and r["jersey_worn"] is not None and r["listed"] != r["jersey_worn"]:
            flags.append(f"{r['team_id']}: {who} wore #{r['jersey_worn']}; roster lists #{r['listed']}")
    for tid, used in rosters.items():
        if used is None:
            continue
        for r in conn.execute("SELECT jersey, COUNT(*) n FROM roster_entries WHERE team_id = ? AND season = ? "
                              "AND jersey IS NOT NULL GROUP BY jersey HAVING n > 1", (tid, used)):
            flags.append(f"{tid}: #{r['jersey']} is shared by {r['n']} players on the {used} roster")
    for r in conn.execute("SELECT team_id, classification FROM events WHERE game_id = ? AND kind = 'penalty' "
                          "AND minutes IS NULL", (game_id,)):
        flags.append(f"{r['team_id']}: {r['classification'] or 'penalty'} has no minutes; PIM excludes it")
    return flags


def snapshot(conn, game_id: str) -> dict:
    """The data one render reads: teams, players with numbers, events, and derived statistics."""
    game = _game(conn, game_id)
    season = game["season"]
    people = {}
    for r in conn.execute(
            "SELECT p.id, p.first_name, p.last_name, p.display_name, gp.team_id, gp.jersey_worn, gp.jersey_label, "
            "re.jersey AS listed, re.jersey_label AS listed_label FROM game_players gp JOIN people p ON p.id = gp.person_id "
            "LEFT JOIN roster_entries re ON re.person_id = gp.person_id AND re.team_id = gp.team_id AND re.season = "
            "(SELECT MAX(season) FROM roster_entries WHERE team_id = gp.team_id AND season <= ?) WHERE gp.game_id = ?",
            (season, game_id)):
        worn = r["jersey_worn"]
        jersey = worn if worn is not None else r["listed"]
        label = (r["jersey_label"] if worn is not None else r["listed_label"]) or (None if jersey is None else str(jersey))
        people[r["id"]] = {"id": r["id"], "name": r["display_name"] or f"{r['first_name']} {r['last_name']}",
                           "jersey": jersey, "jersey_label": label}
    teams = {}
    for t in conn.execute("SELECT t.*, gt.side FROM teams t JOIN game_teams gt ON gt.team_id = t.id "
                          "WHERE gt.game_id = ? ORDER BY t.id", (game_id,)):
        teams[t["id"]] = {"name": t["name"], "short_name": t["short_name"], "own_team": bool(t["own_team"]),
                          "side": t["side"], "colors": json.loads(t["colors"]), "kits": json.loads(t["kits"]),
                          "assets": json.loads(t["assets"]), "extra": json.loads(t["extra"])}
    events, stats = [], {tid: {} for tid in teams}
    team_pim = {tid: 0.0 for tid in teams}
    first_event = {}

    def stat_row(tid, pid):
        first_event.setdefault(pid, len(first_event))
        return stats[tid].setdefault(pid, {**people[pid], "g": 0, "a": 0, "pts": 0, "pim": 0.0})

    for e in conn.execute("SELECT * FROM events WHERE game_id = ? ORDER BY id", (game_id,)):
        out = {k: e[k] for k in ("id", "kind", "ordinal", "parent_event_id", "period", "clock", "scene",
                                 "infraction", "classification", "minutes", "source")}
        out["team"] = e["team_id"]
        out["person"] = people[e["person_id"]] if e["person_id"] else None
        if e["kind"] == "goal":
            out["assists"] = [people[a["person_id"]] for a in conn.execute(
                "SELECT person_id FROM event_assists WHERE event_id = ? ORDER BY ordinal", (e["id"],))]
            out["unassisted"] = not out["assists"]
            stat_row(e["team_id"], e["person_id"])["g"] += 1
            for a in out["assists"]:
                stat_row(e["team_id"], a["id"])["a"] += 1
        elif e["minutes"] is not None:
            team_pim[e["team_id"]] += e["minutes"]
            if e["person_id"]:
                stat_row(e["team_id"], e["person_id"])["pim"] += e["minutes"]
        elif e["person_id"]:
            stat_row(e["team_id"], e["person_id"])
        events.append(out)
    tables = {}
    for tid, rows in stats.items():
        for r in rows.values():
            r["pts"] = r["g"] + r["a"]
            r["pim"] = _whole(r["pim"])
        # Points, then goals, then PIM; ties keep the order of first involvement in the game.
        tables[tid] = sorted(rows.values(), key=lambda r: (-r["pts"], -r["g"], -r["pim"], first_event[r["id"]]))
    goals = {tid: sum(e["kind"] == "goal" and e["team"] == tid for e in events) for tid in teams}
    return {"schema_version": SNAPSHOT_VERSION, "game": {"id": game_id, "date": game["date"], "season": season,
            "focus_team": game["focus_team_id"], "folder": game["folder"]}, "teams": teams, "score": goals,
            "events": events, "stats": tables, "team_pim": {t: _whole(v) for t, v in team_pim.items()},
            "flags": game_flags(conn, game_id)}


def write_snapshot(conn, game_id: str, out: str | Path | None = None) -> Path:
    data = snapshot(conn, game_id)
    if out is None:
        if not data["game"]["folder"]:
            raise ValueError(f"game {game_id} has no folder; give --out")
        out = Path(data["game"]["folder"]) / SNAPSHOT_FILE
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-dir", help="config folder (default ~/hockey or $HHG_CONFIG_DIR)")
    ap.add_argument("--db", help="database file (default <config>/hockey.db)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    t = sub.add_parser("import-team")
    t.add_argument("path")
    t.add_argument("--season", required=True)
    e = sub.add_parser("import-events")
    e.add_argument("path")
    e.add_argument("--game", required=True)
    e.add_argument("--date", required=True)
    e.add_argument("--teams", required=True, help="comma-separated team ids")
    e.add_argument("--focus")
    e.add_argument("--folder")
    s = sub.add_parser("snapshot")
    s.add_argument("game")
    s.add_argument("--out")
    f = sub.add_parser("flags")
    f.add_argument("game")
    sub.add_parser("people-flags")
    m = sub.add_parser("merge-people")
    m.add_argument("keep", type=int)
    m.add_argument("drop", type=int)
    args = ap.parse_args(argv)
    conn = connect(args.db or db_path(args.config_dir))
    if args.cmd == "init":
        print(f"[team_db] schema version {conn.execute('PRAGMA user_version').fetchone()[0]}")
    elif args.cmd == "import-team":
        print(f"[team_db] imported {import_team(conn, args.path, args.season)} for {args.season}")
    elif args.cmd == "import-events":
        ids = args.teams.split(",")
        add_game(conn, args.game, args.date, ids, args.focus, args.folder)
        by_name = {r["name"]: r["id"] for r in conn.execute(
            f"SELECT id, name FROM teams WHERE id IN ({','.join('?' * len(ids))})", ids)}
        data = json.loads(Path(args.path).read_text())
        n = import_events(conn, args.game, data["events"], by_name, source=data.get("source", "user"))
        print(f"[team_db] {args.game}: {n} events")
    elif args.cmd == "snapshot":
        print(f"[team_db] wrote {write_snapshot(conn, args.game, args.out)}")
    elif args.cmd == "flags":
        for flag in game_flags(conn, args.game):
            print(f"[flag] {flag}")
    elif args.cmd == "people-flags":
        for flag in people_flags(conn):
            print(f"[flag] {flag}")
    elif args.cmd == "merge-people":
        merge_people(conn, args.keep, args.drop)
        print(f"[team_db] merged person {args.drop} into {args.keep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
