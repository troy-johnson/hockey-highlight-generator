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
- Penalty classes follow the game's rulebook (USA Hockey by default, or NHL). A penalty
  without minutes gets the minutes that rulebook puts in the records (game misconduct: 10).
- Every record keeps its source: sheet, override, user, or import.

Renders read a per-game snapshot (game_data.json in the Game Folder) written
by write_snapshot(), so a past render can be repeated after a Roster changes.

  team_db.py init
  team_db.py import-team <teams/x.json> --season 2026-2027
  team_db.py import-events <events.json> --game <id> --date YYYY-MM-DD --teams a,b [--focus a] [--folder DIR] [--rulebook usah|nhl]
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
import math
import os
import re
import sqlite3
import sys
import unicodedata
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from recap_options import config_dir

DB_FILE = "hockey.db"
SNAPSHOT_FILE = "game_data.json"
SNAPSHOT_VERSION = 3
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
    """
    ALTER TABLE games ADD COLUMN rulebook TEXT NOT NULL DEFAULT 'usah';
    ALTER TABLE events ADD COLUMN minutes_rule TEXT;
    """,
    """
    CREATE TABLE people_dismissals (
        person_a INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
        person_b INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
        source TEXT NOT NULL,
        CHECK (person_a < person_b),
        PRIMARY KEY (person_a, person_b)
    );
    ALTER TABLE games ADD COLUMN sheet_data TEXT NOT NULL DEFAULT '{}';
    ALTER TABLE events ADD COLUMN goal_type TEXT;
    """,
]

# Penalty classes and the minutes each puts in the records (not the time served on the clock).
# USA Hockey 2025-29 Official Rules, Rules 401-406 (Adult: minors 2:00, majors 5:00).
# NHL Official Rules 2026-2027, Section 4, Rules 16-24 (values unchanged from 2024-2025).
RULEBOOKS = {
    "usah": {
        "MINOR": (2, "USA Hockey 402"),
        "BENCH MINOR": (2, "USA Hockey 402(b)"),
        "DOUBLE MINOR": (4, "USA Hockey 402, two minors"),
        "MAJOR": (5, "USA Hockey 403(a)"),
        "MISCONDUCT": (10, "USA Hockey 404(a)"),
        "GAME MISCONDUCT": (10, "USA Hockey 404(b) Note 2"),
        "MATCH": (10, "USA Hockey 405(a) Note"),
        "PENALTY SHOT": (0, "USA Hockey 406"),
    },
    "nhl": {
        "MINOR": (2, "NHL 16.1"),
        "BENCH MINOR": (2, "NHL 17.1"),
        "DOUBLE MINOR": (4, "NHL 18.1"),
        "MAJOR": (5, "NHL 20.1"),
        "MISCONDUCT": (10, "NHL 22.1"),
        "GAME MISCONDUCT": (10, "NHL 23.1"),
        "MATCH": (15, "NHL 21.1"),
        "PENALTY SHOT": (0, "NHL 24.1"),
    },
}
CLASS_ALIASES = {"GM": "GAME MISCONDUCT", "GMC": "GAME MISCONDUCT", "MATCH PENALTY": "MATCH",
                 "10 MIN MISCONDUCT": "MISCONDUCT", "DOUBLE-MINOR": "DOUBLE MINOR", "BENCH": "BENCH MINOR"}


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

def season_for(date: str, start_month: int = 8) -> str:
    """Use the League's Season start month, or August by default."""
    if not isinstance(date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise ValueError(f"date must be YYYY-MM-DD, not {date!r}")
    d = dt.date.fromisoformat(date)
    if type(start_month) is not int or not 1 <= start_month <= 12:
        raise ValueError("season_start_month must be an integer from 1 to 12")
    start = d.year if d.month >= start_month else d.year - 1
    return f"{start}-{start + 1}"


def validate_season(season: str) -> str:
    if not isinstance(season, str) or not re.fullmatch(r"\d{4}-\d{4}", season):
        raise ValueError(f"season must look like 2026-2027, not {season!r}")
    first, last = map(int, season.split("-"))
    if last != first + 1:
        raise ValueError("season years must be consecutive")
    return season


@contextmanager
def transaction(conn):
    """Keep nested imports and edits in the caller's transaction."""
    if not conn.in_transaction:
        conn.execute("BEGIN")
        try:
            yield
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return
    conn.execute("SAVEPOINT team_db_edit")
    try:
        yield
        conn.execute("RELEASE team_db_edit")
    except Exception:
        conn.execute("ROLLBACK TO team_db_edit")
        conn.execute("RELEASE team_db_edit")
        raise


def read_connection(path: str | Path):
    """Read without creating, migrating, or changing a database."""
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > len(MIGRATIONS) or version < 3:
        conn.close()
        raise RuntimeError(f"{path} has schema version {version}; run team_db.py init with compatible code")
    return conn


def team_configs(conn, season: str | None = None) -> dict:
    """Read Team Configs without reducing shared numbers to one player."""
    teams = {}
    for t in conn.execute("SELECT * FROM teams ORDER BY id"):
        cfg = json.loads(t["extra"])
        cfg.update({"name": t["name"], "short_name": t["short_name"], "own_team": bool(t["own_team"]),
                    "colors": json.loads(t["colors"]), "kits": json.loads(t["kits"])})
        for key, value in json.loads(t["assets"]).items():
            cfg[key] = value["path"]
        cfg.pop("roster", None)
        cfg.pop("listed_roster", None)
        cfg["_source"] = "database"
        if season is not None:
            used = _roster_season(conn, t["id"], season)
            rows = conn.execute("SELECT p.*, r.jersey, r.jersey_label, r.position, r.role, r.source jersey_source "
                                "FROM roster_entries r JOIN people p ON p.id = r.person_id "
                                "WHERE r.team_id = ? AND r.season = ? ORDER BY r.id", (t["id"], used)).fetchall()
            cfg["listed_roster"] = [{**dict(r), "name": r["display_name"] or f"{r['first_name']} {r['last_name']}"}
                                    for r in rows]
            # The legacy map includes only numbers that identify one person.
            numbers = [r["jersey"] for r in rows]
            cfg["roster"] = {r["jersey_label"] or str(r["jersey"]): r["display_name"] or f"{r['first_name']} {r['last_name']}"
                             for r in rows if r["jersey"] is not None and numbers.count(r["jersey"]) == 1}
            cfg["_roster_season"] = used
        teams[t["id"]] = cfg
    return teams


def source_of(value: str | None, default: str = "user") -> str:
    """A source must start with sheet, override, user, or import; details may follow."""
    text = " ".join(str(value or default).split())
    if not re.match(rf"(?:{'|'.join(SOURCES)})\b", text.casefold()):
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
        if value < 0:
            raise ValueError(f"jersey {value!r} is not a number")
        return value, None
    text = str(value).strip().removeprefix("#")
    if not text.isdigit():
        raise ValueError(f"jersey {value!r} is not a number")
    number = int(text)
    return number, (text if text != str(number) else None)


def parse_minutes(value) -> float | None:
    """2, '2', '2+10' (12), '5:00' (5). None means not recorded."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError(f"penalty minutes {value!r} are not a number")
    if isinstance(value, (int, float)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"penalty minutes {value!r} must be a finite number of 0 or more")
        return float(value)
    text = str(value).strip()
    if re.fullmatch(r"\d+(\.\d+)?(\s*\+\s*\d+(\.\d+)?)*", text):
        return float(sum(float(p) for p in text.split("+")))
    if re.fullmatch(r"\d+:[0-5]\d", text):
        m, s = map(int, text.split(":"))
        return m + s / 60
    raise ValueError(f"penalty minutes {value!r} are not a number, a sum like 2+10, or m:ss")


def penalty_class(value) -> str | None:
    """Return the rulebook class name, such as 'GAME MISCONDUCT'. None means not recorded."""
    if value is None or not str(value).strip():
        return None
    text = " ".join(str(value).upper().replace("_", " ").split())
    text = CLASS_ALIASES.get(text, text)
    if text not in RULEBOOKS["usah"]:
        raise ValueError(f"penalty class {value!r} is not one of {', '.join(RULEBOOKS['usah'])}; "
                         "give each class as its own (additional) penalty")
    return text


def penalty_minutes(rulebook: str, classification, minutes) -> tuple[str | None, float | None, str | None]:
    """(class, record minutes, rule). Given minutes win; missing minutes come from the rulebook."""
    cls = penalty_class(classification)
    given = parse_minutes(minutes)
    if given is not None or cls is None:
        return cls, given, None
    record, rule = RULEBOOKS[rulebook][cls]
    return cls, float(record), rule


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


def _match_for_import(conn, team_id: str, first: str, last: str, jersey: int | None, used: set[int],
                      exact: bool = False) -> int | None:
    """Reuse a person already on this team (any Season) on a reimport; never merge two rows of one file.

    exact=True matches only a person who wore this number in some Season.
    """
    key = name_key(f"{first} {last}")
    hits = {}
    for r in _team_people(conn, team_id):
        if r["id"] not in used and key in _keys(r):
            hits.setdefault(r["id"], set()).add(r["jersey"])
    if exact:
        hits = {pid: nums for pid, nums in hits.items() if jersey is not None and jersey in nums}
    elif len(hits) > 1 and jersey is not None:
        same = {pid: nums for pid, nums in hits.items() if jersey in nums}
        if not same:
            raise ValueError(f"{team_id}: {first} {last} #{jersey} could be any of {len(hits)} people with that name; "
                             "keep their earlier numbers or merge by hand")
        hits = same
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
    return store_team(conn, path.stem, cfg, season, source or f"import {path.name}")


def store_team(conn, team_id: str, cfg: dict, season: str, source: str) -> str:
    """Store a Team Config and Roster from the JSON fallback or an explicit import."""
    source = source_of(source)
    validate_season(season)
    known = {"name", "short_name", "own_team", "colors", "kits", "logo", "lockup", "crest",
             "roster", "listed_roster", "family_links"}
    assets = {k: {"path": cfg[k], "source": source_of(cfg.get(f"{k}_source") or source)}
              for k in ("logo", "lockup", "crest") if cfg.get(k)}
    extra = {k: v for k, v in cfg.items() if k not in known}
    rows = cfg.get("listed_roster")
    if rows is None:  # older files have only the number -> name map
        rows = [{"jersey": num, "name": name} for num, name in (cfg.get("roster") or {}).items()]
    with transaction(conn):
        conn.execute(
            "INSERT INTO teams (id, name, short_name, own_team, colors, kits, assets, extra, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET name = excluded.name, "
            "short_name = excluded.short_name, own_team = excluded.own_team, colors = excluded.colors, "
            "kits = excluded.kits, assets = excluded.assets, extra = excluded.extra, source = excluded.source",
            (team_id, clean_name(cfg["name"]), cfg.get("short_name"), int(bool(cfg.get("own_team"))),
             json.dumps(cfg.get("colors") or {}), json.dumps(cfg.get("kits") or {}), json.dumps(assets),
             json.dumps(extra), source))
        parsed = []
        for i, row in enumerate(rows, start=1):
            if row.get("first_name") and row.get("last_name"):
                first, last = clean_name(row["first_name"]), clean_name(row["last_name"])
            elif row.get("name"):
                first, last = split_name(row["name"])
            else:
                raise ValueError(f"{team_id}: roster row {i} has no name")
            parsed.append((row, first, last, *parse_jersey(row.get("jersey"))))
        # Rows whose name and number both match a known person claim that person first,
        # so row order cannot give an old player's record to a new namesake.
        used: set[int] = set()
        matched = {}
        for i, (row, first, last, jersey, _) in enumerate(parsed):
            pid = _match_for_import(conn, team_id, first, last, jersey, used, exact=True)
            if pid is not None:
                matched[i] = pid
                used.add(pid)
        for i, (row, first, last, jersey, label) in enumerate(parsed):
            row_source = source_of(row.get("jersey_source") or row.get("source") or source)
            pid = matched.get(i)
            if pid is None:
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
            same = [r for r in hits if r["jersey"] == number]
            if not same:
                listed = ", ".join(f"#{r['jersey']}" for r in hits)
                raise ValueError(f"{team_id}: {clean_name(name)!r} is {len(hits)} people ({listed}); none wears #{number}")
            hits = same
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
    for pid in (keep, drop):
        if not conn.execute("SELECT 1 FROM people WHERE id = ?", (pid,)).fetchone():
            raise ValueError(f"no person with id {pid}")
    if conn.execute("SELECT 1 FROM family_links WHERE (parent_id = ? AND child_id = ?) OR (parent_id = ? AND child_id = ?)",
                    (keep, drop, drop, keep)).fetchone():
        raise ValueError(f"people {keep} and {drop} are linked as family; not one person")
    with transaction(conn):
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
        for row in conn.execute("SELECT * FROM people_dismissals WHERE person_a = ? OR person_b = ?",
                                (drop, drop)).fetchall():
            pair = sorted(keep if p == drop else p for p in (row["person_a"], row["person_b"]))
            if pair[0] != pair[1]:
                conn.execute("INSERT OR IGNORE INTO people_dismissals VALUES (?, ?, ?)", (*pair, row["source"]))
        conn.execute("DELETE FROM people_dismissals WHERE person_a = ? OR person_b = ?", (drop, drop))
        conn.execute("DELETE FROM people WHERE id = ?", (drop,))


def people_flags(conn) -> list[str]:
    """Different person records with the same name: possibly one person on two teams."""
    groups: dict[str, list] = {}
    for r in conn.execute("SELECT p.id, p.first_name, p.last_name, p.display_name, "
                          "GROUP_CONCAT(DISTINCT r.team_id) teams FROM people p "
                          "LEFT JOIN roster_entries r ON r.person_id = p.id GROUP BY p.id"):
        groups.setdefault(name_key(f"{r['first_name']} {r['last_name']}"), []).append(r)
    flags = []
    dismissed = {tuple(r) for r in conn.execute("SELECT person_a, person_b FROM people_dismissals")}
    for rows in groups.values():
        unresolved = []
        unresolved_ids = set()
        for i, a in enumerate(rows):
            for b in rows[i + 1:]:
                pair = tuple(sorted((a["id"], b["id"])))
                if pair in dismissed or (set((a["teams"] or "").split(",")) & set((b["teams"] or "").split(",")) - {""}):
                    continue
                who = ", ".join(f"id {r['id']} ({r['teams'] or 'no roster'})" for r in (a, b))
                unresolved.append(who)
                unresolved_ids.update(pair)
        if unresolved:
            count = len(unresolved_ids)
            flags.append(f"{rows[0]['first_name']} {rows[0]['last_name']} has {count} records; possible duplicate pairs: "
                         + "; ".join(unresolved) + "; if one person, run merge-people; otherwise run dismiss-people")
    return flags


def _person_exists(conn, person: int):
    if not conn.execute("SELECT 1 FROM people WHERE id = ?", (person,)).fetchone():
        raise ValueError(f"no person with id {person}")


def add_roster_player(conn, team: str, season: str, name: str | None = None, jersey=None,
                      person: int | None = None) -> int:
    """Add a new person, or reuse a known person, on one Season Roster."""
    validate_season(season)
    number, label = parse_jersey(jersey)
    with transaction(conn):
        if not conn.execute("SELECT 1 FROM teams WHERE id = ?", (team,)).fetchone():
            raise ValueError(f"no team {team!r}")
        if person is None:
            if not name or not clean_name(name):
                raise ValueError("a new player needs a name")
            first, last = split_name(name)
            person = _new_person(conn, first, last, "user roster-add")
        else:
            _person_exists(conn, person)
            if name is not None:
                raise ValueError("give a person id or a new name, not both")
        conn.execute("INSERT INTO roster_entries (team_id, season, person_id, jersey, jersey_label, source) "
                     "VALUES (?, ?, ?, ?, ?, 'user roster-add')", (team, season, person, number, label))
    return person


def rename_player(conn, person: int, name: str):
    if not name or not clean_name(name):
        raise ValueError("a player needs a name")
    first, last = split_name(name)
    with transaction(conn):
        _person_exists(conn, person)
        conn.execute("UPDATE people SET first_name = ?, last_name = ?, source = 'user roster-rename' WHERE id = ?",
                     (first, last, person))


def renumber_player(conn, team: str, season: str, person: int, jersey):
    validate_season(season)
    number, label = parse_jersey(jersey)
    with transaction(conn):
        result = conn.execute("UPDATE roster_entries SET jersey = ?, jersey_label = ?, source = 'user roster-number' "
                              "WHERE team_id = ? AND season = ? AND person_id = ?", (number, label, team, season, person))
        if result.rowcount != 1:
            raise ValueError(f"person {person} is not on the {team} {season} roster")


def set_display_name(conn, person: int, name: str | None):
    with transaction(conn):
        _person_exists(conn, person)
        conn.execute("UPDATE people SET display_name = ?, source = 'user display-name' WHERE id = ?",
                     (clean_name(name) if name and clean_name(name) else None, person))


def set_game_jersey(conn, game: str, team: str, person: int, jersey):
    number, label = parse_jersey(jersey)
    with transaction(conn):
        season = _game(conn, game)["season"]
        if not conn.execute("SELECT 1 FROM game_teams WHERE game_id = ? AND team_id = ?", (game, team)).fetchone():
            raise ValueError(f"{team} is not a team in game {game}")
        used = _roster_season(conn, team, season)
        if not conn.execute("SELECT 1 FROM roster_entries WHERE team_id = ? AND season = ? AND person_id = ?",
                            (team, used, person)).fetchone():
            raise ValueError(f"person {person} is not on the {team} {used or season} roster")
        old = conn.execute("SELECT team_id FROM game_players WHERE game_id = ? AND person_id = ?", (game, person)).fetchone()
        if old and old[0] != team:
            raise ValueError(f"person {person} already plays for {old[0]} in game {game}")
        conn.execute("INSERT INTO game_players (game_id, team_id, person_id, jersey_worn, jersey_label, source) "
                     "VALUES (?, ?, ?, ?, ?, 'user game-number') ON CONFLICT(game_id, person_id) DO UPDATE SET "
                     "jersey_worn = excluded.jersey_worn, jersey_label = excluded.jersey_label, source = excluded.source",
                     (game, team, person, number, label))


def dismiss_people(conn, person_a: int, person_b: int):
    if person_a == person_b:
        raise ValueError("a dismissal needs two different people")
    with transaction(conn):
        for person in (person_a, person_b):
            _person_exists(conn, person)
        conn.execute("INSERT OR REPLACE INTO people_dismissals VALUES (?, ?, 'user dismiss-people')",
                     tuple(sorted((person_a, person_b))))


# --- games ------------------------------------------------------------------------------

def add_game(conn, game_id: str, date: str, team_ids: list[str], focus: str | None = None,
             folder: str | None = None, source: str = "user", rulebook: str = "usah",
             season: str | None = None) -> str:
    """rulebook gives the penalty minutes in the records: 'usah' (USA Hockey) or 'nhl'."""
    default_season = season_for(date)
    season = validate_season(season) if season is not None else default_season
    source = source_of(source)
    if rulebook not in RULEBOOKS:
        raise ValueError(f"rulebook {rulebook!r} is not one of {', '.join(RULEBOOKS)}")
    with transaction(conn):
        conn.execute("INSERT INTO games (id, date, season, focus_team_id, folder, source, rulebook) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?) "
                     "ON CONFLICT(id) DO UPDATE SET date = excluded.date, season = excluded.season, "
                     "focus_team_id = excluded.focus_team_id, folder = excluded.folder, rulebook = excluded.rulebook, "
                     "source = excluded.source",
                     (game_id, date, season, focus, folder, source, rulebook))
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
    if not (name and clean_name(name)) and number is None:
        raise ValueError(f"{team_id}: a player needs a name or a jersey")
    pid = find_person(conn, team_id, season, name=name, jersey=number)
    old = conn.execute("SELECT jersey_worn, team_id FROM game_players WHERE game_id = ? AND person_id = ?",
                        (game_id, pid)).fetchone()
    if old and old["team_id"] != team_id:
        raise ValueError(f"person {pid} already plays for {old['team_id']} in game {game_id}")
    if old and old[0] is not None and number is not None and old[0] != number:
        raise ValueError(f"{team_id}: {clean_name(name or '#' + str(number))} wore #{old[0]} and #{number} "
                         "in this game; fix one event")
    conn.execute("INSERT INTO game_players (game_id, team_id, person_id, jersey_worn, jersey_label, note, source) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(game_id, person_id) DO UPDATE SET "
                 "jersey_worn = COALESCE(excluded.jersey_worn, game_players.jersey_worn), "
                 "jersey_label = COALESCE(excluded.jersey_label, game_players.jersey_label), "
                  "note = COALESCE(excluded.note, game_players.note), source = excluded.source",
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
    with transaction(conn):
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
            if kind == "penalty":
                cls, minutes, rule = penalty_minutes(game["rulebook"], e.get("classification"), e.get("minutes"))
            else:
                cls, minutes, rule = None, None, None
            assists = e.get("assists") or []
            if kind == "goal":
                assist_ids = _assist_ids(conn, game_id, team_id, season, in_game, assists, pid, src, n)
            elif assists:
                raise ValueError(f"event {n}: a penalty has no assists")
            if e.get("additional") and kind != "penalty":
                raise ValueError(f"event {n}: only a penalty can have additional penalties")
            eid = conn.execute(
                "INSERT INTO events (game_id, team_id, kind, ordinal, person_id, period, clock, scene, infraction, "
                 "classification, minutes, minutes_rule, unassisted, match_note, note, source, goal_type) "
                 "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (game_id, team_id, kind, order[kind], pid, e.get("period"), e.get("game_clock"), e.get("scene"),
                 e.get("infraction"), cls, minutes, rule,
                  int(not assists) if kind == "goal" else None, e.get("roster_match"), e.get("note"), src, e.get("type"))).lastrowid
            if kind == "goal":
                for i, apid in enumerate(assist_ids, start=1):
                    conn.execute("INSERT INTO event_assists (event_id, ordinal, person_id, source) VALUES (?, ?, ?, ?)",
                                 (eid, i, apid, src))
            for extra in e.get("additional") or []:
                if not extra.get("classification"):
                    raise ValueError(f"event {n}: an additional penalty needs a classification")
                cls, minutes, rule = penalty_minutes(game["rulebook"], extra["classification"], extra.get("minutes"))
                order["penalty"] += 1
                conn.execute(
                    "INSERT INTO events (game_id, team_id, kind, ordinal, parent_event_id, person_id, period, scene, "
                    "classification, minutes, minutes_rule, note, source) "
                    "VALUES (?, ?, 'penalty', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (game_id, team_id, order["penalty"], eid, pid, e.get("period"), e.get("scene"),
                     cls, minutes, rule, extra.get("note"), src))
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
    book = RULEBOOKS[game["rulebook"]]
    for r in conn.execute("SELECT team_id, classification, minutes FROM events WHERE game_id = ? AND kind = 'penalty' "
                          "AND minutes IS NOT NULL AND classification IS NOT NULL", (game_id,)):
        record, rule = book[r["classification"]]
        if r["minutes"] != record:
            flags.append(f"{r['team_id']}: {r['classification']} has {_whole(r['minutes'])} min; "
                         f"{rule} records {record}")
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
                                 "infraction", "classification", "minutes", "minutes_rule", "note", "source")}
        out["team"] = e["team_id"]
        out["type"] = e["goal_type"] if "goal_type" in e.keys() else None
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
    sheet = json.loads(game["sheet_data"]) if "sheet_data" in game.keys() else {}
    for rows in sheet.get("goalkeeping", {}).values():
        for row in rows:
            row["person"] = people[row.pop("person_id")]
    return {"schema_version": SNAPSHOT_VERSION, "game": {"id": game_id, "date": game["date"], "season": season,
            "rulebook": game["rulebook"],
            "focus_team": game["focus_team_id"], "folder": game["folder"]}, "teams": teams, "score": goals,
            "events": events, "stats": tables, "team_pim": {t: _whole(v) for t, v in team_pim.items()},
            "flags": game_flags(conn, game_id), "sheet": sheet}


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
    e.add_argument("--rulebook", default="usah", choices=sorted(RULEBOOKS))
    s = sub.add_parser("snapshot")
    s.add_argument("game")
    s.add_argument("--out")
    f = sub.add_parser("flags")
    f.add_argument("game")
    sub.add_parser("people-flags")
    m = sub.add_parser("merge-people")
    m.add_argument("keep", type=int)
    m.add_argument("drop", type=int)
    a = sub.add_parser("roster-add")
    a.add_argument("team")
    a.add_argument("--season", required=True)
    identity = a.add_mutually_exclusive_group(required=True)
    identity.add_argument("--name")
    identity.add_argument("--person", type=int)
    a.add_argument("--jersey")
    r = sub.add_parser("roster-rename")
    r.add_argument("person", type=int)
    r.add_argument("name")
    n = sub.add_parser("roster-number")
    n.add_argument("team")
    n.add_argument("person", type=int)
    n.add_argument("jersey", help="number, or '-' for no number")
    n.add_argument("--season", required=True)
    j = sub.add_parser("game-number")
    j.add_argument("game")
    j.add_argument("team")
    j.add_argument("person", type=int)
    j.add_argument("jersey", help="number, or '-' to use the Roster number")
    d = sub.add_parser("display-name")
    d.add_argument("person", type=int)
    d.add_argument("name", help="display name, or empty text to clear it")
    p = sub.add_parser("dismiss-people")
    p.add_argument("person_a", type=int)
    p.add_argument("person_b", type=int)
    args = ap.parse_args(argv)
    conn = connect(args.db or db_path(args.config_dir))
    if args.cmd == "init":
        print(f"[team_db] schema version {conn.execute('PRAGMA user_version').fetchone()[0]}")
    elif args.cmd == "import-team":
        print(f"[team_db] imported {import_team(conn, args.path, args.season)} for {args.season}")
    elif args.cmd == "import-events":
        ids = args.teams.split(",")
        add_game(conn, args.game, args.date, ids, args.focus, args.folder, rulebook=args.rulebook)
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
    elif args.cmd == "roster-add":
        person = add_roster_player(conn, args.team, args.season, args.name, args.jersey, args.person)
        print(f"[team_db] added person {person} to {args.team} {args.season}")
    elif args.cmd == "roster-rename":
        rename_player(conn, args.person, args.name)
    elif args.cmd == "roster-number":
        renumber_player(conn, args.team, args.season, args.person, None if args.jersey == "-" else args.jersey)
    elif args.cmd == "game-number":
        set_game_jersey(conn, args.game, args.team, args.person, None if args.jersey == "-" else args.jersey)
    elif args.cmd == "display-name":
        set_display_name(conn, args.person, args.name)
    elif args.cmd == "dismiss-people":
        dismiss_people(conn, args.person_a, args.person_b)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
