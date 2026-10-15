# v3/scripts/team_db.py
"""
Team, Roster, and game data in one SQLite database (ADR-0004).

The database is <config>/hockey.db (see recap_options.config_dir). It is outside
the repository because it holds the user's teams and Rosters. The schema is
versioned with PRAGMA user_version; connect() applies missing migrations.

Renders read a per-game snapshot (game_data.json in the Game Folder) written
by write_snapshot(), so a past render can be repeated after a Roster changes.

  team_db.py init
  team_db.py import-team <teams/x.json> --season 2026-2027
  team_db.py import-events <events.json> --game <id> --date YYYY-MM-DD --teams a,b [--focus a] [--folder DIR]
  team_db.py snapshot <game id> [--out FILE]
  team_db.py flags <game id>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from recap_options import config_dir

DB_FILE = "hockey.db"
SNAPSHOT_FILE = "game_data.json"
SNAPSHOT_VERSION = 1

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
]


def db_path(override: str | None = None) -> Path:
    return config_dir(override) / DB_FILE


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Open the database and apply missing migrations."""
    path = Path(path) if path else db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for number, sql in enumerate(MIGRATIONS[version:], start=version + 1):
        with conn:
            conn.executescript(sql)
            conn.execute(f"PRAGMA user_version = {number}")
    return conn


def season_for(date: str) -> str:
    """Adult league seasons start in August: 2026-10-11 belongs to 2026-2027."""
    d = dt.date.fromisoformat(date)
    start = d.year if d.month >= 8 else d.year - 1
    return f"{start}-{start + 1}"


def _person(conn, team_id: str, first: str, last: str, source: str, display: str | None = None) -> int:
    """Find a person already on this team's rosters by name, or add one."""
    row = conn.execute(
        "SELECT p.id FROM people p JOIN roster_entries r ON r.person_id = p.id "
        "WHERE r.team_id = ? AND p.first_name = ? AND p.last_name = ? LIMIT 1", (team_id, first, last)).fetchone()
    if row:
        if display:
            conn.execute("UPDATE people SET display_name = ? WHERE id = ?", (display, row[0]))
        return row[0]
    return conn.execute("INSERT INTO people (first_name, last_name, display_name, source) VALUES (?, ?, ?, ?)",
                        (first, last, display, source)).lastrowid


def import_team(conn, path: str | Path, season: str, source: str | None = None) -> str:
    """Import one Team Config JSON file and its listed Roster for a season."""
    path = Path(path)
    cfg = json.loads(path.read_text())
    team_id = path.stem
    source = source or f"import {path.name}"
    known = {"name", "short_name", "own_team", "colors", "kits", "logo", "lockup", "crest",
             "roster", "listed_roster", "family_links"}
    assets = {k: cfg[k] for k in ("logo", "lockup", "crest") if cfg.get(k)}
    extra = {k: v for k, v in cfg.items() if k not in known}
    with conn:
        conn.execute(
            "INSERT INTO teams (id, name, short_name, own_team, colors, kits, assets, extra, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET name = excluded.name, "
            "short_name = excluded.short_name, own_team = excluded.own_team, colors = excluded.colors, "
            "kits = excluded.kits, assets = excluded.assets, extra = excluded.extra, source = excluded.source",
            (team_id, cfg["name"], cfg.get("short_name"), int(bool(cfg.get("own_team"))),
             json.dumps(cfg.get("colors") or {}), json.dumps(cfg.get("kits") or {}), json.dumps(assets),
             json.dumps(extra), source))
        rows = cfg.get("listed_roster")
        if rows is None:  # older files have only the number -> name map
            rows = []
            for num, name in (cfg.get("roster") or {}).items():
                first, _, last = str(name).partition(" ")
                rows.append({"jersey": int(num), "first_name": first, "last_name": last or first})
        for row in rows:
            pid = _person(conn, team_id, row["first_name"], row["last_name"], row.get("source") or source,
                          row.get("display_name"))
            conn.execute(
                "INSERT INTO roster_entries (team_id, season, person_id, jersey, position, role, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(team_id, season, person_id) DO UPDATE SET "
                "jersey = excluded.jersey, position = excluded.position, source = excluded.source",
                (team_id, season, pid, row.get("jersey"), row.get("position"),
                 "goalie" if row.get("position") == "G" else "player",
                 row.get("jersey_source") or row.get("source") or source))
        for link in cfg.get("family_links") or []:
            ids = []
            for key in ("parent", "child"):
                first, _, last = link[key]["name"].partition(" ")
                hit = find_person(conn, team_id, season, name=link[key]["name"], jersey=link[key].get("jersey"))
                ids.append(hit if hit else _person(conn, team_id, first, last, source))
            conn.execute("INSERT OR REPLACE INTO family_links VALUES (?, ?, ?, ?)",
                         (ids[0], ids[1], link["relation"], link.get("source") or source))
    return team_id


def find_person(conn, team_id: str, season: str, name: str | None = None, jersey: int | None = None) -> int | None:
    """Match a name (display or first+last), and the jersey when the name is missing or ambiguous."""
    rows = conn.execute(
        "SELECT p.id, p.first_name, p.last_name, p.display_name, r.jersey FROM roster_entries r "
        "JOIN people p ON p.id = r.person_id WHERE r.team_id = ? AND r.season = ?", (team_id, season)).fetchall()
    if name:
        key = name.casefold()
        hits = [r for r in rows if key in {f"{r['first_name']} {r['last_name']}".casefold(),
                                            (r["display_name"] or "").casefold()}]
        if len(hits) > 1 and jersey is not None:
            hits = [r for r in hits if r["jersey"] == jersey]
        if len(hits) == 1:
            return hits[0]["id"]
        if hits:
            raise ValueError(f"{team_id}: name {name!r} matches {len(hits)} people")
        return None
    if jersey is not None:
        hits = [r for r in rows if r["jersey"] == jersey]
        if len(hits) == 1:
            return hits[0]["id"]
        if hits:
            raise ValueError(f"{team_id}: jersey #{jersey} is shared by {len(hits)} people; give a name")
    return None


def add_game(conn, game_id: str, date: str, team_ids: list[str], focus: str | None = None,
             folder: str | None = None, sides: dict | None = None, source: str = "user") -> str:
    with conn:
        conn.execute("INSERT INTO games (id, date, season, focus_team_id, folder, source) VALUES (?, ?, ?, ?, ?, ?) "
                     "ON CONFLICT(id) DO UPDATE SET date = excluded.date, season = excluded.season, "
                     "focus_team_id = excluded.focus_team_id, folder = excluded.folder",
                     (game_id, date, season_for(date), focus, folder, source))
        for tid in team_ids:
            conn.execute("INSERT OR REPLACE INTO game_teams VALUES (?, ?, ?)", (game_id, tid, (sides or {}).get(tid)))
    return game_id


def _game(conn, game_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
    if row is None:
        raise KeyError(f"no game {game_id!r}")
    return row


def _game_player(conn, game_id, team_id, season, name, jersey, source, note=None) -> int:
    pid = find_person(conn, team_id, season, name=name, jersey=jersey)
    if pid is None:
        if not name:
            raise ValueError(f"{team_id}: #{jersey} is not on the {season} roster; give a name")
        first, _, last = name.partition(" ")
        pid = conn.execute("INSERT INTO people (first_name, last_name, source) VALUES (?, ?, ?)",
                           (first, last or first, source)).lastrowid
    conn.execute("INSERT INTO game_players (game_id, team_id, person_id, jersey_worn, note, source) "
                 "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(game_id, person_id) DO UPDATE SET "
                 "jersey_worn = COALESCE(excluded.jersey_worn, game_players.jersey_worn), "
                 "note = COALESCE(excluded.note, game_players.note)",
                 (game_id, team_id, pid, jersey, note, source))
    return pid


def import_events(conn, game_id: str, events: list[dict], team_names: dict[str, str],
                  source: str = "user") -> int:
    """Replace a game's events. team_names maps an event 'team' name to a team id."""
    game = _game(conn, game_id)
    season = game["season"]
    with conn:
        conn.execute("DELETE FROM event_assists WHERE event_id IN (SELECT id FROM events WHERE game_id = ?)", (game_id,))
        conn.execute("DELETE FROM events WHERE game_id = ?", (game_id,))
        conn.execute("DELETE FROM game_players WHERE game_id = ?", (game_id,))
        order = {"goal": 0, "penalty": 0}
        for e in events:
            kind = e["event"].lower()
            team_id = team_names[e["team"]]
            src = e.get("source") or source
            order[kind] += 1
            name = e.get("scorer") if kind == "goal" else e.get("player")
            pid = _game_player(conn, game_id, team_id, season, name, e.get("jersey"), src, e.get("jersey_note"))
            eid = conn.execute(
                "INSERT INTO events (game_id, team_id, kind, ordinal, person_id, period, clock, scene, infraction, "
                "classification, minutes, unassisted, note, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (game_id, team_id, kind, order[kind], pid, e.get("period"), e.get("game_clock"), e.get("scene"),
                 e.get("infraction"), e.get("classification"), e.get("minutes"),
                 None if kind != "goal" else int(not e.get("assists")), e.get("roster_match"), src)).lastrowid
            for i, a in enumerate(e.get("assists") or [], start=1):
                apid = _game_player(conn, game_id, team_id, season, a["name"], a.get("jersey"), src)
                conn.execute("INSERT INTO event_assists VALUES (?, ?, ?)", (eid, i, apid))
            for extra in e.get("additional") or []:
                order["penalty"] += 1
                conn.execute(
                    "INSERT INTO events (game_id, team_id, kind, ordinal, parent_event_id, person_id, period, scene, "
                    "classification, minutes, note, source) VALUES (?, ?, 'penalty', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (game_id, team_id, order["penalty"], eid, pid, e.get("period"), e.get("scene"),
                     extra["classification"], extra.get("minutes"), extra.get("note"), src))
    return sum(order.values())


def game_flags(conn, game_id: str) -> list[str]:
    """Review flags: numbers worn that differ from the season Roster, shared or missing numbers."""
    game = _game(conn, game_id)
    flags = []
    for r in conn.execute(
            "SELECT gp.team_id, gp.jersey_worn, p.first_name, p.last_name, p.display_name, re.jersey AS listed "
            "FROM game_players gp JOIN people p ON p.id = gp.person_id LEFT JOIN roster_entries re "
            "ON re.person_id = gp.person_id AND re.team_id = gp.team_id AND re.season = ? WHERE gp.game_id = ?",
            (game["season"], game_id)):
        who = r["display_name"] or f"{r['first_name']} {r['last_name']}"
        if r["listed"] is None and r["jersey_worn"] is None:
            flags.append(f"{r['team_id']}: {who} has no number in this game or on the {game['season']} roster")
        elif r["listed"] is not None and r["jersey_worn"] is not None and r["listed"] != r["jersey_worn"]:
            flags.append(f"{r['team_id']}: {who} wore #{r['jersey_worn']}; {game['season']} roster lists #{r['listed']}")
    for r in conn.execute(
            "SELECT team_id, jersey, COUNT(*) n FROM roster_entries WHERE season = ? AND jersey IS NOT NULL AND "
            "team_id IN (SELECT team_id FROM game_teams WHERE game_id = ?) GROUP BY team_id, jersey HAVING n > 1",
            (game["season"], game_id)):
        flags.append(f"{r['team_id']}: #{r['jersey']} is shared by {r['n']} players on the {game['season']} roster")
    for r in conn.execute("SELECT team_id, classification FROM events WHERE game_id = ? AND kind = 'penalty' "
                          "AND minutes IS NULL", (game_id,)):
        flags.append(f"{r['team_id']}: {r['classification'] or 'penalty'} has no minutes; PIM excludes it")
    return flags


def snapshot(conn, game_id: str) -> dict:
    """The data one render reads: teams, players with numbers worn, events, and derived statistics."""
    game = _game(conn, game_id)
    numbers = {r["person_id"]: r["jersey_worn"] for r in
               conn.execute("SELECT person_id, jersey_worn FROM game_players WHERE game_id = ?", (game_id,))}
    names = {}
    for r in conn.execute("SELECT p.id, p.first_name, p.last_name, p.display_name FROM people p "
                          "JOIN game_players gp ON gp.person_id = p.id WHERE gp.game_id = ?", (game_id,)):
        names[r["id"]] = r["display_name"] or f"{r['first_name']} {r['last_name']}"
    teams = {}
    for t in conn.execute("SELECT t.*, gt.side FROM teams t JOIN game_teams gt ON gt.team_id = t.id "
                          "WHERE gt.game_id = ? ORDER BY t.id", (game_id,)):
        teams[t["id"]] = {"name": t["name"], "short_name": t["short_name"], "own_team": bool(t["own_team"]),
                          "side": t["side"], "colors": json.loads(t["colors"]), "kits": json.loads(t["kits"]),
                          "assets": json.loads(t["assets"]), "extra": json.loads(t["extra"])}
    player = lambda pid: {"id": pid, "name": names[pid], "jersey": numbers.get(pid)}  # noqa: E731
    events, stats = [], {tid: {} for tid in teams}

    def line(tid, pid):
        return stats[tid].setdefault(pid, {**player(pid), "g": 0, "a": 0, "pts": 0, "pim": 0})

    for e in conn.execute("SELECT * FROM events WHERE game_id = ? ORDER BY id", (game_id,)):
        out = {k: e[k] for k in ("id", "kind", "ordinal", "parent_event_id", "period", "clock", "scene",
                                 "infraction", "classification", "minutes", "note", "source")}
        out["team"] = e["team_id"]
        out["person"] = player(e["person_id"]) if e["person_id"] else None
        if e["kind"] == "goal":
            out["assists"] = [player(a["person_id"]) for a in conn.execute(
                "SELECT person_id FROM event_assists WHERE event_id = ? ORDER BY ordinal", (e["id"],))]
            out["unassisted"] = not out["assists"]
            line(e["team_id"], e["person_id"])["g"] += 1
            for a in out["assists"]:
                line(e["team_id"], a["id"])["a"] += 1
        elif e["person_id"]:
            row = line(e["team_id"], e["person_id"])
            if e["minutes"] is not None:
                row["pim"] += e["minutes"]
        events.append(out)
    tables = {}
    for tid, rows in stats.items():
        for r in rows.values():
            r["pts"] = r["g"] + r["a"]
            r["pim"] = int(r["pim"]) if float(r["pim"]).is_integer() else r["pim"]
        tables[tid] = sorted(rows.values(), key=lambda r: (-r["pts"], -r["g"], -r["pim"], r["name"]))
    goals = {tid: sum(e["kind"] == "goal" and e["team"] == tid for e in events) for tid in teams}
    return {"schema_version": SNAPSHOT_VERSION, "game": {"id": game_id, "date": game["date"], "season": game["season"],
            "focus_team": game["focus_team_id"], "folder": game["folder"]}, "teams": teams, "score": goals,
            "events": events, "stats": tables, "flags": game_flags(conn, game_id)}


def write_snapshot(conn, game_id: str, out: str | Path | None = None) -> Path:
    data = snapshot(conn, game_id)
    out = Path(out) if out else Path(data["game"]["folder"]) / SNAPSHOT_FILE
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    args = ap.parse_args(argv)
    conn = connect(args.db)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
