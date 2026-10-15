"""Connect the applied Game Sheet to the database and frozen graphics data."""
from __future__ import annotations

import copy
import datetime as dt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import team_db as db

SIDES = ("home", "away")


def sheet_source(row: dict) -> str:
    value = row.get("source") or ""
    return "override game_sheet_overrides.json" if str(value).startswith("override") else "sheet game_sheet.json"


def _sheet_player(sheet: dict, side: str, value) -> dict:
    if isinstance(value, dict):
        return {"name": value.get("name"), "jersey": value.get("jersey")}
    number, _ = db.parse_jersey(value)
    matches = [v for k, v in (sheet.get("rosters") or {}).get(side, {}).items()
               if db.parse_jersey(k)[0] == number]
    names = {v.get("name") if isinstance(v, dict) else v for v in matches}
    if len(names) > 1:
        raise ValueError(f"{side}: sheet jersey #{number} has different names; fix the Game Sheet")
    return {"name": next(iter(names), None), "jersey": value}


def _player_source(sheet: dict, side: str, value) -> str:
    if isinstance(value, dict):
        return sheet_source(value)
    number, _ = db.parse_jersey(value)
    for key, source in (sheet.get("roster_sources") or {}).get(side, {}).items():
        if db.parse_jersey(key)[0] == number and source.startswith("override"):
            return "override game_sheet_overrides.json"
    return "sheet game_sheet.json"


def _team_ids(conn, sheet: dict, options: dict, season: str) -> dict:
    configs = db.team_configs(conn)
    fallback = options.get("teams") or {}
    ids = {}
    for side in SIDES:
        name = sheet["teams"][side]
        choices = {tid for tid, cfg in {**fallback, **configs}.items()
                   if db.name_key(name) in {db.name_key(tid), db.name_key(cfg["name"]),
                                           db.name_key(cfg.get("short_name") or "")}}
        if len(choices) > 1:
            raise ValueError(f"{side}: team name {name!r} matches multiple teams; use a team id")
        tid = next(iter(choices), name)
        if tid not in configs:
            cfg = fallback.get(tid) or {"name": name}
            db.store_team(conn, tid, cfg, season,
                          f"import teams/{tid}.json" if tid in fallback else "sheet game_sheet.json")
        ids[side] = tid
    if ids["home"] == ids["away"]:
        raise ValueError("home and away must be different teams")
    return ids


def import_sheet(conn, game_id: str, sheet: dict, options: dict, folder: str | Path) -> dict:
    """Replace one game's applied sheet in one transaction. Unknown people stop the import."""
    date = sheet.get("date") or options.get("date")
    if date:
        for pattern in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%m/%d/%Y"):
            try:
                date = dt.datetime.strptime(date, pattern).date().isoformat()
                break
            except ValueError:
                continue
    rules = options.get("league_rules") or {}
    season, rulebook = db.league_game_rules(date, rules)
    with db.transaction(conn):
        ids = _team_ids(conn, sheet, options, season)
        focus = options.get("focus_team")
        if focus and focus not in ids.values():
            matches = [tid for tid in ids.values() if db.name_key(focus) == db.name_key(
                conn.execute("SELECT name FROM teams WHERE id = ?", (tid,)).fetchone()[0])]
            if len(matches) != 1:
                raise ValueError(f"Focus Team {focus!r} is not a team in the Game Sheet")
            focus = matches[0]
        db.add_game(conn, game_id, date, list(ids.values()), focus, str(folder),
                    source="sheet game_sheet.json", rulebook=rulebook, season=season)
        db.set_game_sides(conn, game_id, ids, "sheet game_sheet.json")
        events = []
        for key, kind in (("goals", "GOAL"), ("penalties", "PENALTY")):
            for side in SIDES:
                for row in (sheet.get(key) or {}).get(side, []):
                    field = "scorer" if kind == "GOAL" else "player"
                    player = _sheet_player(sheet, side, row.get(field))
                    source = sheet_source(row)
                    values = [row.get(field)] + (row.get("assists") or [])
                    if any(_player_source(sheet, side, value).startswith("override") for value in values):
                        source = "override game_sheet_overrides.json"
                    per = row.get("per")
                    period = int(per) if str(per).isdigit() else {"OT": 4, "SO": 5}.get(str(per).upper())
                    event = {"event": kind, "team": ids[side], field: player["name"], "jersey": player["jersey"],
                             "period": period, "game_clock": row.get("time") or row.get("off"),
                             "source": source, "type": row.get("type"), "scene": row.get("scene"),
                             "note": row.get("note"), "infraction": row.get("infraction"),
                             "classification": row.get("classification"), "minutes": row.get("minutes"),
                             "additional": row.get("additional")}
                    if kind == "GOAL":
                        event["assists"] = [_sheet_player(sheet, side, v) for v in row.get("assists") or []]
                    else:
                        event["classification"] = db.sheet_penalty_class(row.get("classification"), row.get("infraction"),
                            row.get("minutes"), not player["name"] and player["jersey"] in (None, ""))
                    events.append(event)
        db.import_events(conn, game_id, events, {tid: tid for tid in ids.values()}, "sheet game_sheet.json",
                         preserve_sheet_players=False)
        details = {key: copy.deepcopy(sheet[key]) for key in ("shots_on_goal", "score_by_periods", "final_score", "flags")
                   if key in sheet}
        details["goalkeeping"] = {}
        for side in SIDES:
            rows = []
            for row in (sheet.get("goalkeeping") or {}).get(side, []):
                player = _sheet_player(sheet, side, row["player"])
                # A goalie name on the applied sheet is also a claim, not a new person.
                name = row.get("name") or player["name"]
                source = sheet_source(row)
                if _player_source(sheet, side, row["player"]).startswith("override"):
                    source = "override game_sheet_overrides.json"
                pid = db.record_game_player(conn, game_id, ids[side], season, name, player["jersey"], source)
                rows.append({**{k: v for k, v in row.items() if k not in ("player", "name")},
                             "person_id": pid, "source": source})
            details["goalkeeping"][ids[side]] = rows
        db.set_sheet_data(conn, game_id, details)
        data = db.snapshot(conn, game_id)
    return data


def write_data(data: dict, path: Path):
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def import_snapshot(conn, game_id: str, sheet: dict, options: dict, folder: Path) -> dict:
    """Keep the previous database game when snapshot publication fails."""
    if conn.in_transaction:
        raise ValueError("snapshot publication must own its database transaction")
    path = folder / db.SNAPSHOT_FILE
    previous = path.read_bytes() if path.exists() else None
    published = False
    try:
        with db.transaction(conn):
            data = import_sheet(conn, game_id, sheet, options, folder)
            write_data(data, path)
            published = True
        return data
    except Exception:
        if published:  # A database commit failed after the file replacement.
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                temp = path.with_name(path.name + ".tmp")
                temp.write_bytes(previous)
                os.replace(temp, path)
        raise


def jersey_label(person: dict) -> str:
    return person.get("jersey_label") or (str(person["jersey"]) if person["jersey"] is not None else "?")


def graphics_inputs(data: dict, options: dict) -> tuple[dict, dict]:
    """Adapt the frozen snapshot to the current graphics interface, without a database read."""
    if data.get("schema_version") != db.SNAPSHOT_VERSION:
        raise ValueError("unsupported game_data.json schema; rerun scoresheet")
    ids = {team.get("side"): tid for tid, team in data["teams"].items()}
    if any(side not in ids for side in SIDES):
        raise ValueError("game_data.json needs home and away sides; rerun scoresheet")
    sheet = {"date": data["game"]["date"], "teams": {}, "rosters": {}, "goals": {}, "penalties": {},
             "goalkeeping": {}, "_numbers": {}, "flags": list(data.get("flags") or [])
             + list(data.get("sheet", {}).get("flags") or [])}
    final = {"score": [data["score"][ids[s]] for s in SIDES], "table": {}}
    configs = {}
    for side in SIDES:
        tid, team = ids[side], data["teams"][ids[side]]
        cfg = copy.deepcopy(team.get("extra") or {})
        cfg.update({k: copy.deepcopy(team[k]) for k in ("name", "short_name", "own_team", "colors", "kits")})
        cfg.update({k: v["path"] for k, v in (team.get("assets") or {}).items()})
        configs[tid] = cfg
        sheet["teams"][side] = team["name"]
        sheet["rosters"][side], sheet["_numbers"][side] = {}, {}

        def player(person):
            if person is None:
                return "?"
            key = str(person["id"])
            sheet["rosters"][side][key] = person["name"]
            sheet["_numbers"][side][key] = jersey_label(person)
            return key

        for kind, target in (("goal", "goals"), ("penalty", "penalties")):
            rows = []
            for event in data["events"]:
                if event["team"] != tid or event["kind"] != kind or event.get("parent_event_id") is not None:
                    continue
                rows.append({"scorer" if kind == "goal" else "player": player(event["person"]),
                             "assists": [player(p) for p in event.get("assists", [])],
                             "per": str(event["period"] or ""), "time": event.get("clock"),
                             "type": event.get("type"), "minutes": event.get("minutes"),
                             "infraction": event.get("infraction"), "team_only": event["person"] is None})
            sheet[target][side] = rows
        sheet["goalkeeping"][side] = [{**r, "player": player(r["person"]), "name": r["person"]["name"]}
                                         for r in (data.get("sheet", {}).get("goalkeeping") or {}).get(tid, [])]
        final["table"][side] = {"skaters": [{"num": jersey_label(r),
            **{k: r[k] for k in ("name", "g", "a", "pts", "pim")}} for r in data["stats"][tid]], "goalies": []}
    sheet["shots_on_goal"] = copy.deepcopy(data.get("sheet", {}).get("shots_on_goal") or {})
    # Goalie figures use only fields frozen in the snapshot.
    from recap_graphics import final_data
    goalie_tables = final_data(sheet, sheet["flags"])["table"]
    for side in SIDES:
        final["table"][side]["goalies"] = goalie_tables[side]["goalies"]
        for row in final["table"][side]["goalies"]:
            row["num"] = sheet["_numbers"][side].get(row["num"], row["num"])
    sheet["_final"] = final
    frozen_options = copy.deepcopy(options)
    frozen_options.update(teams=configs, date=data["game"]["date"], focus_team=data["game"]["focus_team"])
    return sheet, frozen_options


def load_graphics_inputs(root: Path, options: dict) -> tuple[dict, dict]:
    return graphics_inputs(json.loads((root / db.SNAPSHOT_FILE).read_text()), options)


def validate_selection(sheet: dict, selection: dict):
    """Check frozen goal identities and sheet clocks before using Selection timing."""
    from scoresheet import parse_time
    from selection import elapsed_time
    rows = {f"{side}:{i + 1}": row for side in SIDES for i, row in enumerate(sheet["goals"][side])}
    goals = selection.get("goals") or []
    if len(goals) != len(rows) or {g["id"] for g in goals} != set(rows):
        raise ValueError("selection goals do not match game_data.json; rerun selection")
    for goal in goals:
        row = rows[goal["id"]]
        expected_period = int(row["per"]) if row["per"].isdigit() else None
        if goal.get("period") is not None and goal["period"] != expected_period:
            raise ValueError("selection periods do not match game_data.json; rerun selection")
        claim = goal.get("sheet")
        if not isinstance(claim, dict):
            continue
        per = str(claim.get("per") or "")
        period = {"OT": "4", "SO": "5"}.get(per.upper(), per)
        clock = claim.get("time_s")
        if clock is None:
            clock = parse_time(str(claim.get("time") or ""))
        if period != row["per"] or clock != parse_time(str(row.get("time") or "")):
            raise ValueError("selection clocks do not match game_data.json; rerun selection")
        if clock is not None and goal.get("elapsed_s") is not None and goal.get("period_s") is not None:
            direction = (selection.get("rules") or {}).get("time_direction")
            directions = [direction] if direction else ["remaining", "elapsed"]
            expected = [elapsed_time(clock, goal["period_s"], d) for d in directions]
            if goal["elapsed_s"] not in expected:
                raise ValueError("selection elapsed times do not match game_data.json; rerun selection")
