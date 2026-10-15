"""Pipeline integration and edit commands for hhg-3r5.107. All databases are temporary."""
import json
import sqlite3

import pytest

import game_data as gd
import recap_graphics as graphics
import recap_options as options
import recap_runner as runner
import scoresheet
import team_db as db


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cfg = tmp_path / "config"
    (cfg / "teams").mkdir(parents=True)
    (cfg / "leagues").mkdir()
    monkeypatch.setenv("HHG_CONFIG_DIR", str(cfg))
    game = tmp_path / "visitors_10112026"
    game.mkdir()
    conn = db.connect(cfg / "hockey.db")
    for tid, name, own, rows in (
        ("ice", "Ice", True, [{"name": "Ann Lee", "jersey": 8}, {"name": "Bob Stone", "jersey": 10}]),
        ("visitors", "Visitors", False, [{"name": "Kenyon Ross", "jersey": 77},
                                        {"name": "Justin Davidson", "jersey": 77}]),
    ):
        data = {"name": name, "own_team": own, "listed_roster": rows, "league": "rec"}
        (cfg / "teams" / f"{tid}.json").write_text(json.dumps(data))
        db.import_team(conn, cfg / "teams" / f"{tid}.json", "2026-2027")
    opts = options.resolve_options(str(game), cfg_override=str(cfg))
    sheet = {"teams": {"home": "Ice", "away": "Visitors"},
             "rosters": {"home": {"08": "Ann Lee", "10": "Bob Stone"}, "away": {"7": "Kenyon Ross"}},
             "goals": {"home": [{"per": "1", "time": "10:00", "scorer": "8", "assists": ["10"], "type": "PP"}],
                       "away": []},
             "penalties": {"home": [], "away": [{"per": "2", "player": "7", "classification": "MAJOR",
                 "minutes": 5, "additional": [{"classification": "GAME MISCONDUCT"}]}]}, "flags": []}
    yield cfg, game, conn, opts, sheet
    conn.close()


def imported(setup):
    cfg, game, conn, opts, sheet = setup
    data = gd.import_sheet(conn, game.name, sheet, opts, game)
    gd.write_data(data, game / "game_data.json")
    return data


def test_database_identity_and_roster_win_over_json(setup):
    cfg, game, conn, opts, sheet = setup
    conn.execute("UPDATE teams SET name = 'Database Ice' WHERE id = 'ice'")
    conn.commit()
    # A database team must not even read its legacy file.
    (cfg / "teams" / "ice.json").write_text("not JSON")
    resolved = options.resolve_options(str(game), cfg_override=str(cfg))
    assert resolved["teams"]["ice"]["name"] == "Database Ice"
    assert resolved["teams"]["ice"]["roster"] == {"8": "Ann Lee", "10": "Bob Stone"}
    assert len(resolved["teams"]["visitors"]["listed_roster"]) == 2
    assert resolved["teams"]["visitors"]["roster"] == {}  # shared #77 is not one person
    assert resolved["_layers"]["team"] == "hockey.db"
    assert not any("JSON fallback" in f for f in resolved["_flags"])


def test_json_fallback_only_for_missing_team_is_reported(setup):
    cfg, game, conn, opts, sheet = setup
    (cfg / "teams" / "third.json").write_text(json.dumps({"name": "Third", "roster": {"2": "Cal Doe"}}))
    resolved = options.resolve_options(str(game), cfg_override=str(cfg))
    assert resolved["teams"]["third"]["roster"] == {"2": "Cal Doe"}
    assert any("third: no database team record; using JSON fallback" in f for f in resolved["_flags"])


def test_empty_database_roster_never_uses_json_roster(setup):
    cfg, game, conn, opts, sheet = setup
    conn.execute("DELETE FROM roster_entries WHERE team_id = 'ice'")
    conn.commit()
    resolved = options.resolve_options(str(game), cfg_override=str(cfg))
    assert resolved["teams"]["ice"]["listed_roster"] == []
    assert resolved["teams"]["ice"]["roster"] == {}


def test_options_read_does_not_migrate_or_change_database(setup):
    cfg, game, conn, opts, sheet = setup
    path = cfg / "hockey.db"
    before = path.read_bytes()
    options.resolve_options(str(game), cfg_override=str(cfg))
    assert path.read_bytes() == before


def test_options_read_does_not_create_database(tmp_path):
    game = tmp_path / "x_10112026"
    game.mkdir()
    cfg = tmp_path / "config"
    options.resolve_options(str(game), cfg_override=str(cfg))
    assert not (cfg / "hockey.db").exists()


@pytest.mark.parametrize("rules,season,pim", [
    ({}, "2026-2027", 10),
    ({"season_start_month": 11, "rulebook": "nhl"}, "2025-2026", 15),
    ({"season": "2024-2025", "rulebook": "nhl"}, "2024-2025", 15),
])
def test_league_controls_season_and_rulebook(setup, rules, season, pim):
    cfg, game, conn, opts, sheet = setup
    # Supply matching Rosters for each League Season.
    for tid in ("ice", "visitors"):
        db.import_team(conn, cfg / "teams" / f"{tid}.json", season)
    (cfg / "leagues" / "rec.json").write_text(json.dumps(rules))
    resolved = options.resolve_options(str(game), cfg_override=str(cfg))
    assert resolved["season"] == season
    sheet["penalties"]["away"][0].update(classification="MATCH", minutes=None, additional=[])
    data = gd.import_sheet(conn, game.name, sheet, resolved, game)
    assert data["game"]["season"] == season
    assert data["team_pim"]["visitors"] == pim


def test_import_applied_sheet_keeps_numbers_stats_and_rulebook(setup):
    cfg, game, conn, opts, sheet = setup
    data = imported(setup)
    assert data["score"] == {"ice": 1, "visitors": 0}
    assert data["team_pim"]["visitors"] == 15
    assert data["stats"]["visitors"][0]["pim"] == 15
    assert data["stats"]["visitors"][0]["jersey"] == 7
    assert data["stats"]["ice"][0]["name"] == "Ann Lee"
    assert data["stats"]["ice"][1]["a"] == 1
    assert all(e["source"].startswith("sheet") for e in data["events"])
    assert any("wore #7; roster lists #77" in f for f in data["flags"])
    assert json.loads((game / "game_data.json").read_text()) == data


def test_override_roster_name_marks_event_source(setup):
    cfg, game, conn, opts, sheet = setup
    sheet["rosters"]["home"]["08"] = "Unclear Lee"
    applied = scoresheet.apply_overrides(sheet, {"schema_version": 1,
        "rosters": {"home": {"08": "Ann Lee"}}, "goals": {"home": {"1": {"time": "9:00"}}}})
    data = gd.import_sheet(conn, game.name, applied, opts, game)
    goal = data["events"][0]
    assert goal["clock"] == "9:00" and goal["source"].startswith("override")
    assert conn.execute("SELECT source FROM event_assists").fetchone()[0].startswith("override")


def test_roster_only_override_marks_event_source(setup):
    cfg, game, conn, opts, sheet = setup
    applied = scoresheet.apply_overrides(sheet, {"schema_version": 1, "rosters": {"home": {"08": "Ann Lee"}}})
    data = gd.import_sheet(conn, game.name, applied, opts, game)
    assert data["events"][0]["source"].startswith("override")


def test_unknown_name_stops_and_rolls_back_whole_game(setup):
    cfg, game, conn, opts, sheet = setup
    original = imported(setup)
    sheet["rosters"]["home"]["08"] = "Anne Lee"
    opts["date"] = "2026-10-12"
    with pytest.raises(ValueError, match="did you mean.*Ann Lee"):
        gd.import_sheet(conn, game.name, sheet, opts, game)
    assert db.snapshot(conn, game.name) == original
    assert json.loads((game / "game_data.json").read_text()) == original


def test_failure_rolls_back_new_teams_and_game(setup):
    cfg, game, conn, opts, sheet = setup
    sheet["teams"]["away"] = "Unknowns"
    with pytest.raises(ValueError, match="Kenyon Ross.*not on"):
        gd.import_sheet(conn, game.name, sheet, opts, game)
    assert not conn.execute("SELECT 1 FROM games").fetchone()
    assert not conn.execute("SELECT 1 FROM teams WHERE name = 'Unknowns'").fetchone()


def test_reimport_replaces_without_duplicates(setup):
    cfg, game, conn, opts, sheet = setup
    first = imported(setup)
    second = imported(setup)
    assert second == first
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 3


def test_graphics_uses_frozen_snapshot_after_roster_and_team_edits(setup):
    cfg, game, conn, opts, sheet = setup
    original = imported(setup)
    frozen, frozen_opts = gd.load_graphics_inputs(game, opts)
    expected_goal = graphics.goal_data(frozen, "home:1")
    expected_final = graphics.final_data(frozen, [])
    pid = db.find_person(conn, "ice", opts["season"], name="Ann Lee")
    db.rename_player(conn, pid, "Changed Name")
    db.renumber_player(conn, "ice", opts["season"], pid, 99)
    conn.execute("UPDATE teams SET name = 'Changed Team' WHERE id = 'ice'")
    conn.commit()
    (game / "game_sheet.json").write_text("corrupt live sheet")
    live = options.resolve_options(str(game), cfg_override=str(cfg))
    again, again_opts = gd.load_graphics_inputs(game, live)
    assert graphics.goal_data(again, "home:1") == expected_goal
    assert graphics.final_data(again, []) == expected_final
    assert again_opts["teams"]["ice"]["name"] == "Ice"
    assert again_opts["teams"] == frozen_opts["teams"]
    assert json.loads((game / "game_data.json").read_text()) == original


def test_shared_jerseys_do_not_combine_stats(setup):
    cfg, game, conn, opts, sheet = setup
    sheet["penalties"]["away"][0]["player"] = {"name": "Kenyon Ross", "jersey": 77}
    sheet["goals"]["away"] = [{"per": "2", "scorer": {"name": "Justin Davidson", "jersey": 77}, "assists": []}]
    data = gd.import_sheet(conn, game.name, sheet, opts, game)
    frozen, _ = gd.graphics_inputs(data, opts)
    rows = graphics.final_data(frozen, [])["table"]["away"]["skaters"]
    assert [(r["name"], r["num"], r["pim"]) for r in rows] == [("Justin Davidson", "77", 0), ("Kenyon Ross", "77", 15)]
    assert graphics.goal_data(frozen, "away:1")["scorer"] == "Justin Davidson"


def test_scoresheet_stage_writes_snapshot_and_failure_keeps_previous(setup, monkeypatch):
    cfg, game, conn, opts, sheet = setup
    (game / "game_sheet.json").write_text(json.dumps(sheet))
    ctx = runner.Context(game, opts, runner.Reporter())
    monkeypatch.setattr(ctx, "run_cmd", lambda *args: (0, []))
    assert runner._run_scoresheet(ctx).state in ("done", "flagged")
    previous = (game / "game_data.json").read_bytes()
    sheet["rosters"]["home"]["08"] = "Anne Lee"
    (game / "game_sheet.json").write_text(json.dumps(sheet))
    stage = next(s for s in runner.STAGES if s.name == "scoresheet")
    monkeypatch.setattr(runner.Context, "run_cmd", lambda *args: (0, []))
    status = runner.run_game(game, opts, stages=[stage])
    assert status["stages"]["scoresheet"]["state"] == "failed"
    assert "did you mean" in " ".join(status["flags"])
    assert (game / "game_data.json").read_bytes() == previous


def test_graphics_fingerprints_include_snapshot_not_live_roster(setup):
    cfg, game, conn, opts, sheet = setup
    imported(setup)
    (game / "recap_assembly.json").write_text(json.dumps({"output": "recap.mp4"}))
    ctx = runner.Context(game, opts, runner.Reporter())
    before = runner._fp_graphics(ctx)
    opts["teams"]["ice"]["name"] = "Live change"
    opts["teams"]["ice"]["roster"] = {"99": "Not rendered"}
    assert runner._fp_graphics(ctx) == before
    data = json.loads((game / "game_data.json").read_text())
    data["stats"]["ice"][0]["pim"] = 2
    gd.write_data(data, game / "game_data.json")
    assert runner._fp_graphics(ctx) != before
    assert "game_data.json" in before["inputs"] and "game_sheet.json" not in before["inputs"]


def test_edit_commands_keep_game_and_season_numbers_separate(setup):
    cfg, game, conn, opts, sheet = setup
    imported(setup)
    path = str(cfg / "hockey.db")
    assert db.main(["--db", path, "roster-add", "ice", "--season", opts["season"], "--name", "Cal Doe", "--jersey", "08"]) == 0
    pid = db.find_person(conn, "ice", opts["season"], name="Cal Doe")
    for command in (["roster-rename", str(pid), "Calvin Doe"], ["display-name", str(pid), "Cal Doe"],
                    ["roster-number", "ice", str(pid), "9", "--season", opts["season"]],
                    ["game-number", game.name, "ice", str(pid), "08"]):
        assert db.main(["--db", path, *command]) == 0
    person = conn.execute("SELECT * FROM people WHERE id = ?", (pid,)).fetchone()
    roster = conn.execute("SELECT * FROM roster_entries WHERE person_id = ?", (pid,)).fetchone()
    worn = conn.execute("SELECT * FROM game_players WHERE person_id = ?", (pid,)).fetchone()
    assert person["first_name"] == "Calvin" and person["display_name"] == "Cal Doe"
    assert roster["jersey"] == 9 and worn["jersey_worn"] == 8 and worn["jersey_label"] == "08"
    assert all(row["source"].startswith("user ") for row in (person, roster, worn))


def test_failed_edit_rolls_back_new_person(setup):
    cfg, game, conn, opts, sheet = setup
    before = list(conn.execute("SELECT * FROM people"))
    with pytest.raises(ValueError, match="no team"):
        db.add_roster_player(conn, "missing", opts["season"], "New Person", 8)
    assert list(conn.execute("SELECT * FROM people")) == before
    pid = before[0]["id"]
    with pytest.raises(sqlite3.IntegrityError):
        db.add_roster_player(conn, "ice", opts["season"], person=pid)
    assert list(conn.execute("SELECT * FROM people")) == before


def test_dismissal_survives_reopen_and_does_not_hide_other_pairs(setup):
    cfg, game, conn, opts, sheet = setup
    a = db.find_person(conn, "ice", opts["season"], name="Ann Lee")
    b = db.add_roster_player(conn, "visitors", opts["season"], "Ann Lee", 20)
    c = db.add_roster_player(conn, "visitors", opts["season"], "Ann Lee", 21)
    assert db.people_flags(conn)
    db.main(["--db", str(cfg / "hockey.db"), "dismiss-people", str(b), str(a)])
    reopened = db.connect(cfg / "hockey.db")
    try:
        flags = " ".join(db.people_flags(reopened))
        assert f"id {c} " in flags and f"id {b} " not in flags
        db.dismiss_people(reopened, a, c)
        assert not db.people_flags(reopened)
        assert reopened.execute("SELECT source FROM people_dismissals").fetchone()[0].startswith("user ")
    finally:
        reopened.close()


@pytest.mark.parametrize("infraction,minutes,team_only,classification,pim", [
    ("TRIPPING", "2", False, "MINOR", 2),
    ("HIGH STICK", "4", False, "DOUBLE MINOR", 4),
    ("FIGHTING", "5", False, "MAJOR", 5),
    ("GAME MISCONDUCT", None, False, "GAME MISCONDUCT", 10),
    ("TOO MANY MEN", "2", True, "BENCH MINOR", 2),
])
def test_sheet_penalty_classification_uses_recorded_evidence(setup, infraction, minutes, team_only, classification, pim):
    cfg, game, conn, opts, sheet = setup
    sheet["penalties"]["away"] = [{"per": "1", "player": None if team_only else "7",
                                  "infraction": infraction, "minutes": minutes}]
    data = gd.import_sheet(conn, game.name, sheet, opts, game)
    penalty = data["events"][-1]
    assert penalty["classification"] == classification
    assert penalty["minutes"] == pim
    assert data["team_pim"]["visitors"] == pim


def test_ambiguous_sheet_penalty_classification_stops_with_correction_path(setup):
    cfg, game, conn, opts, sheet = setup
    original = imported(setup)
    sheet["penalties"]["away"] = [{"per": "1", "player": "7", "infraction": "ABUSE", "minutes": 10}]
    with pytest.raises(ValueError, match="classification.*game_sheet_overrides.json"):
        gd.import_sheet(conn, game.name, sheet, opts, game)
    assert db.snapshot(conn, game.name) == original


def test_corrected_sheet_date_selects_its_season_roster(setup):
    cfg, game, conn, opts, sheet = setup
    for tid in ("ice", "visitors"):
        db.import_team(conn, cfg / "teams" / f"{tid}.json", "2025-2026")
    applied = scoresheet.apply_overrides(sheet, {"schema_version": 1, "date": "2026-07-31"})
    data = gd.import_sheet(conn, game.name, applied, opts, game)
    assert data["game"]["date"] == "2026-07-31"
    assert data["game"]["season"] == "2025-2026"


def test_snapshot_publication_failure_rolls_back_database(setup, monkeypatch):
    cfg, game, conn, opts, sheet = setup
    previous = imported(setup)
    sheet["goals"]["home"][0]["time"] = "2:00"
    (game / "game_sheet.json").write_text(json.dumps(sheet))
    monkeypatch.setattr(runner.Context, "run_cmd", lambda *args: (0, []))
    def fail(*args):
        raise OSError("snapshot publication failed")
    monkeypatch.setattr(gd, "write_data", fail)
    stage = next(s for s in runner.STAGES if s.name == "scoresheet")
    status = runner.run_game(game, opts, stages=[stage])
    assert status["stages"]["scoresheet"]["state"] == "failed"
    assert db.snapshot(conn, game.name) == previous
    assert json.loads((game / "game_data.json").read_text()) == previous


@pytest.mark.parametrize("classification,infraction,minutes", [
    (None, "MAJOR + GAME MISCONDUCT", "5+10"),
    (None, "MAJOR + GAME MISCONDUCT", None),
    (" ", "ABUSE", 10),
    (None, "MINOR + MINOR", None),
    (None, "GAME MISCONDUCT", "5+10"),
])
def test_ambiguous_combined_sheet_classes_need_explicit_correction(classification, infraction, minutes):
    with pytest.raises(ValueError, match="classification.*game_sheet_overrides.json"):
        db.sheet_penalty_class(classification, infraction, minutes, False)


def test_snapshot_change_reruns_both_graphics_stages_in_saved_status(setup, monkeypatch):
    import recap_assembly
    cfg, game, conn, opts, sheet = setup
    data = imported(setup)
    opts["audio"] = {"manifest": str(cfg / "missing-manifest.json")}
    (game / "recap_assembly.json").write_text(json.dumps({"output": "recap.mp4"}))
    monkeypatch.setattr(recap_assembly, "rife_ready", lambda: False)
    calls = []
    def render(name):
        def run(ctx):
            calls.append(name)
            (game / f"{name}.out").write_text("rendered")
            return runner.StageResult("done")
        return run
    stages = [runner.Stage(name, name, fingerprint, lambda ctx, n=name: [f"{n}.out"], render(name))
              for name, fingerprint in (("graphics", runner._fp_graphics), ("short", runner._fp_short))]
    first = runner.run_game(game, opts, stages=stages)
    assert calls == ["graphics", "short"]
    calls.clear()
    runner.run_game(game, opts, stages=stages)
    assert calls == []
    data["stats"]["ice"][0]["pim"] = 2
    gd.write_data(data, game / "game_data.json")
    runner.run_game(game, opts, stages=stages)
    assert calls == ["graphics", "short"]
    saved = json.loads((game / "recap_status.json").read_text())
    assert all(first["stages"][name]["fingerprint"] != saved["stages"][name]["fingerprint"]
               for name in ("graphics", "short"))
