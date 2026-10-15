"""Adversarial regressions for the SQLite pipeline; no personal data is read."""
import copy
import json
from types import SimpleNamespace

import pytest

import game_data
import recap_assembly
import recap_graphics
import recap_options
import recap_runner
import scoresheet
import selection
import short_clips
import team_db


SEASON = "2026-2027"


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    cfg = tmp_path / "config"
    (cfg / "teams").mkdir(parents=True)
    (cfg / "leagues").mkdir()
    monkeypatch.setenv("HHG_CONFIG_DIR", str(cfg))
    return cfg


@pytest.fixture
def pipeline(tmp_path, isolated_config):
    game = tmp_path / "visitors_10112026"
    game.mkdir()
    conn = team_db.connect(isolated_config / "hockey.db")
    team_db.store_team(conn, "ice", {
        "name": "Ice", "own_team": True,
        "listed_roster": [{"name": "Ann Lee", "jersey": 8},
                          {"name": "Bob Stone", "jersey": 10},
                          {"name": "Gail Keeper", "jersey": 30}],
    }, SEASON, "import test")
    team_db.store_team(conn, "visitors", {
        "name": "Visitors", "listed_roster": [
            {"name": "Kenyon Ross", "jersey": 77},
            {"name": "Justin Davidson", "jersey": 77}],
    }, SEASON, "import test")
    options = recap_options.resolve_options(str(game))
    sheet = {
        "date": "2026-10-11", "teams": {"home": "Ice", "away": "Visitors"},
        "rosters": {"home": {"8": "Ann Lee", "10": "Bob Stone", "30": "Gail Keeper"}, "away": {}},
        "goals": {"home": [{"per": "1", "time": "10:00", "scorer": "8", "assists": ["10"]}], "away": []},
        "penalties": {"home": [], "away": []},
        "goalkeeping": {"home": [{"player": "30", "saves": 20}], "away": []},
        "flags": [],
    }
    state = SimpleNamespace(cfg=isolated_config, game=game, conn=conn, options=options, sheet=sheet)
    yield state
    conn.close()


def freeze(state):
    data = game_data.import_sheet(state.conn, state.game.name, state.sheet, state.options, state.game)
    game_data.write_data(data, state.game / "game_data.json")
    return data


def goal_selection(sheet):
    return {"goals": selection.prepare_goals(sheet, {
        "period_minutes": 20, "time_direction": "remaining",
    }), "plays": []}


def test_database_roster_replaces_stale_per_game_roster(pipeline):
    p = pipeline
    path = p.game / "recap_options.json"
    saved = json.loads(path.read_text())
    saved["teams"] = {"ice": {"roster": {"99": "Deleted Player"}}}
    path.write_text(json.dumps(saved))

    resolved = recap_options.resolve_options(str(p.game))

    assert resolved["teams"]["ice"]["roster"] == {
        "8": "Ann Lee", "10": "Bob Stone", "30": "Gail Keeper",
    }


def test_zero_padded_roster_override_corrects_the_same_jersey(pipeline):
    p = pipeline
    p.sheet["rosters"]["home"]["8"] = "Anne Lee"
    applied = scoresheet.apply_overrides(p.sheet, {
        "schema_version": 1, "rosters": {"home": {"08": "Ann Lee"}},
    })

    data = game_data.import_sheet(p.conn, p.game.name, applied, p.options, p.game)

    assert data["events"][0]["person"]["name"] == "Ann Lee"
    assert data["events"][0]["source"].startswith("override")


def test_roster_only_goalie_override_keeps_override_provenance(pipeline):
    p = pipeline
    p.sheet["rosters"]["home"]["30"] = "Gale Keeper"
    applied = scoresheet.apply_overrides(p.sheet, {
        "schema_version": 1, "rosters": {"home": {"30": "Gail Keeper"}},
    })
    game_data.import_sheet(p.conn, p.game.name, applied, p.options, p.game)

    source = p.conn.execute(
        "SELECT gp.source FROM game_players gp JOIN people p ON p.id = gp.person_id "
        "WHERE p.first_name = 'Gail'",
    ).fetchone()[0]
    assert source.startswith("override"), source


def test_public_event_reimport_preserves_frozen_goalkeeper_references(pipeline):
    p = pipeline
    original = freeze(p)
    events = [{"event": "GOAL", "team": "Ice", "scorer": "Ann Lee", "jersey": 8,
               "period": 1, "game_clock": "09:00"}]
    team_db.import_events(p.conn, p.game.name, events, {"Ice": "ice"}, "user corrected events")

    updated = team_db.snapshot(p.conn, p.game.name)

    assert updated["sheet"]["goalkeeping"] == original["sheet"]["goalkeeping"]
    assert updated["events"][0]["clock"] == "09:00"


def test_merge_people_rewrites_goalkeeper_references_in_sheet_data(pipeline):
    p = pipeline
    original = freeze(p)
    keeper = team_db.find_person(p.conn, "ice", SEASON, name="Gail Keeper")
    duplicate = team_db.add_roster_player(p.conn, "visitors", SEASON, "Gail Keeper", 31)

    team_db.merge_people(p.conn, duplicate, keeper)
    merged = team_db.snapshot(p.conn, p.game.name)

    row = merged["sheet"]["goalkeeping"]["ice"][0]
    assert row["person"]["id"] == duplicate
    assert row["saves"] == original["sheet"]["goalkeeping"]["ice"][0]["saves"]


def test_snapshot_adapter_preserves_unresolved_scoresheet_flags(pipeline):
    p = pipeline
    warning = "home goal 1: scorer needs review"
    p.sheet["flags"] = [warning]
    data = freeze(p)

    frozen_sheet, _ = game_data.graphics_inputs(data, p.options)

    assert warning in frozen_sheet["flags"]


def test_graphics_rejects_stale_selection_after_same_count_clock_correction(pipeline):
    p = pipeline
    stale = goal_selection(p.sheet)
    p.sheet["goals"]["home"][0]["time"] = "02:00"
    data = freeze(p)
    sheet, options = game_data.graphics_inputs(data, p.options)
    options["start"] = "play"
    plan = {"fps": 30, "output": "recap.mp4", "clips": [
        {"goal_id": "home:1", "kind": "goal", "start_s": 10., "moment_s": 15.,
         "end_s": 20., "speed": 1., "frames": 300}], "replays": []}
    audio = {"video": "recap.mp4", "duration_s": 10., "cues": []}

    with pytest.raises(ValueError, match="rerun selection"):
        recap_graphics.write_props(p.game, sheet, stale, plan, audio, options)


@pytest.mark.parametrize("correction", ["duplicate", "clock"])
def test_short_rejects_inconsistent_selection_before_media_loading(pipeline, monkeypatch, correction):
    import cue_library

    p = pipeline
    stale = goal_selection(p.sheet)
    if correction == "duplicate":
        stale["goals"].append(copy.deepcopy(stale["goals"][0]))
    else:
        p.sheet["goals"]["home"][0]["time"] = "02:00"
    freeze(p)
    (p.game / "selection.json").write_text(json.dumps(stale))
    calls = []

    def stop_at_media(*args, **kwargs):
        calls.append("media loading reached")
        raise ValueError("test stopped before media loading")

    def unavailable_manifest(*args, **kwargs):
        raise ValueError("test has no music")

    monkeypatch.setattr(recap_assembly, "load_layouts", stop_at_media)
    monkeypatch.setattr(cue_library, "load_manifest", unavailable_manifest)
    options = {"audio": {"manifest": str(p.cfg / "music.json")}}

    assert short_clips.main([str(p.game), "--options", json.dumps(options)]) == 1
    assert calls == [], "inconsistent selection passed snapshot validation"


def test_short_team_only_penalty_caption_does_not_invent_a_player(pipeline):
    p = pipeline
    p.sheet["penalties"]["home"] = [
        {"per": "1", "player": None, "classification": "BENCH MINOR", "infraction": "TOO MANY MEN"}]
    data = freeze(p)
    sheet, _ = game_data.graphics_inputs(data, p.options)
    selected = {"goals": [], "plays": [
        {"id": "penalty:home:1", "kind": "penalty", "primary_cam": "cam1",
         "detection_s": 20., "score": .9, "label": "TOO MANY MEN"},
        {"id": "save:1", "kind": "save", "primary_cam": "cam1", "detection_s": 40., "score": .8},
        {"id": "save:2", "kind": "save", "primary_cam": "cam1", "detection_s": 60., "score": .7}]}
    layouts = {"cam1": [{"start": 0., "end": 100., "seek": 0.,
                          "files": [(str(p.game / "synthetic.mp4"), 100.)]}]}

    plan = short_clips.plan_short(selected, layouts, sheet=sheet)

    caption = next(c["caption"] for c in plan["clips"] if c["kind"] == "penalty")
    assert "#?" not in caption, caption
    assert "TOO MANY MEN" in caption, caption


def test_failed_identity_change_restores_date_teams_goalkeepers_and_events(pipeline):
    p = pipeline
    original = freeze(p)
    p.options["date"] = "2027-10-11"
    p.sheet["teams"]["away"] = "New Visitors"
    p.sheet["goals"]["home"][0]["scorer"] = {"name": "Anne Lee", "jersey": 8}

    with pytest.raises(ValueError, match="did you mean.*Ann Lee"):
        game_data.import_sheet(p.conn, p.game.name, p.sheet, p.options, p.game)

    assert team_db.snapshot(p.conn, p.game.name) == original
    assert not p.conn.execute("SELECT 1 FROM teams WHERE name = 'New Visitors'").fetchone()


def test_edit_methods_do_not_commit_the_callers_transaction(pipeline):
    p = pipeline
    person = team_db.find_person(p.conn, "ice", SEASON, name="Ann Lee")
    p.conn.execute("BEGIN")
    team_db.rename_player(p.conn, person, "Anne Lee")
    team_db.renumber_player(p.conn, "ice", SEASON, person, 99)
    team_db.set_display_name(p.conn, person, "Anne")
    p.conn.rollback()

    assert team_db.find_person(p.conn, "ice", SEASON, name="Ann Lee", jersey=8) == person
    assert team_db.team_configs(p.conn, SEASON)["ice"]["roster"]["8"] == "Ann Lee"


def test_shared_jersey_people_remain_independent_in_short_goal_captions(pipeline):
    p = pipeline
    p.sheet["goals"]["away"] = [
        {"per": "1", "time": "09:00", "scorer": {"name": "Kenyon Ross", "jersey": 77}},
        {"per": "1", "time": "08:00", "scorer": {"name": "Justin Davidson", "jersey": 77}}]
    data = freeze(p)
    sheet, _ = game_data.graphics_inputs(data, p.options)
    selected = {"goals": [
        {"id": gid, "period": 1, "elapsed_s": i * 10., "interest": .8,
         "chosen": {"primary_cam": "cam1", "detection_s": i * 20.}}
        for i, gid in enumerate(("home:1", "away:1", "away:2"), 1)], "plays": []}
    layouts = {"cam1": [{"start": 0., "end": 100., "seek": 0.,
                          "files": [(str(p.game / "synthetic.mp4"), 100.)]}]}

    plan = short_clips.plan_short(selected, layouts, sheet=sheet)

    assert [c["caption"] for c in plan["clips"]] == [
        "GOAL Ann Lee", "GOAL Kenyon Ross", "GOAL Justin Davidson"]


def test_snapshot_and_fingerprint_ignore_live_display_name_edits(pipeline):
    p = pipeline
    freeze(p)
    (p.game / "recap_assembly.json").write_text(json.dumps({"output": "recap.mp4"}))
    ctx = recap_runner.Context(p.game, p.options, recap_runner.Reporter())
    before = recap_runner._fp_graphics(ctx)
    frozen, _ = game_data.load_graphics_inputs(p.game, p.options)
    goal = recap_graphics.goal_data(frozen, "home:1")
    person = team_db.find_person(p.conn, "ice", SEASON, name="Ann Lee")
    team_db.set_display_name(p.conn, person, "New Display Name")
    ctx.options = recap_options.resolve_options(str(p.game))

    assert recap_runner._fp_graphics(ctx) == before
    frozen, _ = game_data.load_graphics_inputs(p.game, ctx.options)
    assert recap_graphics.goal_data(frozen, "home:1") == goal


@pytest.mark.parametrize("claim", ["scorer", "assist", "goalie"])
def test_unknown_name_with_known_jersey_never_guesses_a_person(pipeline, claim):
    p = pipeline
    original = freeze(p)
    if claim == "scorer":
        p.sheet["rosters"]["home"]["8"] = "Anne Lee"
    elif claim == "assist":
        p.sheet["rosters"]["home"]["10"] = "Bobb Stone"
    else:
        p.sheet["goalkeeping"]["home"][0]["name"] = "Gale Keeper"

    with pytest.raises(ValueError, match="not on.*roster.*did you mean"):
        game_data.import_sheet(p.conn, p.game.name, p.sheet, p.options, p.game)

    assert team_db.snapshot(p.conn, p.game.name) == original


def test_reimport_changes_sides_season_and_rulebook_without_old_identity(pipeline):
    p = pipeline
    freeze(p)
    p.sheet["teams"] = {"home": "Visitors", "away": "Ice"}
    for key in ("rosters", "goals", "penalties", "goalkeeping"):
        p.sheet[key] = {"home": p.sheet[key]["away"], "away": p.sheet[key]["home"]}
    p.sheet["penalties"]["home"] = [
        {"per": "1", "player": {"name": "Kenyon Ross", "jersey": 7}, "classification": "MATCH"}]
    p.options["league_rules"] = {"season": "2027-2028", "rulebook": "nhl"}

    data = freeze(p)

    assert data["game"]["season"] == "2027-2028"
    assert data["game"]["rulebook"] == "nhl"
    assert data["teams"]["ice"]["side"] == "away"
    assert data["teams"]["visitors"]["side"] == "home"
    assert data["score"] == {"ice": 1, "visitors": 0}
    assert data["team_pim"]["visitors"] == 15
    assert any("no 2027-2028 roster; using 2026-2027" in f for f in data["flags"])
    assert len(data["sheet"]["goalkeeping"]["ice"]) == 1


def test_people_dismissal_survives_cli_and_failed_outer_edit(pipeline):
    p = pipeline
    ann = team_db.find_person(p.conn, "ice", SEASON, name="Ann Lee")
    namesake = team_db.add_roster_player(p.conn, "visitors", SEASON, "Ann Lee", 20)
    assert team_db.people_flags(p.conn)
    assert team_db.main(["--db", str(p.cfg / "hockey.db"), "dismiss-people", str(ann), str(namesake)]) == 0
    p.conn.execute("BEGIN")
    team_db.rename_player(p.conn, namesake, "Different Person")
    p.conn.rollback()
    reopened = team_db.connect(p.cfg / "hockey.db")
    try:
        assert team_db.people_flags(reopened) == []
        assert reopened.execute("SELECT source FROM people_dismissals").fetchone()[0].startswith("user")
    finally:
        reopened.close()


def test_public_event_reimport_cannot_move_preserved_goalie_to_other_team(pipeline):
    p = pipeline
    original = freeze(p)
    keeper = team_db.find_person(p.conn, "ice", SEASON, name="Gail Keeper")
    team_db.add_roster_player(p.conn, "visitors", SEASON, person=keeper, jersey=31)
    events = [{"event": "GOAL", "team": "Visitors", "scorer": "Gail Keeper", "jersey": 31,
               "period": 1, "game_clock": "09:00"}]

    with pytest.raises(ValueError, match="already plays|team|goalkeeper"):
        team_db.import_events(p.conn, p.game.name, events, {"Visitors": "visitors"}, "user test")

    assert team_db.snapshot(p.conn, p.game.name) == original


def test_public_event_reimport_keeps_goalie_worn_number_when_new_event_has_no_number(pipeline):
    p = pipeline
    p.sheet["rosters"]["home"].pop("30")
    p.sheet["rosters"]["home"]["31"] = "Gail Keeper"
    p.sheet["goalkeeping"]["home"][0]["player"] = "31"
    original = freeze(p)
    events = [{"event": "GOAL", "team": "Ice", "scorer": "Gail Keeper",
               "period": 1, "game_clock": "09:00"}]

    team_db.import_events(p.conn, p.game.name, events, {"Ice": "ice"}, "user test")
    updated = team_db.snapshot(p.conn, p.game.name)

    assert updated["sheet"]["goalkeeping"] == original["sheet"]["goalkeeping"]
    assert updated["events"][0]["person"]["jersey"] == 31


def test_snapshot_publication_does_not_outlive_rolled_back_outer_transaction(pipeline):
    p = pipeline
    original = freeze(p)
    path = p.game / "game_data.json"
    previous = path.read_bytes()
    p.sheet["goals"]["home"][0]["time"] = "02:00"
    p.conn.execute("BEGIN")
    try:
        try:
            game_data.import_snapshot(p.conn, p.game.name, p.sheet, p.options, p.game)
        except ValueError as exc:
            # Rejecting publication inside a caller's transaction is also safe.
            assert "transaction" in str(exc).lower()
    finally:
        p.conn.rollback()

    assert team_db.snapshot(p.conn, p.game.name) == original
    assert path.read_bytes() == previous, "snapshot contains a game import that the database rolled back"


@pytest.mark.parametrize("has_previous", [False, True])
def test_failed_database_commit_restores_previous_snapshot(pipeline, monkeypatch, has_previous):
    import sqlite3

    p = pipeline
    original = freeze(p) if has_previous else None
    path = p.game / "game_data.json"
    previous = path.read_bytes() if has_previous else None
    p.sheet["goals"]["home"][0]["time"] = "02:00"
    publish = game_data.write_data

    def fail_at_commit(data, target):
        publish(data, target)
        # SQLite accepts this row now, but rejects it when the transaction commits.
        p.conn.execute("PRAGMA defer_foreign_keys = ON")
        p.conn.execute("INSERT INTO family_links (parent_id, child_id, relation, source) "
                       "VALUES (1000000, 1000001, 'parent', 'user test')")

    monkeypatch.setattr(game_data, "write_data", fail_at_commit)
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        game_data.import_snapshot(p.conn, p.game.name, p.sheet, p.options, p.game)

    if has_previous:
        assert team_db.snapshot(p.conn, p.game.name) == original
        assert path.read_bytes() == previous
    else:
        assert not p.conn.execute("SELECT 1 FROM games").fetchone()
        assert not path.exists()
    assert not path.with_name(path.name + ".tmp").exists()
    assert not p.conn.in_transaction


def test_selection_period_must_agree_with_frozen_goal_period(pipeline):
    p = pipeline
    selected = copy.deepcopy(goal_selection(p.sheet))
    selected["goals"][0]["period"] = 2
    data = freeze(p)
    sheet, _ = game_data.graphics_inputs(data, p.options)

    with pytest.raises(ValueError, match="rerun selection"):
        game_data.validate_selection(sheet, selected)


@pytest.mark.parametrize("infraction", ["MINOR + MISCONDUCT", "MAJOR + GAME MISCONDUCT"])
def test_multiple_penalty_classes_need_explicit_resolution(pipeline, infraction):
    p = pipeline
    original = freeze(p)
    p.sheet["penalties"]["home"] = [{"per": "1", "player": "8", "infraction": infraction}]

    with pytest.raises(ValueError, match="classification.*game_sheet_overrides.json"):
        game_data.import_sheet(p.conn, p.game.name, p.sheet, p.options, p.game)

    assert team_db.snapshot(p.conn, p.game.name) == original


def test_corrected_date_uses_different_roster_at_season_boundary(pipeline):
    p = pipeline
    team_db.store_team(p.conn, "ice", {
        "name": "Ice", "own_team": True, "listed_roster": [
            {"name": "Previous Scorer", "jersey": 8},
            {"name": "Previous Assist", "jersey": 10},
            {"name": "Previous Keeper", "jersey": 30}],
    }, "2025-2026", "import previous roster")
    p.sheet["rosters"]["home"] = {"8": "Previous Scorer", "10": "Previous Assist", "30": "Previous Keeper"}
    p.sheet = scoresheet.apply_overrides(p.sheet, {"schema_version": 1, "date": "2026-07-31"})

    data = game_data.import_snapshot(p.conn, p.game.name, p.sheet, p.options, p.game)

    assert data["game"]["date"] == "2026-07-31"
    assert data["game"]["season"] == "2025-2026"
    assert data["events"][0]["person"]["name"] == "Previous Scorer"
    assert data["events"][0]["assists"][0]["name"] == "Previous Assist"
    assert data["sheet"]["goalkeeping"]["ice"][0]["person"]["name"] == "Previous Keeper"
