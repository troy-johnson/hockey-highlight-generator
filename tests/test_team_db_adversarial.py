# tests/test_team_db_adversarial.py — adversarial review of team_db (ADR-0004)
# Each test asserts the correct behaviour. A failure is a defect or an open design question.
import json
import sqlite3

import pytest

import team_db as db

S = "2026-2027"
NAMES = {"Ice Pak": "icepak", "Buzz Light Beer": "blb"}


@pytest.fixture
def conn(tmp_path):
    return db.connect(tmp_path / "hockey.db")


def _team(root, tid, cfg):
    root.mkdir(exist_ok=True)
    p = root / f"{tid}.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture
def teams(tmp_path):
    root = tmp_path / "teams"
    _team(root, "icepak", {"name": "Ice Pak", "own_team": True, "listed_roster": [
        {"jersey": 8, "first_name": "Marcus", "last_name": "Capps"},
        {"jersey": 4, "first_name": "Dave", "last_name": "Mohr"},
        {"jersey": 56, "first_name": "John David", "last_name": "Mohr", "display_name": "JD Mohr"},
        {"jersey": 12, "first_name": "José", "last_name": "Núñez"},
    ]})
    _team(root, "blb", {"name": "Buzz Light Beer", "listed_roster": [
        {"jersey": 8, "first_name": "Jenner", "last_name": "Smith"},
        {"jersey": 77, "first_name": "Kenyon", "last_name": "Ross"},
    ]})
    return root


@pytest.fixture
def game(conn, teams):
    for t in ("icepak", "blb"):
        db.import_team(conn, teams / f"{t}.json", S)
    db.add_game(conn, "g1", "2026-10-11", ["icepak", "blb"], focus="icepak", folder=None)
    return "g1"


def people(conn):
    return conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]


def goal(scorer, jersey=None, team="Ice Pak", assists=()):
    return {"event": "GOAL", "team": team, "scorer": scorer, "jersey": jersey, "assists": list(assists)}


# --- duplicate people from name variants ---------------------------------------------------

@pytest.mark.parametrize("variant", ["marcus capps", "MARCUS CAPPS", "Marcus  Capps", " Marcus Capps ",
                                     "Marcus Capps\t"])
def test_name_variant_does_not_create_duplicate_person(conn, game, variant):
    before = people(conn)
    db.import_events(conn, game, [goal(variant, 8)], NAMES)
    assert people(conn) == before, f"{variant!r} created a second Marcus Capps"


@pytest.mark.parametrize("variant", ["Jose Nunez", "José Núñez"])
def test_accent_variant_does_not_create_duplicate_person(conn, game, variant):
    # "José Núñez" in NFD form (decomposed accents) must match the NFC roster spelling too.
    import unicodedata
    before = people(conn)
    db.import_events(conn, game, [goal(unicodedata.normalize("NFD", variant), 12)], NAMES)
    assert people(conn) == before


# User decision (2026-10-15): an unknown name stops the import and suggests Roster matches.

def test_known_jersey_with_unknown_name_variant_stops_import(conn, game):
    before = people(conn)
    with pytest.raises(ValueError, match="#56 JD Mohr"):
        db.import_events(conn, game, [goal("JD", 56)], NAMES)
    assert people(conn) == before


def test_player_not_on_roster_stops_import_with_suggestion(conn, game):
    before = people(conn)
    with pytest.raises(ValueError, match="did you mean #8 Marcus Capps"):
        db.import_events(conn, game, [goal("Marc Capps", 8)], NAMES)
    assert people(conn) == before


def test_old_roster_map_with_three_part_name_matches_listed_person(conn, tmp_path, teams):
    db.import_team(conn, teams / "icepak.json", S)
    old = _team(tmp_path / "old", "icepak", {"name": "Ice Pak", "roster": {"56": "John David Mohr"}})
    db.import_team(conn, old, "2027-2028")
    assert conn.execute("SELECT COUNT(*) FROM people WHERE last_name LIKE '%Mohr'").fetchone()[0] == 2


# --- same person across teams and seasons -------------------------------------------------

def test_same_person_next_season_keeps_identity(conn, teams):
    db.import_team(conn, teams / "icepak.json", S)
    db.import_team(conn, teams / "icepak.json", "2027-2028")
    assert db.find_person(conn, "icepak", S, name="Marcus Capps") == \
        db.find_person(conn, "icepak", "2027-2028", name="Marcus Capps")


def test_same_person_on_two_teams_is_one_person_after_review(conn, tmp_path, teams):
    # User decision (2026-10-15): one person across teams; a same-name match is flagged and merged once.
    db.import_team(conn, teams / "icepak.json", S)
    other = _team(tmp_path / "t2", "rivals", {"name": "Rivals", "listed_roster": [
        {"jersey": 9, "first_name": "Marcus", "last_name": "Capps"}]})
    db.import_team(conn, other, S)
    keep = db.find_person(conn, "icepak", S, name="Marcus Capps")
    drop = db.find_person(conn, "rivals", S, name="Marcus Capps")
    assert any("Marcus Capps has 2 records" in f for f in db.people_flags(conn))
    db.merge_people(conn, keep, drop)
    assert db.find_person(conn, "rivals", S, name="Marcus Capps") == keep
    assert not db.people_flags(conn)
    db.import_team(conn, other, S)  # a reimport keeps the merged identity
    assert db.find_person(conn, "rivals", S, name="Marcus Capps") == keep


def test_game_in_season_without_roster_reuses_known_person(conn, teams):
    db.import_team(conn, teams / "icepak.json", S)
    db.import_team(conn, teams / "blb.json", S)
    db.add_game(conn, "g2", "2027-09-01", ["icepak", "blb"])  # 2027-2028, no roster imported yet
    before = people(conn)
    db.import_events(conn, "g2", [goal("Marcus Capps", 8)], NAMES)
    assert people(conn) == before, "find_person ignores earlier seasons and duplicates the player"


# --- reimport and atomicity ---------------------------------------------------------------

def test_reimport_leaves_no_orphan_people(conn, game):
    with pytest.raises(ValueError):
        db.import_events(conn, game, [goal("Ghost Player", 99)], NAMES)
    db.import_events(conn, game, [goal("Marcus Capps", 8)], NAMES)
    orphans = conn.execute(
        "SELECT first_name, last_name FROM people WHERE id NOT IN (SELECT person_id FROM roster_entries) "
        "AND id NOT IN (SELECT person_id FROM game_players)").fetchall()
    assert not orphans, [tuple(r) for r in orphans]


def test_failed_import_keeps_previous_events(conn, game):
    db.import_events(conn, game, [goal("Marcus Capps", 8)], NAMES)
    bad = [goal("Dave Mohr", 4), goal("Nobody", 8, team="Unknown Team")]
    with pytest.raises(KeyError):
        db.import_events(conn, game, bad, NAMES)
    rows = conn.execute("SELECT COUNT(*) FROM events WHERE game_id = ?", (game,)).fetchone()[0]
    assert rows == 1
    assert conn.execute("SELECT COUNT(*) FROM game_players WHERE game_id = ?", (game,)).fetchone()[0] == 1


def test_failed_import_with_bad_kind_rolls_back(conn, game):
    db.import_events(conn, game, [goal("Marcus Capps", 8)], NAMES)
    with pytest.raises(Exception):
        db.import_events(conn, game, [goal("Dave Mohr", 4), {"event": "SHOT", "team": "Ice Pak"}], NAMES)
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


def test_event_for_team_not_in_game_is_rejected_at_import(conn, game, tmp_path):
    third = _team(tmp_path / "t3", "rivals", {"name": "Rivals", "listed_roster": [
        {"jersey": 3, "first_name": "Ann", "last_name": "Lee"}]})
    db.import_team(conn, third, S)
    with pytest.raises(ValueError):
        db.import_events(conn, game, [goal("Ann Lee", 3, team="Rivals")], {**NAMES, "Rivals": "rivals"})


# --- assists ------------------------------------------------------------------------------

def test_assist_by_other_team_player_is_rejected(conn, game):
    with pytest.raises(ValueError):
        db.import_events(conn, game, [goal("Marcus Capps", 8, assists=[{"name": "Kenyon Ross", "jersey": 77}])],
                         NAMES)


def test_scorer_cannot_assist_own_goal(conn, game):
    with pytest.raises(ValueError):
        db.import_events(conn, game, [goal("Marcus Capps", 8, assists=[{"name": "Marcus Capps", "jersey": 8}])],
                         NAMES)


def test_same_assist_twice_is_rejected(conn, game):
    with pytest.raises(ValueError):
        db.import_events(conn, game, [goal("Marcus Capps", 8, assists=[{"name": "Dave Mohr", "jersey": 4},
                                                                       {"name": "Dave Mohr", "jersey": 4}])], NAMES)


def test_more_than_two_assists_is_rejected(conn, game):
    a = [{"name": n, "jersey": j} for n, j in (("Dave Mohr", 4), ("JD Mohr", 56), ("José Núñez", 12))]
    with pytest.raises(ValueError):
        db.import_events(conn, game, [goal("Marcus Capps", 8, assists=a)], NAMES)


# --- penalties ----------------------------------------------------------------------------

def pen(minutes, additional=None, player="Kenyon Ross", jersey=77):
    return {"event": "PENALTY", "team": "Buzz Light Beer", "player": player, "jersey": jersey,
            "classification": "MINOR", "minutes": minutes, "additional": additional or []}


@pytest.mark.parametrize("minutes, pim", [("2+10", 12), ("5:00", 5), ("2", 2)])
def test_string_penalty_minutes(conn, game, minutes, pim):
    try:
        db.import_events(conn, game, [pen(minutes)], NAMES)
    except ValueError:
        return  # rejecting the value at import is also correct
    snap = db.snapshot(conn, game)
    ross = next(r for r in snap["stats"]["blb"] if r["name"] == "Kenyon Ross")
    assert ross["pim"] == pim


def test_additional_penalty_minutes_count_in_pim(conn, game):
    db.import_events(conn, game, [pen(2, [{"classification": "MISCONDUCT", "minutes": 10}])], NAMES)
    snap = db.snapshot(conn, game)
    assert next(r for r in snap["stats"]["blb"] if r["name"] == "Kenyon Ross")["pim"] == 12


def test_additional_penalty_on_a_goal_is_rejected(conn, game):
    g = goal("Marcus Capps", 8)
    g["additional"] = [{"classification": "MISCONDUCT", "minutes": 10}]
    with pytest.raises(ValueError):
        db.import_events(conn, game, [g], NAMES)


def test_additional_penalty_without_classification_gives_clear_error(conn, game):
    with pytest.raises(ValueError):
        db.import_events(conn, game, [pen(2, [{"minutes": 10}])], NAMES)


def test_bench_penalty_without_player_can_be_stored(conn, game):
    # User decision (2026-10-15): a team-only penalty counts for team PIM, not for a player.
    db.import_events(conn, game, [pen(2, player=None, jersey=None)], NAMES)
    assert conn.execute("SELECT COUNT(*) FROM events WHERE kind = 'penalty'").fetchone()[0] == 1
    snap = db.snapshot(conn, game)
    assert snap["team_pim"]["blb"] == 2 and snap["stats"]["blb"] == []


# --- snapshot -----------------------------------------------------------------------------

def test_snapshot_score_matches_goal_events(conn, game):
    db.import_events(conn, game, [goal("Marcus Capps", 8), goal("Dave Mohr", 4),
                                  goal("Kenyon Ross", 77, team="Buzz Light Beer"), pen(2)], NAMES)
    snap = db.snapshot(conn, game)
    assert snap["score"] == {"icepak": 2, "blb": 1}
    assert sum(r["g"] for r in snap["stats"]["icepak"]) == snap["score"]["icepak"]


def test_conflicting_numbers_worn_in_one_game_stop_import(conn, game):
    with pytest.raises(ValueError, match="wore #8 and #18"):
        db.import_events(conn, game, [goal("Marcus Capps", 8), goal("Marcus Capps", 18)], NAMES)
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


def test_snapshot_without_folder_or_out_gives_clear_error(conn, game):
    with pytest.raises(ValueError):
        db.write_snapshot(conn, game)


# --- seasons ------------------------------------------------------------------------------

@pytest.mark.parametrize("date, season", [("2026-07-31", "2025-2026"), ("2026-08-01", "2026-2027"),
                                          ("2026-12-31", "2026-2027"), ("2027-01-01", "2026-2027")])
def test_season_for_edges(date, season):
    assert db.season_for(date) == season


@pytest.mark.parametrize("date", ["2026-13-01", "2026-02-30", "10/11/2026", "", "2026-10-11; DROP TABLE games"])
def test_season_for_rejects_invalid_dates(date):
    with pytest.raises(ValueError):
        db.season_for(date)


def test_add_game_with_invalid_date_writes_nothing(conn):
    with pytest.raises(ValueError):
        db.add_game(conn, "bad", "2026-02-30", [])
    assert conn.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 0


# --- migrations ---------------------------------------------------------------------------

def test_newer_database_version_is_refused(tmp_path):
    path = tmp_path / "hockey.db"
    raw = sqlite3.connect(path)
    raw.execute(f"PRAGMA user_version = {len(db.MIGRATIONS) + 5}")
    raw.commit()
    raw.close()
    with pytest.raises(Exception):
        db.connect(path)


# --- injection-like names -----------------------------------------------------------------

def test_sql_like_names_are_stored_verbatim(conn, tmp_path):
    evil = "Robert'); DROP TABLE people;--"
    p = _team(tmp_path / "t", "evil", {"name": "x' OR '1'='1", "listed_roster": [
        {"jersey": 1, "first_name": evil, "last_name": "O'Brien"}]})
    db.import_team(conn, p, S)
    assert db.find_person(conn, "evil", S, name=f"{evil} O'Brien")
    assert conn.execute("SELECT name FROM teams").fetchone()[0] == "x' OR '1'='1"


# --- family links -------------------------------------------------------------------------

def test_ambiguous_family_link_rolls_back_team_import(conn, tmp_path):
    p = _team(tmp_path / "t", "fam", {"name": "Fam", "listed_roster": [
        {"jersey": 4, "first_name": "Dave", "last_name": "Mohr"},
        {"jersey": 5, "first_name": "Dave", "last_name": "Mohr"},
        {"jersey": 56, "first_name": "JD", "last_name": "Mohr"}],
        "family_links": [{"parent": {"name": "Dave Mohr"}, "child": {"name": "JD Mohr"}, "relation": "father"}]})
    with pytest.raises(ValueError):
        db.import_team(conn, p, S)
    assert conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0] == 0
    assert people(conn) == 0


def test_two_same_name_rows_are_two_people(conn, tmp_path):
    p = _team(tmp_path / "t", "fam", {"name": "Fam", "listed_roster": [
        {"jersey": 4, "first_name": "Dave", "last_name": "Mohr"},
        {"jersey": 5, "first_name": "Dave", "last_name": "Mohr"}]})
    db.import_team(conn, p, S)
    assert conn.execute("SELECT COUNT(*) FROM roster_entries").fetchone()[0] == 2


def test_family_link_to_person_off_roster_is_idempotent(conn, tmp_path):
    p = _team(tmp_path / "t", "fam", {"name": "Fam", "listed_roster": [
        {"jersey": 56, "first_name": "JD", "last_name": "Mohr"}],
        "family_links": [{"parent": {"name": "Dave Mohr"}, "child": {"name": "JD Mohr"}, "relation": "father"}]})
    db.import_team(conn, p, S)
    first = people(conn)
    db.import_team(conn, p, S)
    assert people(conn) == first, "reimport adds another off-roster Dave Mohr"
    assert conn.execute("SELECT COUNT(*) FROM family_links").fetchone()[0] == 1


# --- listed_roster shape ------------------------------------------------------------------

def test_listed_roster_row_with_only_name_is_accepted_or_clearly_rejected(conn, tmp_path):
    p = _team(tmp_path / "t", "x", {"name": "X", "listed_roster": [
        {"jersey": 1, "first_name": "Ok", "last_name": "Row"}, {"jersey": 2, "name": "Only Name"}]})
    try:
        db.import_team(conn, p, S)
    except ValueError:
        assert conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0] == 0
        return
    assert db.find_person(conn, "x", S, name="Only Name")


def test_listed_roster_row_without_names_rolls_back(conn, tmp_path):
    p = _team(tmp_path / "t", "x", {"name": "X", "listed_roster": [
        {"jersey": 1, "first_name": "Ok", "last_name": "Row"}, {"jersey": 2}]})
    with pytest.raises((ValueError, KeyError)):
        db.import_team(conn, p, S)
    assert conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0] == 0
    assert people(conn) == 0


# --- jersey types -------------------------------------------------------------------------

def test_string_jersey_in_event_matches_int_roster(conn, game):
    db.import_events(conn, game, [goal(None, "8")], NAMES)
    snap = db.snapshot(conn, game)
    assert snap["stats"]["icepak"][0]["name"] == "Marcus Capps"


def test_string_jersey_in_roster_matches_int_event(conn, tmp_path):
    p = _team(tmp_path / "t", "x", {"name": "X", "listed_roster": [
        {"jersey": "08", "first_name": "Zero", "last_name": "Eight"}]})
    db.import_team(conn, p, S)
    assert db.find_person(conn, "x", S, jersey=8)


def test_jersey_08_is_number_8_and_keeps_its_printed_form(conn, tmp_path):
    # User decision (2026-10-15): 08 and 8 are the same number; the label keeps "08".
    p = _team(tmp_path / "t", "x", {"name": "X", "listed_roster": [
        {"jersey": "08", "first_name": "Zero", "last_name": "Eight"},
        {"jersey": 8, "first_name": "Plain", "last_name": "Eight"}]})
    db.import_team(conn, p, S)
    with pytest.raises(ValueError, match="#8 is shared by 2"):
        db.find_person(conn, "x", S, jersey=8)
    assert conn.execute("SELECT jersey_label FROM roster_entries WHERE jersey_label IS NOT NULL").fetchone()[0] == "08"


def test_non_numeric_jersey_is_rejected(conn, tmp_path):
    p = _team(tmp_path / "t", "x", {"name": "X", "listed_roster": [
        {"jersey": "A", "first_name": "Cap", "last_name": "Tain"}]})
    with pytest.raises(ValueError):
        db.import_team(conn, p, S)
