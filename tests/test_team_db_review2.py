# tests/test_team_db_review2.py — second adversarial review of fix commit c855160 (ADR-0004)
# Each test asserts the correct behaviour. A failure is a defect or an open design question.
import json
import sqlite3

import pytest

import team_db as db

S = "2026-2027"
NAMES = {"Ice Pak": "icepak", "Buzz Light Beer": "blb"}


def _team(root, tid, cfg):
    root.mkdir(exist_ok=True)
    p = root / f"{tid}.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture
def conn(tmp_path):
    return db.connect(tmp_path / "hockey.db")


@pytest.fixture
def game(conn, tmp_path):
    root = tmp_path / "teams"
    db.import_team(conn, _team(root, "icepak", {"name": "Ice Pak", "own_team": True, "listed_roster": [
        {"jersey": 8, "first_name": "Marcus", "last_name": "Capps"},
        {"jersey": 4, "first_name": "Dave", "last_name": "Mohr"},
        {"jersey": 9, "first_name": "Ann", "last_name": "Lee"}]}), S)
    db.import_team(conn, _team(root, "blb", {"name": "Buzz Light Beer", "listed_roster": [
        {"jersey": 8, "first_name": "Jenner", "last_name": "Smith"},
        {"jersey": 77, "first_name": "Kenyon", "last_name": "Ross"}]}), S)
    db.add_game(conn, "g1", "2026-10-11", ["icepak", "blb"], focus="icepak")
    return "g1"


def pid(conn, first, last):
    return conn.execute("SELECT id FROM people WHERE first_name = ? AND last_name = ?", (first, last)).fetchone()[0]


def goal(scorer, jersey=None, team="Ice Pak", assists=()):
    return {"event": "GOAL", "team": team, "scorer": scorer, "jersey": jersey, "assists": list(assists)}


def count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# --- connect(), migrations, transactions ----------------------------------------------------

def test_with_block_is_transactional_after_connect(conn, game):
    db.import_events(conn, game, [goal("Marcus Capps")], NAMES)
    assert not conn.in_transaction
    with pytest.raises(RuntimeError):
        with conn:
            conn.execute("DELETE FROM event_assists")
            conn.execute("DELETE FROM events")
            assert conn.in_transaction
            raise RuntimeError("boom")
    assert count(conn, "events") == 1


def test_failed_import_events_restores_old_events(conn, game):
    db.import_events(conn, game, [goal("Marcus Capps", assists=[{"name": "Dave Mohr"}])], NAMES)
    with pytest.raises(ValueError):
        db.import_events(conn, game, [goal("Ann Lee"), goal("Nobody Here")], NAMES)
    assert count(conn, "events") == 1 and count(conn, "event_assists") == 1 and count(conn, "game_players") == 2


def test_failed_import_team_rolls_back_all_rows(conn, tmp_path):
    p = _team(tmp_path / "t", "x", {"name": "X", "listed_roster": [
        {"jersey": 1, "name": "A One"}, {"jersey": "bad", "name": "B Two"}]})
    with pytest.raises(ValueError):
        db.import_team(conn, p, S)
    assert count(conn, "teams") == 0 and count(conn, "people") == 0


def test_merge_rollback_on_clash_changes_nothing(conn, game):
    db.import_events(conn, game, [goal("Marcus Capps", assists=[{"name": "Dave Mohr"}])], NAMES)
    keep, drop = pid(conn, "Marcus", "Capps"), pid(conn, "Dave", "Mohr")
    with pytest.raises(ValueError):
        db.merge_people(conn, keep, drop)
    assert count(conn, "people") == 5
    assert conn.execute("SELECT person_id FROM event_assists").fetchone()[0] == drop


def _v1_db(path):
    c = sqlite3.connect(path, isolation_level=None)
    c.executescript(db.MIGRATIONS[0])
    c.execute("PRAGMA user_version = 1")
    c.execute("INSERT INTO teams (id, name, source) VALUES ('t', 'T', 'user')")
    c.execute("INSERT INTO people (first_name, last_name, source) VALUES ('A', 'B', 'user')")
    c.execute("INSERT INTO roster_entries (team_id, season, person_id, jersey, source) VALUES ('t', ?, 1, 8, 'user')", (S,))
    return c


def test_migration_2_upgrades_v1_and_keeps_data(tmp_path):
    _v1_db(tmp_path / "v1.db").close()
    c = db.connect(tmp_path / "v1.db")
    assert c.execute("PRAGMA user_version").fetchone()[0] == len(db.MIGRATIONS)
    r = c.execute("SELECT jersey, jersey_label FROM roster_entries").fetchone()
    assert (r[0], r[1]) == (8, None)


def test_failed_migration_2_rolls_back_ddl(tmp_path):
    c = _v1_db(tmp_path / "v1.db")
    c.execute("ALTER TABLE events ADD COLUMN match_note TEXT")  # last ALTER of migration 2 will fail
    c.close()
    with pytest.raises(sqlite3.OperationalError):
        db.connect(tmp_path / "v1.db")
    c = sqlite3.connect(tmp_path / "v1.db")
    assert c.execute("PRAGMA user_version").fetchone()[0] == 1
    cols = [r[1] for r in c.execute("PRAGMA table_info(roster_entries)")]
    assert "jersey_label" not in cols


def test_newer_schema_is_refused_and_untouched(tmp_path):
    c = sqlite3.connect(tmp_path / "new.db")
    c.execute(f"PRAGMA user_version = {len(db.MIGRATIONS) + 1}")
    c.close()
    with pytest.raises(RuntimeError, match="Update the code"):
        db.connect(tmp_path / "new.db")
    assert sqlite3.connect(tmp_path / "new.db").execute("PRAGMA user_version").fetchone()[0] == len(db.MIGRATIONS) + 1


# --- merge_people -----------------------------------------------------------------------------

def _two_team_same_player(conn, tmp_path):
    root = tmp_path / "m"
    db.import_team(conn, _team(root, "a", {"name": "A", "listed_roster": [{"jersey": 5, "name": "Pat Doe"},
                                                                          {"jersey": 6, "name": "Kim Doe"}],
                                           "family_links": [{"parent": {"name": "Kim Doe"}, "child": {"name": "Pat Doe"},
                                                             "relation": "parent"}]}), S)
    db.import_team(conn, _team(root, "b", {"name": "B", "listed_roster": [{"jersey": 15, "name": "Pat Doe"},
                                                                          {"jersey": 16, "name": "Kim Doe"}],
                                           "family_links": [{"parent": {"name": "Kim Doe"}, "child": {"name": "Pat Doe"},
                                                             "relation": "parent"}]}), S)
    ids = lambda n: [r[0] for r in conn.execute(
        "SELECT p.id FROM people p JOIN roster_entries r ON r.person_id = p.id WHERE first_name = ? ORDER BY r.team_id", (n,))]
    return ids("Pat"), ids("Kim")


def test_merge_dedupes_family_links_and_keeps_fk_integrity(conn, tmp_path):
    (pa, pb), (ka, kb) = _two_team_same_player(conn, tmp_path)
    db.merge_people(conn, pa, pb)
    db.merge_people(conn, ka, kb)
    assert [tuple(r) for r in conn.execute("SELECT parent_id, child_id FROM family_links")] == [(ka, pa)]
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert db.people_flags(conn) == []


def test_merge_does_not_leave_a_self_link(conn, tmp_path):
    db.import_team(conn, _team(tmp_path / "f", "c", {"name": "C", "listed_roster": [{"jersey": 1, "name": "Kid Ray"}],
                   "family_links": [{"parent": {"name": "Dad Ray"}, "child": {"name": "Kid Ray"}, "relation": "parent"}]}), S)
    # A wrong merge of parent into child must refuse or not create a parent-of-self row.
    try:
        db.merge_people(conn, pid(conn, "Kid", "Ray"), pid(conn, "Dad", "Ray"))
    except ValueError:
        return
    assert conn.execute("SELECT COUNT(*) FROM family_links WHERE parent_id = child_id").fetchone()[0] == 0


def test_merge_moves_goals_and_assists_across_games(conn, game, tmp_path):
    db.import_events(conn, game, [goal("Marcus Capps", assists=[{"name": "Dave Mohr"}])], NAMES)
    root = tmp_path / "other"
    db.import_team(conn, _team(root, "c", {"name": "C", "listed_roster": [{"jersey": 3, "name": "Dave Mohr"},
                                                                          {"jersey": 4, "name": "Al Fox"}]}), S)
    db.add_game(conn, "g2", "2026-10-12", ["c", "blb"])
    db.import_events(conn, "g2", [goal("Al Fox", team="C", assists=[{"name": "Dave Mohr"}])], {"C": "c"})
    rows = [r[0] for r in conn.execute("SELECT id FROM people WHERE last_name = 'Mohr' ORDER BY id")]
    db.merge_people(conn, rows[0], rows[1])
    assert conn.execute("SELECT COUNT(*) FROM event_assists WHERE person_id = ?", (rows[0],)).fetchone()[0] == 2
    assert db.snapshot(conn, "g2")["stats"]["c"][1]["name"] == "Dave Mohr"


def test_merge_scorer_and_assister_same_game_refused(conn, game):
    db.import_events(conn, game, [goal("Marcus Capps", assists=[{"name": "Ann Lee"}])], NAMES)
    with pytest.raises(ValueError):
        db.merge_people(conn, pid(conn, "Marcus", "Capps"), pid(conn, "Ann", "Lee"))


def test_merge_unknown_drop_is_an_error(conn, game):
    with pytest.raises((ValueError, KeyError)):
        db.merge_people(conn, pid(conn, "Ann", "Lee"), 9999)


# --- people_flags -----------------------------------------------------------------------------

def test_people_flags_same_name_two_teams(conn, game, tmp_path):
    db.import_team(conn, _team(tmp_path / "f", "c", {"name": "C", "listed_roster": [{"jersey": 1, "name": "Ann Lee"}]}), S)
    assert len(db.people_flags(conn)) == 1


def test_people_flags_same_team_namesakes_not_flagged(conn, tmp_path):
    db.import_team(conn, _team(tmp_path / "f", "c", {"name": "C", "listed_roster": [
        {"jersey": 1, "name": "Ann Lee"}, {"jersey": 2, "name": "Ann Lee"}]}), S)
    assert db.people_flags(conn) == []


def test_people_flags_third_record_on_other_team_still_flagged(conn, tmp_path):
    db.import_team(conn, _team(tmp_path / "f", "c", {"name": "C", "listed_roster": [
        {"jersey": 1, "name": "Ann Lee"}, {"jersey": 2, "name": "Ann Lee"}]}), S)
    db.import_team(conn, _team(tmp_path / "f", "d", {"name": "D", "listed_roster": [{"jersey": 3, "name": "Ann Lee"}]}), S)
    assert len(db.people_flags(conn)) == 1  # the team-d record may be one of the team-c people


# --- _family_member ---------------------------------------------------------------------------

def test_off_roster_relative_reimport_is_idempotent(conn, tmp_path):
    cfg = {"name": "C", "listed_roster": [{"jersey": 1, "name": "Kid Ray"}],
           "family_links": [{"parent": {"name": "Dad Ray"}, "child": {"name": "Kid Ray"}, "relation": "parent"}]}
    p = _team(tmp_path / "f", "c", cfg)
    db.import_team(conn, p, S)
    db.import_team(conn, p, S)
    db.import_team(conn, p, "2027-2028")
    assert count(conn, "people") == 2 and count(conn, "family_links") == 1


# --- _match_for_import ------------------------------------------------------------------------

def test_reimport_new_season_number_change_keeps_person(conn, tmp_path):
    root = tmp_path / "r"
    db.import_team(conn, _team(root, "c", {"name": "C", "listed_roster": [{"jersey": 10, "name": "Sam Hill"}]}), S)
    old = pid(conn, "Sam", "Hill")
    db.import_team(conn, _team(root, "c", {"name": "C", "listed_roster": [{"jersey": 11, "name": "Sam Hill"}]}), "2027-2028")
    assert count(conn, "people") == 1 and pid(conn, "Sam", "Hill") == old


def test_new_namesake_does_not_steal_identity_by_row_order(conn, tmp_path):
    root = tmp_path / "r"
    db.import_team(conn, _team(root, "c", {"name": "C", "listed_roster": [{"jersey": 10, "name": "Sam Hill"}]}), S)
    old = pid(conn, "Sam", "Hill")
    db.import_team(conn, _team(root, "c", {"name": "C", "listed_roster": [
        {"jersey": 30, "name": "Sam Hill"}, {"jersey": 10, "name": "Sam Hill"}]}), "2027-2028")
    got = conn.execute("SELECT person_id FROM roster_entries WHERE season = '2027-2028' AND jersey = 10").fetchone()[0]
    assert got == old


def test_namesakes_both_renumbered_is_not_silent_duplicate(conn, tmp_path):
    root = tmp_path / "r"
    rows = [{"jersey": 10, "name": "Sam Hill"}, {"jersey": 20, "name": "Sam Hill"}]
    db.import_team(conn, _team(root, "c", {"name": "C", "listed_roster": rows}), S)
    new = [{"jersey": 11, "name": "Sam Hill"}, {"jersey": 20, "name": "Sam Hill"}]
    try:
        db.import_team(conn, _team(root, "c", {"name": "C", "listed_roster": new}), "2027-2028")
    except ValueError:
        return  # refusing is correct
    # Otherwise the extra record must at least be flagged for review.
    assert count(conn, "people") == 2 or db.people_flags(conn)


# --- _roster_season / find_person --------------------------------------------------------------

def test_roster_season_picks_latest_earlier_across_decades(conn, tmp_path):
    root = tmp_path / "s"
    for season, j in (("2009-2010", 1), ("2010-2011", 2), ("2025-2026", 3)):
        db.import_team(conn, _team(root, "c", {"name": "C", "listed_roster": [{"jersey": j, "name": "Sam Hill"}]}), season)
    assert db._roster_season(conn, "c", S) == "2025-2026"
    assert db._roster_season(conn, "c", "2010-2011") == "2010-2011"


def test_game_before_only_roster_raises_clear_error(conn, tmp_path):
    db.import_team(conn, _team(tmp_path / "s", "c", {"name": "C", "listed_roster": [{"jersey": 1, "name": "Sam Hill"}]}),
                   "2027-2028")
    db.add_game(conn, "old", "2026-10-11", ["c"])
    assert "c: no roster for 2026-2027 or earlier" in db.game_flags(conn, "old")
    with pytest.raises(ValueError, match="not on the 2026-2027 roster"):
        db.import_events(conn, "old", [goal("Sam Hill", team="C")], {"C": "c"})


def test_suggestion_lists_close_name_and_jersey(conn, game):
    with pytest.raises(ValueError) as e:
        db.find_person(conn, "icepak", S, name="Marcus Caps", jersey=4)
    assert "#8 Marcus Capps" in str(e.value) and "#4 Dave Mohr" in str(e.value)


def test_namesakes_with_wrong_jersey_do_not_say_not_on_roster(conn, tmp_path):
    db.import_team(conn, _team(tmp_path / "f", "c", {"name": "C", "listed_roster": [
        {"jersey": 1, "name": "Ann Lee"}, {"jersey": 2, "name": "Ann Lee"}]}), S)
    with pytest.raises(ValueError) as e:
        db.find_person(conn, "c", S, name="Ann Lee", jersey=3)
    assert "is not on the" not in str(e.value)


# --- _assist_ids ------------------------------------------------------------------------------

def test_opponent_assist_is_named(conn, game):
    with pytest.raises(ValueError, match="plays for blb"):
        db.import_events(conn, game, [goal("Marcus Capps", assists=[{"name": "Kenyon Ross"}])], NAMES)


def test_shared_name_on_both_teams_assist_resolves_own_team(conn, game, tmp_path):
    db.import_events(conn, game, [goal("Jenner Smith", team="Buzz Light Beer", assists=[{"jersey": 77}])], NAMES)
    assert count(conn, "event_assists") == 1


def test_goal_without_scorer_is_value_error(conn, game):
    with pytest.raises(ValueError):
        db.import_events(conn, game, [goal(None)], NAMES)


# --- parse_jersey / parse_minutes ---------------------------------------------------------------

@pytest.mark.parametrize("value,want", [("0", (0, None)), ("00", (0, "00")), ("#08", (8, "08")), (8, (8, None))])
def test_parse_jersey_ok(value, want):
    assert db.parse_jersey(value) == want


@pytest.mark.parametrize("value", ["-5", "+", "8.0", 8.0, 8.5, "1 2", -5, "²", "##8"])
def test_parse_jersey_rejects(value):
    with pytest.raises(ValueError):
        db.parse_jersey(value)


@pytest.mark.parametrize("value,want", [("0", 0), ("2 + 10", 12), ("2+10", 12), ("5:00", 5), (2.5, 2.5), ("10:00", 10)])
def test_parse_minutes_ok(value, want):
    assert db.parse_minutes(value) == want


@pytest.mark.parametrize("value", ["+", "2+", "1:5", "1:60", "-2", -2, float("nan"), float("inf"), True])
def test_parse_minutes_rejects(value):
    with pytest.raises(ValueError):
        db.parse_minutes(value)


# --- source_of --------------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["User-confirmed 2026-10-14 by Troy", "import x.json", "sheet", "override  game 1"])
def test_source_of_accepts_real_patterns(value):
    assert db.source_of(value) == " ".join(value.split())


@pytest.mark.parametrize("value", ["username", "importer", "sheetrock", "guess"])
def test_source_of_requires_word_boundary(value):
    with pytest.raises(ValueError):
        db.source_of(value)


# --- snapshot ---------------------------------------------------------------------------------

def test_snapshot_ties_follow_first_involvement(conn, game):
    db.import_events(conn, game, [goal("Ann Lee"), goal("Marcus Capps"), goal("Dave Mohr")], NAMES)
    assert [r["name"] for r in db.snapshot(conn, game)["stats"]["icepak"]] == ["Ann Lee", "Marcus Capps", "Dave Mohr"]


def test_snapshot_team_pim_counts_additional_and_bench(conn, game):
    db.import_events(conn, game, [
        {"event": "PENALTY", "team": "Ice Pak", "player": "Ann Lee", "minutes": 2,
         "additional": [{"classification": "misconduct", "minutes": 10}]},
        {"event": "PENALTY", "team": "Ice Pak", "minutes": "2", "classification": "bench minor"}], NAMES)
    snap = db.snapshot(conn, game)
    assert snap["team_pim"]["icepak"] == 14
    assert snap["stats"]["icepak"][0]["pim"] == 12


def test_snapshot_roster_jersey_fallback_and_label(conn, tmp_path):
    db.import_team(conn, _team(tmp_path / "j", "c", {"name": "C", "listed_roster": [{"jersey": "08", "name": "Sam Hill"}]}), S)
    db.add_game(conn, "g", "2026-10-11", ["c"])
    db.import_events(conn, "g", [goal("Sam Hill", team="C")], {"C": "c"})
    p = db.snapshot(conn, "g")["events"][0]["person"]
    assert (p["jersey"], p["jersey_label"]) == (8, "08")


def test_reimport_with_additional_penalties_under_fk(conn, game):
    ev = [{"event": "PENALTY", "team": "Ice Pak", "player": "Ann Lee", "minutes": 5,
           "additional": [{"classification": "game misconduct", "minutes": 10}, {"classification": "misconduct", "minutes": 10}]}]
    db.import_events(conn, game, ev, NAMES)
    db.import_events(conn, game, ev, NAMES)
    assert count(conn, "events") == 3
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
