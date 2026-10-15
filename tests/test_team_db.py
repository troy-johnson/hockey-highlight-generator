# tests/test_team_db.py — team, Roster, and game data in SQLite (ADR-0004, hhg-3r5.107)
import json

import pytest

import team_db as db


@pytest.fixture
def conn(tmp_path):
    return db.connect(tmp_path / "hockey.db")


@pytest.fixture
def teams(tmp_path):
    root = tmp_path / "teams"
    root.mkdir()
    (root / "icepak.json").write_text(json.dumps({
        "name": "Ice Pak", "short_name": "IPK", "own_team": True,
        "colors": {"primary": "#1D3160", "secondary": "#FFFFFF"}, "kits": {"red": {"fill": "#A60735"}},
        "logo": "/nas/icepak.png", "team_manager": "Sample Manager",
        "listed_roster": [
            {"row": 1, "jersey": 8, "first_name": "Marcus", "last_name": "Capps"},
            {"row": 2, "jersey": 4, "first_name": "Dave", "last_name": "Mohr"},
            {"row": 3, "jersey": None, "first_name": "Chris", "last_name": "Browne"},
            {"row": 4, "jersey": 56, "first_name": "John David", "last_name": "Mohr", "display_name": "JD Mohr"},
        ],
        "family_links": [{"parent": {"jersey": 4, "name": "Dave Mohr"}, "child": {"jersey": 56, "name": "JD Mohr"},
                          "relation": "father", "source": "user"}],
    }))
    (root / "blb.json").write_text(json.dumps({
        "name": "Buzz Light Beer", "short_name": "BLB",
        "listed_roster": [
            {"jersey": 8, "first_name": "Jenner", "last_name": "Smith"},
            {"jersey": 77, "first_name": "Justin", "last_name": "Davidson"},
            {"jersey": 77, "first_name": "Kenyon", "last_name": "Ross"},
            {"jersey": 60, "first_name": "Trevor", "last_name": "Tilk", "position": "G"},
        ],
    }))
    return root


@pytest.fixture
def game(conn, teams):
    for t in ("icepak", "blb"):
        db.import_team(conn, teams / f"{t}.json", "2026-2027")
    db.add_game(conn, "blb_10112026", "2026-10-11", ["icepak", "blb"], focus="icepak", folder="/games/blb")
    events = [
        {"event": "GOAL", "team": "Ice Pak", "scene": 2, "scorer": "Marcus Capps", "jersey": 8, "assists": []},
        {"event": "GOAL", "team": "Ice Pak", "scene": 6, "scorer": "JD Mohr", "jersey": 56,
         "assists": [{"name": "Chris Browne", "jersey": None}]},
        {"event": "PENALTY", "team": "Buzz Light Beer", "scene": 13, "player": "Kenyon Ross", "jersey": 7,
         "infraction": "BUTT-ENDING", "classification": "MAJOR", "minutes": 5,
         "additional": [{"classification": "GAME MISCONDUCT", "minutes": None}]},
        {"event": "GOAL", "team": "Buzz Light Beer", "scene": 10, "scorer": None, "jersey": 8, "assists": []},
    ]
    db.import_events(conn, "blb_10112026", events, {"Ice Pak": "icepak", "Buzz Light Beer": "blb"})
    return "blb_10112026"


def test_migrations_are_applied_once(tmp_path):
    path = tmp_path / "hockey.db"
    db.connect(path).close()
    conn = db.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == len(db.MIGRATIONS)


@pytest.mark.parametrize("date, season", [("2026-10-11", "2026-2027"), ("2027-03-01", "2026-2027"),
                                          ("2027-08-01", "2027-2028")])
def test_season_for(date, season):
    assert db.season_for(date) == season


def test_import_team_keeps_identity_assets_and_extra_fields(conn, teams):
    db.import_team(conn, teams / "icepak.json", "2026-2027")
    row = conn.execute("SELECT * FROM teams WHERE id = 'icepak'").fetchone()
    assert json.loads(row["kits"])["red"]["fill"] == "#A60735"
    assert json.loads(row["assets"]) == {"logo": "/nas/icepak.png"}
    assert json.loads(row["extra"])["team_manager"] == "Sample Manager"


def test_import_team_is_idempotent(conn, teams):
    for _ in range(2):
        db.import_team(conn, teams / "icepak.json", "2026-2027")
    assert conn.execute("SELECT COUNT(*) FROM roster_entries").fetchone()[0] == 4
    assert conn.execute("SELECT COUNT(*) FROM people").fetchone()[0] == 4


def test_display_name_and_family_link(conn, teams):
    db.import_team(conn, teams / "icepak.json", "2026-2027")
    jd = db.find_person(conn, "icepak", "2026-2027", name="JD Mohr")
    assert jd == db.find_person(conn, "icepak", "2026-2027", name="John David Mohr")
    link = conn.execute("SELECT * FROM family_links").fetchone()
    assert link["child_id"] == jd and link["relation"] == "father"


def test_shared_jersey_needs_a_name(conn, teams):
    db.import_team(conn, teams / "blb.json", "2026-2027")
    with pytest.raises(ValueError, match="shared"):
        db.find_person(conn, "blb", "2026-2027", jersey=77)
    assert db.find_person(conn, "blb", "2026-2027", name="Kenyon Ross", jersey=77)


def test_snapshot_stats_score_and_numbers_worn(conn, game):
    snap = db.snapshot(conn, game)
    assert snap["score"] == {"icepak": 2, "blb": 1}
    ipk = {r["name"]: r for r in snap["stats"]["icepak"]}
    assert (ipk["JD Mohr"]["g"], ipk["JD Mohr"]["jersey"]) == (1, 56)
    assert (ipk["Chris Browne"]["a"], ipk["Chris Browne"]["jersey"]) == (1, None)
    blb = {r["name"]: r for r in snap["stats"]["blb"]}
    assert blb["Kenyon Ross"]["pim"] == 5          # game misconduct has no minutes
    assert blb["Jenner Smith"]["g"] == 1           # matched by jersey only
    goals = [e for e in snap["events"] if e["kind"] == "goal"]
    assert goals[0]["unassisted"] and not goals[1]["unassisted"]


def test_number_worn_wins_and_is_flagged(conn, game):
    snap = db.snapshot(conn, game)
    pen = next(e for e in snap["events"] if e["classification"] == "MAJOR")
    assert pen["person"]["jersey"] == 7
    flags = db.game_flags(conn, game)
    assert any("Kenyon Ross wore #7" in f and "#77" in f for f in flags)
    assert any("#77 is shared by 2" in f for f in flags)
    assert any("Chris Browne has no number" in f for f in flags)
    assert any("GAME MISCONDUCT has no minutes" in f for f in flags)


def test_misconduct_is_linked_to_its_major(conn, game):
    rows = conn.execute("SELECT id, parent_event_id, classification FROM events WHERE kind = 'penalty' "
                        "ORDER BY id").fetchall()
    assert rows[1]["parent_event_id"] == rows[0]["id"] and rows[1]["classification"] == "GAME MISCONDUCT"


def test_reimport_replaces_events(conn, game):
    db.import_events(conn, game, [{"event": "GOAL", "team": "Ice Pak", "scorer": "Marcus Capps", "jersey": 8}],
                     {"Ice Pak": "icepak", "Buzz Light Beer": "blb"})
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


def test_unknown_number_without_name_is_rejected(conn, game):
    with pytest.raises(ValueError, match="not on the 2026-2027 roster"):
        db.import_events(conn, game, [{"event": "GOAL", "team": "Ice Pak", "scorer": None, "jersey": 99}],
                         {"Ice Pak": "icepak"})


def test_foreign_keys_are_enforced(conn):
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        with conn:
            conn.execute("INSERT INTO roster_entries (team_id, season, person_id, source) VALUES ('none', 's', 99, 'x')")


def test_write_snapshot_and_cli(tmp_path, conn, game):
    out = db.write_snapshot(conn, game, tmp_path / "game_data.json")
    data = json.loads(out.read_text())
    assert data["schema_version"] == db.SNAPSHOT_VERSION and data["game"]["season"] == "2026-2027"
    assert db.main(["--db", str(tmp_path / "hockey.db"), "flags", game]) == 0
