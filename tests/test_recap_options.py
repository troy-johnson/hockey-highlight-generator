# tests/test_recap_options.py — option layers of hockeyrecap (hhg-3r5.28)
import json

import pytest

import recap_options as ro


@pytest.fixture
def cfg(tmp_path):
    root = tmp_path / "hockey"
    (root / "leagues").mkdir(parents=True)
    (root / "teams").mkdir()
    (root / "leagues" / "ahl_rec.json").write_text(json.dumps({
        "periods": 3, "period_length_min": 15,
        "options": {"live_play_speed": 1.2, "detection": {"thresh_pct": 94, "cooldown_s": 10}},
    }))
    (root / "teams" / "icepak.json").write_text(json.dumps({
        "name": "Ice Pak", "own_team": True, "league": "ahl_rec",
        "options": {"detection": {"thresh_pct": 93}},
    }))
    (root / "teams" / "wild.json").write_text(json.dumps({"name": "Wild", "short_name": "WLD"}))
    return root


@pytest.fixture
def game(tmp_path):
    g = tmp_path / "wild_03012026"
    g.mkdir()
    return g


def test_parse_folder_name():
    assert ro.parse_folder_name("wild_03012026") == ("wild", "2026-03-01")
    assert ro.parse_folder_name("north_stars_12152025") == ("north_stars", "2025-12-15")
    assert ro.parse_folder_name("misc footage") == ("misc footage", None)


def test_parse_set():
    assert ro.parse_set(["detection.thresh_pct=90", "opponent=wild", "x.y=true"]) == {
        "detection": {"thresh_pct": 90}, "opponent": "wild", "x": {"y": True}}
    with pytest.raises(ValueError):
        ro.parse_set(["novalue"])


def test_first_run_creates_game_file_with_inferred_values(game, cfg):
    opts = ro.resolve_options(str(game), cfg_override=str(cfg))
    assert opts["_created"] is True
    data = json.loads((game / ro.OPTIONS_FILE).read_text())
    for key in ro.GAME_KEYS:
        assert data[key]["inferred"] is True, key
        assert data[key]["source"]
    assert data["opponent"]["value"] == "wild"
    assert data["date"]["value"] == "2026-03-01"
    assert data["focus_team"]["value"] == "icepak"
    assert data["league"]["value"] == "ahl_rec"
    assert data["perspective"]["value"] == "focus"
    assert ro.resolve_options(str(game), cfg_override=str(cfg))["_created"] is False


def test_layers_override_in_order(game, cfg):
    opts = ro.resolve_options(str(game), cfg_override=str(cfg))
    assert opts["detection"]["cooldown_s"] == 10        # League
    assert opts["detection"]["thresh_pct"] == 93        # Team Config over League
    assert opts["detection"]["fps"] == 12               # default
    assert opts["league_rules"]["periods"] == 3

    # per-game file over Team Config
    data = json.loads((game / ro.OPTIONS_FILE).read_text())
    data["detection"] = {"thresh_pct": 92}
    data["live_play_speed"] = {"value": 1.0, "inferred": False}
    (game / ro.OPTIONS_FILE).write_text(json.dumps(data))
    opts = ro.resolve_options(str(game), cfg_override=str(cfg))
    assert opts["detection"]["thresh_pct"] == 92
    assert opts["live_play_speed"] == 1.0               # own value kept, not the League's 1.2 or inferred

    # CLI over everything, for one run only
    opts = ro.resolve_options(str(game), cli={"detection": {"thresh_pct": 91}, "perspective": "neutral"},
                              cfg_override=str(cfg))
    assert opts["detection"]["thresh_pct"] == 91
    assert opts["perspective"] == "neutral"
    assert json.loads((game / ro.OPTIONS_FILE).read_text())["detection"]["thresh_pct"] == 92


def test_user_value_is_kept_and_inferred_value_is_updated(game, cfg):
    ro.resolve_options(str(game), cfg_override=str(cfg))
    data = json.loads((game / ro.OPTIONS_FILE).read_text())
    data["opponent"] = {"value": "Minnesota Wild", "inferred": False}
    (game / ro.OPTIONS_FILE).write_text(json.dumps(data))
    (cfg / "teams" / "wild.json").unlink()
    opts = ro.resolve_options(str(game), cfg_override=str(cfg))
    assert opts["opponent"] == "Minnesota Wild"


def test_no_config_dir_gives_defaults_and_flags(game, tmp_path):
    opts = ro.resolve_options(str(game), cfg_override=str(tmp_path / "nothing"))
    assert opts["focus_team"] is None
    assert opts["perspective"] == "neutral"
    assert opts["live_play_speed"] == 1.10
    assert opts["detection"] == ro.DEFAULTS["detection"]
    assert any("No Team Config" in f for f in opts["_flags"])


def test_config_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HHG_CONFIG_DIR", str(tmp_path))
    assert ro.config_dir() == tmp_path
    assert ro.config_dir(str(tmp_path / "x")) == tmp_path / "x"


def test_timing_rules_follow_game_and_cli_layers(game, cfg):
    ro.resolve_options(str(game), cfg_override=str(cfg))
    path = game / ro.OPTIONS_FILE
    data = json.loads(path.read_text())
    data["league_rules"] = {"period_minutes": 13, "clock": "stop", "time_direction": "remaining"}
    path.write_text(json.dumps(data))
    options = ro.resolve_options(str(game), cli={"league_rules": {"period_minutes": 12}}, cfg_override=str(cfg))
    assert options["league_rules"] == {
        "periods": 3, "period_length_min": 15, "period_minutes": 12,
        "clock": "stop", "time_direction": "remaining",
    }
    assert json.loads(path.read_text())["league_rules"]["period_minutes"] == 13


def test_game_team_assets_survive_option_resolution(game, cfg):
    ro.resolve_options(str(game), cfg_override=str(cfg))
    path = game / ro.OPTIONS_FILE
    data = json.loads(path.read_text())
    data["teams"] = {"icepak": {"logo": "/assets/crest.png", "lockup": "/assets/lockup.png"}}
    path.write_text(json.dumps(data))
    options = ro.resolve_options(str(game), cli={"teams": {"icepak": {"wordmark": "/assets/wordmark.png"}}},
                                 cfg_override=str(cfg))
    assert options["teams"]["icepak"]["name"] == "Ice Pak"
    assert options["teams"]["icepak"]["logo"] == "/assets/crest.png"
    assert options["teams"]["icepak"]["lockup"] == "/assets/lockup.png"
    assert options["teams"]["icepak"]["wordmark"] == "/assets/wordmark.png"


def test_kit_is_inferred_from_the_focus_team_default_kit(game, cfg):
    team = json.loads((cfg / "teams" / "icepak.json").read_text())
    team["default_kit"] = "red"
    (cfg / "teams" / "icepak.json").write_text(json.dumps(team))
    opts = ro.resolve_options(str(game), cfg_override=str(cfg))
    data = json.loads((game / ro.OPTIONS_FILE).read_text())
    assert data["kit"] == {"value": "red", "inferred": True, "source": "default_kit of icepak"}
    assert opts["kit"] == "red"
    data["kit"] = {"value": "navy", "inferred": False, "source": "user"}
    (game / ro.OPTIONS_FILE).write_text(json.dumps(data))
    assert ro.resolve_options(str(game), cfg_override=str(cfg))["kit"] == "navy"
    assert ro.resolve_options(str(game), cli={"kit": "white"}, cfg_override=str(cfg))["kit"] == "white"


def test_kit_is_empty_without_a_default_kit(game, cfg):
    ro.resolve_options(str(game), cfg_override=str(cfg))
    data = json.loads((game / ro.OPTIONS_FILE).read_text())
    assert data["kit"]["value"] is None
    assert data["kit"]["source"] == "no default_kit in Team Config"


def test_kit_follows_the_selected_focus_team_not_the_inferred_one(game, cfg):
    for tid, kit in (("icepak", "red"), ("pak2", "navy")):
        path = cfg / "teams" / f"{tid}.json"
        team = json.loads(path.read_text()) if path.exists() else {"name": "Pak Two", "own_team": True}
        team["default_kit"] = kit
        path.write_text(json.dumps(team))
    # Two own teams: nothing is inferred until the user pins one.
    ro.resolve_options(str(game), cfg_override=str(cfg))
    data = json.loads((game / ro.OPTIONS_FILE).read_text())
    assert data["kit"]["value"] is None
    data["focus_team"] = {"value": "pak2", "inferred": False, "source": "user"}
    (game / ro.OPTIONS_FILE).write_text(json.dumps(data))
    assert ro.resolve_options(str(game), cfg_override=str(cfg))["kit"] == "navy"
    assert json.loads((game / ro.OPTIONS_FILE).read_text())["kit"]["value"] == "navy"
    # A CLI Focus Team changes the kit for that run only.
    assert ro.resolve_options(str(game), cli={"focus_team": "icepak"}, cfg_override=str(cfg))["kit"] == "red"
    assert json.loads((game / ro.OPTIONS_FILE).read_text())["kit"]["value"] == "navy"
    assert ro.resolve_options(str(game), cli={"focus_team": "icepak", "kit": "white"},
                              cfg_override=str(cfg))["kit"] == "white"
