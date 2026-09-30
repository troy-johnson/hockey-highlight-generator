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
