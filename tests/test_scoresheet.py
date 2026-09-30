# tests/test_scoresheet.py
"""Pure checks of scoresheet.py. No model, no OCR: inputs are what the reads return."""
import scoresheet as ss


ROSTER = {"47": "Andersen", "9": "E. Capps", "17": "G. Capps", "57": "Troy Johnson", "00": "Ben Smith"}


def goal(per="2", time="1:43", scorer="47", assist="57", typ="PP"):
    return {"goal": "1", "per": per, "time": time, "scorer": scorer, "assist": assist, "type": typ}


def pen(per="2", player="73", minutes="2", off="1:52", start="1:52", on="1:43"):
    return {"per": per, "player": player, "infraction": "trip", "minutes": minutes,
            "off": off, "start": start, "on": on}


def test_parse_time():
    assert ss.parse_time("9:30") == 570
    assert ss.parse_time("9.30") == 570
    assert ss.parse_time("0i50") == 50
    assert ss.parse_time("1:75") is None
    assert ss.parse_time("") is None


def test_normalize_goal_type():
    assert ss.normalize_goal_type("es") == ("ES", False)
    assert ss.normalize_goal_type("BP") == ("PP", True)
    assert ss.normalize_goal_type("") == ("", False)


def test_split_assists():
    assert ss.split_assists("9-47") == ["9", "47"]
    assert ss.split_assists("-") == []
    assert ss.split_assists(" 9 / 47 ") == ["9", "47"]


def test_check_goals_ok_and_skips_empty_rows():
    rows = ss.check_goals([goal(typ="PP"), goal(per="", time="", scorer="", assist="", typ="")], ROSTER)
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["time_s"] == 103
    assert rows[0]["assists"] == ["57"]


def test_check_goals_flags_bad_cells():
    [row] = ss.check_goals([goal(per="4", time="1:99", scorer="88", typ="BP")], ROSTER)
    assert row["status"] == "review"
    text = " ".join(row["reasons"])
    assert "period" in text and "clock time" in text and "#88" in text and "read as PP" in text


def test_check_penalties_ok():
    [row] = ss.check_penalties([pen(player="9", start="6:06", on="4:06")], ROSTER)
    assert row["status"] == "ok"


def test_check_penalties_ended_by_power_play_goal():
    opp = ss.check_goals([goal(per="2", time="1:43")], ROSTER)
    roster = {"73": ""}
    [ended] = ss.check_penalties([pen()], roster, opp)
    assert ended["status"] == "ok"
    [unexplained] = ss.check_penalties([pen()], roster, [])
    assert unexplained["status"] == "review"
    assert "not 2 min apart" in unexplained["reasons"][0]


def test_power_play_goal_in_other_period_does_not_end_penalty():
    opp = ss.check_goals([goal(per="3", time="1:43")], ROSTER)
    [row] = ss.check_penalties([pen()], {"73": ""}, opp)
    assert row["status"] == "review"


def test_mark_disagreements():
    first = ss.check_goals([goal(time="1:10")], ROSTER)
    second = ss.check_goals([goal(time="1:16"), goal(time="2:00")], ROSTER)
    out = ss.mark_disagreements(first, second)
    assert len(out) == 2
    assert out[0]["status"] == "review" and "reads disagree on time" in out[0]["reasons"][0]
    assert out[1]["reasons"][-1] == "only the second read found this row"


def test_check_score_by_periods():
    away = ss.check_goals([goal(per="1"), goal(per="2"), goal(per="3")], ROSTER)
    periods = {"home": ["0", "6", "", "", ""], "away": ["1", "1", "", "", ""]}
    flags = ss.check_score_by_periods(periods, [], away)
    assert flags == ["home period 2: table says 6, goal rows count 0",
                     "away period 3: table empty, 1 goal row(s)"]


def test_compare_periods():
    a = {"home": ["0", "0", ""], "away": ["1", "3"]}
    b = {"home": ["0", "6", ""], "away": ["1", "3", "", "", ""]}
    assert ss.compare_periods(a, b) == ["home period 2: reads disagree '0' vs '6'"]


def test_merge_roster_adds_handwritten_numbers_and_snaps_names():
    printed = {"47": "Andersen", "15": "Chad Linville"}
    model = [{"number": "57", "name": "Troy Johnson", "other_numbers": []},
             {"number": "15", "name": "Chad Linvile", "other_numbers": ["142"]},
             {"number": "7", "name": "", "other_numbers": []},
             {"number": "Blake", "name": "Blake Moss"},
             "junk"]
    roster = ss.merge_roster(printed, model)
    assert roster["57"] == "Troy Johnson"
    assert roster["142"] == "Chad Linville"
    assert roster["7"] == ""
    assert roster["47"] == "Andersen"
    assert "Blake" not in roster


def test_merge_roster_tolerates_bad_model_output():
    assert ss.merge_roster({"2": "Ned White"}, {"error": "x"}) == {"2": "Ned White"}
    assert ss.merge_roster({"2": "Ned White"}, None) == {"2": "Ned White"}


def test_parse_rosters_groups_boxes_into_rows():
    lines = [("HOME: Salty Boys", (0.05, 0.10, 0.10, 0.02)),
             ("2", (0.05, 0.14, 0.01, 0.015)), ("Ned White", (0.07, 0.14, 0.06, 0.015)),
             ("15 Chad Linville 142", (0.05, 0.17, 0.10, 0.015)),
             ("AWAY: Ice Pak", (0.55, 0.10, 0.10, 0.02)),
             ("47 Andersen", (0.55, 0.14, 0.07, 0.015))]
    assert ss.parse_team_names(lines) == {"home": "Salty Boys", "away": "Ice Pak"}
    rosters = ss.parse_rosters(lines)
    assert rosters["home"] == {"2": "Ned White", "15": "Chad Linville", "142": "Chad Linville"}
    assert rosters["away"] == {"47": "Andersen"}


def test_roster_boxes():
    lines = [("HOME: A", (0.05, 0.10, 0.10, 0.02)), ("AWAY: B", (0.55, 0.10, 0.10, 0.02))]
    tables = {"home_scoring": (300, 150, 400, 300), "away_scoring": (1300, 150, 400, 300)}
    boxes = ss.roster_boxes(lines, tables, 2000, 1000)
    assert boxes["home_roster"] == (0, 100, 300, 350)
    assert boxes["away_roster"] == (1700, 100, 100, 350)
    assert boxes["home_margin"] == (0, 100, 60, 350)
    assert boxes["away_margin"] == (1770, 100, 30, 350)


def test_find_scoresheet_photos_skips_pipeline_images(tmp_path):
    (tmp_path / "IMG_9973.jpg").write_bytes(b"x" * 10)
    (tmp_path / "scoresheet_periods_b.png").write_bytes(b"x" * 100)
    (tmp_path / "rois_preview.png").write_bytes(b"x" * 100)
    (tmp_path / "scoresheet_crops").mkdir()
    assert ss.find_scoresheet_photos(str(tmp_path)) == [str(tmp_path / "IMG_9973.jpg")]
