import gamesheet_pdf as gp

ROSTERS = {"home": {"5": {"name": "Eric Janes"}, "17": {"name": "Nick Gavern"}, "39": {"name": "Tyson"}},
           "away": {"8": {"name": "Marcus Capps"}, "57": {"name": "Troy Johnson"}, "34": {"name": "M W"}}}
META = {"photo": "x.pdf", "date": "Aug 9, 2026", "teams": {"home": "Salty Boys", "away": "Ice Pak"},
        "final_score": {"home": 1, "away": 2}}


def test_goal_row_joins_assists():
    assert gp.goal_row(["2", "01:41", "8", "57", ""]) == \
        {"per": "2", "time": "01:41", "scorer": "8", "assist": "57", "type": ""}


def test_penalty_row_maps_start_and_off():
    row = gp.penalty_row(["2", "17", "2", "INT", "", "01:41", "02:10", ""])
    assert (row["off"], row["start"], row["on"], row["infraction"]) == ("02:10", "02:10", "01:41", "INT")


def test_penalty_row_with_all_three_times():
    row = gp.penalty_row(["2", "2", "2", "SL-MIN", "", "11:23", "11:23", "09:23"])
    assert (row["off"], row["start"], row["on"]) == ("11:23", "11:23", "09:23")


def test_penalty_cells_take_code_from_row_text():
    cells = gp.penalty_cells("3 17 2 RGH-MIN 08:28 08:28 06:28", ["3", "17", "2", "GH-MIN", "", "08:28", "08:28", "06:28"])
    assert cells == ["3", "17", "2", "RGH-MIN", "", "08:28", "08:28", "06:28"]


def test_sheet_from_tables_ok_and_power_play_end():
    tables = {"scoring": {"home": [["1", "01:53", "39", "17", ""]],
                          "away": [["1", "12:24", "34", "", ""], ["2", "01:41", "8", "57", ""]]},
              "penalties": {"home": [["2", "17", "2", "INT", "", "01:41", "02:10", ""]], "away": []}}
    s = gp.sheet_from_tables(META, ROSTERS, tables)
    assert s["flags"] == []
    assert [g["status"] for g in s["goals"]["away"]] == ["ok", "ok"]
    assert s["penalties"]["home"][0]["status"] == "ok"      # ended by the away goal at 01:41


def test_final_score_mismatch_is_flagged():
    assert gp.check_final_score({"home": 1, "away": 5}, {"home": [{}], "away": [{}]}) == \
        ["away: final score 5, goal rows count 1"]
    assert gp.check_final_score(None, {"home": [], "away": []}) == ["final score not found"]


def test_goal_types_from_players_in_the_box():
    tables = {"scoring": {"home": [["1", "01:53", "39", "17", ""], ["1", "08:00", "5", "", ""]],
                          "away": [["1", "12:24", "34", "", ""], ["2", "01:41", "8", "57", ""]]},
              "penalties": {"home": [["2", "17", "2", "INT", "", "01:41", "02:10", ""],
                                     ["1", "5", "2", "TR", "", "07:00", "09:00", ""]], "away": []}}
    s = gp.sheet_from_tables(META, ROSTERS, tables)
    assert [g["type"] for g in s["goals"]["away"]] == ["ES", "PP"]   # PP goal at 01:41 ends the minor
    assert [g["type"] for g in s["goals"]["home"]] == ["ES", "SH"]   # #5 scores at 08:00 while own #5 sits


def test_goal_one_second_after_the_on_time_is_still_power_play():
    tables = {"scoring": {"home": [["3", "13:30", "39", "", ""]], "away": []},
              "penalties": {"home": [], "away": [["3", "34", "2", "TR", "", "13:29", "15:29", ""]]}}
    s = gp.sheet_from_tables({**META, "final_score": {"home": 1, "away": 0}}, ROSTERS, tables)
    assert s["goals"]["home"][0]["type"] == "PP"
