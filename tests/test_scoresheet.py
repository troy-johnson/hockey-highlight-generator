# tests/test_scoresheet.py
"""Pure checks of scoresheet.py. No model, no OCR: inputs are what the reads return."""
import json

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
    assert ss.normalize_goal_type("ev") == ("ES", False)


def test_normalize_minutes():
    assert ss.normalize_minutes("2:00") == "2"
    assert ss.normalize_minutes("10.00") == "10"
    assert ss.normalize_minutes("2") == "2"
    assert ss.normalize_minutes("7:30") == "7:30"


def test_period_cell_reads_d_as_zero():
    assert ss.period_cell("D") == "0"
    assert ss.period_cell(" 2 ") == "2"
    assert ss.period_cell("") == ""


def test_check_penalties_accepts_minutes_written_as_clock():
    rows = ss.check_penalties([{"per": "1", "player": "7", "infraction": "TRIP", "minutes": "2:00",
                                "off": "11:18", "start": "11:18", "on": "9:18"}], {"7": "A"})
    assert rows[0]["status"] == "ok" and rows[0]["minutes"] == "2"


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


def test_find_scoresheet_photos_skips_non_sheet_images_and_old_outputs(tmp_path):
    # hhg-3r5.28: the largest image used to win, even a recap graphic or a thumbnail.
    (tmp_path / "IMG_2041.HEIC").write_bytes(b"x" * 10)
    for name in ("recap.png", "yt_thumbnail.png", "yt_thumbnail_2.png", "yt_thumb_1.3.1.png",
                 "Still 2026-03-01.png", "score_overlay.png", "GOPR0017.JPG", ".hidden.png"):
        (tmp_path / name).write_bytes(b"x" * 1000)
    for name in ("cam1.mp4", "cam2.mp4", "events.csv", "markers.csv", "markers.edl", "markers.fcpxml"):
        (tmp_path / name).write_bytes(b"x" * 1000)
    (tmp_path / "editprep").mkdir()
    (tmp_path / "editprep" / "big.png").write_bytes(b"x" * 5000)
    assert ss.find_scoresheet_photos(str(tmp_path)) == [str(tmp_path / "IMG_2041.HEIC")]


def test_is_non_sheet_file():
    assert ss.is_non_sheet_file("recap.png")
    assert ss.is_non_sheet_file("markers.pdf")
    assert not ss.is_non_sheet_file("IMG_9973.jpg")
    assert not ss.is_non_sheet_file("GameSheet.pdf")


def test_find_scoresheet_pdfs_skips_old_outputs(tmp_path):
    (tmp_path / "gamesheet.pdf").write_bytes(b"%PDF")
    (tmp_path / "markers.pdf").write_bytes(b"%PDF")
    assert ss.find_scoresheet_pdfs(str(tmp_path)) == [str(tmp_path / "gamesheet.pdf")]


def _read_sheet():
    return {"photo": "IMG_1.jpg", "teams": {"home": "Salty Boys", "away": "Ice Pak"},
            "rosters": {"home": {"73": ""}, "away": {"47": "Andersen", "00": "Ben Smith BLANKV"}},
            "goals": {"home": [], "away": [
                {"per": "1", "time": "9:30", "time_s": 570, "scorer": "47", "assists": [], "type": "ES",
                 "status": "ok", "reasons": []},
                {"per": "2", "time": "1:43", "time_s": 103, "scorer": "47", "assists": ["57"], "type": "PP",
                 "status": "review", "reasons": ["type 'BP' read as PP"]}]},
            "penalties": {"home": [], "away": []},
            "score_by_periods": {"home": [], "away": []}, "flags": ["table not found: home_penalties"]}


def test_overrides_apply_on_every_run_without_a_second_read(tmp_path, monkeypatch):
    (tmp_path / "IMG_1.jpg").write_bytes(b"sheet photo")
    reads = []
    monkeypatch.setattr(ss, "read_scoresheet", lambda photo, work: reads.append(photo) or _read_sheet())
    assert ss.main(str(tmp_path)) == 0
    first = json.loads((tmp_path / "game_sheet.json").read_text())
    assert first["goals"]["away"][1]["type"] == "PP"
    assert json.loads((tmp_path / "game_sheet_read.json").read_text())["sheet"]["teams"]["home"] == "Salty Boys"

    (tmp_path / "game_sheet_overrides.json").write_text(json.dumps({
        "schema_version": 1,
        "teams": {"home": "The Salty Boys"},
        "rosters": {"home": {"73": "J. Tripper", "59": "Joe Shiflett"}, "away": {"00": "Ben Smith"}},
        "goals": {"away": {"2": {"was": {"type": "PP"}, "type": "SH", "time": "1:10"},
                           "3": {"per": "3", "time": "4:00", "scorer": "00", "assists": ["47"], "type": "ES"}}},
        "penalties": {"home": {"1": {"per": "2", "player": "73", "infraction": "trip", "minutes": "2",
                                     "off": "1:52"}}},
        "shots_on_goal": {"home": [10, 4, 6], "away": [10, 13, 9]},
        "goalkeeping": {"home": [{"player": "59", "name": "Joe Shiflett"}]},
    }))
    assert ss.main(str(tmp_path)) == 0
    assert len(reads) == 1
    sheet = json.loads((tmp_path / "game_sheet.json").read_text())
    assert sheet["teams"] == {"home": "The Salty Boys", "away": "Ice Pak"}
    assert sheet["rosters"]["home"] == {"73": "J. Tripper", "59": "Joe Shiflett"}
    assert sheet["rosters"]["away"]["00"] == "Ben Smith"
    fixed, added = sheet["goals"]["away"][1], sheet["goals"]["away"][2]
    assert (fixed["type"], fixed["time"], fixed["time_s"], fixed["status"], fixed["reasons"]) == \
        ("SH", "1:10", 70, "ok", [])
    assert fixed["source"] == "override" and fixed["scorer"] == "47"
    assert (added["scorer"], added["assists"], added["time_s"], added["status"]) == ("00", ["47"], 240, "ok")
    pen = sheet["penalties"]["home"][0]
    assert (pen["player"], pen["infraction"], pen["minutes"], pen["status"]) == ("73", "TRIP", "2", "ok")
    assert sheet["shots_on_goal"] == {"home": [10, 4, 6], "away": [10, 13, 9]}
    assert sheet["goalkeeping"]["home"] == [{"player": "59", "name": "Joe Shiflett"}]
    assert json.loads((tmp_path / "game_sheet_read.json").read_text())["sheet"]["goals"]["away"][1]["type"] == "PP"


def test_stale_and_removing_overrides(tmp_path, monkeypatch):
    (tmp_path / "IMG_1.jpg").write_bytes(b"sheet photo")
    monkeypatch.setattr(ss, "read_scoresheet", lambda photo, work: _read_sheet())
    (tmp_path / "game_sheet_overrides.json").write_text(json.dumps({
        "schema_version": 1,
        "goals": {"away": {"1": None, "2": {"was": {"type": "ES"}, "type": "SH"}}},
        "rosters": {"home": {"73": None}},
    }))
    assert ss.main(str(tmp_path)) == 0
    sheet = json.loads((tmp_path / "game_sheet.json").read_text())
    assert [g["time"] for g in sheet["goals"]["away"]] == ["1:43"]
    assert sheet["goals"]["away"][0]["type"] == "PP" and sheet["goals"]["away"][0]["status"] == "review"
    assert any("away goal 2" in f and "not applied" in f for f in sheet["flags"])
    assert sheet["rosters"]["home"] == {}


def test_invalid_overrides_stop_instead_of_being_ignored(tmp_path, monkeypatch):
    (tmp_path / "IMG_1.jpg").write_bytes(b"sheet photo")
    monkeypatch.setattr(ss, "read_scoresheet", lambda photo, work: _read_sheet())
    (tmp_path / "game_sheet_overrides.json").write_text(json.dumps({"schema_version": 1, "goal": {}}))
    assert ss.main(str(tmp_path)) == 1
    assert not (tmp_path / "game_sheet.json").exists()


def test_a_changed_sheet_photo_is_read_again(tmp_path, monkeypatch):
    photo = tmp_path / "IMG_1.jpg"
    photo.write_bytes(b"sheet photo")
    reads = []
    monkeypatch.setattr(ss, "read_scoresheet", lambda p, work: reads.append(p) or _read_sheet())
    ss.main(str(tmp_path))
    photo.write_bytes(b"a retaken, larger sheet photo")
    ss.main(str(tmp_path))
    assert len(reads) == 2


def test_added_rows_need_their_required_fields(tmp_path, monkeypatch):
    (tmp_path / "IMG_1.jpg").write_bytes(b"sheet photo")
    monkeypatch.setattr(ss, "read_scoresheet", lambda photo, work: _read_sheet())
    (tmp_path / "game_sheet_overrides.json").write_text(json.dumps({
        "schema_version": 1, "goals": {"away": {"3": {"time": "4:00", "scorer": "00"}}}}))
    assert ss.main(str(tmp_path)) == 1
    assert not (tmp_path / "game_sheet.json").exists()


def test_edits_keep_unknown_type_and_cover_pdf_fields_and_checked_flags(tmp_path, monkeypatch):
    (tmp_path / "IMG_1.jpg").write_bytes(b"sheet photo")
    read = _read_sheet()
    read["goals"]["away"][0]["type"] = ""
    monkeypatch.setattr(ss, "read_scoresheet", lambda photo, work: read)
    (tmp_path / "game_sheet_overrides.json").write_text(json.dumps({
        "schema_version": 1,
        "goals": {"away": {"1": {"scorer": "00"}}},
        "date": "2026-02-01",
        "final_score": {"home": 0, "away": 7},
        "dismiss_flags": ["table not found: home_penalties", "a flag the reader never wrote"],
    }))
    assert ss.main(str(tmp_path)) == 0
    sheet = json.loads((tmp_path / "game_sheet.json").read_text())
    assert sheet["goals"]["away"][0]["type"] == ""
    assert sheet["date"] == "2026-02-01" and sheet["final_score"] == {"home": 0, "away": 7}
    assert "table not found: home_penalties" not in sheet["flags"]
    assert any("a flag the reader never wrote" in f and "not dismissed" in f for f in sheet["flags"])


def test_a_new_reader_version_reads_the_sheet_again(tmp_path, monkeypatch):
    (tmp_path / "IMG_1.jpg").write_bytes(b"sheet photo")
    reads = []
    monkeypatch.setattr(ss, "read_scoresheet", lambda p, work: reads.append(p) or _read_sheet())
    ss.main(str(tmp_path))
    monkeypatch.setattr(ss, "READER_VERSION", ss.READER_VERSION + 1)
    ss.main(str(tmp_path))
    assert len(reads) == 2
