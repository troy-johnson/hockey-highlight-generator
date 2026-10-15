# tests/conftest.py
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "v2", "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "v3", "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "v3", "resolve_scripts"))


import pytest


@pytest.fixture
def freeze_sheet(tmp_path):
    """Build graphics fixtures through a temporary database, as the pipeline does."""
    import copy
    import team_db as db
    from game_data import import_sheet, write_data

    def freeze(sheet, options=None):
        sheet = copy.deepcopy(sheet)
        conn = db.connect(tmp_path / "fixture.db")
        configs = {}
        sheet.setdefault("rosters", {"home": {}, "away": {}})
        for side in ("home", "away"):
            config = {"name": sheet["teams"][side], "roster": dict(sheet["rosters"][side])}
            for row in (sheet.get("goalkeeping") or {}).get(side, []):
                config["roster"][row["player"]] = row["name"]
                sheet["rosters"][side][row["player"]] = row["name"]
            config.update(next((t for t in ((options or {}).get("teams") or {}).values()
                                if t["name"] == config["name"]), {}))
            configs[side] = config
        data = import_sheet(conn, "fixture", sheet, {"date": "2026-10-11", "teams": configs,
                            "focus_team": (options or {}).get("focus_team")}, tmp_path)
        conn.close()
        write_data(data, tmp_path / "game_data.json")
        return data

    return freeze
