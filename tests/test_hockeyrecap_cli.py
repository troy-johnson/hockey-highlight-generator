# tests/test_hockeyrecap_cli.py — hockeyrecap CLI: run, batch summary, status
import json

import pytest

import hockeyrecap as H
import recap_runner as rr
from recap_runner import Stage, StageResult


def _stages():
    def run(name):
        def f(ctx):
            (ctx.game_folder / f"{name}.out").write_text("x")
            if name == "b":
                (ctx.cache_dir / "signals").mkdir(parents=True, exist_ok=True)
                (ctx.cache_dir / "signals" / "signals_k.npz").write_bytes(b"x" * 3000)
                return StageResult("flagged", ["b needs a look"])
            return StageResult("done")
        return f
    return [Stage(n, n.upper(), lambda ctx: 1, lambda ctx, n=n: [f"{n}.out"], run(n))
            for n in ("a", "b")]


@pytest.fixture
def fake_stages(monkeypatch, tmp_path):
    monkeypatch.setattr(rr, "STAGES", _stages())
    monkeypatch.setenv("HHG_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("COLUMNS", "300")


def test_run_status_and_rerun(tmp_path, fake_stages, capsys):
    game = tmp_path / "wild_03012026"
    game.mkdir()
    assert H.main(["run", str(game)]) == 0
    out = capsys.readouterr().out
    assert "Created recap_options.json" in out
    assert "Summary" in out and "flag(s)" in out and "2.9 KB" in out
    assert "b needs a look" in out
    opts = json.loads((game / "recap_options.json").read_text())
    assert opts["opponent"]["value"] == "wild" and opts["opponent"]["inferred"] is True

    assert H.main(["status", str(game)]) == 0
    out = capsys.readouterr().out
    assert "b needs a look" in out and "Signal cache: 2.9 KB" in out

    assert H.main(["run", str(game)]) == 0
    assert json.loads((game / rr.STATUS_FILE).read_text())["stages"]["a"]["reused"] is True


def test_batch_run_continues_after_missing_folder(tmp_path, fake_stages, capsys):
    game = tmp_path / "wild_03012026"
    game.mkdir()
    rc = H.main(["run", str(tmp_path / "nope_01012026"), str(game)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Game Folder not found" in out
    assert (game / rr.STATUS_FILE).exists()
    assert "nope_01012026" in out and "wild_03012026" in out


def test_status_without_run(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    assert H.main(["status", str(tmp_path)]) == 1
    assert "has not run" in capsys.readouterr().out


def test_cli_flags_cover_game_keys():
    import recap_options
    args = H.build_parser().parse_args(["run", "x"])
    for key in recap_options.GAME_KEYS:
        if key != "date":
            assert hasattr(args, key), key


def test_strip_keeps_plain_brackets(monkeypatch):
    monkeypatch.setattr(H, "HAVE_RICH", False)
    assert H._strip("[green]done[/green] [red]x[/red]") == "done x"
    assert H._strip("[bold]game [home][/bold]") == "game [home]"
