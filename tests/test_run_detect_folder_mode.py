from pathlib import Path


def test_folder_mode_references_v3_discovery_and_sync_flow():
    script = (Path(__file__).resolve().parents[1] / "run_detect.sh").read_text()

    assert 'SCRIPTS_V3="$REPO_DIR/v3/scripts"' in script
    assert 'discover.py' in script
    assert 'gopro_meta.py' in script
    assert 'chapters.json' in script
    assert 'cam1_concat.txt' in script
    assert 'cam2_concat.txt' in script


def _fake_repo(tmp_path):
    """run_detect.sh with stub scripts: auto_roi fails, the picker leaves a mark if it runs."""
    import shutil
    repo = tmp_path / "repo"
    (repo / "v2" / "scripts").mkdir(parents=True)
    (repo / "v3" / "scripts").mkdir(parents=True)
    shutil.copy(Path(__file__).resolve().parents[1] / "run_detect.sh", repo / "run_detect.sh")
    stubs = {
        "v3/scripts/discover.py": "import json, sys, os\n"
            "json.dump({'cam1': ['a.mp4'], 'cam2': ['b.mp4']}, open(os.path.join(sys.argv[1], 'chapters.json'), 'w'))\n",
        "v3/scripts/gopro_meta.py": "",
        "v3/scripts/auto_roi.py": "raise SystemExit(1)\n",
        "v2/scripts/roi_picker.py": "open('PICKER_RAN', 'w').close()\n",
    }
    for rel, body in stubs.items():
        (repo / rel).write_text(body)
    game = tmp_path / "game"
    game.mkdir()
    return repo, game


def test_unattended_run_stops_instead_of_opening_the_roi_picker(tmp_path):
    import os, subprocess, sys
    repo, game = _fake_repo(tmp_path)
    env = {**os.environ, "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]}
    r = subprocess.run(["bash", str(repo / "run_detect.sh"), str(game)], cwd=tmp_path, env=env,
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
    assert "unattended" in r.stdout and "roi_picker.py" in r.stdout
    assert not (tmp_path / "PICKER_RAN").exists()
