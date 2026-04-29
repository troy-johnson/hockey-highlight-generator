from pathlib import Path


def test_folder_mode_references_v3_discovery_and_sync_flow():
    script = (Path(__file__).resolve().parents[1] / "run_detect.sh").read_text()

    assert 'SCRIPTS_V3="$REPO_DIR/v3/scripts"' in script
    assert 'discover.py' in script
    assert 'gopro_meta.py' in script
    assert 'chapters.json' in script
    assert 'cam1_concat.txt' in script
    assert 'cam2_concat.txt' in script
