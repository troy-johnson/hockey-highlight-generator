# tests/test_auto_roi.py — automatic net/slot ROIs from the goal frame (hhg-3hk.3, spec 002 §5.4)
import json
import numpy as np
import pytest
from pathlib import Path

import auto_roi as A


def test_rois_from_goal_box_match_hand_drawn_ghost_pirates():
    # HockeyAI goal box on the Ghost Pirates cam A median background, scaled to 1280x720
    rois = A.rois_from_goal_box((430.0, 100.4, 741.5, 339.9), 1280, 720)
    for key, hand in (("net", [380, 180, 410, 240]), ("slot", [380, 60, 410, 140])):
        assert all(abs(a - b) <= 3 for a, b in zip(rois[key], hand)), (key, rois[key], hand)


def test_rois_are_clamped_to_the_frame():
    rois = A.rois_from_goal_box((5.0, 5.0, 1275.0, 700.0), 1280, 720)
    for x, y, w, h in rois.values():
        assert x >= 0 and y >= 0 and x + w <= 1280 and y + h <= 720 and w > 0 and h > 0


def test_pick_goal_prefers_confident_large_box():
    dets = [("goal", 0.40, (10, 10, 40, 30)), ("player", 0.9, (0, 0, 500, 500)), ("goal", 0.77, (430, 100, 741, 340))]
    assert A.pick_goal_box(dets) == ((430, 100, 741, 340), 0.77)
    assert A.pick_goal_box([("player", 0.9, (0, 0, 5, 5))]) is None


def test_main_writes_rois_json(tmp_path, monkeypatch):
    (tmp_path / "chapters.json").write_text(json.dumps({"cam1": ["/a/1.MP4"], "cam2": ["/b/2.MP4"],
                                                        "recordings": {"cam1": [["/a/1.MP4"]], "cam2": [["/b/2.MP4"]]}}))
    monkeypatch.setattr(A, "median_background", lambda paths, width=1280: np.zeros((720, 1280), np.uint8))
    boxes = iter([((430.0, 100.4, 741.5, 339.9), 0.77), ((430.0, 100.4, 741.5, 339.9), 0.74)])
    monkeypatch.setattr(A, "detect_goal_box", lambda img: next(boxes))
    assert A.main(str(tmp_path)) == 0
    rois = json.loads((tmp_path / "rois.json").read_text())
    assert set(rois) == {"camera_1", "camera_2"} and set(rois["camera_1"]) == {"net", "slot"}
    info = json.loads((tmp_path / "rois_auto.json").read_text())
    assert info["camera_1"]["confidence"] == 0.77
    assert (tmp_path / "rois_preview.png").exists()


def test_main_fails_without_goal_so_picker_runs(tmp_path, monkeypatch):
    (tmp_path / "chapters.json").write_text(json.dumps({"cam1": ["/a/1.MP4"], "cam2": ["/b/2.MP4"]}))
    monkeypatch.setattr(A, "median_background", lambda paths, width=1280: np.zeros((720, 1280), np.uint8))
    monkeypatch.setattr(A, "detect_goal_box", lambda img: None)
    assert A.main(str(tmp_path)) != 0
    assert not (tmp_path / "rois.json").exists()


def test_run_detect_tries_auto_roi_before_picker():
    script = (Path(__file__).resolve().parents[1] / "run_detect.sh").read_text()
    assert script.index("auto_roi.py") < script.index("roi_picker.py")


# Review findings (GPT-6.1 review of #20)

def test_analysis_size_follows_source_aspect_like_signals(monkeypatch):
    monkeypatch.setattr(A, "_source_dims", lambda path: (4000, 3000))      # 4:3 capture
    assert A.analysis_size("/x/GX010001.MP4") == (1280, 960)
    monkeypatch.setattr(A, "_source_dims", lambda path: (3840, 2160))
    assert A.analysis_size("/x/GX010001.MP4") == (1280, 720)


def test_rois_clamp_to_the_real_analysis_height():
    rois = A.rois_from_goal_box((400.0, 600.0, 800.0, 950.0), 1280, 960)
    for x, y, w, h in rois.values():
        assert y + h <= 960 and y + h > 720          # uses the 4:3 frame, not 720


def test_interrupted_download_leaves_no_weights(tmp_path, monkeypatch):
    target = tmp_path / "w.pt"
    monkeypatch.setattr(A, "WEIGHTS_PATH", target)
    monkeypatch.setattr(A, "_MODEL", None)
    monkeypatch.setattr(A.urllib.request, "urlretrieve", lambda url, dest: Path(dest).write_bytes(b"partial"))

    class Broken:
        def __init__(self, path):
            raise RuntimeError("corrupt weights")

    import types, sys
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=Broken))
    with pytest.raises(RuntimeError):
        A._model()
    assert not target.exists() and not list(tmp_path.glob("*.part*"))


def test_first_download_loads_from_a_pt_name_and_ends_on_the_final_path(tmp_path, monkeypatch):
    """Second review of #20: Ultralytics loads a checkpoint only from a '.pt' name."""
    target = tmp_path / "HockeyAI_model_weight.pt"
    monkeypatch.setattr(A, "WEIGHTS_PATH", target)
    monkeypatch.setattr(A, "_MODEL", None)
    monkeypatch.setattr(A.urllib.request, "urlretrieve", lambda url, dest: Path(dest).write_bytes(b"weights"))
    loaded = []

    class FakeYOLO:
        def __init__(self, path):
            assert Path(path).suffix == ".pt" and Path(path).exists()
            loaded.append(path)

    import types, sys
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    A._model()
    assert target.exists() and loaded[-1] == str(target)
    assert not [p for p in tmp_path.iterdir() if p != target]
