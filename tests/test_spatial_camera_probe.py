import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3/scripts"))
import spatial_camera_probe as P


def inputs(tmp_path):
    game = tmp_path / "game"
    game.mkdir()
    source = game / "raw.MP4"
    source.write_bytes(b"synthetic source; do not decode")
    roi = game / "rois.json"
    roi.write_text(json.dumps({"camera_1": {"net": [40, 40, 20, 20], "slot": [40, 20, 20, 20]}}))
    identity = {"file": str(source), "size": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns}
    cache = tmp_path / "frames.npz"
    np.savez(cache, frames=np.zeros((2, 100, 100, 3), dtype=np.uint8), times=np.array([10., 11.]),
             identity=json.dumps(identity))
    return cache, roi, source, tmp_path / "results" / "spatial.json"


def test_spatial_probe_binds_inputs_and_reuses_valid_predictions_without_model_calls(tmp_path):
    cache, roi, source, output = inputs(tmp_path)
    before = {p: p.read_bytes() for p in (cache, roi, source)}
    calls = []
    def predict(frame):
        calls.append(frame.shape)
        return [{"class": "player", "confidence": .8, "xyxy": [48, 42, 52, 50]}]
    first = P.analyze_cache(cache, roi, "camera_1", output, {"model": "synthetic"}, predict)
    assert len(calls) == 2
    assert first["input_identity"]["frame_cache_sha256"] == P.sha256(cache)
    assert first["observations"][0]["evidence"]["near_players"] == 1
    second = P.analyze_cache(cache, roi, "camera_1", output, {"model": "synthetic"},
                             lambda frame: pytest.fail("valid inference cache must be reused"))
    assert second == first
    assert all(p.read_bytes() == value for p, value in before.items())


def test_spatial_probe_refuses_changed_raw_identity_before_using_prediction_cache(tmp_path):
    cache, roi, source, output = inputs(tmp_path)
    P.analyze_cache(cache, roi, "camera_1", output, {}, lambda frame: [])
    before = output.read_bytes()
    source.write_bytes(b"different raw file")
    with pytest.raises(ValueError, match="raw source identity changed"):
        P.analyze_cache(cache, roi, "camera_1", output, {}, lambda frame: [])
    assert output.read_bytes() == before


@pytest.mark.parametrize("change", ["model", "roi", "stride", "scope"])
def test_spatial_probe_refuses_changed_inputs_without_overwriting_outputs(tmp_path, change):
    cache, roi, source, output = inputs(tmp_path)
    P.analyze_cache(cache, roi, "camera_1", output, {}, lambda frame: [])
    before = output.read_bytes()
    producer, kwargs = {}, {}
    if change == "model":
        producer = {"model": "different"}
    elif change == "roi":
        roi.write_text(json.dumps({"camera_1": {"net": [41, 40, 20, 20], "slot": [40, 20, 20, 20]}}))
    elif change == "stride":
        kwargs["stride"] = 2
    else:
        kwargs["scope"] = "source_selected"
    with pytest.raises(ValueError, match="identity changed"):
        P.analyze_cache(cache, roi, "camera_1", output, producer, lambda frame: [], **kwargs)
    assert output.read_bytes() == before


def test_cached_missing_predictions_remain_unavailable_not_empty(tmp_path):
    cache, roi, source, output = inputs(tmp_path)
    first = P.analyze_cache(cache, roi, "camera_1", output, {}, lambda frame: None)
    second = P.analyze_cache(cache, roi, "camera_1", output, {}, lambda frame: pytest.fail("cache miss"))
    assert first == second
    assert all(o["evidence"]["status"] == "unavailable" for o in second["observations"])


def test_tampered_camera_authorization_is_rejected(tmp_path):
    cache, roi, source, output = inputs(tmp_path)
    P.analyze_cache(cache, roi, "camera_1", output, {}, lambda frame: [])
    data = json.loads(output.read_text())
    data["can_reassign_camera"] = True
    output.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        P.analyze_cache(cache, roi, "camera_1", output, {}, lambda frame: pytest.fail("no reprocessing"))


def test_tampered_spatial_count_is_rejected_without_reprocessing(tmp_path):
    cache, roi, source, output = inputs(tmp_path)
    P.analyze_cache(cache, roi, "camera_1", output, {}, lambda frame: [])
    data = json.loads(output.read_text())
    data["observations"][0]["evidence"]["near_players"] = 20
    output.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="invalid cached spatial observation"):
        P.analyze_cache(cache, roi, "camera_1", output, {}, lambda frame: pytest.fail("no reprocessing"))


@pytest.mark.parametrize("invalid", ["times", "frames"])
def test_invalid_frame_cache_does_not_create_an_empty_observation(tmp_path, invalid):
    cache, roi, source, output = inputs(tmp_path)
    with np.load(cache) as data:
        arrays = {name: data[name].copy() for name in data.files}
    if invalid == "times":
        arrays["times"][0] = float("nan")
    else:
        arrays["frames"] = arrays["frames"].astype(np.float32)
    np.savez(cache, **arrays)
    with pytest.raises(ValueError, match="invalid spatial frame cache"):
        P.analyze_cache(cache, roi, "camera_1", output, {}, lambda frame: pytest.fail("must not infer"))
    assert not output.exists()


def test_probe_refuses_output_in_game_folder(tmp_path):
    cache, roi, source, output = inputs(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        P.analyze_cache(cache, roi, "camera_1", roi.parent / "spatial.json", {}, lambda frame: [])
    assert not (roi.parent / "spatial.json").exists()


def test_missing_local_weights_fail_before_import_or_download(tmp_path, capsys):
    cache, roi, source, output = inputs(tmp_path)
    with pytest.raises(SystemExit) as caught:
        P.main(["--frame-cache", str(cache), "--rois", str(roi), "--camera", "camera_1",
                "--weights", str(tmp_path / "missing.pt"), "--output", str(output)])
    assert caught.value.code == 2
    assert "provide existing local" in capsys.readouterr().err
    assert not output.exists()
