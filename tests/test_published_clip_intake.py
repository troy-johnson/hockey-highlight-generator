"""Keep draft matches separate from answer keys and raw inputs."""
from pathlib import Path
import hashlib
import json
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3" / "scripts"))
import published_clip_intake as I


def recording(camera="cam1", seed=12):
    rng = np.random.default_rng(seed)
    background = cv2.GaussianBlur(rng.integers(110, 200, (180, 320), dtype=np.uint8), (3, 3), 0)
    frames = []
    for i in range(8):
        frame = background.copy()
        cv2.rectangle(frame, (40 + i * 20, 60), (50 + i * 20, 78), 20, -1)
        frames.append(frame)
    return {"camera": camera, "index": 0, "start_s": 0.0, "seek_s": 0.0,
            "background": background, "frames": np.stack(frames), "times": np.arange(8.0)}


def test_motion_proposal_never_assigns_a_goal_label():
    raw = recording()
    result = I.propose_clip(raw["frames"][3], [raw])
    assert result["status"] == "proposed"
    assert result["best"]["camera"] == "cam1"
    assert result["best"]["detection_s"] == 3
    assert "goal" not in result


def test_two_equally_good_cameras_require_review():
    first = recording()
    second = dict(first, camera="cam2")
    assert I.propose_clip(first["frames"][3], [first, second])["status"] == "needs_review"


def test_trimmed_camera_match_cannot_propose_a_negative_reference():
    raw = dict(recording(), seek_s=10.0)
    assert I.propose_clip(raw["frames"][3], [raw])["status"] == "needs_review"


def test_refinement_checks_full_frames_but_still_requires_review_of_labels(tmp_path, monkeypatch):
    raw = dict(recording(), files=["raw.MP4"])
    query = raw["frames"][3]
    proposal = I.propose_clip(query, [raw])
    calls = []
    def sample(files, cache, **settings):
        calls.append(settings)
        return raw["frames"], raw["times"], cache / "refinement.npz"
    monkeypatch.setattr(I, "sample_recording", sample)
    monkeypatch.setattr(I, "probe", lambda _: {"duration_s": 8.0})
    result = I.refine_clip(query, proposal, [raw], tmp_path)
    assert result["status"] == "refined_proposal"
    assert result["best"]["detection_s"] == 3
    assert calls[0]["keyframes_only"] is False
    assert calls[0]["fps"] == 12.0
    assert "goal" not in result


def test_output_cannot_overlap_original_inputs(tmp_path):
    source = tmp_path / "game"
    for output in (source, source / "output", tmp_path):
        with pytest.raises(ValueError, match="overlaps"):
            I.separate_output(output, [source])


def test_intake_requires_processing_approval_before_reading_footage(tmp_path):
    with pytest.raises(ValueError, match="approval"):
        I.run_intake(tmp_path / "edit.edl", tmp_path / "recap.mp4", tmp_path / "game",
                     tmp_path / "output", tmp_path / "cache")


def test_run_intake_writes_only_review_drafts(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    edl, recap, raw = (inputs / name for name in ("edit.edl", "recap.mp4", "chapter.MP4"))
    edl.write_text("001 AX V C 00:01:00:00 00:01:04:00 01:00:05:00 01:00:09:00\n"
                   "* FROM CLIP NAME: Game v1\n")
    recap.write_bytes(b"published recap")
    raw.write_bytes(b"raw footage")
    game = tmp_path / "game"
    game.mkdir()
    before = {p: p.read_bytes() for p in inputs.iterdir()}
    record = recording()
    monkeypatch.setattr(I, "manifest_parts", lambda _: [{"start_s": 0.0, "seek_s": 0.0,
                                                         "files": [str(raw)]}])
    monkeypatch.setattr(I, "sample_recording", lambda files, cache:
                        (record["frames"], record["times"], cache / "frames.npz"))
    monkeypatch.setattr(I, "read_frame", lambda *args, **kwargs: record["frames"][3])
    output = tmp_path / "draft"
    report = I.run_intake(edl, recap, game, output, tmp_path / "cache", allow_processing=True)
    assert report["status"] == "review_draft"
    assert report["clips"][0]["goal"] is None
    assert report["clips"][0]["camera"] is None
    assert report["clips"][0]["motion"]["status"] == "needs_review"
    assert (output / "draft.json").is_file()
    assert not list(game.iterdir())
    assert all(p.read_bytes() == data for p, data in before.items())
    with pytest.raises(FileExistsError):
        I.run_intake(edl, recap, game, output, tmp_path / "cache", allow_processing=True)


def test_published_labels_bind_to_video_without_assigning_a_camera(tmp_path):
    recap, references = tmp_path / "recap.mp4", tmp_path / "references.json"
    recap.write_bytes(b"published video")
    parsed = {"clips": [{"id": "edl:003", "recap_start_s": 5.0, "recap_end_s": 15.0,
                         "camera": None, "goal": None}]}
    data = {"recap_sha256": hashlib.sha256(recap.read_bytes()).hexdigest(),
            "references": [{"id": "goal:1", "recap_s": 10.0, "goal": True,
                            "evidence": "Published GOAL banner at 11 seconds."}]}
    references.write_text(json.dumps(data))
    result = I.published_references(references, recap, parsed)
    assert result[0]["goal"] is True
    assert result[0]["camera"] is None
    assert result[0]["edl_id"] == "edl:003"
    recap.write_bytes(b"different video")
    with pytest.raises(ValueError, match="do not match"):
        I.published_references(references, recap, parsed)


@pytest.mark.parametrize("field,value", [("goal", None), ("recap_s", float("nan")),
                                          ("recap_s", 15), ("evidence", ""), ("id", "../../outside")])
def test_ambiguous_published_labels_block(tmp_path, field, value):
    recap, references = tmp_path / "recap.mp4", tmp_path / "references.json"
    recap.write_bytes(b"published video")
    ref = {"id": "goal:1", "recap_s": 10.0, "goal": True, "evidence": "Verified banner."}
    ref[field] = value
    references.write_text(json.dumps({"recap_sha256": hashlib.sha256(recap.read_bytes()).hexdigest(),
                                      "references": [ref]}))
    with pytest.raises(ValueError):
        I.published_references(references, recap, {"clips": [{"recap_start_s": 5, "recap_end_s": 15}]})


def intake_case(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    edl, recap, raw = (inputs / name for name in ("edit.edl", "recap.mp4", "chapter.MP4"))
    edl.write_text("001 AX V C 00:01:00:00 00:01:04:00 01:00:05:00 01:00:09:00\n"
                   "* FROM CLIP NAME: Game v1\n")
    recap.write_bytes(b"published recap")
    raw.write_bytes(b"raw footage")
    game = tmp_path / "game"
    game.mkdir()
    record = recording()
    monkeypatch.setattr(I, "manifest_parts", lambda _: [{"start_s": 0.0, "seek_s": 0.0,
                                                         "files": [str(raw)]}])
    monkeypatch.setattr(I, "sample_recording", lambda files, cache, **kwargs:
                        (record["frames"], record["times"], cache / "frames.npz"))
    monkeypatch.setattr(I, "read_frame", lambda *args, **kwargs: record["frames"][3])
    return edl, recap, game, tmp_path / "draft", tmp_path / "cache", raw


def test_resume_preserves_coarse_work_after_interrupted_refinement(tmp_path, monkeypatch):
    edl, recap, game, output, cache, _ = intake_case(tmp_path, monkeypatch)
    def interrupt(*args):
        raise RuntimeError("interrupted refinement")
    monkeypatch.setattr(I, "refine_clip", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        I.run_intake(edl, recap, game, output, cache, allow_processing=True, refine=True)
    checkpoint = output / "checkpoints" / "edl_001.json"
    assert json.loads(checkpoint.read_text())["stage"] == "coarse_done"
    assert not (output / "draft.json").exists()
    def no_repeat(*args, **kwargs):
        raise AssertionError("coarse matching must not repeat")
    monkeypatch.setattr(I, "propose_clip", no_repeat)
    monkeypatch.setattr(I, "refine_clip", lambda *args: {"status": "refined_proposal"})
    report = I.run_intake(edl, recap, game, output, cache, allow_processing=True, refine=True, resume=True)
    assert report["clips"][0]["refinement"]["status"] == "refined_proposal"
    assert json.loads(checkpoint.read_text())["stage"] == "complete"
    saved = (output / "draft.json").read_bytes()
    monkeypatch.setattr(I, "sample_recording", no_repeat)
    again = I.run_intake(edl, recap, game, output, cache, allow_processing=True, refine=True, resume=True)
    assert again == report
    assert (output / "draft.json").read_bytes() == saved


@pytest.mark.parametrize("change", ["raw", "edl", "recap", "settings"])
def test_resume_blocks_changed_input_identity(tmp_path, monkeypatch, change):
    edl, recap, game, output, cache, raw = intake_case(tmp_path, monkeypatch)
    def interrupt(*args):
        raise RuntimeError("interrupted refinement")
    monkeypatch.setattr(I, "refine_clip", interrupt)
    with pytest.raises(RuntimeError):
        I.run_intake(edl, recap, game, output, cache, allow_processing=True, refine=True)
    if change in ("raw", "edl", "recap"):
        path = {"raw": raw, "edl": edl, "recap": recap}[change]
        path.write_bytes(path.read_bytes() + b"\n")
    def no_decode(*args, **kwargs):
        raise AssertionError("changed inputs must block before decoding")
    monkeypatch.setattr(I, "sample_recording", no_decode)
    with pytest.raises(ValueError, match="identity"):
        I.run_intake(edl, recap, game, output, cache, allow_processing=True,
                     refine=change != "settings", resume=True)


def test_refinement_seeks_directly_in_the_relevant_chapter(tmp_path, monkeypatch):
    raw = dict(recording(), files=["first.MP4", "second.MP4"], start_s=10.0, seek_s=2.0)
    proposal = {"alternatives": [{"camera": "cam1", "recording": 0, "raw_s": 8.0,
                                  "transform": np.eye(3).tolist()}]}
    calls = []
    monkeypatch.setattr(I, "probe", lambda _: {"duration_s": 5.0})
    def sample(files, cache, **settings):
        calls.append((files, settings))
        times = settings["start_s"] + np.arange(2.0)
        return raw["frames"][:2], times, cache / "frames.npz"
    monkeypatch.setattr(I, "sample_recording", sample)
    monkeypatch.setattr(I, "match_motion", lambda q, eb, frames, times, bg, matrix, **kwargs:
                        {"status": "matched", "raw_s": float(times[1]), "score": 0.99})
    result = I.refine_clip(raw["frames"][3], proposal, [raw], tmp_path)
    assert calls[0][0] == [Path("second.MP4")]
    assert calls[0][1]["start_s"] == 1.0
    assert result["best"]["raw_s"] == 7.0
    assert result["best"]["detection_s"] == 15.0


def test_refinement_preserves_times_across_chapter_boundary(tmp_path, monkeypatch):
    raw = dict(recording(), files=["first.MP4", "second.MP4"])
    proposal = {"alternatives": [{"camera": "cam1", "recording": 0, "raw_s": 4.5,
                                  "transform": np.eye(3).tolist()}]}
    calls, seen = [], []
    monkeypatch.setattr(I, "probe", lambda _: {"duration_s": 5.0})
    def sample(files, cache, **settings):
        calls.append((files, settings))
        times = settings["start_s"] + np.arange(0, settings["duration_s"], 0.5)
        return raw["frames"][:len(times)], times, cache / "frames.npz"
    def match(q, eb, frames, times, bg, matrix, **kwargs):
        seen.extend(times)
        return {"status": "matched", "raw_s": float(times[0]), "score": 0.99}
    monkeypatch.setattr(I, "sample_recording", sample)
    monkeypatch.setattr(I, "match_motion", match)
    I.refine_clip(raw["frames"][3], proposal, [raw], tmp_path)
    assert [files for files, _ in calls] == [[Path("first.MP4")], [Path("second.MP4")]]
    assert seen == [2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]


def test_resume_blocks_a_checkpoint_from_another_input(tmp_path, monkeypatch):
    edl, recap, game, output, cache, _ = intake_case(tmp_path, monkeypatch)
    def interrupt(*args):
        raise RuntimeError("interrupted refinement")
    monkeypatch.setattr(I, "refine_clip", interrupt)
    with pytest.raises(RuntimeError):
        I.run_intake(edl, recap, game, output, cache, allow_processing=True, refine=True)
    path = output / "checkpoints" / "edl_001.json"
    value = json.loads(path.read_text())
    value["signature"] = "another-input"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checkpoint identity"):
        I.run_intake(edl, recap, game, output, cache, allow_processing=True, refine=True, resume=True)


def test_atomic_report_does_not_replace_an_existing_file(tmp_path):
    path = tmp_path / "draft.json"
    I.atomic_json(path, {"status": "review_draft"})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        I.atomic_json(path, {"status": "changed"})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
