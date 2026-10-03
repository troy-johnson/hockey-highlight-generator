import hashlib
import json
import subprocess
import wave

import pytest

import audio_mix as am
import cue_library as cl


def media(root, name="beds/bed-01.wav", duration=4):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * (8000 * duration))
    proof = root / "provenance/license.txt"
    proof.parent.mkdir(exist_ok=True)
    proof.write_text("Synthetic fixture license proof; no real track approval.")
    return path


def entry(root, name="bed-01", kind="beds"):
    file = f"{kind}/{name}.wav"
    path = media(root, file)
    item = {"id": name, "file": file, "source": "YouTube Audio Library",
            "content_id_safe": True, "approved": True, "duration_s": 4.0,
            "provenance": {"url": "https://www.youtube.com/audiolibrary",
                           "license": "YouTube Audio Library License",
                           "proof": "provenance/license.txt",
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "approved_by": "Fixture", "approved_at": "2026-10-02"}}
    if kind == "beds":
        item.update(bpm=120.0, bar_s=2.0, beats=[0, .5, 1, 1.5, 2, 2.5, 3, 3.5],
                    beats_per_bar=4, tags=["synthetic"], lufs=-23.0)
    else:
        item["gain_db"] = 0.0
    return item


def manifest(root, beds=None, cues=None):
    data = {"schema_version": 1, "beds": beds or [], "cues": cues or []}
    path = root / "manifest.json"
    path.write_text(json.dumps(data))
    return path


def test_schema_one_library_loads_without_changing_the_mixer_contract(tmp_path):
    bed = entry(tmp_path)
    cues = [entry(tmp_path, cid, "cues") for cid in am.CUE_IDS]
    path = manifest(tmp_path, [bed], cues)
    library = cl.load_manifest(path)
    mixed, flags = am.load_manifest(path)
    assert mixed is not None
    assert not flags
    assert mixed["beds"][0]["path"] == str(tmp_path / bed["file"])
    assert {c["id"] for c in library["cues"]} == set(am.CUE_IDS)
    assert library["beds"][0]["provenance"]["approved_by"] == "Fixture"


@pytest.mark.parametrize("field,value", [
    ("id", ""), ("duration_s", 0), ("duration_s", True),
    ("duration_s", float("nan")), ("bpm", -1), ("bar_s", float("inf")),
    ("beats", [0, .5, .5]), ("beats", [0, 5]), ("beats_per_bar", 0),
    ("beats_per_bar", 2.5), ("tags", "rock"), ("lufs", float("nan")),
    ("file", "../outside.wav"), ("file", "/tmp/bed.wav"),
    ("source", "unknown"), ("content_id_safe", False),
    ("content_id_safe", "true"), ("approved", False), ("provenance", {}),
])
def test_invalid_entries_are_rejected_before_use(tmp_path, field, value):
    bed = entry(tmp_path)
    bed[field] = value
    with pytest.raises(ValueError, match=field):
        cl.load_manifest(manifest(tmp_path, [bed]))


@pytest.mark.parametrize("data", [[], None, {"schema_version": 2},
                                      {"schema_version": True, "beds": [], "cues": []},
                                      {"schema_version": 1, "beds": {}, "cues": []},
                                      {"schema_version": 1, "beds": [None], "cues": []}])
def test_invalid_manifest_shapes_raise_value_error(tmp_path, data):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        cl.load_manifest(path)


def test_duplicate_ids_and_unknown_cues_are_rejected(tmp_path):
    bed = entry(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        cl.load_manifest(manifest(tmp_path, [bed, bed]))
    with pytest.raises(ValueError, match="cue id"):
        cl.load_manifest(manifest(tmp_path, cues=[entry(tmp_path, "invented", "cues")]))


def test_stings_must_be_two_to_four_seconds(tmp_path):
    sting = entry(tmp_path, "sting_penalty", "cues")
    sting["duration_s"] = 1.0
    with pytest.raises(ValueError, match="2-4"):
        cl.load_manifest(manifest(tmp_path, cues=[sting]))


@pytest.mark.parametrize("field,value", [("url", ""), ("license", ""),
                                       ("proof", "missing.txt"), ("sha256", "0" * 64),
                                       ("approved_by", ""), ("approved_at", "yesterday")])
def test_provenance_requires_source_license_proof_and_approval(tmp_path, field, value):
    bed = entry(tmp_path)
    bed["provenance"][field] = value
    with pytest.raises(ValueError, match=field):
        cl.load_manifest(manifest(tmp_path, [bed]))


def test_freesound_requires_cc0_and_media_hash_detects_replacement(tmp_path):
    cue = entry(tmp_path, "horn", "cues")
    cue.update(source="Freesound CC0")
    cue["provenance"].update(url="https://freesound.org/s/1/", license="CC-BY")
    path = manifest(tmp_path, cues=[cue])
    with pytest.raises(ValueError, match="CC0"):
        cl.load_manifest(path)
    cue["provenance"]["license"] = "CC0"
    path = manifest(tmp_path, cues=[cue])
    assert cl.load_manifest(path)["cues"]
    (tmp_path / cue["file"]).write_bytes(b"replaced")
    with pytest.raises(ValueError, match="sha256"):
        cl.load_manifest(path)


def test_missing_files_and_escaping_symlinks_are_rejected(tmp_path):
    bed = entry(tmp_path)
    path = manifest(tmp_path, [bed])
    audio = tmp_path / bed["file"]
    audio.unlink()
    with pytest.raises(ValueError, match="file"):
        cl.load_manifest(path)
    audio.symlink_to(tmp_path.parent / "outside.wav")
    with pytest.raises(ValueError, match="file"):
        cl.load_manifest(path)


def test_empty_library_is_valid_and_reports_unfilled_slots(tmp_path):
    path = manifest(tmp_path)
    assert cl.load_manifest(path)["beds"] == []
    flags = cl.library_flags(cl.load_manifest(path))
    assert any("Beds" in f for f in flags)
    assert all(any(cid in f for f in flags) for cid in am.CUE_IDS)


def test_manual_and_detected_grids_use_the_mixer_bar_convention():
    grid = cl.derive_grid(4, bpm=120, beats_per_bar=3)
    assert grid == {"bpm": 120.0, "bar_s": 1.5,
                    "beats": [0, .5, 1, 1.5, 2, 2.5, 3, 3.5], "beats_per_bar": 3}
    detected = cl.derive_grid(4, beats=[.2, .7, 1.2, 1.7, 2.2, 2.7, 3.2, 3.7])
    assert detected["bpm"] == pytest.approx(120)
    assert detected["bar_s"] == pytest.approx(2)
    assert am.bar_seconds(detected) == pytest.approx(2)
    override = cl.derive_grid(4, bpm=120, bar_s=2.1)
    assert override["bar_s"] == 2.1


@pytest.mark.parametrize("kwargs", [{"beats": [0]}, {"beats": [0, 0]},
                                    {"beats": [-1, 0]}, {"beats": [0, 5]},
                                    {"bpm": 0}, {"bpm": True},
                                    {"bpm": 120, "beats_per_bar": 0},
                                    {"bpm": 120, "bar_s": float("nan")}])
def test_invalid_grid_input_does_not_create_timestamps(kwargs):
    with pytest.raises(ValueError):
        cl.derive_grid(4, **kwargs)


def test_beat_tool_updates_only_grids_and_preserves_provenance(tmp_path):
    bed = entry(tmp_path)
    for key in ("bpm", "bar_s", "beats", "beats_per_bar"):
        del bed[key]
    path = manifest(tmp_path, [bed])
    called = []

    def detector(audio):
        called.append(audio)
        return [0, .5, 1, 1.5, 2, 2.5, 3, 3.5]

    cl.update_grids(path, detector=detector)
    updated = cl.load_manifest(path)["beds"][0]
    assert called == [tmp_path / bed["file"]]
    assert updated["provenance"] == bed["provenance"]
    assert updated["bar_s"] == 2.0
    mixed, _flags = am.load_manifest(path)
    assert mixed is not None
    assert mixed["beds"][0]["bpm"] == 120.0


def test_failed_grid_analysis_keeps_the_manifest_intact(tmp_path):
    path = manifest(tmp_path, [entry(tmp_path)])
    original = path.read_bytes()
    with pytest.raises(ValueError):
        cl.update_grids(path, detector=lambda audio: [0])
    assert path.read_bytes() == original


@pytest.mark.parametrize("mode", ["detected", "manual", "cli", "cli-bar"])
def test_regridding_preserves_each_beds_meter_unless_overridden(tmp_path, mode):
    beds = [entry(tmp_path, f"bed-{i}") for i in range(3)]
    beds[0].update(beats_per_bar=3, bar_s=1.5)
    del beds[2]["beats_per_bar"]
    path = manifest(tmp_path, beds)
    if mode == "detected":
        cl.update_grids(path, detector=lambda audio: [0, .5, 1, 1.5, 2, 2.5, 3, 3.5])
    elif mode == "manual":
        cl.update_grids(path, bpm=120)
    else:
        args = ["grid", str(path), "--bpm", "120"]
        if mode == "cli-bar":
            args += ["--bar-s", "2.1"]
        assert cl.main(args) == 0
    updated = cl.load_manifest(path)["beds"]
    assert [bed["beats_per_bar"] for bed in updated] == [3, 4, 4]
    assert [am.bar_seconds(bed) for bed in updated] == (
        [2.1, 2.1, 2.1] if mode == "cli-bar" else [1.5, 2.0, 2.0])


def test_cli_meter_override_is_explicit_and_invalid_values_do_not_change_manifest(tmp_path):
    bed = entry(tmp_path)
    bed.update(beats_per_bar=3, bar_s=1.5)
    path = manifest(tmp_path, [bed])
    assert cl.main(["grid", str(path), "--bpm", "120", "--beats-per-bar", "5"]) == 0
    updated = cl.load_manifest(path)["beds"][0]
    assert updated["beats_per_bar"] == 5 and am.bar_seconds(updated) == 2.5
    original = path.read_bytes()
    assert cl.main(["grid", str(path), "--bpm", "120", "--beats-per-bar", "0"]) == 1
    assert path.read_bytes() == original


def test_rotation_avoids_all_previous_game_beds_and_reuses_saved_choices(tmp_path):
    beds = [{"id": f"bed-{i}", "bar_s": 2} for i in range(6)]
    history = tmp_path / "rotation.json"
    first, flags = cl.select_beds(beds, "game-1", [1, 2, 3], history)
    assert len({bed["id"] for bed in first.values()}) == 3
    assert not flags and not history.exists()
    cl.record_rotation(history, "game-1", first)
    second, flags = cl.select_beds(beds, "game-2", [1, 2, 3], history)
    assert not flags
    assert {b["id"] for b in first.values()}.isdisjoint(b["id"] for b in second.values())
    cl.record_rotation(history, "game-2", second)
    rerun, flags = cl.select_beds(list(reversed(beds)), "game-1", [1, 2, 3], history)
    assert rerun == first and not flags
    cl.record_rotation(history, "game-1", rerun)
    assert json.loads(history.read_text())["games"][-1]["key"] == "game-2"


def test_small_library_exhausts_unused_tracks_before_reusing_and_flags(tmp_path):
    beds = [{"id": f"bed-{i}"} for i in range(4)]
    history = tmp_path / "rotation.json"
    cl.record_rotation(history, "old", {1: beds[0], 2: beds[1], 3: beds[2]})
    chosen, flags = cl.select_beds(beds, "new", [1, 2, 3], history)
    assert chosen[1]["id"] == "bed-3"
    assert len({b["id"] for b in chosen.values()}) == 3
    assert any("reuse" in f for f in flags)
    empty, flags = cl.select_beds([], "new", [1], history)
    assert not empty and flags


def test_corrupt_history_is_not_overwritten_and_missing_saved_bed_is_flagged(tmp_path):
    history = tmp_path / "rotation.json"
    history.write_text('{"schema_version":1,"games":{}}')
    with pytest.raises(ValueError, match="rotation"):
        cl.select_beds([{"id": "a"}], "new", [1], history)
    with pytest.raises(ValueError, match="rotation"):
        cl.record_rotation(history, "new", {1: {"id": "a"}})
    history.unlink()
    cl.record_rotation(history, "old", {1: {"id": "removed"}})
    chosen, flags = cl.select_beds([{"id": "a"}], "old", [1], history)
    assert chosen[1]["id"] == "a"
    assert any("removed" in f for f in flags)


def test_mixer_plan_uses_rotation_choices_at_bar_boundaries():
    beds = [{"id": "a", "bar_s": 2}, {"id": "b", "bar_s": 3}]
    tl = [{"out_start": 0, "out_dur": 5, "moment_s": 5},
          {"out_start": 5, "out_dur": 5, "moment_s": 15}]
    periods = [{"n": 1, "start": 0, "end": 10}, {"n": 2, "start": 10, "end": 20}]
    segments = am.plan_bed(tl, periods, beds, "game", bed_choices={1: beds[0], 2: beds[1]})
    assert [s["track"] for s in segments] == ["a", "b"]
    assert segments[1]["out_start"] == 6


def test_cli_initializes_an_empty_library_and_validates_without_inference(tmp_path, capsys):
    root = tmp_path / "library"
    assert cl.main(["init", str(root)]) == 0
    assert (root / "beds").is_dir()
    assert (root / "cues").is_dir()
    assert (root / "provenance").is_dir()
    assert cl.load_manifest(root / "manifest.json") == {"schema_version": 1, "beds": [], "cues": []}
    assert cl.main(["validate", str(root / "manifest.json")]) == 0
    assert "no approved audio" in capsys.readouterr().out
    assert cl.main(["init", str(root)]) == 1


def test_cli_registers_only_explicitly_approved_metadata_and_probes_media(tmp_path):
    bed = entry(tmp_path)
    bed["duration_s"] = 100  # The tool measures the actual four-second WAV.
    del bed["provenance"]["sha256"]
    metadata = tmp_path / "track.json"
    metadata.write_text(json.dumps(bed))
    path = manifest(tmp_path)
    assert cl.main(["add", str(path), "beds", str(metadata)]) == 0
    added = cl.load_manifest(path)["beds"][0]
    assert added["duration_s"] == 4.0
    assert added["provenance"]["sha256"] == hashlib.sha256((tmp_path / bed["file"]).read_bytes()).hexdigest()
    original = path.read_bytes()
    bed["approved"] = False
    metadata.write_text(json.dumps(bed))
    assert cl.main(["add", str(path), "beds", str(metadata)]) == 1
    assert path.read_bytes() == original


def test_cli_manual_grid_works_without_the_optional_model(tmp_path):
    path = manifest(tmp_path, [entry(tmp_path)])
    assert cl.main(["grid", str(path), "--id", "bed-01", "--bpm", "100", "--beats-per-bar", "3"]) == 0
    bed = cl.load_manifest(path)["beds"][0]
    assert bed["bar_s"] == pytest.approx(1.8)
    assert bed["bpm"] == 100


def test_replacing_a_removed_bed_reserves_other_periods_saved_tracks(tmp_path):
    beds = [{"id": f"bed-{i}"} for i in range(6)]
    path = tmp_path / "rotation.json"
    initial, _ = cl.select_beds(beds, "game", [1, 2, 3], path)
    cl.record_rotation(path, "game", {1: {"id": "removed"}, 2: initial[1], 3: initial[2]})
    chosen, flags = cl.select_beds(beds, "game", [1, 2, 3], path)
    assert chosen[2] == initial[1] and chosen[3] == initial[2]
    assert len({b["id"] for b in chosen.values()}) == 3
    assert not any("insufficient" in f for f in flags)


def test_authoring_an_untimed_bed_runs_the_beat_adapter(tmp_path):
    bed = entry(tmp_path)
    for key in ("bpm", "beats", "bar_s", "beats_per_bar"):
        del bed[key]
    metadata = tmp_path / "track.json"
    metadata.write_text(json.dumps(bed))
    path = manifest(tmp_path)
    cl.add_track(path, "beds", metadata, detector=lambda audio: [0, .5, 1, 1.5, 2, 2.5, 3, 3.5])
    assert cl.load_manifest(path)["beds"][0]["bpm"] == 120


def test_authoring_a_bar_only_bed_preserves_explicit_timing(tmp_path):
    bed = entry(tmp_path)
    del bed["bpm"], bed["beats"]
    metadata = tmp_path / "track.json"
    metadata.write_text(json.dumps(bed))
    path = manifest(tmp_path)
    assert cl.main(["add", str(path), "beds", str(metadata)]) == 0
    assert cl.load_manifest(path)["beds"][0]["bar_s"] == 2


@pytest.mark.parametrize("history_state", ["normal", "corrupt", "read-only", "failed-mix"])
def test_real_synthetic_mix_records_rotation_only_after_success_and_flags_history_errors(tmp_path, history_state):
    library = tmp_path / "library"
    library.mkdir()
    path = manifest(library, [entry(library)])
    history = library / "rotation.json"
    if history_state == "corrupt":
        history.write_text("not JSON")
    game = tmp_path / "game"
    game.mkdir()
    output = "2026-10-01_A-vs-B_Recap.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=black:s=64x64:r=30:d=6", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=6", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(game / output)], check=True)
    clip = {"goal_id": "play:1", "kind": "play", "play_kind": "neutral", "camera": "cam1",
            "moment_s": 3, "start_s": 0, "end_s": 6, "speed": 1, "frames": 180,
            "parts": [{"file": str(game / output), "seek_s": 0, "duration_s": 6}]}
    plan = {"schema_version": 3, "clips": [clip], "replays": [], "duration_s": 6, "output": output}
    (game / "recap_assembly.json").write_text(json.dumps(plan))
    (game / "game_sheet.json").write_text(json.dumps({"teams": {"home": "A", "away": "B"}}))
    (game / "music_spans.json").write_text('{"spans":[]}')
    (game / "selection.json").write_text('{"periods":[{"n":1,"start":0,"end":6}]}')
    if history_state == "failed-mix":
        (game / output).unlink()
    if history_state == "read-only":
        library.chmod(0o555)
    try:
        result = am.main([str(game), "--options", json.dumps({"audio": {"manifest": str(path)}})])
    finally:
        library.chmod(0o755)
    if history_state == "failed-mix":
        assert result == 1 and not history.exists()
        return
    assert result == 0
    report = json.loads((game / "recap_audio.json").read_text())
    assert report["bed"][0]["track"] == "bed-01"
    assert (game / am.mixed_output_name(output)).is_file()
    if history_state == "normal":
        assert json.loads(history.read_text())["games"][0]["beds"] == {"1": "bed-01"}
    else:
        assert any("rotation" in flag for flag in report["flags"])
        if history_state == "corrupt":
            assert history.read_text() == "not JSON"
        else:
            assert not history.exists()
