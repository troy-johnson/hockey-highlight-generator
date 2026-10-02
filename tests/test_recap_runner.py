# tests/test_recap_runner.py — hockeyrecap stage runner (hhg-3r5.28, hhg-3r5.58)
import json
from pathlib import Path

import pytest

import recap_runner as rr
from recap_runner import Stage, StageResult, run_game, run_batch, load_status


class Recorder(rr.Reporter):
    def __init__(self):
        self.events = []

    def stage_start(self, name, title):
        self.events.append(("start", name))

    def stage_end(self, name, record, reused):
        self.events.append(("end", name, record["state"], reused))


def make_stages(calls, behavior=None, inputs=None):
    """Three fake stages a -> b -> c. Each writes <name>.out; the fingerprint is inputs[name]."""
    behavior = behavior or {}
    inputs = inputs if inputs is not None else {}

    def stage(name, needs=()):
        def run(ctx):
            calls.append(name)
            b = behavior.get(name)
            if callable(b):
                return b(ctx)
            (ctx.game_folder / f"{name}.out").write_text("x")
            return b or StageResult("done")
        return Stage(name, name.upper(), lambda ctx: inputs.get(name, 0),
                     lambda ctx: [f"{name}.out"], run, needs=needs)

    return [stage("a"), stage("b", needs=("a",)), stage("c", needs=("b",))]


def states(status):
    return {n: r["state"] for n, r in status["stages"].items()}


def test_stages_run_in_order_and_status_file_written(tmp_path):
    calls, rep = [], Recorder()
    st = run_game(tmp_path, {}, rep, stages=make_stages(calls))
    assert calls == ["a", "b", "c"]
    assert [e[1] for e in rep.events if e[0] == "start"] == ["a", "b", "c"]
    assert st["state"] == "done"
    saved = load_status(tmp_path)
    assert saved["state"] == "done" and states(saved) == {"a": "done", "b": "done", "c": "done"}
    assert (tmp_path / rr.LOG_FILE).exists()


def test_second_run_reuses_unchanged_stages(tmp_path):
    calls = []
    run_game(tmp_path, {}, stages=make_stages(calls))
    calls.clear()
    rep = Recorder()
    st = run_game(tmp_path, {}, rep, stages=make_stages(calls))
    assert calls == []
    assert all(e[3] for e in rep.events if e[0] == "end")
    assert st["state"] == "done"


def test_changed_input_reruns_that_stage_only(tmp_path):
    calls, inputs = [], {}
    run_game(tmp_path, {}, stages=make_stages(calls, inputs=inputs))
    calls.clear()
    inputs["b"] = 1
    run_game(tmp_path, {}, stages=make_stages(calls, inputs=inputs))
    assert calls == ["b"]


def test_missing_output_reruns_stage(tmp_path):
    calls = []
    run_game(tmp_path, {}, stages=make_stages(calls))
    calls.clear()
    (tmp_path / "c.out").unlink()
    run_game(tmp_path, {}, stages=make_stages(calls))
    assert calls == ["c"]


def test_rerun_from_runs_that_stage_and_later(tmp_path):
    calls = []
    run_game(tmp_path, {}, stages=make_stages(calls))
    calls.clear()
    st = run_game(tmp_path, {}, from_stage="b", stages=make_stages(calls))
    assert calls == ["b", "c"]
    assert st["stages"]["a"]["reused"] is True


def test_rerun_from_unknown_stage_raises(tmp_path):
    with pytest.raises(ValueError):
        run_game(tmp_path, {}, from_stage="nope", stages=make_stages([]))


def test_interrupted_run_resumes_at_the_interrupted_stage(tmp_path):
    calls = []

    def boom(ctx):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_game(tmp_path, {}, stages=make_stages(calls, behavior={"b": boom}))
    saved = load_status(tmp_path)
    assert saved["state"] == "stopped"
    assert saved["stages"]["b"]["state"] == "interrupted"

    calls.clear()
    st = run_game(tmp_path, {}, stages=make_stages(calls))
    assert calls == ["b", "c"]
    assert st["state"] == "done"


def test_failed_stage_is_a_flag_and_dependents_are_skipped(tmp_path):
    calls = []
    fail = StageResult("failed", ["b failed: no goal frame"])
    st = run_game(tmp_path, {}, stages=make_stages(calls, behavior={"b": fail}))
    assert calls == ["a", "b"]
    assert st["state"] == "done"                       # not a stop: the run finished with flags
    assert states(st) == {"a": "done", "b": "failed", "c": "skipped"}
    assert "b failed: no goal frame" in st["flags"]
    assert any("c skipped" in f for f in st["flags"])
    # a failed stage runs again next time
    calls.clear()
    run_game(tmp_path, {}, stages=make_stages(calls))
    assert calls == ["b", "c"]


def test_exception_in_stage_is_a_flag(tmp_path):
    def bug(ctx):
        raise RuntimeError("oops")

    st = run_game(tmp_path, {}, stages=make_stages([], behavior={"a": bug}))
    assert st["state"] == "done"
    assert st["stages"]["a"]["state"] == "failed"
    assert any("oops" in f for f in st["flags"])


def test_fatal_stage_stops_the_run(tmp_path):
    calls = []
    fatal = StageResult("failed", ["no footage"], "no usable camera", fatal=True)
    st = run_game(tmp_path, {}, stages=make_stages(calls, behavior={"a": fatal}))
    assert calls == ["a"]
    assert st["state"] == "stopped"
    assert st["stop_reason"] == "no usable camera"
    assert "b" not in st["stages"] and "c" not in st["stages"]


def test_flagged_stage_counts_as_ok_for_later_stages(tmp_path):
    calls = []

    def flagged(ctx):
        (ctx.game_folder / "a.out").write_text("x")
        return StageResult("flagged", ["cam2 Recording 1 is black"])

    st = run_game(tmp_path, {}, stages=make_stages(calls, behavior={"a": flagged}))
    assert calls == ["a", "b", "c"]
    assert st["flags"] == ["cam2 Recording 1 is black"]


def test_needs_two_cameras_skips_with_one_camera(tmp_path):
    (tmp_path / "chapters.json").write_text(json.dumps({"cam1": ["x.MP4"], "recordings": {"cam1": [["x.MP4"]]}}))
    calls = []
    stages = make_stages(calls)
    stages[2].needs_two_cameras = True
    st = run_game(tmp_path, {}, stages=stages)
    assert calls == ["a", "b"]
    assert st["stages"]["c"]["state"] == "skipped"
    assert any("two usable cameras" in f for f in st["flags"])


def test_older_outputs_are_copied_before_first_overwrite(tmp_path):
    (tmp_path / "a.out").write_text("old")
    run_game(tmp_path, {}, stages=make_stages([]))
    assert (tmp_path / rr.PREVIOUS_DIR / "a.out").read_text() == "old"
    assert (tmp_path / "a.out").read_text() == "x"


def test_batch_one_failed_game_does_not_stop_the_others(tmp_path):
    g1, g2, g3 = tmp_path / "g1", tmp_path / "g2", tmp_path / "g3"
    for g in (g1, g2, g3):
        g.mkdir()
    calls = []

    def make_options(folder):
        if Path(folder).name == "g2":
            raise ValueError("bad options file")
        return {}

    fatal = StageResult("failed", [], "no usable camera", fatal=True)

    def stages_for_all():
        return make_stages(calls)

    summaries = run_batch([str(g1), str(g2), str(g3)], make_options, lambda f: rr.Reporter(),
                          stages=stages_for_all())
    assert [s["state"] for s in summaries] == ["done", "failed", "done"]
    assert summaries[1]["error"] == "bad options file"
    assert summaries[0]["stages"] == {"a": "done", "b": "done", "c": "done"}

    summaries = run_batch([str(g1)], lambda f: {}, lambda f: rr.Reporter(), from_stage="a",
                          stages=make_stages([], behavior={"a": fatal}))
    assert summaries[0]["state"] == "stopped" and summaries[0]["error"] == "no usable camera"


def test_real_stage_list_order():
    assert rr.STAGE_NAMES == ["discovery", "sync", "rois", "detection", "audio", "coverage", "scoresheet"]


def test_audio_stage_wiring(tmp_path):
    stage = next(s for s in rr.STAGES if s.name == "audio")
    assert stage.needs == ("sync",) and stage.needs_two_cameras
    assert stage.outputs(None) == ["audio_signals.json", "music_spans.json"]
    ctx = rr.Context(tmp_path, {"detection": {"fps": 10, "width": 960, "audio_weight": 0}}, rr.Reporter(),
                     use_cache=False)
    argv = rr.audio_argv(ctx)
    assert argv[1].endswith("audio_signals.py") and argv[2] == str(tmp_path)
    assert argv[argv.index("--fps") + 1] == "10" and argv[argv.index("--width") + 1] == "960"
    assert "--no-cache" in argv and "--flow_audio" not in argv
    fp = stage.fingerprint(ctx)
    assert set(fp) >= {"manifests", "files", "music_stack", "scripts"}


def _audio_result(tmp_path, monkeypatch, rc, lines):
    ctx = rr.Context(tmp_path, {}, rr.Reporter())

    def fake_run(argv, on_line=None):
        for ln in lines:
            if on_line:
                on_line(ln)
        return rc, lines

    monkeypatch.setattr(ctx, "run_cmd", fake_run)
    return rr._run_audio(ctx)


def test_audio_stage_flags_missing_music_stack(tmp_path, monkeypatch):
    res = _audio_result(tmp_path, monkeypatch, 0, [
        "[audio] decoded 10/20 s",
        "[audio] 5 whistles, 2 stoppages, 0 music spans (0 s muted)",
        "[audio] flag: needs the optional ML stack (pip install -r requirements-ml.txt)"])
    assert res.state == "flagged" and "ML stack" in res.flags[0]
    assert res.message.startswith("5 whistles")
    assert not res.fatal


def test_audio_stage_done_and_failed(tmp_path, monkeypatch):
    res = _audio_result(tmp_path, monkeypatch, 0, ["[audio] 1 whistles, 0 stoppages, 0 music spans (0 s muted)"])
    assert res.state == "done"
    res = _audio_result(tmp_path, monkeypatch, 1, ["[audio] no camera audio could be read: x"])
    assert res.state == "failed" and not res.fatal


def test_coverage_stage_wiring(tmp_path):
    stage = next(s for s in rr.STAGES if s.name == "coverage")
    assert set(stage.needs) == {"audio", "detection"} and stage.needs_two_cameras
    assert stage.outputs(None) == ["coverage.json"]
    rules = {"periods": 2, "period_minutes": 20, "clock": "running", "break_minutes": 1, "penalty": 2}
    ctx = rr.Context(tmp_path, {"detection": {"fps": 10, "width": 960}, "league_rules": rules},
                     rr.Reporter())
    argv = rr.coverage_argv(ctx)
    assert argv[1].endswith("coverage.py") and argv[2] == str(tmp_path)
    assert argv[argv.index("--fps") + 1] == "10" and argv[argv.index("--width") + 1] == "960"
    import json
    assert json.loads(argv[argv.index("--league") + 1]) == {
        "periods": 2, "period_minutes": 20, "clock": "running", "break_minutes": 1}
    fp = stage.fingerprint(ctx)
    assert fp["league"] == rr.league_timing_rules(ctx)
    assert "--flow_audio" not in argv and fp["flow"]["audio"] is False
    ctx.options["detection"]["audio_weight"] = 0.3
    assert "--flow_audio" in rr.coverage_argv(ctx)


def _coverage_result(tmp_path, monkeypatch, rc, lines):
    ctx = rr.Context(tmp_path, {}, rr.Reporter())

    def fake_run(argv, on_line=None):
        for ln in lines:
            if on_line:
                on_line(ln)
        return rc, lines

    monkeypatch.setattr(ctx, "run_cmd", fake_run)
    return rr._run_coverage(ctx)


def test_coverage_stage_flags_done_and_failed(tmp_path, monkeypatch):
    summary = "[coverage] 3 periods (starts 6:15, 26:38, 46:56), game 6:15-66:53"
    res = _coverage_result(tmp_path, monkeypatch, 0, [
        summary, "[coverage] flag: no League period timing; assumed 3 periods"])
    assert res.state == "flagged" and "League" in res.flags[0] and not res.fatal
    assert res.message.startswith("3 periods")
    assert _coverage_result(tmp_path, monkeypatch, 0, [summary]).state == "done"
    res = _coverage_result(tmp_path, monkeypatch, 1, ["[ERROR] audio_signals.json is missing"])
    assert res.state == "failed" and not res.fatal


def test_detection_argv_uses_options_and_signal_cache(tmp_path):
    from recap_options import DEFAULTS, deep_merge
    opts = deep_merge(DEFAULTS, {"detection": {"thresh_pct": 90}})
    ctx = rr.Context(tmp_path, opts, rr.Reporter(), use_cache=True)
    argv = rr.detection_argv(ctx)
    assert argv[argv.index("--thresh_pct") + 1] == "90"
    assert argv[argv.index("--audio_weight") + 1] == "0"
    assert "--replay_markers" in argv
    assert "--signal_cache" in argv
    ctx.use_cache = False
    assert "--signal_cache" not in rr.detection_argv(ctx)


def test_camera_files_skip_old_outputs(tmp_path):
    for n in ("GX010017.MP4", "cam1.mp4", "cam2.mp4", "recap.mp4"):
        (tmp_path / n).write_bytes(b"x")
    assert [p.name for p in rr.camera_files(tmp_path)] == ["GX010017.MP4"]


# --- review follow-ups -------------------------------------------------------

def _ctx(tmp_path, options=None):
    return rr.Context(tmp_path, options or {}, rr.Reporter())


def _touch_ns(path, ns):
    import os
    os.utime(path, ns=(ns, ns))


def test_roi_mode_auto_when_no_rois(tmp_path):
    assert rr._roi_mode(_ctx(tmp_path)) == "auto"


def test_roi_mode_keeps_rois_from_older_pipeline(tmp_path):
    (tmp_path / "rois.json").write_text("{}")
    assert rr._roi_mode(_ctx(tmp_path)) == "keep"


def test_roi_mode_auto_after_auto_roi_run(tmp_path):
    (tmp_path / "rois.json").write_text("{}")
    (tmp_path / "rois_auto.json").write_text("{}")
    _touch_ns(tmp_path / "rois.json", 1_000_000_000_000)
    _touch_ns(tmp_path / "rois_auto.json", 1_000_000_000_001)
    assert rr._roi_mode(_ctx(tmp_path)) == "auto"


def test_roi_mode_keeps_rois_changed_by_hand_after_auto_run(tmp_path):
    (tmp_path / "rois.json").write_text("{}")
    (tmp_path / "rois_auto.json").write_text("{}")
    _touch_ns(tmp_path / "rois_auto.json", 1_000_000_000_000)
    _touch_ns(tmp_path / "rois.json", 2_000_000_000_000)
    assert rr._roi_mode(_ctx(tmp_path)) == "keep"
    assert rr._roi_mode(_ctx(tmp_path, {"rois": {"source": "auto"}})) == "auto"


def test_discovery_fingerprint_covers_recordings_py(tmp_path, monkeypatch):
    base = rr._fp_discovery(_ctx(tmp_path))
    real = rr._file_hash
    monkeypatch.setattr(rr, "_file_hash",
                        lambda p: "changed" if Path(p).name == "recordings.py" else real(p))
    assert rr._fp_discovery(_ctx(tmp_path)) != base


def test_cache_size_is_in_status_and_batch_summary(tmp_path):
    def with_cache(ctx):
        (ctx.cache_dir / "signals").mkdir(parents=True, exist_ok=True)
        (ctx.cache_dir / "signals" / "signals_x.npz").write_bytes(b"x" * 2048)
        (ctx.game_folder / "a.out").write_text("x")
        return StageResult("done")

    calls = []
    st = run_game(tmp_path, {}, stages=make_stages(calls, {"a": with_cache}))
    assert st["cache_bytes"] == 2048 == rr.cache_size(tmp_path)
    assert load_status(tmp_path)["cache_bytes"] == 2048
    [summary] = run_batch([str(tmp_path)], lambda f: {}, lambda f: rr.Reporter(),
                          stages=make_stages([], {"a": with_cache}))
    assert summary["cache_bytes"] == 2048
    assert rr.human_size(2048) == "2.0 KB"


def test_fingerprint_error_is_a_flag_not_a_crash(tmp_path):
    calls = []
    stages = make_stages(calls)
    def boom(ctx):
        raise OSError("disk gone")
    stages[1] = Stage("b", "B", boom, lambda ctx: ["b.out"], stages[1].run, needs=("a",))
    st = run_game(tmp_path, {}, stages=stages)
    assert calls == ["a", "b", "c"]
    assert st["state"] == "done"


def test_audio_fingerprint_follows_model_file_and_audio_weight(tmp_path, monkeypatch):
    stage = next(s for s in rr.STAGES if s.name == "audio")
    monkeypatch.setattr(rr, "_music_stack_present", lambda: True)
    m1, m2 = tmp_path / "m1.tflite", tmp_path / "m2.tflite"
    m1.write_bytes(b"a")
    m2.write_bytes(b"bb")
    ctx = rr.Context(tmp_path, {"detection": {"audio_weight": 0}}, rr.Reporter())
    monkeypatch.setenv("HHG_YAMNET_MODEL", str(m1))
    fp1 = stage.fingerprint(ctx)
    monkeypatch.setenv("HHG_YAMNET_MODEL", str(m2))
    assert stage.fingerprint(ctx) != fp1
    ctx2 = rr.Context(tmp_path, {"detection": {"audio_weight": 0.3}}, rr.Reporter())
    assert stage.fingerprint(ctx2) != stage.fingerprint(ctx)


def test_audio_fingerprint_retries_while_model_is_missing(tmp_path, monkeypatch):
    stage = next(s for s in rr.STAGES if s.name == "audio")
    monkeypatch.setattr(rr, "_music_stack_present", lambda: True)
    monkeypatch.setenv("HHG_YAMNET_MODEL", str(tmp_path / "missing.tflite"))
    ctx = rr.Context(tmp_path, {}, rr.Reporter())
    a = stage.fingerprint(ctx)
    import time
    time.sleep(0.01)
    assert stage.fingerprint(ctx) != a
