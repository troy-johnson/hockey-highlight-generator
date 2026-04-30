import json

import pytest

from compile_reel import (
    _get_resolve,
    _find_chapter_for_frame,
    _resolve_assemble,
    _tc_to_frames,
    main,
    Event,
    calc_source_frames,
    load_events,
    score_to_color,
    select_events,
)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.99, "Blue"),
        (1.00, "Blue"),
        (1.01, "Yellow"),
        (1.30, "Yellow"),
        (1.31, "Orange"),
        (1.60, "Orange"),
        (1.61, "Red"),
    ],
)
def test_score_to_color_thresholds(score, expected):
    assert score_to_color(score) == expected


def test_load_events_parses_typed_rows_and_derived_color(tmp_path):
    csv_path = tmp_path / "events.csv"
    csv_path.write_text(
        "start_s,end_s,score,primary_cam,confidence\n"
        "12.5,18.0,1.31,cam2,0.87\n",
        encoding="utf-8",
    )

    events = load_events(csv_path)

    assert events == [
        Event(
            start_s=12.5,
            end_s=18.0,
            score=1.31,
            primary_cam="cam2",
            confidence=0.87,
            color="Orange",
        )
    ]
    assert events[0].duration_s == 5.5


def test_select_events_returns_chronological_order_when_under_cap():
    events = [
        Event(20.0, 24.0, 1.61, "cam1", 0.95, "Red"),
        Event(5.0, 9.0, 1.01, "cam2", 0.80, "Yellow"),
    ]

    selected = select_events(events, max_reel_s=20.0)

    assert [event.start_s for event in selected] == [5.0, 20.0]


def test_select_events_drops_blue_before_higher_priority_events():
    blue = Event(12.0, 20.0, 0.95, "cam1", 0.70, "Blue")
    yellow = Event(2.0, 10.0, 1.05, "cam2", 0.82, "Yellow")

    selected = select_events([blue, yellow], max_reel_s=8.0)

    assert selected == [yellow]


def test_select_events_drops_lowest_score_yellow_before_orange():
    yellow_low = Event(2.0, 8.0, 1.05, "cam1", 0.70, "Yellow")
    yellow_high = Event(10.0, 16.0, 1.20, "cam2", 0.84, "Yellow")
    orange = Event(20.0, 26.0, 1.40, "cam1", 0.90, "Orange")

    selected = select_events([orange, yellow_low, yellow_high], max_reel_s=12.0)

    assert selected == [yellow_high, orange]


def test_select_events_never_drops_red_events():
    red = Event(2.0, 12.0, 1.70, "cam1", 0.95, "Red")
    orange = Event(20.0, 30.0, 1.40, "cam2", 0.88, "Orange")

    selected = select_events([red, orange], max_reel_s=5.0)

    assert selected == [red]


def test_calc_source_frames_applies_preroll_postroll_and_zero_clamp():
    event = Event(1.0, 5.0, 1.31, "cam2", 0.87, "Orange")

    src_in, src_out = calc_source_frames(
        event,
        detect_offset_s=0.5,
        timeline_fps=30,
    )

    assert (src_in, src_out) == (0, 225)


def test_get_resolve_raises_clear_error_when_scripting_module_missing(monkeypatch):
    import importlib

    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "DaVinciResolveScript":
            raise ModuleNotFoundError(name)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="DaVinci Resolve scripting module"):
        _get_resolve()


def test_main_exits_cleanly_when_resolve_is_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(
        "compile_reel._get_resolve",
        lambda: (_ for _ in ()).throw(RuntimeError("DaVinci Resolve scripting module not available")),
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert "DaVinci Resolve scripting module not available" in capsys.readouterr().out


def test_main_exits_cleanly_when_folder_selection_is_cancelled(monkeypatch, capsys):
    class _FakeFusion:
        @staticmethod
        def RequestDir(_prompt):
            return ""

    class _FakeResolve:
        @staticmethod
        def Fusion():
            return _FakeFusion()

    monkeypatch.setattr("compile_reel._get_resolve", lambda: _FakeResolve())

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert "No folder selected" in capsys.readouterr().out


def test_main_exits_cleanly_when_events_csv_is_missing(monkeypatch, capsys, tmp_path):
    class _FakeFusion:
        @staticmethod
        def RequestDir(_prompt):
            return str(tmp_path)

    class _FakeResolve:
        @staticmethod
        def Fusion():
            return _FakeFusion()

    monkeypatch.setattr("compile_reel._get_resolve", lambda: _FakeResolve())

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    output = capsys.readouterr().out
    assert "Missing required file" in output
    assert "events.csv" in output


def test_main_exits_cleanly_when_sync_info_json_is_missing(monkeypatch, capsys, tmp_path):
    (tmp_path / "events.csv").write_text(
        "start_s,end_s,score,primary_cam,confidence\n",
        encoding="utf-8",
    )

    class _FakeFusion:
        @staticmethod
        def RequestDir(_prompt):
            return str(tmp_path)

    class _FakeResolve:
        @staticmethod
        def Fusion():
            return _FakeFusion()

    monkeypatch.setattr("compile_reel._get_resolve", lambda: _FakeResolve())

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    output = capsys.readouterr().out
    assert "Missing required file" in output
    assert "sync_info.json" in output


def test_main_exits_cleanly_when_sync_info_json_is_invalid(monkeypatch, capsys, tmp_path):
    (tmp_path / "events.csv").write_text(
        "start_s,end_s,score,primary_cam,confidence\n",
        encoding="utf-8",
    )
    (tmp_path / "sync_info.json").write_text("{not-json", encoding="utf-8")

    class _FakeFusion:
        @staticmethod
        def RequestDir(_prompt):
            return str(tmp_path)

    class _FakeResolve:
        @staticmethod
        def Fusion():
            return _FakeFusion()

    monkeypatch.setattr("compile_reel._get_resolve", lambda: _FakeResolve())

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    output = capsys.readouterr().out
    assert "Failed to parse sync_info.json" in output


def test_main_exits_cleanly_when_sync_info_json_missing_required_keys(
    monkeypatch, capsys, tmp_path
):
    (tmp_path / "events.csv").write_text(
        "start_s,end_s,score,primary_cam,confidence\n",
        encoding="utf-8",
    )
    (tmp_path / "sync_info.json").write_text("{}", encoding="utf-8")

    class _FakeFusion:
        @staticmethod
        def RequestDir(_prompt):
            return str(tmp_path)

    class _FakeResolve:
        @staticmethod
        def Fusion():
            return _FakeFusion()

    monkeypatch.setattr("compile_reel._get_resolve", lambda: _FakeResolve())

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    output = capsys.readouterr().out
    assert "Missing required sync field" in output
    assert "cam1_detect_offset_s" in output


def test_main_exits_cleanly_when_sync_offsets_are_non_numeric(
    monkeypatch, capsys, tmp_path
):
    (tmp_path / "events.csv").write_text(
        "start_s,end_s,score,primary_cam,confidence\n",
        encoding="utf-8",
    )
    (tmp_path / "sync_info.json").write_text(
        '{"cam1_detect_offset_s":"x","cam2_detect_offset_s":0.0}',
        encoding="utf-8",
    )

    class _FakeFusion:
        @staticmethod
        def RequestDir(_prompt):
            return str(tmp_path)

    class _FakeResolve:
        @staticmethod
        def Fusion():
            return _FakeFusion()

    monkeypatch.setattr("compile_reel._get_resolve", lambda: _FakeResolve())

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    output = capsys.readouterr().out
    assert "Invalid sync field type" in output
    assert "cam1_detect_offset_s" in output


def test_main_loads_events_when_required_inputs_are_valid(monkeypatch, tmp_path):
    (tmp_path / "events.csv").write_text(
        "start_s,end_s,score,primary_cam,confidence\n",
        encoding="utf-8",
    )
    (tmp_path / "sync_info.json").write_text(
        '{"cam1_detect_offset_s":0.0,"cam2_detect_offset_s":0.0}',
        encoding="utf-8",
    )

    class _FakeFusion:
        @staticmethod
        def RequestDir(_prompt):
            return str(tmp_path)

    class _FakeResolve:
        @staticmethod
        def Fusion():
            return _FakeFusion()

    calls: list[str] = []

    def _fake_load_events(path):
        calls.append(str(path))
        return []

    monkeypatch.setattr("compile_reel._get_resolve", lambda: _FakeResolve())
    monkeypatch.setattr("compile_reel.load_events", _fake_load_events)
    monkeypatch.setattr("compile_reel._resolve_assemble", lambda *_args, **_kwargs: None)

    main()

    assert calls == [str(tmp_path / "events.csv")]


def test_main_calls_select_events_when_required_inputs_are_valid(monkeypatch, tmp_path):
    (tmp_path / "events.csv").write_text(
        "start_s,end_s,score,primary_cam,confidence\n",
        encoding="utf-8",
    )
    (tmp_path / "sync_info.json").write_text(
        '{"cam1_detect_offset_s":0.0,"cam2_detect_offset_s":0.0}',
        encoding="utf-8",
    )

    class _FakeFusion:
        @staticmethod
        def RequestDir(_prompt):
            return str(tmp_path)

    class _FakeResolve:
        @staticmethod
        def Fusion():
            return _FakeFusion()

    loaded_events = [
        Event(1.0, 2.0, 1.31, "cam1", 0.95, "Orange"),
    ]
    calls = []

    def _fake_load_events(path):
        calls.append(("load_events", str(path)))
        return loaded_events

    def _fake_select_events(events, max_reel_s=900.0):
        calls.append(("select_events", events, max_reel_s))
        return events

    monkeypatch.setattr("compile_reel._get_resolve", lambda: _FakeResolve())
    monkeypatch.setattr("compile_reel.load_events", _fake_load_events)
    monkeypatch.setattr("compile_reel.select_events", _fake_select_events)
    monkeypatch.setattr("compile_reel._resolve_assemble", lambda *_args, **_kwargs: None)

    main()

    assert calls == [
        ("load_events", str(tmp_path / "events.csv")),
        ("select_events", loaded_events, 900.0),
    ]


def test_main_calls_resolve_assemble_with_selected_events_and_sync_info(monkeypatch, tmp_path):
    (tmp_path / "events.csv").write_text(
        "start_s,end_s,score,primary_cam,confidence\n",
        encoding="utf-8",
    )
    sync_info = {"cam1_detect_offset_s": 0.0, "cam2_detect_offset_s": 1.5}
    (tmp_path / "sync_info.json").write_text(
        '{"cam1_detect_offset_s":0.0,"cam2_detect_offset_s":1.5}',
        encoding="utf-8",
    )

    class _FakeFusion:
        @staticmethod
        def RequestDir(_prompt):
            return str(tmp_path)

    class _FakeResolve:
        @staticmethod
        def Fusion():
            return _FakeFusion()

    loaded_events = [Event(1.0, 2.0, 1.31, "cam1", 0.95, "Orange")]
    selected_events = [Event(3.0, 4.0, 1.61, "cam2", 0.98, "Red")]
    calls = []

    monkeypatch.setattr("compile_reel._get_resolve", lambda: _FakeResolve())
    monkeypatch.setattr("compile_reel.load_events", lambda _path: loaded_events)
    monkeypatch.setattr("compile_reel.select_events", lambda _events, max_reel_s=900.0: selected_events)

    def _fake_resolve_assemble(resolve, game_folder, events, parsed_sync_info):
        calls.append((resolve, game_folder, events, parsed_sync_info))

    monkeypatch.setattr("compile_reel._resolve_assemble", _fake_resolve_assemble, raising=False)

    main()

    assert len(calls) == 1
    resolve_obj, game_folder_arg, events_arg, sync_info_arg = calls[0]
    assert isinstance(resolve_obj, _FakeResolve)
    assert game_folder_arg == str(tmp_path)
    assert events_arg == selected_events
    assert sync_info_arg == sync_info


def test_resolve_assemble_exits_when_cam1_folder_missing(capsys, tmp_path):
    with pytest.raises(SystemExit) as exc_info:
        _resolve_assemble(
            resolve=object(),
            game_folder=str(tmp_path),
            events=[],
            sync_info={"cam1_detect_offset_s": 0.0, "cam2_detect_offset_s": 0.0},
        )

    assert exc_info.value.code == 1
    output = capsys.readouterr().out
    assert "Missing required folder" in output
    assert "cam1" in output


def test_resolve_assemble_exits_when_cam2_folder_missing(capsys, tmp_path):
    (tmp_path / "cam1").mkdir()

    with pytest.raises(SystemExit) as exc_info:
        _resolve_assemble(
            resolve=object(),
            game_folder=str(tmp_path),
            events=[],
            sync_info={"cam1_detect_offset_s": 0.0, "cam2_detect_offset_s": 0.0},
        )

    assert exc_info.value.code == 1
    output = capsys.readouterr().out
    assert "Missing required folder" in output
    assert "cam2" in output


def test_resolve_assemble_exits_when_chapters_json_missing(capsys, tmp_path):
    (tmp_path / "cam1").mkdir()
    (tmp_path / "cam2").mkdir()

    with pytest.raises(SystemExit) as exc_info:
        _resolve_assemble(
            resolve=object(),
            game_folder=str(tmp_path),
            events=[],
            sync_info={"cam1_detect_offset_s": 0.0, "cam2_detect_offset_s": 0.0},
        )

    assert exc_info.value.code == 1
    output = capsys.readouterr().out
    assert "Missing required file" in output
    assert "chapters.json" in output


# ---------------------------------------------------------------------------
# Fake Resolve hierarchy for API-level tests
# ---------------------------------------------------------------------------


def _setup_game_folder(tmp_path, cam1_files=None, cam2_files=None):
    """Create cam1/, cam2/ dirs and chapters.json; return chapters dict."""
    (tmp_path / "cam1").mkdir(exist_ok=True)
    (tmp_path / "cam2").mkdir(exist_ok=True)
    chapters = {
        "cam1": cam1_files or [str(tmp_path / "cam1" / "GOPRO1801.MP4")],
        "cam2": cam2_files or [str(tmp_path / "cam2" / "GOPRO1901.MP4")],
    }
    (tmp_path / "chapters.json").write_text(json.dumps(chapters), encoding="utf-8")
    return chapters


class _FakeTimelineItem:
    def __init__(self, start, duration):
        self._start = start
        self._duration = duration

    def GetStart(self):
        return self._start

    def GetDuration(self):
        return self._duration


class _FakeMediaItem:
    def __init__(self, path, duration_tc="01:00:00:00"):
        self._path = path
        self._duration_tc = duration_tc

    def GetClipProperty(self, key):
        if key == "File Path":
            return self._path
        if key == "Duration":
            return self._duration_tc
        return None


class _FakePlacedItem:
    def __init__(self):
        self.angles_set = []

    def SetCurrentVideoItem(self, angle):
        self.angles_set.append(angle)


class _FakeMediaPool:
    def __init__(self, import_returns, multicam_returns, append_returns):
        self._import_returns = import_returns
        self._multicam_returns = multicam_returns
        self._append_returns = append_returns
        self.root_folder = object()
        self.imported_paths = None
        self.append_calls = []
        self.multicam_opts = None

    def GetRootFolder(self):
        return self.root_folder

    def AddSubFolder(self, folder, name):
        return object()

    def SetCurrentFolder(self, folder):
        pass

    def ImportMedia(self, paths):
        self.imported_paths = list(paths)
        return self._import_returns

    def CreateMultiCamClip(self, items, opts):
        self.multicam_opts = opts
        return self._multicam_returns

    def AppendToTimeline(self, clip_list):
        self.append_calls.append(clip_list)
        return self._append_returns


class _FakeTimeline:
    def __init__(self, items=None):
        self._items = items or []

    def GetName(self):
        return "Test Timeline"

    def GetItemListInTrack(self, track_type, track_num):
        return self._items


class _FakeProject:
    def __init__(self, media_pool, timeline, fps="60"):
        self._media_pool = media_pool
        self._timeline = timeline
        self._fps = fps

    def GetCurrentTimeline(self):
        return self._timeline

    def GetMediaPool(self):
        return self._media_pool

    def GetSetting(self, key):
        return self._fps if key == "timelineFrameRate" else None


class _FakeProjectManager:
    def __init__(self, project):
        self._project = project

    def GetCurrentProject(self):
        return self._project


class _FakeResolve:
    def __init__(self, pm):
        self._pm = pm

    def GetProjectManager(self):
        return self._pm


_UNSET = object()


def _make_resolve(tmp_path, import_returns, multicam_returns=_UNSET, append_returns=_UNSET,
                  timeline_items=None, fps="60"):
    placed = _FakePlacedItem()
    media_pool = _FakeMediaPool(
        import_returns=import_returns,
        multicam_returns=object() if multicam_returns is _UNSET else multicam_returns,
        append_returns=[placed] if append_returns is _UNSET else append_returns,
    )
    timeline = _FakeTimeline(items=timeline_items)
    proj = _FakeProject(media_pool, timeline, fps=fps)
    pm = _FakeProjectManager(proj)
    return _FakeResolve(pm), media_pool, placed


def test_resolve_assemble_exits_when_no_project_manager(capsys, tmp_path):
    _setup_game_folder(tmp_path)

    class _NoPmResolve:
        def GetProjectManager(self):
            return None

    with pytest.raises(SystemExit) as exc_info:
        _resolve_assemble(
            resolve=_NoPmResolve(),
            game_folder=str(tmp_path),
            events=[],
            sync_info={"cam1_detect_offset_s": 0.0, "cam2_detect_offset_s": 0.0},
        )

    assert exc_info.value.code == 1
    assert "No project manager" in capsys.readouterr().out


def test_resolve_assemble_exits_when_no_current_project(capsys, tmp_path):
    _setup_game_folder(tmp_path)

    class _NoProjPm:
        def GetCurrentProject(self):
            return None

    class _Resolve:
        def GetProjectManager(self):
            return _NoProjPm()

    with pytest.raises(SystemExit) as exc_info:
        _resolve_assemble(
            resolve=_Resolve(),
            game_folder=str(tmp_path),
            events=[],
            sync_info={"cam1_detect_offset_s": 0.0, "cam2_detect_offset_s": 0.0},
        )

    assert exc_info.value.code == 1
    assert "No active project" in capsys.readouterr().out


def test_resolve_assemble_exits_when_no_current_timeline(capsys, tmp_path):
    _setup_game_folder(tmp_path)

    class _NoTlProject:
        def GetCurrentTimeline(self):
            return None

    class _Pm:
        def GetCurrentProject(self):
            return _NoTlProject()

    class _Resolve:
        def GetProjectManager(self):
            return _Pm()

    with pytest.raises(SystemExit) as exc_info:
        _resolve_assemble(
            resolve=_Resolve(),
            game_folder=str(tmp_path),
            events=[],
            sync_info={"cam1_detect_offset_s": 0.0, "cam2_detect_offset_s": 0.0},
        )

    assert exc_info.value.code == 1
    assert "No active timeline" in capsys.readouterr().out


def test_resolve_assemble_exits_when_import_media_returns_nothing(capsys, tmp_path):
    _setup_game_folder(tmp_path)
    resolve, media_pool, _ = _make_resolve(tmp_path, import_returns=[])

    with pytest.raises(SystemExit) as exc_info:
        _resolve_assemble(
            resolve=resolve,
            game_folder=str(tmp_path),
            events=[],
            sync_info={"cam1_detect_offset_s": 0.0, "cam2_detect_offset_s": 0.0},
        )

    assert exc_info.value.code == 1
    assert "ImportMedia" in capsys.readouterr().out


def test_resolve_assemble_falls_back_to_dual_track_when_multicam_fails(tmp_path):
    chapters = _setup_game_folder(tmp_path)
    cam1_path = chapters["cam1"][0]
    cam2_path = chapters["cam2"][0]
    placed = _FakePlacedItem()
    resolve, media_pool, _ = _make_resolve(
        tmp_path,
        import_returns=[_FakeMediaItem(cam1_path), _FakeMediaItem(cam2_path)],
        multicam_returns=None,
        append_returns=[placed],
    )

    events = [
        Event(10.0, 20.0, 1.61, "cam1", 0.95, "Red"),
        Event(30.0, 40.0, 1.31, "cam2", 0.88, "Orange"),
    ]
    sync_info = {
        "cam1_detect_offset_s": 0.0,
        "cam2_detect_offset_s": 0.0,
        "sync_method": "timecode",
    }

    # Must not raise SystemExit
    _resolve_assemble(
        resolve=resolve,
        game_folder=str(tmp_path),
        events=events,
        sync_info=sync_info,
    )

    assert len(media_pool.append_calls) == 2
    # cam1 event → Track 1
    assert media_pool.append_calls[0][0]["trackIndex"] == 1
    # cam2 event → Track 2
    assert media_pool.append_calls[1][0]["trackIndex"] == 2


def test_resolve_assemble_uses_audio_sync_for_non_timecode_sync_method(tmp_path):
    chapters = _setup_game_folder(tmp_path)
    cam1_path = chapters["cam1"][0]
    cam2_path = chapters["cam2"][0]
    resolve, media_pool, _ = _make_resolve(
        tmp_path,
        import_returns=[_FakeMediaItem(cam1_path), _FakeMediaItem(cam2_path)],
    )

    _resolve_assemble(
        resolve=resolve,
        game_folder=str(tmp_path),
        events=[],
        sync_info={
            "cam1_detect_offset_s": 0.0,
            "cam2_detect_offset_s": 0.0,
            "sync_method": "creation_time",
        },
    )

    assert media_pool.multicam_opts["syncType"] == "audio"


def test_resolve_assemble_uses_timecode_sync_for_timecode_sync_method(tmp_path):
    chapters = _setup_game_folder(tmp_path)
    cam1_path = chapters["cam1"][0]
    cam2_path = chapters["cam2"][0]
    resolve, media_pool, _ = _make_resolve(
        tmp_path,
        import_returns=[_FakeMediaItem(cam1_path), _FakeMediaItem(cam2_path)],
    )

    _resolve_assemble(
        resolve=resolve,
        game_folder=str(tmp_path),
        events=[],
        sync_info={
            "cam1_detect_offset_s": 0.0,
            "cam2_detect_offset_s": 0.0,
            "sync_method": "timecode",
        },
    )

    assert media_pool.multicam_opts["syncType"] == "timecode"


def test_resolve_assemble_places_clips_with_correct_source_frames(tmp_path):
    chapters = _setup_game_folder(tmp_path)
    cam1_path = chapters["cam1"][0]
    cam2_path = chapters["cam2"][0]
    multicam = object()
    placed = _FakePlacedItem()
    resolve, media_pool, _ = _make_resolve(
        tmp_path,
        import_returns=[_FakeMediaItem(cam1_path), _FakeMediaItem(cam2_path)],
        multicam_returns=multicam,
        append_returns=[placed],
        fps="60",
    )

    events = [
        Event(10.0, 20.0, 1.61, "cam1", 0.95, "Red"),
        Event(30.0, 40.0, 1.31, "cam2", 0.88, "Orange"),
    ]
    sync_info = {"cam1_detect_offset_s": 0.0, "cam2_detect_offset_s": 1.5}

    _resolve_assemble(
        resolve=resolve,
        game_folder=str(tmp_path),
        events=events,
        sync_info=sync_info,
    )

    assert len(media_pool.append_calls) == 2

    # cam1 event: offset=0.0, start=10, end=20, preroll=3, postroll=2, fps=60
    # src_in  = max(0, 0 + 10 - 3) = 7s  → 420 frames
    # src_out = 0 + 20 + 2 = 22s         → 1320 frames
    first = media_pool.append_calls[0][0]
    assert first["startFrame"] == 420
    assert first["endFrame"] == 1320
    assert first["mediaPoolItem"] is multicam

    # cam2 event: offset=1.5, start=30, end=40
    # src_in  = max(0, 1.5 + 30 - 3) = 28.5s → 1710 frames
    # src_out = 1.5 + 40 + 2 = 43.5s          → 2610 frames
    second = media_pool.append_calls[1][0]
    assert second["startFrame"] == 1710
    assert second["endFrame"] == 2610

    # Angles: cam1 → 1, cam2 → 2
    assert placed.angles_set == [1, 2]


def test_resolve_assemble_imports_all_chapter_files(tmp_path):
    cam1_files = [str(tmp_path / "cam1" / f"GOPRO180{i}.MP4") for i in range(1, 4)]
    cam2_files = [str(tmp_path / "cam2" / f"GOPRO190{i}.MP4") for i in range(1, 3)]
    chapters = _setup_game_folder(tmp_path, cam1_files=cam1_files, cam2_files=cam2_files)

    all_items = [_FakeMediaItem(p) for p in cam1_files + cam2_files]
    resolve, media_pool, _ = _make_resolve(
        tmp_path,
        import_returns=all_items,
        multicam_returns=object(),
        append_returns=[_FakePlacedItem()],
    )

    _resolve_assemble(
        resolve=resolve,
        game_folder=str(tmp_path),
        events=[],
        sync_info={"cam1_detect_offset_s": 0.0, "cam2_detect_offset_s": 0.0},
    )

    assert set(media_pool.imported_paths) == set(cam1_files + cam2_files)


# ---------------------------------------------------------------------------
# _find_chapter_for_frame / dual-track chapter boundary clamping
# ---------------------------------------------------------------------------


def test_find_chapter_for_frame_single_chapter():
    item = _FakeMediaItem("/a/GOPRO1801.MP4", duration_tc="00:01:00:00")  # 60s = 3600f at 60fps
    chapter_item, local_frame, ch_start, ch_dur = _find_chapter_for_frame([item], 1200, 60)
    assert chapter_item is item
    assert local_frame == 1200
    assert ch_start == 0
    assert ch_dur == _tc_to_frames("00:01:00:00", 60)


def test_find_chapter_for_frame_second_of_two_chapters():
    item1 = _FakeMediaItem("/a/GOPRO1801.MP4", duration_tc="00:01:00:00")  # 3600f
    item2 = _FakeMediaItem("/a/GOPRO1802.MP4", duration_tc="00:01:00:00")  # 3600f
    # Frame 4000 is in chapter 2 (400 frames in)
    chapter_item, local_frame, ch_start, ch_dur = _find_chapter_for_frame(
        [item1, item2], 4000, 60
    )
    assert chapter_item is item2
    assert local_frame == 400
    assert ch_start == 3600


def test_dual_track_fallback_clamps_cross_chapter_event_to_chapter_boundary(tmp_path):
    """An event whose src_out falls into chapter 2 should be clamped to end of chapter 1."""
    cam1_path = str(tmp_path / "cam1" / "GOPRO1801.MP4")
    cam2_path = str(tmp_path / "cam2" / "GOPRO1901.MP4")
    _setup_game_folder(tmp_path)

    # Chapter 1 is 60 s = 3600 frames at 60fps
    # Event: start=55s (src_in≈3120f), end=70s (src_out≈4320f) → spans into chapter 2
    # Expected: local_out clamped to 3600 (end of chapter 1), not 4320-3600=720
    event = Event(55.0, 70.0, 1.61, "cam1", 0.95, "Red")
    items = [
        _FakeMediaItem(cam1_path, duration_tc="00:01:00:00"),  # 3600f
        _FakeMediaItem(cam2_path, duration_tc="00:01:00:00"),
    ]
    append_calls = []

    class _Mp:
        def GetRootFolder(self): return object()
        def AddSubFolder(self, *a): return object()
        def SetCurrentFolder(self, *a): pass
        def ImportMedia(self, paths): return items
        def CreateMultiCamClip(self, *a): return None
        def AppendToTimeline(self, clips):
            append_calls.append(clips)
            return [_FakePlacedItem()]

    from compile_reel import _place_clips_dual_track
    _place_clips_dual_track(
        media_pool=_Mp(),
        cam1_items=[items[0]],
        cam2_items=[items[1]],
        events=[event],
        sync_info={"cam1_detect_offset_s": 0.0, "cam2_detect_offset_s": 0.0},
        timeline_fps=60,
        stinger_end=0,
    )

    assert len(append_calls) == 1
    clip = append_calls[0][0]
    # local_in: src_in = max(0, 0+55-3)=52s → 3120f; local_in=3120
    # local_out clamped to chapter end = 3600 (not 3600+720=4320 or local 720)
    assert clip["startFrame"] == 3120
    assert clip["endFrame"] == 3600


# ---------------------------------------------------------------------------
# --max_reel_s CLI argument
# ---------------------------------------------------------------------------


def test_main_respects_max_reel_s_argument(monkeypatch, tmp_path):
    (tmp_path / "events.csv").write_text(
        "start_s,end_s,score,primary_cam,confidence\n"
        "0.0,600.0,1.61,cam1,0.95\n"   # Red, 600s
        "700.0,800.0,0.5,cam2,0.60\n",  # Blue, 100s — total 700s > 300s cap
        encoding="utf-8",
    )
    (tmp_path / "sync_info.json").write_text(
        '{"cam1_detect_offset_s":0.0,"cam2_detect_offset_s":0.0}',
        encoding="utf-8",
    )

    class _FakeFusion:
        @staticmethod
        def RequestDir(_prompt): return str(tmp_path)

    class _FakeResolve:
        @staticmethod
        def Fusion(): return _FakeFusion()

    calls = []

    def _fake_select_events(events, max_reel_s=900.0):
        calls.append(max_reel_s)
        return events

    monkeypatch.setattr("compile_reel._get_resolve", lambda: _FakeResolve())
    monkeypatch.setattr("compile_reel.select_events", _fake_select_events)
    monkeypatch.setattr("compile_reel._resolve_assemble", lambda *a, **kw: None)

    main(["--max_reel_s", "300"])

    assert calls == [300.0]
