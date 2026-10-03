import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from test_recap_assembly import fixture
from test_recap_graphics import game


def test_cold_open_precedes_stinger_without_reordering_game():
    import recap_assembly as assembly
    import audio_mix as audio
    selection, layouts = fixture(2)
    selection['goals'][0]['interest'] = 10
    selection['goals'][1]['interest'] = 90
    original = copy.deepcopy(selection)
    plan = assembly.plan_recap(selection, layouts, start='cold_open')
    timeline = audio.build_timeline(plan)
    assert [e['kind'] for e in timeline[:3]] == ['cold_open', 'stinger', 'goal']
    assert timeline[0]['goal_id'] == 'home:1'
    assert timeline[1]['out_start'] == timeline[0]['frames'] / 30
    assert timeline[1]['frames'] == 75
    assert [e['goal_id'] for e in timeline if e['kind'] == 'goal'] == ['home:0', 'home:1']
    assert plan['duration_s'] == pytest.approx(sum(e['frames'] for e in timeline) / 30)
    assert selection == original


def test_reserved_open_and_period_wipe_follow_audio_cues(tmp_path):
    import audio_mix as am
    import recap_graphics as graphics
    sheet, selection, plan, audio = game()
    plan['opening'] = [{'kind': 'stinger', 'goal_id': 'open', 'frames': 75,
                        'start_s': 10, 'moment_s': 10, 'speed': 1, 'parts': []}]
    plan['start'] = 'stinger'
    plan['duration_s'] = 42.5
    audio['duration_s'] = 42.5
    audio['cues'] = am.plan_cues(am.build_timeline(plan), duration_s=42.5, period_wipes=[32.5])
    props = graphics.write_props(tmp_path, sheet, selection, plan, audio, {})
    opening = next(e for e in props['events'] if e['kind'] == 'open_stinger')
    wipe = next(e for e in props['events'] if e['kind'] == 'period_wipe')
    period = next(e for e in props['events'] if e['kind'] == 'period')
    assert (opening['startFrame'], opening['durationFrames']) == (0, 75)
    assert (wipe['startFrame'], wipe['durationFrames']) == (975, 30)
    assert (period['startFrame'], period['durationFrames']) == (1005, 60)
    assert wipe['data']['period'] == 2
    assert min(e['startFrame'] for e in props['events'] if e['kind'] == 'scorebug') == 75
    assert props['durationFrames'] == 1275


@pytest.mark.parametrize('start,kinds', [('stinger', ['stinger']), ('play', []),
                                       ('cold_open', ['cold_open', 'stinger'])])
def test_open_options_reach_assembly_and_preserve_cap(start, kinds):
    import recap_assembly as assembly
    import recap_runner as runner
    selection, layouts = fixture(38)
    plan = assembly.plan_recap(selection, layouts, start=start)
    assert [e['kind'] for e in plan['opening']] == kinds
    assert plan['duration_s'] <= 240
    ctx = type('Context', (), {'options': {'start': start}})()
    assert runner.assembly_options(ctx)['start'] == start
    assert runner.graphics_options(ctx)['start'] == start


def test_renderer_cli_rejects_out_of_bounds_stinger_before_render(tmp_path):
    import recap_graphics as graphics
    if not shutil.which('node'):
        pytest.skip('Node is required for the graphics renderer')
    sheet, selection, plan, audio = game()
    props = graphics.write_props(tmp_path, sheet, selection, plan, audio, {})
    props['events'].append({'kind': 'open_stinger', 'startFrame': 1190,
                            'durationFrames': 75, 'data': {'team': None}})
    source = tmp_path / 'props.json'
    source.write_text(json.dumps(props))
    renderer = Path(__file__).resolve().parents[1] / 'v3/graphics/render.mjs'
    output = tmp_path / 'render'
    result = subprocess.run(['node', str(renderer), str(source), str(output), '--validate'],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert 'Event exceeds graphics duration' in result.stderr
    assert not output.exists()


def test_legacy_plan_requires_assembly_for_cold_open(tmp_path):
    import recap_graphics as graphics
    sheet, selection, plan, audio = game()
    with pytest.raises(ValueError, match='cold open requires.*rerun assembly'):
        graphics.write_props(tmp_path, sheet, selection, plan, audio, {'start': 'cold_open'})


def test_renderer_cli_rejects_non_public_logo(tmp_path):
    import recap_graphics as graphics
    if not shutil.which('node'):
        pytest.skip('Node is required for the graphics renderer')
    props = graphics.write_props(tmp_path, *game(), {})
    props['teams']['home']['logo'] = '../outside.png'
    source = tmp_path / 'props.json'
    source.write_text(json.dumps(props))
    renderer = Path(__file__).resolve().parents[1] / 'v3/graphics/render.mjs'
    result = subprocess.run(['node', str(renderer), str(source), str(tmp_path / 'render'), '--validate'],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert 'Logo must be a public asset basename' in result.stderr


def test_cold_open_uses_surviving_play_after_source_validation(monkeypatch):
    import recap_assembly as assembly
    selection, layouts = fixture(2)
    selection['goals'][0]['interest'] = 10
    selection['goals'][1]['interest'] = 90
    plan = assembly.plan_recap(selection, layouts, start='cold_open')
    # The second full clip ends after 63 seconds. Its shorter teaser still fits.
    monkeypatch.setattr(subprocess, 'check_output',
                        lambda argv: b'{"streams": [{"duration": "63"}]}')
    assembly.verify_sources(plan)
    assert [c['goal_id'] for c in plan['clips']] == ['home:0']
    assert plan['opening'][0]['goal_id'] == 'home:0'
    assert [c['kind'] for c in plan['opening']] == ['cold_open', 'stinger']


def test_synthetic_cold_open_renders_exact_reserved_frames(tmp_path):
    import recap_assembly as assembly
    import audio_mix as audio
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg is required for assembly')
    source = tmp_path / 'source.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    'color=c=blue:s=64x36:r=30', '-frames:v', '15', str(source)], check=True)
    clip = {'kind': 'goal', 'goal_id': 'home:0', 'start_s': 0, 'end_s': .5,
            'moment_s': .2, 'frames': 15, 'speed': 1, 'zoom': {'type': 'fixed', 'z': 1},
            'parts': [{'file': str(source), 'seek_s': 0, 'duration_s': .5}]}
    plan = {'clips': [clip], 'replays': [], 'opening': [dict(clip, kind='cold_open'),
            {'kind': 'stinger', 'goal_id': 'open', 'start_s': 0, 'moment_s': 0,
             'frames': 75, 'speed': 1, 'parts': []}], 'duration_s': 3.5, 'flags': []}
    output = tmp_path / 'recap.mp4'
    assembly.render(plan, output)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
                                               '-of', 'json', str(output)]))
    assert [(s['codec_type'], s.get('nb_frames')) for s in probe['streams']] == [('video', '105')]
    cues = audio.plan_cues(audio.build_timeline(plan), duration_s=3.5)
    assert next(c['t'] for c in cues if c['cue'] == 'game_start') == .5
    assert [c['t'] for c in cues if c['cue'] == 'horn'] == [3.2]
