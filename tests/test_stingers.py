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
    assert timeline[1]['frames'] == 135
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
    assert (wipe['startFrame'], wipe['durationFrames']) == (960, 30)
    assert wipe['data']['cutFrame'] == 975
    assert (period['startFrame'], period['durationFrames']) == (990, 82)
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
    # The second full clip ends after 62 seconds. Its shorter teaser still fits.
    monkeypatch.setattr(subprocess, 'check_output',
                        lambda argv: b'{"streams": [{"duration": "62"}]}')
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


def test_opening_exits_on_its_own_without_a_second_frost_pane(tmp_path):
    import audio_mix as am
    import recap_graphics as graphics
    sheet, selection, plan, audio = game()
    plan['opening'] = [{'kind': 'stinger', 'goal_id': 'open', 'frames': 135,
                        'start_s': 10, 'moment_s': 10, 'speed': 1, 'parts': []}]
    plan['start'] = 'stinger'
    plan['duration_s'] = 44.5
    audio['duration_s'] = 44.5
    audio['cues'] = am.plan_cues(am.build_timeline(plan), duration_s=44.5)
    props = graphics.write_props(tmp_path, sheet, selection, plan, audio, {})
    opening = [e for e in props['events'] if e['kind'] == 'open_stinger']
    assert [(e['startFrame'], e['durationFrames']) for e in opening] == [(0, 135)]
    panes = [e for e in props['events'] if e['kind'] == 'replay_pane' and e['startFrame'] < 143]
    assert panes == []


OPENING = {'path': 'opening.mp4', 'frames': 148, 'dissolve': 9}


def test_pre_rendered_opening_sets_the_stinger_length():
    import recap_assembly as assembly
    selection, layouts = fixture(2)
    plain = assembly.plan_recap(selection, layouts, start='stinger')
    plan = assembly.plan_recap(selection, layouts, start='stinger', opening_video=OPENING)
    stinger = plan['opening'][0]
    assert (stinger['kind'], stinger['frames'], stinger['opening_video']) == ('stinger', 139, OPENING)
    assert plan['duration_s'] == pytest.approx(plain['duration_s'] + 4 / 30)
    assert assembly.plan_recap(selection, layouts, start='play', opening_video=OPENING)['opening'] == []


def _video(path, color, frames, size='1920x1080'):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', f'color=c={color}:s={size}:r=30',
                    '-frames:v', str(frames), '-pix_fmt', 'yuv420p', str(path)], check=True)


def test_probe_opening_checks_size_and_rate(tmp_path):
    import recap_assembly as assembly
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg is required for assembly')
    _video(tmp_path / 'good.mp4', 'red', 48)
    _video(tmp_path / 'small.mp4', 'red', 48, '64x36')
    assert assembly.probe_opening(tmp_path / 'good.mp4') == {
        'path': str(tmp_path / 'good.mp4'), 'frames': 48, 'dissolve': 9}
    with pytest.raises(ValueError, match='1920x1080'):
        assembly.probe_opening(tmp_path / 'small.mp4')


def test_pre_rendered_opening_dissolves_into_the_first_play(tmp_path):
    import recap_assembly as assembly
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg is required for assembly')
    opening, source = tmp_path / 'opening.mp4', tmp_path / 'source.mp4'
    _video(opening, 'red', 48)
    _video(source, 'blue', 30, '64x36')
    clip = {'kind': 'goal', 'goal_id': 'home:0', 'start_s': 0, 'end_s': 1,
            'moment_s': .5, 'frames': 30, 'speed': 1, 'zoom': {'type': 'fixed', 'z': 1},
            'parts': [{'file': str(source), 'seek_s': 0, 'duration_s': 1}]}
    video = {'path': str(opening), 'frames': 48, 'dissolve': 9}
    plan = {'clips': [clip], 'replays': [], 'opening': [
            {'kind': 'stinger', 'goal_id': 'open', 'start_s': 0, 'moment_s': 0, 'frames': 39,
             'speed': 1, 'parts': [], 'opening_video': video}], 'duration_s': 69 / 30, 'flags': []}
    output = tmp_path / 'recap.mp4'
    assembly.render(plan, output)
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(output), '-vf', 'scale=4:4',
                                   '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    frames = [raw[i:i + 48] for i in range(0, len(raw), 48)]
    red = [f[0] - f[2] for f in frames]          # red minus blue, one pixel per frame
    assert len(frames) == 69
    assert min(red[:39]) > 200                    # the opening, up to its dissolve
    assert red[38] > red[43] > red[47]            # the dissolve, across 9 frames
    assert -200 < red[43] < 200
    assert max(red[48:]) < -200                   # the first play, at full length


def test_pre_rendered_opening_draws_only_the_matchup_rail(tmp_path):
    import audio_mix as am
    import recap_graphics as graphics
    sheet, selection, plan, audio = game()
    plan['opening'] = [{'kind': 'stinger', 'goal_id': 'open', 'frames': 139, 'start_s': 10,
                        'moment_s': 10, 'speed': 1, 'parts': [], 'opening_video': OPENING}]
    plan['start'] = 'stinger'
    plan['duration_s'] = 40 + 139 / 30
    audio['duration_s'] = plan['duration_s']
    audio['cues'] = am.plan_cues(am.build_timeline(plan), duration_s=plan['duration_s'])
    props = graphics.write_props(tmp_path, sheet, selection, plan, audio, {})
    assert not [e for e in props['events'] if e['kind'] == 'open_stinger']
    rail = [e for e in props['events'] if e['kind'] == 'matchup_rail']
    assert len(rail) == 1 and rail[0]['startFrame'] == 95
    end = rail[0]['startFrame'] + rail[0]['durationFrames']
    later = [e['startFrame'] for e in props['events'] if e['kind'] in ('goal', 'penalty', 'call', 'period')]
    assert end == min([147 + 75 + 10] + [s - 4 for s in later if s >= 95])
    if not shutil.which('node'):
        return
    source = tmp_path / 'props.json'
    source.write_text(json.dumps(props))
    renderer = Path(__file__).resolve().parents[1] / 'v3/graphics/render.mjs'
    result = subprocess.run(['node', str(renderer), str(source), str(tmp_path / 'render'), '--validate'],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_matchup_rail_gives_way_to_the_first_information_rail(tmp_path):
    import audio_mix as am
    import recap_graphics as graphics
    sheet, selection, plan, audio = game()
    plan['opening'] = [{'kind': 'stinger', 'goal_id': 'open', 'frames': 139, 'start_s': 10,
                        'moment_s': 10, 'speed': 1, 'parts': [], 'opening_video': dict(OPENING, frames=1000)}]
    plan['start'] = 'stinger'
    plan['duration_s'] = 40 + 139 / 30
    audio['duration_s'] = plan['duration_s']
    audio['cues'] = am.plan_cues(am.build_timeline(plan), duration_s=plan['duration_s'])
    props = graphics.write_props(tmp_path, sheet, selection, plan, audio, {})
    rail = next(e for e in props['events'] if e['kind'] == 'matchup_rail')
    first = min(e['startFrame'] for e in props['events'] if e['kind'] in ('goal', 'penalty', 'call', 'period'))
    assert rail['startFrame'] + rail['durationFrames'] == first - 4
    assert 'matchup rail shortened to clear the first information rail' in props['flags']


def test_runner_finds_the_kit_opening_video(tmp_path):
    import recap_runner as runner
    layers = {'config_dir': str(tmp_path)}
    team = {'name': 'Ice Pak', 'opening': '/abs/team.mp4',
            'kits': {'red': {'fill': '#A60735', 'line': '#E8354F', 'opening': 'assets/red.mp4'}, 'navy': {}}}
    def options(kit):
        ctx = type('Context', (), {'options': {'focus_team': 'icepak', 'kit': kit, '_layers': layers,
                                               'teams': {'icepak': team}}})()
        return runner.assembly_options(ctx)
    assert options('red')['opening'] == str(tmp_path / 'assets/red.mp4')
    assert options('navy')['opening'] == '/abs/team.mp4'
    team.pop('opening')
    assert 'opening' not in options('navy')
