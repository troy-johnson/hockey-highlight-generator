"""Check published edits without assigning camera or goal labels."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3" / "scripts"))
from answer_key_edl import parse_edl, timecode_frames


def test_non_drop_timecode_uses_explicit_rate():
    assert timecode_frames("00:01:10:49", 60) == 4249
    with pytest.raises(ValueError, match="frame"):
        timecode_frames("00:00:00:49", 30)


def test_parser_keeps_incoming_transition_and_excludes_graphics_and_audio():
    text = """TITLE: Game
FCM: NON-DROP FRAME
001 AX V C 01:00:00:00 01:00:04:30 01:00:00:00 01:00:04:30
* FROM CLIP NAME: icepak_stinger.mp4
002 BL V C 01:00:05:00 01:00:05:00 01:00:05:00 01:00:05:00
002 AX V D 030 00:01:10:49 00:01:15:04 01:00:05:00 01:00:09:15
* FROM CLIP NAME: DTCD .5
* TO CLIP NAME: Game v1
003 AX V C 00:01:15:04 00:01:21:51 01:00:09:15 01:00:16:02
* FROM CLIP NAME: Game v1
004 AX A C 00:00:00:00 00:00:04:00 01:00:08:00 01:00:12:00
* FROM CLIP NAME: Goal - Opponent.mp3
005 AX V C 00:03:00:00 00:03:06:00 01:00:16:02 01:00:22:02
* FROM CLIP NAME: Game v 1
"""
    result = parse_edl(text, fps=60)
    clips = result["clips"]
    assert len(clips) == 3
    assert clips[0]["source_start_s"] == pytest.approx(70 + 49 / 60)
    assert clips[0]["recap_start_s"] == 5
    assert clips[0]["transition_frames"] == 30
    assert clips[2]["source_name"] == "Game v 1"
    assert all(c["camera"] is None and c["goal"] is None for c in clips)
    assert result["audio_cues"][0]["recap_start_s"] == 8
    assert result["audio_cues"][0]["source_name"] == "Goal - Opponent.mp3"


def test_outgoing_transition_does_not_duplicate_kept_clip():
    text = """FCM: NON-DROP FRAME
001 AX V C 00:03:00:00 00:03:00:00 01:00:05:00 01:00:05:00
001 AX V D 030 00:03:00:00 00:03:09:49 01:00:05:00 01:00:14:49
* FROM CLIP NAME: Game v1
* TO CLIP NAME: Game v1
"""
    assert len(parse_edl(text, fps=60)["clips"]) == 1


@pytest.mark.parametrize("text", [
    "FCM: DROP FRAME\n",
    "FCM: NON-DROP FRAME\nM2 AX 060.0 00:00:00:00\n",
    "001 AX V C 00:00:00:00 00:00:04:00 01:00:00:00 01:00:04:00\n",
    "001 AX V C malformed\n",
])
def test_ambiguous_edl_blocks_instead_of_inventing_mapping(text):
    with pytest.raises(ValueError):
        parse_edl(text, fps=60)


def test_absolute_record_origin_is_not_first_kept_clip():
    text = """001 AX V C 00:01:00:00 00:01:04:00 01:00:05:00 01:00:09:00
* FROM CLIP NAME: Game v1
"""
    assert parse_edl(text, fps=60)["clips"][0]["recap_start_s"] == 5


def test_speed_effect_is_retained_as_evidence_not_applied_as_raw_time():
    text = """001 AX V C 00:01:00:00 00:01:04:00 01:00:05:00 01:00:09:00
M2 AX 060.1 00:01:00:00
* FROM CLIP NAME: Game v1
"""
    clip = parse_edl(text, fps=60)["clips"][0]
    assert clip["speed_effect"] == {"source_rate": 60.1, "source_anchor": "00:01:00:00"}
    assert clip["camera"] is None
    assert clip["source_start_s"] == 60
