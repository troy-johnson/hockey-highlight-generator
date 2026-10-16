# v3/scripts/audio_mix.py
"""
Recap audio mix (hhg-3r5.39, spec 002 §5.10).

Lays the audio bed, cues, horns and stings over the plain silent Recap from
the assembly stage. The assembly plan (recap_assembly.json) is the only
cut-timeline source; the rendered video is stream-copied, never re-encoded.

Layers, summed before one two-pass loudnorm to -14 LUFS / -1 dBTP:
- rink sound from the chapter files, under everything
- a low music Bed per period, cut on bars, rotated between games
- cues: Game Start hit, goal horn (Perspective rule), stings, Win/Loss/neutral
  close, per-graphic SFX hooks
- PA music muted inside the spans music_spans.json reports as state "mute"

Cue Library manifest (hhg-3r5.38) is an input contract, schema 1. Without a
manifest the mix runs in placeholder mode: numpy-synthesized, silent-safe cue
audio, flagged for review. Nothing blocks on missing cues.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

FPS = 30
SR = 48000
BED_GAIN_DB = -19.0          # Bed level below full (spec §5.10: 18-20 dB)
DUCK_DB = 6.0                # Bed dip under a cue
CROSSFADE_S = 0.5            # Bed crossfade at period changes
TARGET_I = -14.0             # LUFS integrated
TARGET_TP = -1.0              # dB true peak
ENCODE_TP_MARGIN = 0.35       # loudnorm aims below the ceiling; the limiter and
                              # AAC encoding together overshoot it by up to ~0.5 dB
LIMITER_CEILING = 0.85        # sample-peak safety net after loudnorm (~-1.4 dBFS)
TARGET_LRA = 11.0
PLACEHOLDER_BED_BAR_S = 2.0
DEFAULT_MANIFEST = Path("~/hockey/cues/manifest.json").expanduser()
REPORT_FILE = "recap_audio.json"
ASSEMBLY_FILE = "recap_assembly.json"

PENALTY_KINDS = frozenset({"penalty"})
FIGHT_KINDS = frozenset({"fight"})     # hook only: selection.py has no fight producer yet

# One table drives the placeholder synth: (duration_s, freqs, decay_s). Stings
# hold the spec range of 2-4 s; the manifest may override with real audio.
_PLACEHOLDER_SPEC = {
    "game_start": (1.0, (60.0,), 1.0),
    "horn": (1.6, (220.0, 277.2, 329.6), 0.5),
    "close_win": (2.0, (261.6, 329.6, 392.0), 0.6),
    "close_loss": (2.0, (220.0, 261.6, 311.1), 0.6),
    "close_neutral": (2.0, (246.9, 311.1, 370.0), 0.6),
    "sting_penalty": (2.0, (392.0, 466.2), 0.6),
    "sting_power_play": (2.0, (349.2, 440.0), 0.6),
    "sting_fight": (2.0, (110.0, 116.5), 0.6),
    "sting_comic_call": (2.0, (523.3, 659.3), 0.6),
    "sting_neutral": (2.0, (293.7, 370.0, 440.0), 0.6),
    "sfx_goal_card": (0.25, (880.0,), 0.04),
    "sfx_penalty_card": (0.25, (740.0,), 0.04),
    "sfx_period_wipe": (0.25, (660.0,), 0.04),
    "sfx_final_card": (0.25, (990.0,), 0.04),
    # Game cues (hhg-3r5.110). A song runs from its event to the end of the replay block, so
    # its event sets the length; these placeholder lengths are only for synthesis.
    "song_goal_focus": (4.0, (261.6, 329.6, 392.0), 1.5),
    "song_goal_opponent": (4.0, (220.0, 261.6, 329.6), 1.5),
    "song_penalty_focus": (4.0, (293.7, 349.2), 1.5),
    "song_penalty_opponent": (4.0, (329.6, 415.3, 493.9), 1.5),
    "sfx_replay_pane": (0.25, (1320.0,), 0.04),
    # The period sound on a plain cut; sfx_period_wipe stays the cue of the full period pane.
    "sfx_period_cut": (0.25, (990.0, 1320.0), 0.04),
}

CUE_IDS = tuple(_PLACEHOLDER_SPEC)
PLACEHOLDER_DUR_S = {cue: spec[0] for cue, spec in _PLACEHOLDER_SPEC.items()}

# Game-sound mode cues (hhg-3r5.110): arena music and package sounds over live rink sound,
# with no Bed. Levels are the cue alone in LUFS, over rink sound set to -20 LUFS; a
# manifest entry's measured `lufs` sets its gain. Values are the reviewed previews.
GAME_CUE_LUFS = {
    "horn": -16.0,
    "song_goal_focus": -18.0, "song_goal_opponent": -20.0,
    "song_penalty_focus": -20.0, "song_penalty_opponent": -18.0,
    "close_win": -18.0, "close_loss": -20.0, "close_neutral": -20.0,
    "sfx_replay_pane": -20.0, "sfx_period_cut": -22.0, "sfx_period_wipe": -22.0,
}
GAME_CUES = frozenset(GAME_CUE_LUFS)
SONG_FADE_S = 1.0             # a song fades out over its last second, ending on the cut
PANE_PEAK_S = 8 / FPS         # the replay pane covers the cut on its 8th frame
LIVE_KINDS = ("goal", "play", "cold_open", "stinger", "final")


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

def build_timeline(plan: dict, *, selection: dict | None = None,
                   audio_signals: dict | None = None, live_replay: bool = False) -> list[dict]:
    """
    The output cut timeline: plan clips in plan order, each followed by its
    replays, with output positions. `frames / FPS` is the exact piece length,
    as in recap_assembly.render(). With live_replay, Replays carry live sound
    (live_sound_sequence), and whistle times map through that sound.
    """
    from recap_assembly import recap_sequence
    sequence = recap_sequence(plan)
    if live_replay:
        sequence = live_sound_sequence(sequence)
    tl: list[dict] = []
    out = 0.0
    for entry_index, entry in enumerate(sequence):
        speed = float(entry.get("speed") or 1.0)
        out_dur = float(entry["frames"]) / FPS
        e = dict(entry)
        e["out_start"] = out
        e["out_dur"] = out_dur
        e["moment_out"] = out + (float(entry.get("moment_s", entry["start_s"])) - float(entry["start_s"])) / speed
        timed_penalty = entry.get("kind") == "play" and entry.get("play_kind") == "penalty"
        if (entry.get("kind") == "goal" or timed_penalty) and selection is not None:
            if timed_penalty:
                selected_event = next((g for g in selection.get("plays", []) if g["id"] == entry["goal_id"]), {})
                candidate_id = selected_event.get("candidate_id", "")
            else:
                selected_event = next((g for g in selection.get("goals", []) if g["id"] == entry["goal_id"]), {})
                candidate_id = (selected_event.get("chosen") or {}).get("candidate_id", "")
            whistles = (audio_signals or {}).get("whistles", [])
            confirmation = None
            if candidate_id.startswith("whistle:"):
                try:
                    index = int(candidate_id.split(":", 1)[1])
                    if index >= 0:
                        confirmation = float(whistles[index]["t"])
                except (ValueError, IndexError, KeyError, TypeError):
                    pass
            if timed_penalty and confirmation is None and selected_event.get("whistle_t") is not None:
                # A reviewed penalty still binds to a measured whistle, not a guessed delay.
                confirmation = next((float(w["t"]) for w in whistles
                                     if abs(float(w["t"]) - float(selected_event["whistle_t"])) < .001), None)
            moment = float(entry["moment_s"])
            valid = (confirmation is not None and math.isfinite(confirmation)
                     and moment - 1 / 12 - .001 <= confirmation <= moment + 3.0 + .5 / 12 + .001)
            e["confirmation_out"] = None
            if valid:
                # A whistle omitted by the live cut can occur later in its own Replay.
                # Map through that piece's actual source seek and speed, not the expired live seek.
                cursor = out
                for piece in sequence[entry_index:]:
                    if piece.get("goal_id") != entry["goal_id"]:
                        break
                    piece_speed = float(piece.get("speed") or 1)
                    source_end = min(piece["end_s"], piece["start_s"] + piece["frames"] / FPS * piece_speed)
                    if piece["start_s"] <= confirmation < source_end:
                        e["confirmation_out"] = cursor + (confirmation - piece["start_s"]) / piece_speed
                        break
                    cursor += piece["frames"] / FPS
        parts, tl_pos, out_pos = [], float(entry["start_s"]), out
        for p in entry.get("parts") or []:
            q = dict(p)
            q["tl_start"] = tl_pos
            q["out_start"] = out_pos
            parts.append(q)
            tl_pos += float(p["duration_s"])
            out_pos += float(p["duration_s"]) / speed
        e["parts"] = parts
        tl.append(e)
        out += out_dur
    return tl


def _continued_part(previous: dict, start: float, duration: float) -> dict:
    """The previous piece's last part, continued from `start` (source s) for `duration`."""
    last = previous["parts"][-1]
    offset = start - (previous["start_s"] + sum(p["duration_s"] for p in previous["parts"][:-1]))
    return dict(last, seek_s=last["seek_s"] + offset, duration_s=duration, tl_start=start, continued=True)


def live_sound_sequence(sequence: list[dict]) -> list[dict]:
    """
    Live sound for Replays (hhg-3r5.110). Instead of the slowed Replay audio, the live
    piece's source keeps running at its own speed, so the sound under a Replay follows on
    from the play. A Replay keeps its kind, frames and picture fields; its sound fields
    (start_s, end_s, speed, camera, parts) become the continued live source, and
    `live_sound` marks it. Consecutive Replays of one live piece continue one after another.
    """
    out, live, cursor = [], None, 0.0
    for e in sequence:
        if e.get("kind") in LIVE_KINDS:
            live = e if e.get("parts") and e.get("kind") in ("goal", "play") else None
            if live is not None:
                cursor = live["start_s"] + sum(p["duration_s"] for p in live["parts"])
            out.append(e)
        elif live is not None and e.get("goal_id") == live.get("goal_id"):
            speed = float(live.get("speed") or 1)
            duration = float(e["frames"]) / FPS * speed
            # moment_s keeps the pictured time: Bed and period lookups follow the picture.
            out.append(dict(e, moment_s=e.get("moment_s", e["start_s"]), start_s=cursor,
                            end_s=cursor + duration, speed=speed, camera=live.get("camera"),
                            live_sound=True, parts=[_continued_part(live, cursor, duration)]))
            cursor += duration
        else:
            out.append(e)
    return out


def mute_windows(entry: dict, spans: list[dict]) -> list[tuple[float, float]]:
    """
    PA-mute windows in entry source time (parts laid end to end, before the
    entry speed is applied). Only spans of the entry's camera with state
    "mute" apply; "review" spans are not muted.
    """
    windows: list[tuple[float, float]] = []
    cam = entry.get("camera")
    prefix = 0.0
    for p in entry.get("parts") or []:
        dur = float(p.get("duration_s") or 0.0)
        t0 = float(p.get("tl_start", entry.get("start_s") or 0.0))
        for s in spans or []:
            if s.get("camera") != cam or s.get("state") != "mute":
                continue
            a = max(float(s["start"]) - t0, 0.0)
            b = min(float(s["end"]) - t0, dur)
            if a < b:
                windows.append((round(prefix + a, 3), round(prefix + b, 3)))
        prefix += dur
    return sorted(windows)


# ---------------------------------------------------------------------------
# Cues
# ---------------------------------------------------------------------------

def goal_side(entry: dict) -> str:
    """"home" / "away" from a goal_id like "home:2"; "" when not a side goal."""
    return (entry.get("goal_id") or "").split(":")[0]


def plan_cues(entries: list[dict], perspective: str = "neutral", focus_side: str | None = None,
              duration_s: float = 0.0, pp_goals=(), comic_calls=(),
              penalty_kinds=PENALTY_KINDS, fight_kinds=FIGHT_KINDS,
              goal_counts: tuple[int, int] | None = None, period_wipes=(),
              cue_durations: dict | None = None, game_cues: bool = False,
              period_cuts=()) -> list[dict]:
    """
    Cue events on the output timeline. The horn follows the Perspective:
    every goal in Neutral, only the Focus Team's goals in Focus. SFX cues are
    per-graphic hooks for hhg-3r5.34; sfx_score_flap is not derivable from the
    plan yet and is skipped. game_cues adds the songs, replay panes and the
    sfx_period_cut sound at period_cuts (hhg-3r5.110); songs need a Focus Team.
    """
    durs = dict(PLACEHOLDER_DUR_S)
    durs.update(cue_durations or {})
    cues: list[dict] = []

    def add(cue: str, t: float, **extra):
        c = {"cue": cue, "t": float(t), "dur_s": float(durs.get(cue, 0.25))}
        c.update(extra)
        cues.append(c)

    opening = next((e for e in entries if e.get("kind") == "stinger"), None)
    open_t = opening["out_start"] if opening else 0.0
    opening_duration = (opening["out_dur"] if opening is not None and
                        "game_start" not in (cue_durations or {}) else durs["game_start"])
    add("game_start", open_t, dur_s=opening_duration)
    if perspective == "neutral":
        add("sting_neutral", open_t)
    for e in entries:
        if e.get("kind") == "goal":
            side = goal_side(e)
            confirmation = e.get("confirmation_out", e["moment_out"])
            if confirmation is None:
                continue
            if perspective != "focus" or focus_side is None or side == focus_side:
                add("horn", confirmation, goal_id=e["goal_id"], side=side)
            if e["goal_id"] in pp_goals:
                add("sting_power_play", e["out_start"], goal_id=e["goal_id"])
            if e["goal_id"] in comic_calls:
                add("sting_comic_call", e["out_start"], goal_id=e["goal_id"])
            add("sfx_goal_card", confirmation, goal_id=e["goal_id"])
        elif e.get("kind") == "play":
            pk = e.get("play_kind")
            if pk in penalty_kinds:
                confirmation = e.get("confirmation_out", e["out_start"])
                if confirmation is not None:
                    add("sting_penalty", confirmation, goal_id=e["goal_id"])
                    add("sfx_penalty_card", confirmation, goal_id=e["goal_id"])
            if pk in fight_kinds:
                add("sting_fight", e["out_start"], goal_id=e["goal_id"])
            if e["goal_id"] in comic_calls:
                add("sting_comic_call", e["out_start"], goal_id=e["goal_id"])
    for w in period_wipes:
        add("sfx_period_wipe", w)
    if game_cues:
        cues.extend(_game_cues(entries, focus_side if perspective == "focus" else None, duration_s,
                               durs["sfx_replay_pane"], period_wipes))
        for t in period_cuts:
            add("sfx_period_cut", t)
    close_id = _close_cue(perspective, focus_side, goal_counts, entries)
    closing = next((e for e in entries if e.get("kind") == "final"), None)
    close_t = closing["out_start"] if closing else max(0.0, duration_s - durs.get(close_id, 2.0))
    add("sfx_final_card", close_t)
    add(close_id, close_t)
    cues.sort(key=lambda c: (c["t"], 1 if c["cue"].startswith("close_") else 0))
    return cues


def _block_end(entries: list[dict], i: int) -> float:
    """Output end of entry i and the replays that follow it (same goal_id)."""
    end = entries[i]["out_start"] + entries[i]["out_dur"]
    for e in entries[i + 1:]:
        if e.get("goal_id") != entries[i].get("goal_id") or e.get("kind") in LIVE_KINDS:
            break
        end = e["out_start"] + e["out_dur"]
    return end


def _game_cues(entries: list[dict], focus_side: str | None, duration_s: float,
               pane_s: float = PLACEHOLDER_DUR_S["sfx_replay_pane"], period_wipes=()) -> list[dict]:
    """
    Songs and replay panes. A goal song starts at the goal confirmation, a penalty song at
    the penalty card, and each runs to the end of its replay block, fading over the last
    SONG_FADE_S. Goal songs follow the scoring team; penalty songs (ids like
    "penalty:home:1") follow the penalized team. A pane sound, pane_s long (the library
    file), starts PANE_PEAK_S early so its peak lands on each replay cut. Panes follow
    recap_graphics: none on a period-wipe cut, within 8 frames of either end, or where a
    Replay exits into FINAL.
    """
    frames = round(duration_s * FPS)
    wipe_frames = {round(w * FPS) for w in period_wipes}
    cues = []
    for i, e in enumerate(entries):
        role, side, t = None, "", None
        if focus_side and e.get("kind") == "goal":
            role, side, t = "goal", goal_side(e), e.get("confirmation_out", e["moment_out"])
        elif focus_side and e.get("kind") == "play" and e.get("play_kind") == "penalty":
            role, t = "penalty", e.get("confirmation_out", e["out_start"])
            fields = (e.get("goal_id") or "").split(":")
            side = fields[1] if len(fields) >= 3 else ""
        if role and t is not None and side in ("home", "away"):
            end = _block_end(entries, i)
            if end > t:
                who = "focus" if side == focus_side else "opponent"
                cues.append({"cue": f"song_{role}_{who}", "t": float(t), "dur_s": end - float(t),
                             "fade_out_s": SONG_FADE_S, "goal_id": e["goal_id"], "side": side})
        if e.get("kind") not in LIVE_KINDS:
            following = entries[i + 1] if i + 1 < len(entries) else None
            for edge, cut in (("in", e["out_start"]), ("out", e["out_start"] + e["out_dur"])):
                frame = round(cut * FPS)
                if frame < 8 or frame + 8 > frames or frame in wipe_frames:
                    continue
                if edge == "out" and following is not None and following.get("kind") == "final":
                    continue
                cues.append({"cue": "sfx_replay_pane", "t": cut - PANE_PEAK_S, "dur_s": pane_s,
                             "goal_id": e.get("goal_id")})
    # Back-to-back replays share a cut: one pane.
    seen, unique = set(), []
    for c in cues:
        key = (c["cue"], round(c["t"], 3))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def period_cut_times(entries: list[dict], periods: list[dict]) -> list[float]:
    """Output times of the hard cuts where the period changes between live pieces."""
    cuts, last = [], None
    for e in entries:
        if e.get("kind") not in ("goal", "play"):
            continue
        n = period_of(float(e.get("moment_s") or e.get("start_s") or 0.0), periods)
        if last is not None and n != last:
            cuts.append(e["out_start"])
        last = n
    return cuts


def _cue_lufs(value) -> float | None:
    """A manifest cue's measured loudness, or None when it is not a plausible measurement."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and -70.0 <= value <= 0.0 else None


def _close_cue(perspective: str, focus_side: str | None,
               goal_counts: tuple[int, int] | None, entries: list[dict]) -> str:
    if perspective != "focus" or focus_side is None:
        return "close_neutral"
    home, away = _goal_counts(goal_counts, entries)
    if home == away:
        return "close_neutral"
    return "close_win" if (home > away) == (focus_side == "home") else "close_loss"


def _goal_counts(goal_counts: tuple[int, int] | None, entries: list[dict]) -> tuple[int, int]:
    if goal_counts:
        return int(goal_counts[0]), int(goal_counts[1])
    home = away = 0
    for e in entries:
        if e.get("kind") == "goal":
            side = goal_side(e)
            if side == "home":
                home += 1
            elif side == "away":
                away += 1
    return home, away


# ---------------------------------------------------------------------------
# Bed
# ---------------------------------------------------------------------------

def bed_index(beds: list[dict], rotation_key: str, period: int = 1) -> int:
    """Deterministic Bed rotation: one slot per game, +1 per period."""
    if not beds:
        return 0
    base = int(hashlib.sha256(rotation_key.encode()).hexdigest()[:8], 16)
    return (base + max(1, int(period)) - 1) % len(beds)


def bar_seconds(bed: dict | None) -> float:
    """Bar length of a Bed: explicit bar_s, then the beat grid, then bpm."""
    if not bed:
        return PLACEHOLDER_BED_BAR_S
    if bed.get("bar_s"):
        return float(bed["bar_s"])
    beats = bed.get("beats") or []
    if len(beats) >= 2:
        intervals = sorted(float(beats[i + 1]) - float(beats[i]) for i in range(len(beats) - 1))
        interval = intervals[len(intervals) // 2]
        if interval > 0:
            return interval * int(bed.get("beats_per_bar") or 4)
    bpm = float(bed.get("bpm") or 0.0)
    if bpm > 0:
        return 60.0 / bpm * int(bed.get("beats_per_bar") or 4)
    return PLACEHOLDER_BED_BAR_S


def loop_bars(duration_s: float, bar_s: float) -> float:
    """A whole number of bars that covers duration_s, for tiling."""
    bars = max(1, int(float(duration_s) / float(bar_s) + 1e-9))
    return bars * float(bar_s)


def period_of(moment: float, periods: list[dict]) -> int:
    best, best_d = None, None
    for p in periods or []:
        start, end = float(p["start"]), float(p["end"])
        if start <= moment < end:
            return int(p["n"])
        d = min(abs(moment - start), abs(moment - end))
        if best_d is None or d < best_d:
            best, best_d = int(p["n"]), d
    return best if best is not None else 1


def plan_bed(tl: list[dict], periods: list[dict], beds: list[dict], rotation_key: str,
             *, bed_choices: dict[int, dict] | None = None) -> list[dict]:
    """
    One Bed segment per run of consecutive entries in the same period. The
    first segment starts at 0 and covers the whole output; later segments
    start at the first bar line of the outgoing Bed at or after the period
    boundary, so the change lands on a bar (masked by the period wipe).
    """
    if not tl:
        return []
    total = sum(e["out_dur"] for e in tl)
    runs: list[tuple[int, list[dict]]] = []
    first_play = next((e for e in tl if e.get("kind") not in ("cold_open", "stinger")), tl[0])
    for e in tl:
        if e.get("kind") == "final" and runs:
            runs[-1][1].append(e)
            continue
        source = first_play if e.get("kind") in ("cold_open", "stinger") else e
        per = period_of(float(source.get("moment_s") or source.get("start_s") or 0.0), periods)
        if runs and runs[-1][0] == per:
            runs[-1][1].append(e)
        else:
            runs.append((per, [e]))
    segments: list[dict] = []
    for i, (per, group) in enumerate(runs):
        boundary = group[0]["out_start"]
        if i == 0:
            seg_start = 0.0
        else:
            prev = segments[-1]
            bar = prev["bar_s"] or 1.0
            k = math.ceil((boundary - prev["out_start"]) / bar - 1e-9)
            seg_start = min(prev["out_start"] + k * bar, total)
            if seg_start <= prev["out_start"]:
                seg_start = boundary
            prev["out_end"] = seg_start
        bed = (bed_choices.get(per) if bed_choices is not None else
               beds[bed_index(beds, rotation_key, per) % len(beds)] if beds else None)
        segments.append({"period": per, "out_start": round(seg_start, 3), "out_end": round(total, 3),
                         "boundary": round(boundary, 3),
                         "track": bed["id"] if bed else "placeholder",
                         "bar_s": round(bar_seconds(bed), 3)})
    return segments


def duck_envelope(n: int, sr: int, events: list[dict], duck_db: float = DUCK_DB,
                  attack_s: float = 0.05, release_s: float = 0.4) -> np.ndarray:
    """Bed gain envelope: dip to -duck_db under each cue, linear in dB."""
    g = np.ones(n, dtype=np.float32)
    floor = 10.0 ** (-float(duck_db) / 20.0)
    for ev in events:
        t0 = max(0, int(float(ev["t"]) * sr))
        hold_end = t0 + max(1, int(float(ev.get("dur_s", 0.0)) * sr))
        a = max(1, int(attack_s * sr))
        r = max(1, int(release_s * sr))
        start = max(0, t0 - a)
        end = min(n, hold_end + r)
        if start >= end:
            continue
        m = end - start
        env = np.full(m, floor, dtype=np.float32)
        ai = min(a, m)
        env[:ai] = np.linspace(1.0, floor, ai, dtype=np.float32)
        rel_start = min(hold_end - start, m)
        if m > rel_start:
            env[rel_start:] = np.linspace(floor, 1.0, m - rel_start, dtype=np.float32)
        g[start:end] = np.minimum(g[start:end], env)
    return g


# ---------------------------------------------------------------------------
# Placeholder audio
# ---------------------------------------------------------------------------

def synth_placeholder(cue: str, *, duration_s: float | None = None) -> np.ndarray:
    dur, freqs, decay = _PLACEHOLDER_SPEC.get(cue, (0.25, (440.0,), 0.04))
    if duration_s is not None:
        dur = duration_s
    if cue == "game_start" and duration_s is not None:
        return _opening_sound(dur)
    n = max(1, int(dur * SR))
    t = np.arange(n, dtype=np.float32) / SR
    wave = sum(np.sin(2.0 * np.pi * f * t) for f in freqs) / max(1, len(freqs))
    x = (wave * np.exp(-t / decay)).astype(np.float32)
    return np.stack([x, x], axis=1)


def _opening_sound(duration_s: float) -> np.ndarray:
    """Original review sound: room tone, scrape, impact, crackle, crack, plate, and pane whoosh."""
    from scipy.signal import butter, sosfilt
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("opening sound duration must be positive and finite")
    n = max(1, round(duration_s * SR))
    t = np.arange(n) / SR
    # Reference choreography lasts 135 frames at 30 fps.
    f = t * 135 / duration_s
    frame_s = duration_s / 135
    rng = np.random.default_rng(1731)
    noise = rng.standard_normal(n)
    band = lambda lo, hi: sosfilt(butter(2, [lo, hi], btype="bandpass", fs=SR, output="sos"), noise)
    blade, air, crisp = band(650, 8500), band(150, 1200), band(3000, 12000)
    age_at = lambda frame: np.maximum(0, t - frame * frame_s) * (f >= frame)
    room = air * .012 * np.clip(f / 4, 0, 1)
    scrape = np.sin(np.pi * np.clip((f - 4) / 22, 0, 1)) ** .8
    hit = age_at(26)
    impact = (np.sin(2 * np.pi * 50 * hit) * np.exp(-hit / .11) * .6 +
              blade * np.exp(-hit / .03) * .25) * (f >= 26)
    # Deterministic crackle densifies while frost grows (frames 26-46).
    grains = np.zeros(n)
    times = np.sort(26 + 20 * np.sqrt(rng.random(70)))
    for frame in times:
        g = age_at(frame)
        grains += crisp * np.exp(-g / .004) * (f >= frame) * .15
    crackle = grains * np.clip((f - 26) / 20, 0, 1) * (f < 48)
    crack_age = age_at(46)
    crack = (crisp * 1.4 + blade * .8) * np.exp(-crack_age / .05) * (f >= 46)
    crack += np.sin(2 * np.pi * 140 * crack_age) * np.exp(-crack_age / .09) * .3 * (f >= 46)
    shing_age = age_at(62)
    shing = sum(np.sin(2 * np.pi * hz * shing_age) for hz in (2400, 3600, 5100))
    shing *= .03 * np.exp(-shing_age / .3) * (f >= 62)
    thunk_age = age_at(66)
    thunk = (np.sin(2 * np.pi * 95 * thunk_age) * .45 + air * .5) * np.exp(-thunk_age / .07) * (f >= 66)
    # The camera push and the ice flash cover the cut at frame 135.
    whoosh = air * 1.2 * np.clip((f - 120) / 14, 0, 1) ** 2
    whoosh += crisp * .35 * np.clip((f - 124) / 10, 0, 1) ** 2
    movement = blade * .2 * scrape + whoosh
    pan = np.interp(f, [4, 26, 120, 135], [.45, 0, -.5, .5])
    centre = room + impact + crackle + crack + shing + thunk
    sound = np.column_stack([movement * np.sqrt((1 - pan) / 2) + centre,
                             movement * np.sqrt((1 + pan) / 2) + centre])
    sound *= min(1, .8 / max(.8, float(np.max(np.abs(sound)))))
    return _edge_fade(sound.astype(np.float32))


def synth_placeholder_bed() -> np.ndarray:
    n = int(8 * PLACEHOLDER_BED_BAR_S * SR)
    t = np.arange(n, dtype=np.float32) / SR
    beat = 0.5
    phase = t % beat
    kick = np.sin(2.0 * np.pi * 55.0 * phase) * np.exp(-phase / 0.06)
    hat_phase = (t + beat / 2.0) % beat
    hat = 0.25 * np.exp(-hat_phase / 0.02) * np.sin(2.0 * np.pi * 6000.0 * hat_phase)
    x = (0.5 * kick + 0.2 * hat).astype(np.float32)
    return np.stack([x, x], axis=1)


# ---------------------------------------------------------------------------
# Cue Library manifest (hhg-3r5.38 contract)
# ---------------------------------------------------------------------------

def load_manifest(path: Path | str) -> tuple[dict | None, list[str]]:
    """
    Schema 1: {"schema_version": 1,
               "beds": [{id, file, source, content_id_safe, duration_s, bpm,
                         bar_s?, beats?, beats_per_bar?, lufs, tags}],
               "cues": [{id, file, duration_s, gain_db?}]}.
    Media paths resolve against the manifest's directory. Entries whose media
    file is missing are dropped with a flag. No manifest file → placeholder
    mode (None returned, flagged).
    """
    p = Path(path).expanduser()
    try:
        data = json.loads(p.read_text())
    except FileNotFoundError:
        return None, [f"placeholder cue audio: no Cue Library manifest at {p}; "
                      "synthesized cues and bed"]
    except OSError as exc:
        return None, [f"placeholder cue audio: Cue Library manifest unreadable ({exc}); "
                      "synthesized cues and bed"]
    except json.JSONDecodeError as exc:
        return None, [f"invalid Cue Library manifest {p.name}: {exc}"]
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        return None, [f"unsupported Cue Library manifest schema "
                      f"{(data or {}).get('schema_version')!r}; expected 1; placeholder mode"]
    base = p.parent
    flags: list[str] = []

    def resolve(entries, kind):
        kept = []
        for e in entries or []:
            name = e.get("file")
            f = base / str(name) if name else None
            if f and Path(f).is_file():
                q = dict(e)
                q["path"] = str(f)
                kept.append(q)
            else:
                flags.append(f"manifest {kind} {e.get('id')}: media file missing: {name}")
        return kept

    return {"schema_version": 1, "path": str(p),
            "beds": resolve(data.get("beds"), "bed"),
            "cues": resolve(data.get("cues"), "cue")}, flags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fit_to_length(buf: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((n, buf.shape[1] if buf.ndim == 2 else 2), dtype=np.float32)
    m = min(n, len(buf))
    if m > 0:
        out[:m] = buf[:m]
    return out


def mixed_output_name(name: str) -> str:
    return str(Path(name).with_suffix("")) + "_audio.mp4"


def parse_loudnorm_json(text: str) -> dict:
    out: dict[str, float] = {}
    for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset"):
        m = re.search(rf'"{k}"\s*:\s*"(-?[0-9.]+)"', text)
        if m:
            out[k] = float(m.group(1))
    return out


def _place(master: np.ndarray, buf: np.ndarray, t: float) -> None:
    start = int(round(float(t) * SR))
    if start < 0 or start >= len(master) or len(buf) == 0:
        return
    end = min(len(master), start + len(buf))
    master[start:end] += buf[:end - start]


def _tile(buf: np.ndarray, n: int) -> np.ndarray:
    if len(buf) == 0:
        return np.zeros((n, 2), dtype=np.float32)
    reps = -(-n // len(buf))
    return np.concatenate([buf] * reps)[:n]


def atempo_filter(speed: float) -> str:
    """
    atempo chain for a play speed. One atempo covers 0.5-100; below 0.5 the
    chain halves first, so slow-mo replays down to 0.25 stay in sync.
    """
    s = max(0.25, float(speed))
    if s >= 0.5:
        return f"atempo={s:.6f}"
    return f"atempo=0.5,atempo={s / 0.5:.6f}"


def decode_audio(path: Path | str, seek_s: float, duration_s: float, speed: float = 1.0) -> np.ndarray:
    """
    Decode a media file to f32le stereo at SR. The seek is always before -i
    (the concat inpoint rule does not apply to sync offsets on GoPro HEVC).
    atempo keeps pitch while applying the cut speed.
    """
    argv = ["ffmpeg", "-v", "error",
            "-ss", f"{max(0.0, float(seek_s)):.3f}", "-t", f"{max(0.0, float(duration_s)):.3f}",
            "-i", str(path), "-vn", "-af", atempo_filter(speed),
            "-ar", str(SR), "-ac", "2", "-f", "f32le", "-"]
    proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode {Path(path).name}: "
                           f"{proc.stderr.decode(errors='replace')[:300].strip()}")
    raw = np.frombuffer(proc.stdout, dtype=np.float32)
    return raw.reshape(-1, 2).copy()


def _faded_zero(buf: np.ndarray, i0: int, i1: int, fade: int) -> None:
    """Zero [i0, i1) with short raised-cosine ramps at both edges, so the mute
    window has no click and rings less in the AAC encode."""
    if i1 <= i0:
        return
    buf[i0:i1] = 0.0
    if fade > 0:
        if i0 > 0:
            k = min(fade, i0)
            ramp = (0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, k))).astype(np.float32)
            buf[i0 - k:i0] *= ramp[:, None]
        if i1 < len(buf):
            k = min(fade, len(buf) - i1)
            ramp = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, k))).astype(np.float32)
            buf[i1:i1 + k] *= ramp[:, None]


def _edge_fade(buf: np.ndarray, ms: float = 5.0) -> np.ndarray:
    """Short fade-in/out on a cue buffer, so hard cue starts ring less."""
    f = int(SR * ms / 1000.0)
    if f > 0 and len(buf) > 2 * f:
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, f))).astype(np.float32)
        buf = buf.copy()
        buf[:f] *= ramp[:, None]
        buf[-f:] *= ramp[::-1][:, None]
    return buf


_DURATIONS: dict[str, float] = {}


def _media_duration(path: str) -> float:
    """Container duration in seconds; 0 when unknown."""
    if path not in _DURATIONS:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
        try:
            _DURATIONS[path] = float(out)
        except ValueError:
            _DURATIONS[path] = 0.0
    return _DURATIONS[path]


NEXT_CHAPTER: dict[str, str] = {}   # chapter path -> next chapter of its Recording


def set_recordings(chapters: dict | None) -> None:
    """Load chapter order from chapters.json. Only discover.py knows a chapter's camera:
    in a flat folder, two cameras' files can share GoPro names, so names alone do not say
    which file continues a Recording."""
    NEXT_CHAPTER.clear()
    for recordings in ((chapters or {}).get("recordings") or {}).values():
        for rec in recordings or []:
            for a, b in zip(rec, rec[1:]):
                NEXT_CHAPTER[os.path.realpath(a)] = b


def next_chapter(path: str) -> str | None:
    """The next chapter of the same camera Recording, from chapters.json; None when unknown."""
    return NEXT_CHAPTER.get(os.path.realpath(path))


def decode_continued(part: dict, speed: float) -> tuple[np.ndarray, float]:
    """
    Decode a continued part (live Replay or FINAL sound) that may run past the end of its
    chapter file: the rest comes from the Recording's next chapter. Returns the audio and
    the source seconds that no chapter could supply.
    """
    path, seek, need = str(part["file"]), float(part["seek_s"]), float(part["duration_s"])
    chunks = []
    for _ in range(64):                       # a Recording has far fewer chapters
        if need <= 1e-3 or not path:
            break
        length = _media_duration(path)
        if length and seek >= length - 1e-3:
            seek -= length
            path = next_chapter(path)
            continue
        take = min(need, length - seek) if length else need
        chunk = decode_audio(path, seek, take, speed)
        chunks.append(chunk)
        # Count what was decoded, not what was asked: a short read means the file ended
        # there, so the next chapter starts at its beginning and the timing stays exact.
        need -= len(chunk) / SR * speed
        seek = 0.0
        path = next_chapter(path)
    buf = np.concatenate(chunks) if chunks else np.zeros((0, 2), dtype=np.float32)
    return buf, max(0.0, need)


def _entry_audio(entry: dict, windows: list[tuple[float, float]],
                 notes: list[str] | None = None) -> np.ndarray:
    """Rink sound for one timeline entry: parts decoded, PA-muted, atempo'd."""
    speed = float(entry.get("speed") or 1.0)
    bufs = []
    for p in entry["parts"]:
        if p.get("continued"):
            buf, missing = decode_continued(p, speed)
            if missing > 0.05 and notes is not None:
                notes.append(f"{entry.get('goal_id')}: live sound ends {missing:.1f} s before the cut "
                             f"(no next chapter after {Path(p['file']).name} in chapters.json); "
                             "silence fills it")
        else:
            buf = decode_audio(p["file"], p["seek_s"], p["duration_s"], speed)
        bufs.append(fit_to_length(buf, int(round(float(p["duration_s"]) / speed * SR))))
    buf = np.concatenate(bufs) if bufs else np.zeros((0, 2), dtype=np.float32)
    # Mute windows are in entry source time; entry buffer position = source / speed.
    fade = int(0.01 * SR)
    for a, b in windows:
        i0 = max(0, int(a / speed * SR))
        i1 = min(len(buf), int(b / speed * SR))
        _faded_zero(buf, i0, i1, fade)
    return buf


def game_sound_entry(entry: dict, windows: list[tuple[float, float]],
                     notes: list[str] | None = None) -> np.ndarray:
    """Keep the source cuts and speed. Lower Replay sound and fade each cut."""
    buf = fit_to_length(_entry_audio(entry, windows, notes), round(entry["out_dur"] * SR))
    if entry["kind"] not in LIVE_KINDS and not entry.get("live_sound"):
        buf = buf * 10 ** (-8 / 20)
    return _edge_fade(buf, ms=15)


def _continuation(previous: dict, start: float, out_dur: float) -> dict:
    """The previous piece's source sound, continued from `start` (source s) for out_dur."""
    duration = out_dur * float(previous.get("speed") or 1)
    tail = dict(previous, out_dur=out_dur, start_s=start, end_s=start + duration)
    tail["parts"] = [_continued_part(previous, start, duration)]
    return tail


def closing_game_sound(closing: dict, previous: dict | None, spans: list[dict],
                       notes: list[str] | None = None) -> np.ndarray:
    """Continue the last source sound under FINAL and fade its final three seconds."""
    n = round(closing["out_dur"] * SR)
    if not previous or not previous.get("parts"):
        return np.zeros((n, 2), dtype=np.float32)
    start = previous["start_s"] + sum(p["duration_s"] for p in previous["parts"])
    tail = _continuation(previous, start, closing["out_dur"])
    buf = game_sound_entry(tail, mute_windows(tail, spans), notes)
    fade = min(len(buf), 3 * SR)
    if fade:
        buf[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
    return buf


def normalize_game_sound(buf: np.ndarray) -> np.ndarray:
    """Set rink sound to -20 LUFS before adding the mastered opening sound."""
    with tempfile.TemporaryDirectory(prefix="hockey-game-sound-") as scratch:
        raw = Path(scratch) / "game.f32"
        raw.write_bytes(buf.tobytes())
        measured = _measure_loudness(raw, raw_input=True, tp=-2.0, target_i=-20.0)
        if not math.isfinite(measured.get("input_i", -math.inf)):
            return buf
        af = ("loudnorm=I=-20:TP=-2:LRA=11"
              f":measured_i={measured['input_i']}:measured_tp={measured['input_tp']}"
              f":measured_lra={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
              f":offset={measured['target_offset']}:linear=true")
        proc = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "2",
                               "-i", str(raw), "-af", af, "-ar", str(SR), "-ac", "2", "-f", "f32le", "-"],
                              capture_output=True, check=True)
        return fit_to_length(np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, 2).copy(), len(buf))


def _add_bed(master: np.ndarray, segments: list[dict], bed_bufs: dict, bed_gain_db: float,
             duck: np.ndarray) -> None:
    n = len(master)
    lin = 10.0 ** (float(bed_gain_db) / 20.0)
    fade = int(CROSSFADE_S * SR)
    for i, seg in enumerate(segments):
        s0 = max(0, int(round(seg["out_start"] * SR)))
        s1 = min(n, int(round(seg["out_end"] * SR)))
        if s1 <= s0:
            continue
        buf = bed_bufs.get(seg["track"])
        if buf is None or len(buf) == 0:
            continue
        env = np.ones(s1 - s0, dtype=np.float32)
        if i > 0 and fade > 0 and len(env) > fade:
            env[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        if i < len(segments) - 1 and fade > 0 and len(env) > fade:
            env[-fade:] = np.linspace(1.0, 0.0, fade, dtype=np.float32)
        gain = env * duck[s0:s1] * lin
        master[s0:s1] += (_tile(buf, s1 - s0) * gain[:, None]).astype(np.float32)


def _measure_loudness(path: Path, raw_input: bool, tp: float = TARGET_TP, target_i: float = TARGET_I) -> dict:
    argv = ["ffmpeg", "-v", "info"]
    if raw_input:
        argv += ["-f", "f32le", "-ar", str(SR), "-ac", "2"]
    argv += ["-i", str(path),
             "-af", f"loudnorm=I={target_i}:TP={tp}:LRA={TARGET_LRA}:print_format=json",
             "-f", "null", "-"]
    proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return parse_loudnorm_json(proc.stdout)


def _probe_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                         "-of", "json", str(path)], stdout=subprocess.PIPE, text=True, check=True)
    return float(json.loads(out.stdout)["format"]["duration"])


def _rotation_key(plan: dict, sheet: dict) -> str:
    m = re.match(r"(\d{4}-\d{2}-\d{2})_", str(plan.get("output") or ""))
    date = m.group(1) if m else "unknown"
    teams = sheet.get("teams") or {}
    return f"{date}|{teams.get('home') or '?'}-vs-{teams.get('away') or '?'}"


def _sheet_goal_counts(sheet: dict) -> tuple[int, int]:
    goals = sheet.get("goals") or {}
    return len(goals.get("home") or []), len(goals.get("away") or [])


def resolve_focus_side(perspective: str, focus_team: str | None,
                       sheet: dict) -> tuple[str | None, str | None]:
    """
    "home"/"away" for the Focus Team, matched case-insensitively like
    selection.py and recap_assembly.output_name. (None, None) when not Focus;
    (None, flag) when the name is not on the Game Sheet.
    """
    if perspective != "focus" or not focus_team:
        return None, None
    teams = sheet.get("teams") or {}
    for side in ("home", "away"):
        name = teams.get(side) or ""
        if name and focus_team.casefold() == name.casefold():
            return side, None
    return None, (f"focus team {focus_team!r} is not on the Game Sheet; "
                  "the horn plays on every goal")


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def period_wipe_times(bed_segments, options) -> list:
    """Period boundaries that get a wipe and its sound. A short Recap uses plain
    cuts; the scorebug shows the new period. Full-game modes opt in."""
    if not options.get("period_transitions"):
        return []
    return [s["boundary"] for s in bed_segments[1:]]


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.game_folder)
    flags: list[str] = []

    def flag(msg: str) -> None:
        flags.append(msg)
        print(f"[mix] flag: {msg}", flush=True)

    plan = _load_json(root / ASSEMBLY_FILE)
    if not plan or not plan.get("clips"):
        print(f"[ERROR] mix: {ASSEMBLY_FILE} is missing or has no clips; run the assembly stage first")
        return 1

    options = json.loads(args.options) if args.options else {}
    audio_opts = dict(options.get("audio") or {})
    opening = next((e for e in plan.get("opening", []) if e.get("opening_video")), None)
    mode = audio_opts.get("mode", "game_sound" if opening else "broadcast")
    if mode not in ("game_sound", "broadcast"):
        raise ValueError("audio.mode must be game_sound or broadcast")
    game_sound = mode == "game_sound"
    perspective = options.get("perspective") or "neutral"
    focus_team = options.get("focus_team")
    comic_calls = list(options.get("comic_calls") or [])
    bed_gain_db = float(audio_opts.get("bed_gain_db", BED_GAIN_DB))
    if not -20.0 <= bed_gain_db <= -18.0:
        flag(f"bed gain {bed_gain_db} dB is outside the spec range (-20 to -18 dB)")

    set_recordings(_load_json(root / "chapters.json"))
    music = _load_json(root / "music_spans.json")
    spans = (music or {}).get("spans") or []
    if music is None:
        flag("music_spans.json is missing; PA music was not muted")
    selection = _load_json(root / "selection.json") or {}
    from recap_assembly import load_play_reviews
    selection = load_play_reviews(root, selection)
    coverage = _load_json(root / "coverage.json")
    if coverage is not None:
        selection = dict(selection, periods=coverage.get("periods") or [])
    if not selection.get("periods"):
        flag("selection.json has no periods; the Bed runs as one segment")
    if (root / "game_data.json").exists():
        from game_data import load_graphics_inputs
        sheet, frozen = load_graphics_inputs(root, options)
        focus_team = frozen.get("focus_team")
        focus_team = (frozen.get("teams", {}).get(focus_team) or {}).get("name", focus_team)
    else:
        sheet = _load_json(root / "game_sheet.json") or {}
    if not sheet:
        flag("game_sheet.json is missing; close kind and power-play stings follow the plan clips only")

    manifest_path = audio_opts.get("manifest") or str(DEFAULT_MANIFEST)
    manifest, man_flags = load_manifest(manifest_path)
    placeholder_mode = manifest is None and not game_sound
    live_replay = bool(audio_opts.get("live_replay", True))
    if placeholder_mode:
        flag(f"placeholder cue audio: no Cue Library manifest at {manifest_path}; "
             "synthesized cues and bed")
    elif game_sound and manifest is None:
        # Game cues are opt-in by library: without one the Recap keeps its rink sound only.
        if Path(manifest_path).expanduser().exists():
            for f in man_flags:
                flag(f"{f}; game cues skipped")
        elif audio_opts.get("manifest"):
            flag(f"no Cue Library manifest at {manifest_path}; game cues skipped")
        else:
            print(f"[mix] no Cue Library at {manifest_path}; game cues off", flush=True)
    else:
        for f in man_flags:
            flag(f)

    focus_side, focus_flag = resolve_focus_side(perspective, focus_team, sheet)
    if focus_flag:
        flag(focus_flag)

    pp_goals = {f"{side}:{i}" for side in ("home", "away")
                for i, g in enumerate((sheet.get("goals") or {}).get(side) or [], 1)
                if g.get("type") == "PP"}

    tl = build_timeline(plan, selection=selection,
                        audio_signals=_load_json(root / "audio_signals.json") or {},
                        live_replay=live_replay)
    for entry in tl:
        if entry["kind"] == "goal" and entry.get("confirmation_out") is None:
            flags.append(f"{entry['goal_id']}: goal confirmation unavailable; horn and GOAL card omitted")
    duration_s = float(plan.get("duration_s") or sum(e["frames"] for e in tl) / FPS)
    n = int(round(duration_s * SR))

    beds = manifest["beds"] if manifest and not game_sound else []
    if manifest is not None and not beds and not game_sound:
        flag("the Cue Library manifest has no usable Beds; synthesized placeholder Bed")
    rotation_key = _rotation_key(plan, sheet)
    bed_choices = None
    history_path = None
    if beds:
        from cue_library import select_beds
        history_path = Path(manifest_path).expanduser().resolve().parent / "rotation.json"
        used_periods = sorted({period_of(float(e.get("moment_s") or e.get("start_s") or 0),
                                        selection.get("periods") or []) for e in tl
                               if e.get("kind") not in ("cold_open", "stinger", "final")})
        try:
            bed_choices, rotation_flags = select_beds(beds, rotation_key, used_periods, history_path)
        except ValueError as exc:
            flag(f"{exc}; using deterministic Bed selection without saved rotation")
            history_path = None
            rotation_flags = []
        for message in rotation_flags:
            flag(message)
    bed_segments = plan_bed(tl, selection.get("periods") or [], beds, rotation_key,
                            bed_choices=bed_choices)
    bed_bufs: dict[str, np.ndarray] = {}
    decoded_beds: set[str] = set()
    for b in beds:
        try:
            buf = decode_audio(b["path"], 0.0, float(b.get("duration_s") or 30.0))
            decoded_beds.add(b["id"])
        except RuntimeError as exc:
            flag(f"Bed {b['id']} could not be decoded ({exc}); synthesized placeholder Bed")
            buf = synth_placeholder_bed()
        # Trim the loop to whole bars, so tile seams land on a downbeat.
        bar = bar_seconds(b)
        bed_bufs[b["id"]] = buf[:int(loop_bars(len(buf) / SR, bar) * SR)]
    if not bed_bufs:
        bed_bufs["placeholder"] = synth_placeholder_bed()

    cue_durations = {c["id"]: float(c.get("duration_s") or 0.0)
                     for c in (manifest or {}).get("cues", [])}
    cue_durations = {k: v for k, v in cue_durations.items() if v > 0}
    for cid, dur_s in cue_durations.items():
        if cid.startswith("sting_") and not 2.0 <= dur_s <= 4.0:
            flag(f"sting {cid} is {dur_s:.1f} s; the spec range is 2-4 s")
    game_cues = game_sound and manifest is not None
    period_wipes = period_wipe_times(bed_segments, options)
    period_cuts = ([] if not game_cues or period_wipes
                   else period_cut_times(tl, selection.get("periods") or []))
    cues = plan_cues(tl, perspective=perspective, focus_side=focus_side, duration_s=duration_s,
                     pp_goals=pp_goals, comic_calls=comic_calls,
                     goal_counts=_sheet_goal_counts(sheet),
                     period_wipes=period_wipes,
                     cue_durations=cue_durations or None, game_cues=game_cues, period_cuts=period_cuts)
    if not any(c["cue"] == "sting_fight" for c in cues):
        flag("no fight clips in the plan: sting_fight did not fire "
             "(selection.py has no fight producer yet)")
    flag("sfx_score_flap is not derivable from the plan yet; skipped (needs hhg-3r5.34 graphics)")

    master = np.zeros((n, 2), dtype=np.float32)
    muted = 0
    sound_notes: list[str] = []
    for i, e in enumerate(tl):
        windows = mute_windows(e, spans)
        try:
            if game_sound and e["kind"] == "final":
                previous = next((piece for piece in reversed(tl[:i]) if piece.get("parts")), None)
                buf = closing_game_sound(e, previous, spans, sound_notes)
            else:
                buf = (game_sound_entry(e, windows, sound_notes) if game_sound
                       else _entry_audio(e, windows, sound_notes))
        except RuntimeError as exc:
            print(f"[ERROR] mix: rink audio: {exc}")
            return 1
        muted += len(windows)
        _place(master, fit_to_length(buf, int(round(e["out_dur"] * SR))), e["out_start"])
        print(f"[mix] rink {i + 1}/{len(tl)} decoded", flush=True)
    for note in sound_notes:
        flag(note)

    from whistle_presentation import confirm_presentations
    presentation_whistles = confirm_presentations(tl, master, SR)
    for record in presentation_whistles:
        if record["onset_t"] is None:
            flag(f"{record['goal_id']}: whistle not confirmed in kept audio; announcement omitted")
    cues = plan_cues(tl, perspective=perspective, focus_side=focus_side, duration_s=duration_s,
                     pp_goals=pp_goals, comic_calls=comic_calls, goal_counts=_sheet_goal_counts(sheet),
                     period_wipes=period_wipes, cue_durations=cue_durations or None,
                     game_cues=game_cues, period_cuts=period_cuts)
    duck = duck_envelope(n, SR, [c for c in cues if not c["cue"].startswith("sfx_")])
    if game_sound:
        master = normalize_game_sound(master)
        if opening:
            video = opening["opening_video"]
            sound = decode_audio(video["path"], 0, video["frames"] / FPS)
            opening_entry = next(e for e in tl if e["kind"] == "stinger")
            _place(master, sound, opening_entry["out_start"])
        else:
            flag("game-sound mode has no mastered opening sound")
    else:
        _add_bed(master, bed_segments, bed_bufs, bed_gain_db, duck)

    manifest_cues = {c["id"]: c for c in (manifest or {}).get("cues", [])}
    skipped_cues: set[str] = set()
    bad_lufs: set[str] = set()
    # The report keeps every cue: graphics place the GOAL, penalty and FINAL cards from them.
    # Game-sound mode plays only the game cues over its rink sound.
    for c in ([] if game_sound and not game_cues else cues):
        if game_sound and c["cue"] not in GAME_CUES:
            continue
        mc = manifest_cues.get(c["cue"])
        if manifest is not None and mc is None:
            if c["cue"] not in skipped_cues:
                skipped_cues.add(c["cue"])
                flag(f"cue {c['cue']} is not in the Cue Library manifest; skipped")
            continue
        if mc is not None:
            try:
                buf = decode_audio(mc["path"], 0.0, float(mc.get("duration_s") or c["dur_s"]))
            except RuntimeError as exc:
                flag(f"cue {c['cue']} could not be decoded ({exc}); skipped")
                continue
            gain_db = float(mc.get("gain_db") or 0.0)
            if game_cues and mc.get("lufs") is not None:
                lufs = _cue_lufs(mc.get("lufs"))
                if lufs is None:
                    if c["cue"] not in bad_lufs:
                        bad_lufs.add(c["cue"])
                        flag(f"cue {c['cue']}: lufs {mc.get('lufs')!r} is not a measured loudness "
                             "(-70 to 0 LUFS); gain_db only")
                else:
                    gain_db += GAME_CUE_LUFS[c["cue"]] - lufs
            buf = buf * (10.0 ** (gain_db / 20.0))
            short = c["dur_s"] - len(buf) / SR
            if c.get("fade_out_s") and short > 0.05:
                flag(f"cue {c['cue']} is {short:.1f} s shorter than its window at {c['t']:.1f} s; "
                     "it fades at its own end and silence follows")
                c = dict(c, dur_s=len(buf) / SR)
        else:
            timed_open = c["cue"] == "game_start" and any(e["kind"] == "stinger" for e in tl)
            buf = synth_placeholder(c["cue"], duration_s=c["dur_s"] if timed_open else None)
        buf = _edge_fade(fit_to_length(buf, int(round(c["dur_s"] * SR))))
        fade = min(len(buf), int(round(float(c.get("fade_out_s") or 0.0) * SR)))
        if fade:
            buf[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)[:, None]
        _place(master, buf, c["t"])

    with tempfile.TemporaryDirectory(prefix="hockey-mix-") as scratch:
        raw = Path(scratch) / "mix.f32"
        raw.write_bytes(master.tobytes())
        encode_tp = TARGET_TP - ENCODE_TP_MARGIN
        measured = _measure_loudness(raw, raw_input=True, tp=encode_tp)
        if "input_i" not in measured or not math.isfinite(measured["input_i"]):
            print("[ERROR] mix: loudness measurement failed on the mixed audio")
            return 1
        video_in = root / plan["output"]
        if not video_in.exists():
            print(f"[ERROR] mix: {plan['output']} is missing; run the assembly stage first")
            return 1
        out_name = mixed_output_name(plan["output"])
        out_path = root / out_name
        tmp_path = out_path.with_name(out_path.stem + ".tmp.mp4")
        af = (f"loudnorm=I={TARGET_I}:TP={encode_tp}:LRA={TARGET_LRA}"
              f":measured_i={measured['input_i']}:measured_tp={measured['input_tp']}"
              f":measured_lra={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
              f":offset={measured['target_offset']}:linear=true,"
              # level=false: alimiter's default auto-level adds 1/limit of make-up
              # gain (+1.4 dB here), which pushes peaks back over the ceiling.
              f"alimiter=limit={LIMITER_CEILING}:level=false")
        if game_sound:
            af = f"alimiter=limit={LIMITER_CEILING}:level=false"
        out_i = out_tp = None
        for attempt in range(3):
            argv = ["ffmpeg", "-y", "-v", "error", "-i", str(video_in),
                    "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", str(raw),
                    "-map", "0:v", "-map", "1:a", "-af", af,
                    # loudnorm resamples to 192 kHz internally; pin the AAC track to SR.
                    "-ar", str(SR), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", "-shortest", str(tmp_path)]
            proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode != 0:
                tmp_path.unlink(missing_ok=True)
                print(f"[ERROR] mix: ffmpeg could not write the mixed Recap: "
                      f"{proc.stderr.decode(errors='replace')[:300].strip()}")
                return 1
            dur = _probe_duration(tmp_path)
            if abs(dur - duration_s) > 0.15:
                tmp_path.unlink(missing_ok=True)
                print(f"[ERROR] mix: mixed Recap duration {dur:.2f}s differs from the plan "
                      f"{duration_s:.2f}s")
                return 1
            check = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(tmp_path),
                                    "-f", "null", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if check.returncode != 0:
                tmp_path.unlink(missing_ok=True)
                print(f"[ERROR] mix: the mixed Recap does not decode cleanly: "
                      f"{check.stderr.decode(errors='replace')[:300].strip()}")
                return 1
            final = _measure_loudness(tmp_path, raw_input=False)
            if "input_i" not in final or "input_tp" not in final:
                flag("the mixed Recap loudness could not be re-measured")
                out_i = out_i if out_i is not None else final.get("input_i", 0.0)
                out_tp = out_tp if out_tp is not None else final.get("input_tp")
                break
            out_i, out_tp = final["input_i"], final["input_tp"]
            if out_tp <= TARGET_TP + 0.02 and (game_sound or out_i <= TARGET_I + 0.5):
                break
            # AAC encoding overshoots the true-peak ceiling by 0.1-0.3 dB on peaky
            # cues; corrected re-encodes from the raw mix bring the file under it.
            # Never chase the ceiling deeper than -1.5 LU below the loudness target.
            gain_db = ((TARGET_TP - 0.3) - out_tp if game_sound else
                       min((TARGET_TP - 0.3) - out_tp, (TARGET_I - 0.2) - out_i))
            if gain_db >= -0.01 or (not game_sound and out_i <= TARGET_I - 1.5):
                break
            print(f"[mix] re-encoding with {gain_db:.2f} dB correction "
                  f"(encoded file measured {out_i:.1f} LUFS / {out_tp:.1f} dBTP)", flush=True)
            af += f",volume={gain_db:.2f}dB"
        if out_i is not None and ((not game_sound and out_i > TARGET_I + 0.5)
                                  or (out_tp is not None and out_tp > TARGET_TP + 0.02)):
            tp_txt = f"{out_tp:.1f}" if out_tp is not None else "?"
            flag(f"output loudness {out_i:.1f} LUFS / {tp_txt} dBTP is off target "
                 f"({TARGET_I} LUFS / {TARGET_TP} dBTP)")
        os.replace(tmp_path, out_path)

    report = {"schema_version": 1, "output": out_name, "video": plan["output"],
              "duration_s": round(duration_s, 3), "perspective": perspective,
              "focus_side": focus_side, "placeholder": placeholder_mode, "mode": mode,
              "live_replay": live_replay, "game_cues": game_cues,
              "opening_sound": opening["opening_video"] if opening and game_sound else None,
              "manifest": manifest_path if manifest is not None else None,
              "bed_gain_db": bed_gain_db, "bed": bed_segments,
              "cues": cues, "muted_windows": muted,
              "presentation_whistles": presentation_whistles,
              "loudnorm": {"measured": measured, "output_i": out_i, "output_tp": out_tp},
              "flags": flags}
    (root / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n")
    if history_path is not None and bed_choices:
        from cue_library import record_rotation
        used_tracks = {s["track"] for s in bed_segments if s["out_end"] > s["out_start"]}
        recorded = {p: b for p, b in bed_choices.items()
                    if b["id"] in used_tracks and b["id"] in decoded_beds}
        try:
            if recorded:
                record_rotation(history_path, rotation_key, recorded)
        except (OSError, ValueError) as exc:
            flag(f"rotation history could not be saved: {exc}; next game may reuse Beds")
            (root / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n")
    i_txt = f"{out_i:.1f} LUFS integrated" if out_i is not None else "loudness not measured"
    tp_txt = f"{out_tp:.1f} dBTP peak" if out_tp is not None else "peak not measured"
    print(f"[mix] {out_name}: {duration_s:.1f}s, {len(cues)} cue(s), {muted} PA span(s) muted, "
          f"{i_txt}, {tp_txt}")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Mix Recap audio (bed, cues, rink sound)")
    ap.add_argument("game_folder")
    ap.add_argument("--options", default="{}", help="runner options as JSON")
    return ap.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
