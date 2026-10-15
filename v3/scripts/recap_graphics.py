"""Write Remotion props and overlay Ice Pak Slab graphics on the mixed Recap."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from audio_mix import build_timeline, period_of

PROPS_FILE = "recap_graphics_props.json"
REPORT_FILE = "recap_graphics.json"
PACKAGE = Path(__file__).resolve().parents[1] / "graphics"


def player_name(sheet: dict, side: str, num) -> str:
    value = (sheet.get("rosters") or {}).get(side, {}).get(str(num))
    return (value.get("name") if isinstance(value, dict) else value) or f"#{num}"


def player_number(sheet: dict, side: str, key) -> str:
    return (sheet.get("_numbers") or {}).get(side, {}).get(str(key), str(key))


def goal_data(sheet: dict, gid: str) -> dict:
    side, index = gid.split(":")
    row = sheet["goals"][side][int(index) - 1]
    num = str(row.get("scorer") or "?")
    return {"id": gid, "team": side, "num": player_number(sheet, side, num), "scorer": player_name(sheet, side, num),
            "assists": [f"{player_number(sheet, side, n)} {player_name(sheet, side, n)}" if not player_name(sheet, side, n).startswith("#")
                        else player_name(sheet, side, n) for n in row.get("assists") or []],
            "type": row.get("type") or "", "period": int(row["per"]) if str(row["per"]).isdigit() else 4}


def penalty_minutes(value) -> float | None:
    text = str(value if value is not None else "").strip()
    if re.fullmatch(r"\d+(?:\.\d+)?(?:\s*\+\s*\d+(?:\.\d+)?)*", text):
        return sum(float(part) for part in text.split("+"))
    if re.fullmatch(r"\d+:[0-5]\d", text):
        minutes, seconds = map(int, text.split(":"))
        return minutes + seconds / 60
    return None


def _count(value):
    """A non-negative whole count from the sheet, or None when unreadable."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) and number >= 0 and number == int(number) else None


MAX_FINAL_SKATERS = 10  # FINAL lists the top skaters per team, not the full roster


def final_data(sheet: dict, flags: list) -> dict:
    if "_final" in sheet:
        data = copy.deepcopy(sheet["_final"])
        for table in data["table"].values():
            table["skaters"] = table["skaters"][:MAX_FINAL_SKATERS]
        return data
    tables = {}
    for side in ("home", "away"):
        rows = {}
        def skater(num):
            num = str(num or "?")
            return rows.setdefault(num, {"num": num, "name": player_name(sheet, side, num),
                                         "g": 0, "a": 0, "pts": 0, "pim": 0})
        for goal in (sheet.get("goals") or {}).get(side, []):
            skater(goal.get("scorer"))["g"] += 1
            for num in goal.get("assists") or []:
                skater(num)["a"] += 1
        for penalty in (sheet.get("penalties") or {}).get(side, []):
            row = skater(penalty.get("player"))
            minutes = penalty_minutes(penalty.get("minutes"))
            if minutes is None:
                row["pim"] = None
                flags.append(f"{side}: penalty minutes unavailable for #{row['num']}; PIM needs review")
            elif row["pim"] is not None:
                row["pim"] += minutes
        keepers = (sheet.get("goalkeeping") or {}).get(side, [])
        for num in list(rows):  # list only skaters with a point or PIM; a goalie is never a zero row
            if not any(rows[num][k] for k in ("g", "a", "pim")) and rows[num]["pim"] is not None:
                del rows[num]
        for row in rows.values():
            row["pts"] = row["g"] + row["a"]
        other = "away" if side == "home" else "home"
        shots = ((sheet.get("shots_on_goal") or {}).get(other))
        opp_shots = sum(shots) if shots and all(isinstance(v, (int, float)) for v in shots) else None
        opp_goals = len((sheet.get("goals") or {}).get(other, []))
        goalies = []
        for row in keepers:
            num = str(row.get("player") or "?")
            single = len(keepers) == 1
            ga = _count(row.get("ga"))
            ga = ga if ga is not None else (opp_goals if single else None)
            sa = _count(row.get("shots"))
            sa = sa if sa is not None else (opp_shots if single else None)
            saves = _count(row.get("saves"))
            if saves is None and sa is not None and ga is not None:
                saves = sa - ga
            if sa is None and saves is not None and ga is not None:
                sa = saves + ga
            if saves is not None and saves < 0:
                flags.append(f"{side}: goalie #{num} shots below goals against; saves need review")
                saves = None
            goalies.append({"num": num, "name": row.get("name") or player_name(sheet, side, num),
                            "sa": None if sa is None else int(sa), "ga": None if ga is None else int(ga),
                            "saves": None if saves is None else int(saves),
                            "sv_pct": round(saves / sa, 3) if saves is not None and sa else None})
        ranked = sorted(rows.values(), key=lambda r: (-r["pts"], -r["g"], -(r["pim"] or 0), r["num"]))
        tables[side] = {"skaters": ranked[:MAX_FINAL_SKATERS],
                        "goalies": goalies}
    return {"score": [len((sheet.get("goals") or {}).get(s, [])) for s in ("home", "away")], "table": tables}


def logo_path(value: str, options: dict) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    base = Path((options.get("_layers") or {}).get("config_dir") or Path.home() / "hockey").expanduser()
    return base / path


PLATE = "#0E1319"          # graphite reading plate (spec 003)
# Matchup rail over a pre-rendered opening (spec 003), in frames from the opening's first
# frame: it enters as the hero shot settles, holds after the dissolve, then exits.
RAIL_IN, RAIL_HOLD, RAIL_EXIT = 95, 75, 10
FROST = "#9FB2C1"          # frost-grey fallback line
_HEX = re.compile(r"#[0-9a-fA-F]{6}")


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio of two #RRGGBB colours."""
    def lum(h):
        c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def close(a: str, b: str) -> bool:
    """True when two #RRGGBB colours are too similar to tell teams apart."""
    return sum((int(a[i:i + 2], 16) - int(b[i:i + 2], 16)) ** 2 for i in (1, 3, 5)) ** .5 < 120


def apply_kit(team: dict, config: dict, kit: str | None, side: str, flags: list) -> str | None:
    """Set the Focus Team kit's fill, line, and text; return its crest path, if any.

    A line must read on the graphite plate (3:1). Lines that fail use frost grey.
    """
    line = None
    crest = None
    kits = config.get("kits") or {}
    if kit is not None:
        values = kits.get(kit)
        if not isinstance(values, dict):
            flags.append(f"{side}: kit '{kit}' not in Team Config; using team colors")
        else:
            bad = [k for k in ("fill", "line", "text") if k in values
                   and not (isinstance(values[k], str) and _HEX.fullmatch(values[k]))]
            if bad or "fill" not in values or "line" not in values:
                flags.append(f"{side}: kit '{kit}' needs #RRGGBB fill and line; using team colors")
            else:
                team["kit"] = kit
                team["primary"] = values["fill"]
                line = values.get("line")
                team["text"] = values.get("text") or (
                    "#FFFFFF" if contrast("#FFFFFF", values["fill"]) >= 4.5 else PLATE)
                crest = values.get("crest")
                if line and contrast(line, PLATE) < 3:
                    flags.append(f"{side}: kit '{kit}' line {line} fails 3:1 on {PLATE}; using {FROST}")
    line = line or team["primary"]
    team["line"] = line if contrast(line, PLATE) >= 3 else FROST
    return crest


def team_theme(root: Path, sheet: dict, options: dict, flags: list,
               assets_dir: Path | None = None) -> tuple[dict, str | None]:
    teams, focus = {}, None
    for side in ("home", "away"):
        name = sheet["teams"][side]
        key, config = next(((k, c) for k, c in (options.get("teams") or {}).items()
                            if c.get("name", "").casefold() == name.casefold()), (None, {}))
        wanted = options.get("focus_team")
        is_focus = wanted is not None and str(wanted).casefold() in {
            str(v).casefold() for v in (key, name, config.get("name")) if v}
        if is_focus:
            focus = side
        words = re.findall(r"[A-Za-z0-9]+", name)
        acronym = "".join(w[0] for w in words)
        if len(acronym) < 3 and words:
            acronym += words[-1].rstrip("sS")[-1:]
        colors = config.get("colors") or {}
        if isinstance(colors, list):
            colors = dict(zip(("primary", "secondary"), colors))
        def color(key, default):
            value = colors.get(key, default)
            if not isinstance(value, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
                flags.append(f"{side}: invalid {key} color; using default")
                return default
            return value
        logo = None
        if config.get("logo"):
            source = logo_path(config["logo"], options)
            if source.is_file() and source.suffix.lower() in (".png", ".jpg", ".jpeg", ".svg", ".webp"):
                assets = assets_dir or root / ".recap_cache" / "graphics" / "public"
                assets.mkdir(parents=True, exist_ok=True)
                logo = side + source.suffix.lower()
                shutil.copyfile(source, assets / logo)
            else:
                flags.append(f"{side}: logo missing; using wordmark")
        teams[side] = {"name": name, "acronym": config.get("short_name") or acronym[:3].upper(),
                       "primary": color("primary", "#5AA4D0" if side == "home" else "#7B2D36"),
                        "secondary": color("secondary", "#FFFFFF"), "logo": logo}
        kit_crest = apply_kit(teams[side], config, options.get("kit") if is_focus else None, side, flags)
        sources = {a: config.get(a) for a in ("wordmark", "lockup", "crest")}
        if kit_crest:
            sources["crest"] = kit_crest
        for asset, value in sources.items():
            if value:
                source = logo_path(value, options)
                if source.is_file() and source.suffix.lower() in (".png", ".jpg", ".jpeg", ".svg", ".webp"):
                    assets = assets_dir or root / ".recap_cache" / "graphics" / "public"
                    assets.mkdir(parents=True, exist_ok=True)
                    name_out = f"{side}_{asset}{source.suffix.lower()}"
                    shutil.copyfile(source, assets / name_out)
                    teams[side][asset] = name_out
                else:
                    flags.append(f"{side}: {asset} missing; using "
                                 + ("logo" if asset == "crest" else "text"))
    if focus is not None:
        # Two similar lines read as one team: the opponent uses its secondary, or frost grey.
        own, other = teams[focus], teams["away" if focus == "home" else "home"]
        if close(other["line"], own["line"]):
            alt = other["secondary"]
            other["line"] = alt if contrast(alt, PLATE) >= 3 and not close(alt, own["line"]) else FROST
    if options.get("perspective") == "focus" and focus is None:
        flags.append("Focus Team does not match game_sheet.json; using neutral cards")
    return teams, focus


def write_props(root: Path, sheet: dict, selection: dict, plan: dict, audio: dict, options: dict,
                *, publish: bool = True, assets_dir: Path | None = None) -> dict:
    fps = int(plan.get("fps") or 30)
    if fps != 30:
        raise ValueError("graphics require the 30 fps Recap timeline")
    entries = plan.get("opening", []) + plan["clips"] + plan.get("replays", []) + plan.get("closing", [])
    if not entries or any(type(e["frames"]) is not int or e["frames"] <= 0 for e in entries):
        raise ValueError("assembly clips need positive integer frame counts")
    frames = sum(e["frames"] for e in entries)
    if abs(float(audio["duration_s"]) - frames / fps) > 1 / fps:
        raise ValueError("audio duration does not match the assembly plan")
    if audio["video"] != plan["output"]:
        raise ValueError("audio source does not match the assembly plan")
    events, flags = [], list(sheet.get("flags") or [])
    timeline = build_timeline(plan)
    frame_cursor = 0
    for entry in timeline:
        entry["frame_start"] = frame_cursor
        frame_cursor += entry["frames"]
        entry["frame_end"] = frame_cursor
    goals = sorted(selection.get("goals") or [], key=lambda g: (g["period"] or 999,
                   g["elapsed_s"] if g["elapsed_s"] is not None else math.inf,
                   g["id"].split(":")[0], int(g["id"].split(":")[1])))
    goal_order = [g["id"] for g in goals]
    sheet_ids = {f"{s}:{i + 1}" for s in ("home", "away") for i, _ in enumerate(sheet["goals"].get(s, []))}
    if set(goal_order) != sheet_ids or len(goal_order) != len(sheet_ids):
        raise ValueError("selection goals do not match game_sheet.json; rerun selection")
    def score_before(gid, inclusive=False):
        stop = goal_order.index(gid) + int(inclusive)
        return [sum(g.startswith(s + ":") for g in goal_order[:stop]) for s in ("home", "away")]

    live = {e["goal_id"]: e for e in timeline if e["kind"] in ("goal", "play")}
    if len(live) != len(plan["clips"]) or any(r["goal_id"] not in live for r in plan.get("replays", [])):
        raise ValueError("assembly has duplicate clips or orphan replays")
    moments = {g["id"]: (g.get("chosen") or {}).get("detection_s", (g.get("window") or {}).get("expected"))
               for g in selection.get("goals") or []}
    moments.update({gid: e["moment_s"] for gid, e in live.items() if e["kind"] == "goal"})
    uncertain_clocks = [g for g in goals if g["period"] is None or g["elapsed_s"] is None]
    if uncertain_clocks:
        goals.sort(key=lambda g: (moments.get(g["id"]) is None,
                                 moments.get(g["id"]) if moments.get(g["id"]) is not None else math.inf))
        goal_order = [g["id"] for g in goals]
    last_score = [0, 0]
    for e in timeline:
        if e["kind"] in ("cold_open", "stinger", "final"):
            continue
        gid = e["goal_id"]
        start, end = e["frame_start"], e["frame_end"]
        moment = e.get("moment_s", live[gid]["moment_s"])
        period = period_of(moment, selection.get("periods") or [])
        def bug(a, b, score):
            if b > a:
                score = [max(old, new) for old, new in zip(last_score, score)]
                last_score[:] = score
                events.append({"kind": "scorebug", "startFrame": a, "durationFrames": b - a,
                               "data": {"score": score, "period": period}})
        if e["kind"] == "goal":
            confirmation = next((c["t"] for c in audio.get("cues", [])
                                 if c["cue"] == "sfx_goal_card" and c.get("goal_id") == gid), None)
            # No confirmation: no announcement, but the verified goal still counts from its
            # moment in the footage (spec 003, goal and decision states, rule 4).
            when = confirmation if confirmation is not None else e.get("moment_out")
            change = end if when is None else max(start, min(end, round(when * fps)))
            bug(start, change, score_before(gid))
            bug(change, end, score_before(gid, True))
        else:
            if gid in goal_order:
                score = score_before(gid, True)
            else:
                score = [sum(g.startswith(s + ":") and moments.get(g) is not None
                              and moments[g] <= moment for g in goal_order) for s in ("home", "away")]
            bug(start, end, score)
            # A replay shot (a goal's, or the best non-goal play's) carries its shot kind, not "play".
            if e["kind"] != "play":
                events.append({"kind": "replay", "startFrame": start, "durationFrames": end - start, "data": {}})
    for cue in audio.get("cues") or []:
        if not cue["cue"].startswith("sfx_"):
            continue
        if not math.isfinite(cue["t"]) or cue["t"] < 0 or round(cue["t"] * fps) >= frames:
            raise ValueError("graphics cue is outside the Recap timeline")
        start = round(cue["t"] * fps)
        if cue["cue"] == "sfx_goal_card":
            events.append({"kind": "goal", "startFrame": start,
                           "durationFrames": min(4 * fps, frames - start),
                           "data": goal_data(sheet, cue["goal_id"])})
        elif cue["cue"] == "sfx_final_card":
            closing = next((e for e in timeline if e.get("kind") == "final"), None)
            start = round(closing["out_start"] * fps) if closing else max(0, min(start, frames - 5 * fps))
            events.append({"kind": "final", "startFrame": start, "durationFrames": frames - start,
                            "data": final_data(sheet, flags)})
        elif cue["cue"] == "sfx_penalty_card":
            gid = cue["goal_id"]
            _, side, index = gid.split(":")
            row = sheet["penalties"][side][int(index) - 1]
            num = str(row.get("player") or "?")
            events.append({"kind": "penalty", "startFrame": start,
                           "durationFrames": min(4 * fps, frames - start),
                            "data": {"id": gid, "team": side, "num": player_number(sheet, side, num),
                                    "name": player_name(sheet, side, num),
                                     "minutes": row.get("minutes") if row.get("minutes") is not None else "?",
                                     "infraction": row.get("infraction") or "PENALTY"}})
        elif cue["cue"] == "sfx_period_wipe":
            boundaries = []
            previous_period = None
            for entry in timeline:
                if entry["kind"] in ("cold_open", "stinger", "final"):
                    continue
                period = period_of(entry.get("moment_s", live[entry["goal_id"]]["moment_s"]),
                                   selection.get("periods") or [])
                if previous_period is not None and period != previous_period:
                    boundaries.append((entry["frame_start"], period, entry["frame_end"]))
                previous_period = period
            nearby = [b for b in boundaries if abs(b[0] - start) <= fps]
            if not nearby:
                flags.append("period transition has no nearby verified source cut; omitted")
                continue
            cut, period, segment_end = min(nearby, key=lambda b: abs(b[0] - start))
            if cut < 15 or cut + 15 > frames:
                flags.append("period transition needs a complete 30-frame pane; omitted")
                continue
            if any(e["kind"] == "period_wipe" and e["data"]["cutFrame"] == cut for e in events):
                continue
            bug_event = next(e for e in events if e["kind"] == "scorebug" and
                             e["startFrame"] <= cut < e["startFrame"] + e["durationFrames"])
            data = {"period": period, "score": bug_event["data"]["score"]}
            events.append({"kind": "period_wipe", "startFrame": cut - 15,
                           "durationFrames": 30, "data": dict(data, cutFrame=cut)})
            if cut + 15 + 82 <= segment_end:
                events.append({"kind": "period", "startFrame": cut + 15,
                               "durationFrames": 82, "data": data})
            else:
                flags.append("period plate has insufficient readable time; omitted")
            if start != cut - 15:
                flags.append("period sound cue differs from pane entry; audio alignment needs review")
    top_cards = sorted((e for e in events if e["kind"] in ("goal", "penalty")),
                       key=lambda e: e["startFrame"])
    for i, card in enumerate(top_cards):
        start = card["startFrame"]
        # A late GOAL card can continue into that goal's replay, not the next play.
        matching = ([e for e in timeline if e.get("goal_id") == card["data"]["id"]]
                    if card["kind"] == "goal" else
                    [e for e in timeline if round(e["out_start"] * fps) <= start <
                     round((e["out_start"] + e["out_dur"]) * fps)])
        clip_end = max(round((e["out_start"] + e["out_dur"]) * fps) for e in matching)
        end = min(start + card["durationFrames"], clip_end)
        if i + 1 < len(top_cards):
            end = min(end, top_cards[i + 1]["startFrame"] - 4)
        card["durationFrames"] = end - start
        minimum = (14 if card["kind"] == "goal" else 12) + 75 + 10
        if card["durationFrames"] < minimum:
            flags.append(f'{card["data"]["id"]}: insufficient readable time; {card["kind"]} rail omitted')
            card["durationFrames"] = 0
    for final in (e for e in events if e["kind"] == "final"):
        if any(e["startFrame"] + e["durationFrames"] > final["startFrame"] for e in top_cards):
            flags.append("FINAL overlaps a late card; review the closing timeline")
    events = [e for e in events if e["durationFrames"] > 0]
    for card in [e for e in events if e["kind"] == "period"]:
        if any(e["kind"] in ("goal", "penalty") and
               card["startFrame"] < e["startFrame"] + e["durationFrames"] + 4 and
               e["startFrame"] < card["startFrame"] + card["durationFrames"] + 4 for e in events):
            events.remove(card)
            flags.append("period plate conflicts with an event rail; omitted")
    period_cuts = {e["data"]["cutFrame"] for e in events if e["kind"] == "period_wipe"}
    replay_cuts = set()
    for replay in [e for e in events if e["kind"] == "replay"]:
        for edge, cut in (("in", replay["startFrame"]),
                          ("out", replay["startFrame"] + replay["durationFrames"])):
            if cut in period_cuts or cut in replay_cuts or cut < 8 or cut + 8 > frames:
                continue
            if edge == "out" and any(e["kind"] == "final" and e["startFrame"] == cut for e in events):
                continue
            events.append({"kind": "replay_pane", "startFrame": cut - 8,
                           "durationFrames": 16, "data": {"cutFrame": cut, "edge": edge}})
            replay_cuts.add(cut)
    events.sort(key=lambda e: 0 if e["kind"] == "scorebug" else 2 if e["kind"] == "final" else 1)
    props = {"schemaVersion": 1, "fps": fps, "width": 1920, "height": 1080,
             "durationFrames": frames, "events": events, "flags": flags}
    props["teams"], props["focusSide"] = team_theme(root, sheet, options, props["flags"], assets_dir)
    props["perspective"] = options.get("perspective") or "neutral"
    props["tokens"] = {"score": "#FFFFFF", "state": "#0D1B2A", "info": "#0A0A0A",
                        "hero": "#FFFFFF", "label": "#5AA4D0",
                        # Black Ice accent: the Focus Team kit line; ice cyan without a Focus Team.
                        "accent": props["teams"][props["focusSide"]]["line"] if props["focusSide"] else "#7FD3F0"}
    stinger = next((e for e in timeline if e["kind"] == "stinger"), None)
    start_mode = plan.get("start", options.get("start", "play"))
    if start_mode == "cold_open" and (stinger is None or not any(e["kind"] == "cold_open" for e in timeline)):
        raise ValueError("cold open requires a reserved teaser and stinger; rerun assembly")
    if stinger is not None or start_mode == "stinger":
        cue = next((c for c in audio.get("cues", []) if c["cue"] == "game_start"), None)
        if cue is None:
            raise ValueError("open stinger requires the game_start audio cue; rerun mix")
        start_frame = round(cue["t"] * fps)
        duration = stinger["frames"] if stinger is not None else min(75, frames - start_frame)
        if stinger is not None and start_frame != round(stinger["out_start"] * fps):
            raise ValueError("open stinger does not match game_start audio cue; rerun mix")
        data = {"team": props["focusSide"], "date": options.get("date") or sheet.get("date")}
        video = (stinger or {}).get("opening_video")
        if video:
            # The opening is in the video; only the matchup rail is drawn. It enters over
            # the hero shot and holds over the first play after the dissolve (spec 003).
            rail_start = start_frame + RAIL_IN
            rail_end = min(frames, start_frame + video["frames"] - 1 + RAIL_HOLD + RAIL_EXIT)
            later = [e["startFrame"] for e in props["events"]
                     if e["kind"] in ("goal", "penalty", "call", "period") and e["startFrame"] >= rail_start]
            if later and min(later) - 4 < rail_end:
                rail_end = min(later) - 4
                props["flags"].append("matchup rail shortened to clear the first information rail")
            if rail_end - rail_start >= RAIL_EXIT + 30:
                props["events"].append({"kind": "matchup_rail", "startFrame": rail_start,
                                         "durationFrames": rail_end - rail_start, "data": data})
            else:
                props["flags"].append("matchup rail left out: no room before the first information rail")
        else:
            props["events"].append({"kind": "open_stinger", "startFrame": start_frame,
                                     "durationFrames": duration, "data": data})
    for event in props["events"]:
        if event["kind"] == "period_wipe":
            event["data"]["team"] = props["focusSide"]
    props["events"].sort(key=lambda e: -1 if e["kind"] == "replay_pane" else 0 if e["kind"] == "scorebug" else
                         2 if e["kind"] in ("open_stinger", "period_wipe", "matchup_rail") else
                         3 if e["kind"] == "final" else 1)
    for goal in goals:
        if moments.get(goal["id"]) is None:
            props["flags"].append(f"{goal['id']}: goal timing unavailable; scorebug needs review")
        elif goal in uncertain_clocks:
            props["flags"].append(f"{goal['id']}: sheet clock unavailable; score order uses camera timing")
    table = final_data(sheet, [])["table"]
    for side in ("home", "away"):
        if not any(g["saves"] is not None for g in table[side]["goalies"]):
            props["flags"].append(f"{side}: goalie saves unavailable in game_sheet.json")
    if publish:
        (root / PROPS_FILE).write_text(json.dumps(props, indent=2) + "\n")
    return props


def graphics_output_name(name: str) -> str:
    path = Path(name)
    return path.stem + "_graphics.mp4"


def probe_video(video: Path, props: dict) -> dict:
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video)],
                           check=True, capture_output=True, text=True)
    stream = next(s for s in json.loads(probe.stdout)["streams"] if s["codec_type"] == "video")
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    if not denominator or numerator / denominator != props["fps"]:
        raise ValueError(f"{video.name}: frame rate does not match the assembly plan")
    if stream.get("nb_frames") in (None, "N/A") or int(stream["nb_frames"]) != props["durationFrames"]:
        raise ValueError(f"{video.name}: frame count does not match the assembly plan")
    return stream


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game_folder", type=Path)
    parser.add_argument("--options", default="{}")
    parser.add_argument("--props-only", action="store_true")
    parser.add_argument("--workspace", type=Path, help="Local directory for graphics intermediates")
    args = parser.parse_args(argv)
    root = args.game_folder.resolve()
    temp = None
    video_published = False
    publication_files = []
    try:
        selection, plan, audio = [json.loads((root / n).read_text()) for n in
                                 ("selection.json", "recap_assembly.json", "recap_audio.json")]
        from recap_assembly import load_play_reviews
        selection = load_play_reviews(root, selection)
        options = json.loads(args.options)
        from game_data import load_graphics_inputs
        sheet, options = load_graphics_inputs(root, options)
        if not plan.get("clips"):
            raise ValueError("recap_assembly.json has no clips")
        key = hashlib.sha256(str(root).encode()).hexdigest()[:20]
        scratch = (args.workspace or Path(tempfile.gettempdir()) / "hockeyrecap-graphics" / key).resolve()
        if scratch == root or root in scratch.parents:
            raise ValueError("graphics workspace must be outside the game folder")
        (scratch / "public").mkdir(parents=True, exist_ok=True)
        props = write_props(root, sheet, selection, plan, audio, options, publish=False,
                            assets_dir=scratch / "public")
        video = root / audio["output"]
        stream = probe_video(video, props)
        props.update(width=int(stream["width"]), height=int(stream["height"]))
        for flag in props["flags"]:
            print(f"[graphics] flag: {flag}", flush=True)
        if args.props_only:
            (root / PROPS_FILE).write_text(json.dumps(props, indent=2) + "\n")
            print(f"[graphics] wrote {PROPS_FILE}", flush=True)
            return 0
        staged_props = scratch / PROPS_FILE
        staged_props.write_text(json.dumps(props, indent=2) + "\n")
        subprocess.run(["node", str(PACKAGE / "render.mjs"), str(staged_props), str(scratch)], check=True)
        output = root / graphics_output_name(plan["output"])
        temp = scratch / output.with_suffix(".tmp.mp4").name
        filters = (f"[1:v]scale={props['width']}:{props['height']}:flags=lanczos,format=rgba[alpha];"
                   "[0:v][alpha]overlay=x=0:y=0:format=rgb:eof_action=endall,"
                   "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
                   "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709[v]")
        subprocess.run(["ffmpeg", "-y", "-v", "warning", "-i", str(video), "-f", "concat", "-safe", "0",
                        "-i", str(scratch / "clips.ffconcat"), "-filter_complex", filters,
                        "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                          "-c:a", "copy", "-colorspace", "bt709", "-color_range", "tv",
                          "-color_primaries", "bt709", "-color_trc", "bt709",
                         "-frames:v", str(props["durationFrames"]), "-movflags", "+faststart", str(temp)],
                        check=True)
        probe_video(temp, props)
        report = {"schema_version": 1, "output": output.name, "video": audio["output"],
                  "props": PROPS_FILE, "duration_s": props["durationFrames"] / props["fps"],
                  "flags": props["flags"]}
        staged_report = scratch / REPORT_FILE
        staged_report.write_text(json.dumps(report, indent=2) + "\n")
        # Copy every artifact before replacing any existing output on the destination filesystem.
        for source, destination in ((temp, output), (staged_props, root / PROPS_FILE),
                                    (staged_report, root / REPORT_FILE)):
            fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=root)
            os.close(fd)
            staged = Path(name)
            publication_files.append((staged, destination))
            shutil.copyfile(source, staged)
            shutil.copymode(source, staged)
        for staged, destination in publication_files:
            os.replace(staged, destination)
            if destination == output:
                video_published = True
        print(f"[graphics] {output.name}: {report['duration_s']:.1f}s, {len(props['events'])} graphics events", flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, IndexError, StopIteration, subprocess.CalledProcessError) as exc:
        if video_published:
            (root / REPORT_FILE).unlink(missing_ok=True)
        print(f"[ERROR] graphics: {exc}", flush=True)
        return 1
    finally:
        for staged, _destination in publication_files:
            staged.unlink(missing_ok=True)
        if temp is not None:
            temp.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
