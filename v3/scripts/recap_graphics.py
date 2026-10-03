"""Write Remotion props and overlay Ice Pak Slab graphics on the mixed Recap."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from audio_mix import build_timeline, period_of

PROPS_FILE = "recap_graphics_props.json"
REPORT_FILE = "recap_graphics.json"
PACKAGE = Path(__file__).resolve().parents[1] / "graphics"


def player_name(sheet: dict, side: str, num) -> str:
    value = (sheet.get("rosters") or {}).get(side, {}).get(str(num))
    return (value.get("name") if isinstance(value, dict) else value) or f"#{num}"


def goal_data(sheet: dict, gid: str) -> dict:
    side, index = gid.split(":")
    row = sheet["goals"][side][int(index) - 1]
    num = str(row.get("scorer") or "?")
    return {"id": gid, "team": side, "num": num, "scorer": player_name(sheet, side, num),
            "assists": [f"{n} {player_name(sheet, side, n)}" if not player_name(sheet, side, n).startswith("#")
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


def final_data(sheet: dict, flags: list) -> dict:
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
        for row in rows.values():
            row["pts"] = row["g"] + row["a"]
        goalies = []
        for row in (sheet.get("goalkeeping") or {}).get(side, []):
            if row.get("saves") is not None:
                num = str(row.get("player") or "?")
                goalies.append({"num": num, "name": row.get("name") or player_name(sheet, side, num),
                                "saves": int(row["saves"])})
        tables[side] = {"skaters": sorted(rows.values(), key=lambda r: (-r["pts"], -r["g"], -(r["pim"] or 0), r["num"])),
                        "goalies": goalies}
    return {"score": [len((sheet.get("goals") or {}).get(s, [])) for s in ("home", "away")], "table": tables}


def logo_path(value: str, options: dict) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    base = Path((options.get("_layers") or {}).get("config_dir") or Path.home() / "hockey").expanduser()
    return base / path


def team_theme(root: Path, sheet: dict, options: dict, flags: list) -> tuple[dict, str | None]:
    teams, focus = {}, None
    for side in ("home", "away"):
        name = sheet["teams"][side]
        key, config = next(((k, c) for k, c in (options.get("teams") or {}).items()
                            if c.get("name", "").casefold() == name.casefold()), (None, {}))
        if key is not None and key == options.get("focus_team"):
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
                assets = root / ".recap_cache" / "graphics" / "public"
                assets.mkdir(parents=True, exist_ok=True)
                logo = side + source.suffix.lower()
                shutil.copyfile(source, assets / logo)
            else:
                flags.append(f"{side}: logo missing; using wordmark")
        teams[side] = {"name": name, "acronym": config.get("short_name") or acronym[:3].upper(),
                       "primary": color("primary", "#5AA4D0" if side == "home" else "#7B2D36"),
                       "secondary": color("secondary", "#FFFFFF"), "logo": logo}
    if options.get("perspective") == "focus" and focus is None:
        flags.append("Focus Team does not match game_sheet.json; using neutral cards")
    return teams, focus


def write_props(root: Path, sheet: dict, selection: dict, plan: dict, audio: dict, options: dict,
                *, publish: bool = True) -> dict:
    fps = int(plan.get("fps") or 30)
    if fps != 30:
        raise ValueError("graphics require the 30 fps Recap timeline")
    entries = plan.get("opening", []) + plan["clips"] + plan.get("replays", [])
    if not entries or any(type(e["frames"]) is not int or e["frames"] <= 0 for e in entries):
        raise ValueError("assembly clips need positive integer frame counts")
    frames = sum(e["frames"] for e in entries)
    if abs(float(audio["duration_s"]) - frames / fps) > 1 / fps:
        raise ValueError("audio duration does not match the assembly plan")
    if audio["video"] != plan["output"]:
        raise ValueError("audio source does not match the assembly plan")
    events, flags = [], []
    timeline = build_timeline(plan)
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
        if e["kind"] in ("cold_open", "stinger"):
            continue
        gid = e["goal_id"]
        start, end = round(e["out_start"] * fps), round((e["out_start"] + e["out_dur"]) * fps)
        moment = e.get("moment_s", live[gid]["moment_s"])
        period = period_of(moment, selection.get("periods") or [])
        def bug(a, b, score):
            if b > a:
                score = [max(old, new) for old, new in zip(last_score, score)]
                last_score[:] = score
                events.append({"kind": "scorebug", "startFrame": a, "durationFrames": b - a,
                               "data": {"score": score, "period": period}})
        if e["kind"] == "goal":
            change = max(start, min(end, round(e["moment_out"] * fps)))
            bug(start, change, score_before(gid))
            bug(change, end, score_before(gid, True))
        elif gid in goal_order:
            bug(start, end, score_before(gid, True))
        else:
            score = [sum(g.startswith(s + ":") and moments.get(g) is not None
                          and moments[g] <= moment for g in goal_order) for s in ("home", "away")]
            bug(start, end, score)
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
            start = max(0, min(start, frames - 5 * fps))
            events.append({"kind": "final", "startFrame": start, "durationFrames": frames - start,
                            "data": final_data(sheet, flags)})
        elif cue["cue"] == "sfx_penalty_card":
            gid = cue["goal_id"]
            _, side, index = gid.split(":")
            row = sheet["penalties"][side][int(index) - 1]
            num = str(row.get("player") or "?")
            events.append({"kind": "penalty", "startFrame": start,
                           "durationFrames": min(4 * fps, frames - start),
                           "data": {"id": gid, "team": side, "num": num,
                                    "name": player_name(sheet, side, num),
                                     "minutes": row.get("minutes") if row.get("minutes") is not None else "?",
                                     "infraction": row.get("infraction") or "PENALTY"}})
        elif cue["cue"] == "sfx_period_wipe":
            # Bar-aligned cues can precede the new clip by a few frames.
            bugs = [e for e in events if e["kind"] == "scorebug"]
            current = next(e for e in bugs if e["startFrame"] <= start < e["startFrame"] + e["durationFrames"])
            bug_event = next((e for e in bugs if start <= e["startFrame"] <= start + fps and
                              e["data"]["period"] != current["data"]["period"]), current)
            data = {"period": bug_event["data"]["period"], "score": bug_event["data"]["score"]}
            events.append({"kind": "period_wipe", "startFrame": start,
                           "durationFrames": min(fps, frames - start), "data": dict(data)})
            if start + fps < frames:
                events.append({"kind": "period", "startFrame": start + fps,
                               "durationFrames": min(2 * fps, frames - start - fps), "data": data})
    top_cards = sorted((e for e in events if e["kind"] in ("goal", "penalty")),
                       key=lambda e: e["startFrame"])
    for i, card in enumerate(top_cards):
        start = card["startFrame"]
        clip_end = next(round((e["out_start"] + e["out_dur"]) * fps) for e in timeline
                        if round(e["out_start"] * fps) <= start < round((e["out_start"] + e["out_dur"]) * fps))
        end = min(start + card["durationFrames"], clip_end)
        if i + 1 < len(top_cards):
            end = min(end, top_cards[i + 1]["startFrame"])
        card["durationFrames"] = end - start
    for final in (e for e in events if e["kind"] == "final"):
        if any(e["startFrame"] + e["durationFrames"] > final["startFrame"] for e in top_cards):
            flags.append("FINAL overlaps a late card; review the closing timeline")
    events = [e for e in events if e["durationFrames"] > 0]
    events.sort(key=lambda e: 0 if e["kind"] == "scorebug" else 2 if e["kind"] == "final" else 1)
    props = {"schemaVersion": 1, "fps": fps, "width": 1920, "height": 1080,
             "durationFrames": frames, "events": events, "flags": flags}
    props["teams"], props["focusSide"] = team_theme(root, sheet, options, props["flags"])
    props["perspective"] = options.get("perspective") or "neutral"
    props["tokens"] = {"score": "#FFFFFF", "state": "#0D1B2A", "info": "#0A0A0A",
                        "hero": "#FFFFFF", "label": "#5AA4D0"}
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
        props["events"].append({"kind": "open_stinger", "startFrame": start_frame,
                                "durationFrames": duration, "data": {"team": props["focusSide"]}})
    for event in props["events"]:
        if event["kind"] == "period_wipe":
            event["data"]["team"] = props["focusSide"]
    props["events"].sort(key=lambda e: 0 if e["kind"] == "scorebug" else
                         2 if e["kind"] in ("open_stinger", "period_wipe") else
                         3 if e["kind"] == "final" else 1)
    for goal in goals:
        if moments.get(goal["id"]) is None:
            props["flags"].append(f"{goal['id']}: goal timing unavailable; scorebug needs review")
        elif goal in uncertain_clocks:
            props["flags"].append(f"{goal['id']}: sheet clock unavailable; score order uses camera timing")
    for side in ("home", "away"):
        if not any(row.get("saves") is not None for row in (sheet.get("goalkeeping") or {}).get(side, [])):
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
    args = parser.parse_args(argv)
    root = args.game_folder.resolve()
    temp = None
    video_published = False
    try:
        sheet, selection, plan, audio = [json.loads((root / n).read_text()) for n in
                                       ("game_sheet.json", "selection.json", "recap_assembly.json", "recap_audio.json")]
        options = json.loads(args.options)
        if not plan.get("clips"):
            raise ValueError("recap_assembly.json has no clips")
        props = write_props(root, sheet, selection, plan, audio, options, publish=False)
        video = root / audio["output"]
        stream = probe_video(video, props)
        props.update(width=int(stream["width"]), height=int(stream["height"]))
        for flag in props["flags"]:
            print(f"[graphics] flag: {flag}", flush=True)
        if args.props_only:
            (root / PROPS_FILE).write_text(json.dumps(props, indent=2) + "\n")
            print(f"[graphics] wrote {PROPS_FILE}", flush=True)
            return 0
        scratch = root / ".recap_cache" / "graphics"
        (scratch / "public").mkdir(parents=True, exist_ok=True)
        staged_props = scratch / PROPS_FILE
        staged_props.write_text(json.dumps(props, indent=2) + "\n")
        subprocess.run(["node", str(PACKAGE / "render.mjs"), str(staged_props), str(scratch)], check=True)
        output = root / graphics_output_name(plan["output"])
        temp = output.with_suffix(".tmp.mp4")
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
        os.replace(temp, output)
        video_published = True
        os.replace(staged_props, root / PROPS_FILE)
        os.replace(staged_report, root / REPORT_FILE)
        print(f"[graphics] {output.name}: {report['duration_s']:.1f}s, {len(props['events'])} graphics events", flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, IndexError, StopIteration, subprocess.CalledProcessError) as exc:
        if video_published:
            (root / REPORT_FILE).unlink(missing_ok=True)
        print(f"[ERROR] graphics: {exc}", flush=True)
        return 1
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
