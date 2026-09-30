# v3/scripts/scoresheet.py
"""
Read a photographed Scoresheet into a Game Sheet, locally (spec 002 §5.5).

1. Apple Vision (on-device OCR, via PyObjC) reads the printed text: table
   labels, team names and the printed rosters. It is reliable on print.
2. Each table (SCORING, PENALTIES, SCORE BY PERIODS) is located from its label
   and the ruled lines around it, so no per-form template is needed.
3. A local vision-language model (Qwen3-VL-8B, 4-bit, MLX) reads the
   handwritten cells of one table crop at a time, numbers only. On the first
   test sheet it read 34/35 goal cells; a whole-page read was clearly worse.
4. Hockey consistency checks flag rows for the review page.

Output: game_sheet.json in the Game Folder. Every goal and penalty row has
"status": "ok" or "review", with "reasons". The Game Sheet is a claim to
verify against the video (CONTEXT.md), never ground truth.

A GameSheet PDF export in the Game Folder is read instead, exactly, by
gamesheet_pdf.py (no model).

Needs the optional ML stack (requirements-ml.txt): mlx-vlm, pyobjc Vision.
Runs on Apple Silicon only. No API key and no per-game cost.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

MODEL = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
PHOTO_EXT = {".jpg", ".jpeg", ".png", ".heic"}
PIPELINE_IMAGES = {"rois_preview.png"}
GOAL_TYPES = ("ES", "PP", "SH", "EN", "PS")
GOAL_TYPE_ALIASES = {"EV": "ES", "EQ": "ES"}  # forms write even strength as EV
MAX_PERIOD_S = 25 * 60          # no League period is longer; used only to flag impossible times

# ---------------------------------------------------------------------------
# Finding the photo
# ---------------------------------------------------------------------------

# Files that are never a Scoresheet: pipeline outputs, editor exports,
# thumbnails and stills from earlier runs (hhg-3r5.28). Names are matched
# case-insensitively on the file name only.
NON_SHEET_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"^rois_preview\.png$",
    r"^scoresheet_",
    r"^recap.*\.(png|jpe?g)$",
    r"^yt_thumb.*",
    r"^still .*",
    r".*_overlay\.(png|jpe?g)$",
    r"^markers\..*",
    r"^events\.csv$",
    r"^cam\d+\.mp4$",
    r"^(GOPR|GP|G[XH])\d+.*\.jpe?g$",   # GoPro photos from the camera card
)]


def is_non_sheet_file(name: str) -> bool:
    """True for files that must never be read as a Scoresheet or GameSheet."""
    return name.startswith(".") or any(p.match(name) for p in NON_SHEET_PATTERNS)


def find_scoresheet_photos(game_folder: str) -> list[str]:
    """Image files in the Game Folder that can be a Scoresheet photo, largest first.

    Only the top level is scanned, so folders such as editprep/ are ignored.
    """
    root = Path(game_folder)
    photos = [p for p in root.iterdir()
              if p.is_file() and p.suffix.lower() in PHOTO_EXT and p.name not in PIPELINE_IMAGES
              and not is_non_sheet_file(p.name)]
    return [str(p) for p in sorted(photos, key=lambda p: -p.stat().st_size)]


# ---------------------------------------------------------------------------
# Printed text (Apple Vision)
# ---------------------------------------------------------------------------

def ocr_lines(path: str) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Printed text lines with boxes (x, y, w, h), normalized, y from the top."""
    import Quartz
    import Vision
    from Foundation import NSURL
    src = Quartz.CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(path), None)
    img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(0)            # accurate
    req.setUsesLanguageCorrection_(False)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None)
    handler.performRequests_error_([req], None)
    out = []
    for o in req.results() or []:
        b = o.boundingBox()
        out.append((str(o.topCandidates_(1)[0].string()),
                    (b.origin.x, 1 - b.origin.y - b.size.height, b.size.width, b.size.height)))
    return out


_ROSTER_ROW = re.compile(r"^\s*(\d{1,3})\s*[|°'.]?\s*([A-Za-z][A-Za-z.'\- ]*[A-Za-z])(?:\s+[vV✓√/]+)?(?:\s+(\d{1,3}))?")


def parse_team_names(lines) -> dict:
    teams = {}
    for text, _ in lines:
        m = re.match(r"^\s*(HOME|AWAY)\s*:\s*(.+)$", text, re.I)
        if m:
            teams[m.group(1).lower()] = m.group(2).strip()
    return teams


def parse_rosters(lines) -> dict[str, dict[str, str]]:
    """
    Printed '<number> <name>' rows under the HOME: and AWAY: headers. OCR often
    returns the number and the name as separate boxes, so boxes in the roster
    column are grouped into rows by height and joined left to right. A number
    written after the name (a scorekeeper's correction) is added as an alias.
    """
    heads = {}
    for text, box in lines:
        m = re.match(r"^\s*(HOME|AWAY)\s*:", text, re.I)
        if m:
            heads[m.group(1).lower()] = box
    rosters: dict[str, dict[str, str]] = {"home": {}, "away": {}}
    for side, (hx, hy, hw, hh) in heads.items():
        cells = [(y + h / 2, x, t) for t, (x, y, w, h) in lines
                 if hx - 0.03 <= x <= hx + 0.12 and hy + hh * 0.5 < y < hy + 0.6]
        rows: list[list[tuple[float, float, str]]] = []
        for c in sorted(cells):
            if rows and abs(c[0] - rows[-1][0][0]) < 0.008:
                rows[-1].append(c)
            else:
                rows.append([c])
        for row in rows:
            text = " ".join(t for _, _, t in sorted(row, key=lambda c: c[1]))
            m = _ROSTER_ROW.match(text)
            if m:
                name = m.group(2).strip()
                rosters[side][m.group(1)] = name
                if m.group(3):
                    rosters[side][m.group(3)] = name
    return rosters


def _jersey(v) -> str:
    s = _clean(v)
    return s if re.fullmatch(r"\d{1,3}", s) else ""


def merge_roster(printed: dict[str, str], model_rows) -> dict[str, str]:
    """
    One roster from two reads: the printed rows (Apple Vision) and the model's
    read of the roster crop, which also sees handwritten numbers. A number from
    either read counts as on the roster; the roster check only flags rows for
    review, so a missing number costs more than an extra one. Names come from
    the printed read when a model name matches one; margin numbers map to "".
    """
    roster = dict(printed)
    names = set(printed.values())
    for row in model_rows if isinstance(model_rows, list) else []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if name and names:
            close = min(names, key=lambda n: _edit_distance(n.lower(), name.lower()))
            if _edit_distance(close.lower(), name.lower()) <= 2:
                name = close
        others = row.get("other_numbers") or []
        for num in [row.get("number")] + (others if isinstance(others, list) else [others]):
            n = _jersey(num)
            if n and n not in roster:
                roster[n] = name
    return roster


# ---------------------------------------------------------------------------
# Locating the tables
# ---------------------------------------------------------------------------

def _grid_boxes(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of ruled regions (tables): long horizontal + vertical lines, dilated."""
    import cv2
    h, w = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    hz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (w // 30, 1)))
    vt = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 30)))
    grid = cv2.dilate(hz | vt, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 0.005 * w * h]


def locate_tables(image: np.ndarray, lines) -> dict[str, tuple[int, int, int, int]]:
    """
    Pixel boxes of the tables: home_scoring, away_scoring, home_penalties,
    away_penalties, periods. Each is the smallest ruled region containing its
    printed label. Home is the left one of a pair (the standard form layout).
    """
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape
    boxes = _grid_boxes(gray)

    def region_at(box):
        x, y, bw, bh = box
        px, py = (x + bw / 2) * w, (y + bh / 2) * h
        hit = [b for b in boxes if b[0] <= px <= b[0] + b[2] and b[1] <= py <= b[1] + b[3]]
        return min(hit, key=lambda b: b[2] * b[3]) if hit else None

    found: dict[str, tuple[int, int, int, int]] = {}
    for label, key in (("SCORING", "scoring"), ("PENALTIES", "penalties")):
        regions = sorted({r for t, b in lines if t.strip().upper() == label and (r := region_at(b))}, key=lambda r: r[0])
        if len(regions) == 2:
            found[f"home_{key}"], found[f"away_{key}"] = regions
    periods = [region_at(b) for t, b in lines if t.strip().upper() == "SCORE BY PERIODS"]
    if periods and periods[0]:
        found["periods"] = periods[0]
    found.update(roster_boxes(lines, found, w, h))
    return found


def roster_boxes(lines, tables: dict, w: int, h: int) -> dict[str, tuple[int, int, int, int]]:
    """
    Pixel boxes of the two rosters: from the HOME:/AWAY: label down to the
    bottom of that side's SCORING table, between the page edge and the table.
    The ruled lines of a roster are often too faint to find, and the margins
    hold handwritten numbers (subs, corrections), so the box includes them.
    """
    out = {}
    for text, (x, y, bw, bh) in lines:
        m = re.match(r"^\s*(HOME|AWAY)\s*:", text, re.I)
        side = m.group(1).lower() if m else None
        if not side or f"{side}_scoring" not in tables:
            continue
        sx, sy, sw, sh = tables[f"{side}_scoring"]
        top, bottom = int(y * h), sy + sh
        if side == "home":
            left, right = max(0, int((x - 0.05) * w)), sx
        else:
            left, right = sx + sw, min(w, int((x + bw + 0.25) * w))
        if right > left and bottom > top:
            bw_px, bh_px = right - left, bottom - top
            out[f"{side}_roster"] = (left, top, bw_px, bh_px)
            # The page-edge margin, read on its own: in the full roster crop the
            # model misses small margin numbers (a sub's '73' was seen only here).
            mw = int(bw_px * (0.2 if side == "home" else 0.3))
            out[f"{side}_margin"] = (left if side == "home" else right - mw, top, mw, bh_px)
    return out


# ---------------------------------------------------------------------------
# Reading the handwriting (local vision-language model)
# ---------------------------------------------------------------------------

PROMPTS = {
    "scoring": ("This is the SCORING table of a hockey scoresheet. Read every filled row. Columns: GOAL (row number), "
                "PER (period 1, 2, 3 or OT), TIME (clock time like 9:30), SCORED BY (jersey number), ASSIST (jersey numbers, "
                "may be two numbers separated by '-', or '-' when there is none), TYPE (ES, PP, SH or EN). "
                "Return only JSON: a list of objects with keys goal, per, time, scorer, assist, type. "
                "Write jersey numbers as digits, never names. Copy what is written; do not guess empty rows."),
    "penalties": ("This is the PENALTIES table of a hockey scoresheet. Read every filled row. Columns: PER (period), "
                  "# (jersey number), PENALTY (infraction, e.g. TRIP, HOOK, ROUGH), # MINUTES, OFF, START, ON (clock times). "
                  "Return only JSON: a list of objects with keys per, player, infraction, minutes, off, start, on. "
                  "Write jersey numbers as digits. Copy what is written; skip rows with no penalty."),
    "periods": ("This is the SCORE BY PERIODS table of a hockey scoresheet, with a HOME row and an AWAY row and columns "
                "1, 2, 3, OT, TOTAL. Return only JSON: {\"home\": [p1, p2, p3, ot, total], \"away\": [...]} "
                "with each cell as written, or \"\" when empty."),
    "roster": ("This is a team roster from a hockey scoresheet. Each row has a jersey number and a printed player name. "
               "Some numbers are handwritten, crossed out and rewritten, written after the name, or written in the margin. "
               "Return only JSON: a list of objects with keys number, name, other_numbers (a list of any other numbers "
               "written on that row or beside it). For a crossed-out number, give the replacement as number and the "
               "crossed-out one in other_numbers. Add numbers in the margin that belong to no row as objects with name \"\"."),
    "margin": ("This is the margin of a hockey scoresheet. List every handwritten number on it. "
               "Return only JSON: a list of strings, or [] if there are none."),
}

_VLM = None


def _vlm():
    global _VLM
    if _VLM is None:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        model, processor = load(MODEL)
        _VLM = (model, processor, load_config(MODEL))
    return _VLM


def read_table(crop_path: str, kind: str):
    """Ask the local model to read one table crop; returns the parsed JSON (list or dict)."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor, config = _vlm()
    prompt = apply_chat_template(processor, config, PROMPTS[kind], num_images=1)
    out = generate(model, processor, prompt, [crop_path], max_tokens=1500, temperature=0.0, verbose=False)
    text = out.text if hasattr(out, "text") else str(out)
    m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
    if not m:
        raise ValueError(f"No JSON in model output for {kind}: {text[:200]!r}")
    return json.loads(m.group(1))


# ---------------------------------------------------------------------------
# Normalizing and checking
# ---------------------------------------------------------------------------

def _clean(v) -> str:
    return re.sub(r"\s+", "", str(v if v is not None else "")).replace("—", "-").replace("–", "-")


def parse_time(v) -> int | None:
    """'9:30' / '9.30' / '0i50' -> seconds, or None."""
    s = _clean(v).replace(".", ":").replace(";", ":").lower().replace("i", ":")
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", s)
    if not m or int(m.group(2)) >= 60:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def _edit_distance(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def normalize_goal_type(v) -> tuple[str, bool]:
    """Nearest valid goal type ('BP' -> 'PP'); True when it had to be changed."""
    s = _clean(v).upper()
    s = GOAL_TYPE_ALIASES.get(s, s)
    if s in GOAL_TYPES or s == "":
        return s, False
    best = min(GOAL_TYPES, key=lambda t: _edit_distance(s, t))
    return best, True


def split_assists(v) -> list[str]:
    s = _clean(v)
    if s in ("", "-", "--", "NONE"):
        return []
    return [p for p in re.split(r"[-,/]", s) if p]


def normalize_minutes(v) -> str:
    """'2:00' / '2.00' -> '2'; other values unchanged."""
    s = _clean(v)
    t = parse_time(s)
    return str(t // 60) if t is not None and t % 60 == 0 else s


def period_cell(v) -> str:
    """A SCORE BY PERIODS cell: a handwritten zero often reads as 'D' or 'O'."""
    s = _clean(v)
    return "0" if s.upper() in ("D", "O", "Ø") else s


def _period_key(v) -> str:
    s = _clean(v).upper()
    return "OT" if s.startswith("O") else s


def _is_empty_row(row: dict, keys: tuple[str, ...]) -> bool:
    return all(_clean(row.get(k)) in ("", "-", "--") for k in keys)


def check_goals(goals: list[dict], roster: dict[str, str]) -> list[dict]:
    out = []
    for g in goals:
        if _is_empty_row(g, ("per", "time", "scorer", "type")):
            continue
        reasons = []
        per = _period_key(g.get("per"))
        if per not in ("1", "2", "3", "OT"):
            reasons.append(f"period '{g.get('per')}' is not 1, 2, 3 or OT")
        t = parse_time(g.get("time"))
        if t is None or t > MAX_PERIOD_S:
            reasons.append(f"time '{g.get('time')}' is not a valid clock time")
        typ, changed = normalize_goal_type(g.get("type"))
        if changed:
            reasons.append(f"type '{g.get('type')}' read as {typ}")
        scorer = _clean(g.get("scorer"))
        assists = split_assists(g.get("assist"))
        for num in [scorer] + assists:
            if num and roster and num not in roster:
                reasons.append(f"#{num} is not on the roster (sub, unlisted player, or misread)")
        out.append({"per": per, "time": _clean(g.get("time")), "time_s": t, "scorer": scorer, "assists": assists,
                    "type": typ, "status": "review" if reasons else "ok", "reasons": reasons})
    return out


def _ended_by_goal(per: str, start: int, on: int, minutes: int, opp_goals: list[dict]) -> bool:
    """A minor ends early when the other team scores on the power play: 'on' is that goal's time."""
    if minutes != 2 or not 0 < abs(start - on) < minutes * 60:
        return False
    return any(g["per"] == per and g["time_s"] is not None and abs(g["time_s"] - on) <= 1 for g in opp_goals)


def check_penalties(pens: list[dict], roster: dict[str, str], opp_goals: list[dict] | None = None) -> list[dict]:
    """opp_goals: the other team's checked goal rows, used to accept a minor ended by a power-play goal."""
    opp_goals = opp_goals or []
    out = []
    for p in pens:
        if _is_empty_row(p, ("player", "infraction", "minutes")):
            continue
        reasons = []
        per = _period_key(p.get("per"))
        if per not in ("1", "2", "3", "OT"):
            reasons.append(f"period '{p.get('per')}' is not 1, 2, 3 or OT")
        mins = normalize_minutes(p.get("minutes"))
        if mins not in ("2", "4", "5", "10"):
            reasons.append(f"minutes '{p.get('minutes')}' is not 2, 4, 5 or 10")
        num = _clean(p.get("player"))
        if num and roster and num not in roster:
            reasons.append(f"#{num} is not on the roster")
        start, on = parse_time(p.get("start")), parse_time(p.get("on"))
        if start is not None and on is not None and mins.isdigit() and abs(abs(start - on) - int(mins) * 60) > 1 \
                and not _ended_by_goal(per, start, on, int(mins), opp_goals):
            reasons.append(f"start {p.get('start')} and on {p.get('on')} are not {mins} min apart "
                           f"and no other-team goal at {p.get('on')} ended it")
        if parse_time(p.get("off")) is None:
            reasons.append(f"time '{p.get('off')}' is not a valid clock time")
        out.append({"per": per, "player": num, "infraction": str(p.get("infraction", "")).strip().upper(),
                    "minutes": mins, "off": _clean(p.get("off")), "start": _clean(p.get("start")), "on": _clean(p.get("on")),
                    "status": "review" if reasons else "ok", "reasons": reasons})
    return out


_COMPARE = ("per", "time", "scorer", "assists", "type", "player", "infraction", "minutes", "off", "start", "on")


def mark_disagreements(first: list[dict], second: list[dict]) -> list[dict]:
    """Flag every field where the two reads differ; a row count mismatch flags the extra rows."""
    for i, row in enumerate(first):
        other = second[i] if i < len(second) else None
        if other is None:
            row["reasons"].append("second read did not find this row")
        else:
            for k in _COMPARE:
                if k in row and row[k] != other.get(k):
                    row["reasons"].append(f"reads disagree on {k}: {row[k]!r} vs {other.get(k)!r}")
        row["status"] = "review" if row["reasons"] else "ok"
    for extra in second[len(first):]:
        first.append(dict(extra, status="review", reasons=extra["reasons"] + ["only the second read found this row"]))
    return first


def check_score_by_periods(periods: dict, home_goals: list[dict], away_goals: list[dict]) -> list[str]:
    """Compare goal rows per period with the SCORE BY PERIODS table; returns flags."""
    flags = []
    for side, goals in (("home", home_goals), ("away", away_goals)):
        cells = [period_cell(c) for c in (periods.get(side) or [])]
        for i, per in enumerate(("1", "2", "3")):
            counted = sum(1 for g in goals if g["per"] == per)
            written = cells[i] if i < len(cells) else ""
            if written == "":
                if counted:
                    flags.append(f"{side} period {per}: table empty, {counted} goal row(s)")
            elif not written.isdigit() or int(written) != counted:
                flags.append(f"{side} period {per}: table says {written}, goal rows count {counted}")
    return flags


def compare_periods(first: dict, second: dict) -> list[str]:
    """Flag SCORE BY PERIODS cells where the two reads differ."""
    flags = []
    cols = ("1", "2", "3", "OT", "TOTAL")
    for side in ("home", "away"):
        a = [period_cell(c) for c in (first.get(side) or [])]
        b = [period_cell(c) for c in (second.get(side) or [])]
        for i, col in enumerate(cols):
            va, vb = (a[i] if i < len(a) else ""), (b[i] if i < len(b) else "")
            if va != vb:
                flags.append(f"{side} period {col}: reads disagree {va!r} vs {vb!r}")
    return flags


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def read_scoresheet(photo: str, work_dir: str) -> dict:
    import cv2
    image = cv2.imread(photo)
    if image is None:
        raise ValueError(f"Cannot read image {photo}")
    lines = ocr_lines(photo)
    teams = parse_team_names(lines)
    rosters = parse_rosters(lines)
    tables = locate_tables(image, lines)
    missing = [k for k in ("home_scoring", "away_scoring", "home_penalties", "away_penalties", "periods",
                           "home_roster", "away_roster") if k not in tables]
    raw, second, flags = {}, {}, []

    def read(path, kind, key):
        try:
            return read_table(path, kind)
        except ValueError as e:     # includes json.JSONDecodeError
            flags.append(f"{key}: model output not readable ({e})")
            return {} if kind == "periods" else []

    for key, (x, y, w, h) in tables.items():
        kind = key.split("_")[-1] if key != "periods" else "periods"
        # Two reads from different crops: the exact table, and a padded, enlarged copy.
        # Cells where they disagree go to review with both values (a silent misread
        # "1:10" -> "1:16" was seen with a single read).
        a = str(Path(work_dir) / f"scoresheet_{key}.png")
        crop = image[y:y + h, x:x + w]
        if kind == "margin":        # small handwriting: enlarge
            crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(a, crop)
        px, py = int(w * 0.04), int(h * 0.02)
        b_img = image[max(0, y - py):y + h + py, max(0, x - px):x + w + px]
        b = str(Path(work_dir) / f"scoresheet_{key}_b.png")
        cv2.imwrite(b, cv2.resize(b_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC))
        raw[key], second[key] = read(a, kind, key), read(b, kind, key)
    # The roster check must use the merged roster (printed + handwritten numbers).
    for side in ("home", "away"):
        rows = []
        for reads in (raw, second):
            roster_rows, margin = reads.get(f"{side}_roster"), reads.get(f"{side}_margin")
            rows += roster_rows if isinstance(roster_rows, list) else []
            rows += [{"number": n, "name": ""} for n in (margin if isinstance(margin, list) else [])]
        rosters[side] = merge_roster(rosters[side], rows)
    goals = {side: mark_disagreements(check_goals(raw.get(f"{side}_scoring") or [], rosters[side]),
                                      check_goals(second.get(f"{side}_scoring") or [], rosters[side]))
             for side in ("home", "away")}
    other = {"home": "away", "away": "home"}
    penalties = {side: mark_disagreements(
        check_penalties(raw.get(f"{side}_penalties") or [], rosters[side], goals[other[side]]),
        check_penalties(second.get(f"{side}_penalties") or [], rosters[side], goals[other[side]]))
        for side in ("home", "away")}
    periods = raw.get("periods") or {}
    second_periods = second.get("periods") or {}
    flags = ([f"table not found: {k}" for k in missing] + flags
             + check_score_by_periods(periods, goals["home"], goals["away"])
             + compare_periods(periods if isinstance(periods, dict) else {},
                               second_periods if isinstance(second_periods, dict) else {}))
    return {
        "photo": photo,
        "teams": teams,
        "rosters": rosters,
        "goals": goals,
        "penalties": penalties,
        "score_by_periods": {side: [period_cell(c) for c in (periods.get(side) or [])] for side in ("home", "away")}
        if isinstance(periods, dict) else {},
        "flags": flags,
    }


def find_scoresheet_pdfs(game_folder: str) -> list[str]:
    """PDF files in the Game Folder (a GameSheet export is exact, so it is tried before a photo)."""
    return sorted(str(p) for p in Path(game_folder).iterdir()
                  if p.is_file() and p.suffix.lower() == ".pdf" and not is_non_sheet_file(p.name))


def main(game_folder: str) -> int:
    sheet = None
    for pdf in find_scoresheet_pdfs(game_folder):
        sys.path.insert(0, str(Path(__file__).parent))
        from gamesheet_pdf import read_gamesheet_pdf
        try:
            sheet = read_gamesheet_pdf(pdf)
            break
        except ValueError as e:
            print(f"[scoresheet] {Path(pdf).name}: {e}", flush=True)
    if sheet is None:
        photos = find_scoresheet_photos(game_folder)
        if not photos:
            print("[scoresheet] No Scoresheet photo or GameSheet PDF in the Game Folder; "
                  "the Game Sheet must be entered in review", flush=True)
            return 2
        crops = Path(game_folder) / "scoresheet_crops"
        crops.mkdir(exist_ok=True)
        try:
            sheet = read_scoresheet(photos[0], str(crops))
        except ImportError as e:
            print(f"[scoresheet] Scoresheet reader needs the optional ML stack "
                  f"(requirements-ml.txt): {e}", flush=True)
            return 4
    (Path(game_folder) / "game_sheet.json").write_text(json.dumps(sheet, indent=2))
    rows = sheet["goals"]["home"] + sheet["goals"]["away"] + sheet["penalties"]["home"] + sheet["penalties"]["away"]
    review = sum(r["status"] == "review" for r in rows)
    print(f"[scoresheet] game_sheet.json written: {len(sheet['goals']['home'])}+{len(sheet['goals']['away'])} goals, "
          f"{len(sheet['penalties']['home'])}+{len(sheet['penalties']['away'])} penalties, "
          f"{review} row(s) and {len(sheet['flags'])} sheet flag(s) for review", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: scoresheet.py <game_folder>")
    sys.exit(main(sys.argv[1]))
