# v3/scripts/gamesheet_pdf.py
"""
Read a GameSheet PDF (the league tracking site's scoresheet export) into a Game Sheet.

The PDF has a text layer, so no OCR and no model are needed: the values are
exact. The text order in the PDF is mixed, so tables are read by position:
each table's header line gives the column x positions, and each row's cells
are read from those columns (PDFKit, via PyObjC).

Column meaning in the GameSheet export (checked on one sheet):
- Clock times are time remaining in the period (they count down).
- Penalty OFF, START and ON are usually filled as on paper (OFF = START =
  time of the penalty, ON = time it ended). One export left ON empty and put
  the end time in OFF; then START is the time of the penalty and OFF is the end.
- There is no goal TYPE column and no SCORE BY PERIODS table; the FINAL SCORE
  is checked against the goal rows instead.

Output has the same keys as scoresheet.read_scoresheet, plus source, date
and final_score.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from scoresheet import check_goals, check_penalties, parse_time  # noqa: E402

PENALTY_HEADER = "PER NO. MIN. Code Infraction OFF START ON"
SCORING_HEADER = "PER TIME G A A"
_PERIOD_ROW = re.compile(r"^(1|2|3|OT|SO)\s")
_PENALTY_TEXT = re.compile(r"^(\S+)\s+(\S+)\s+(\S+)\s+(.*?)\s*(?:\d{1,2}:\d{2}\s*)*$")
_ROSTER_ROW = re.compile(r"^(\d{1,3})\s+(G\s+)?([A-Z][A-Z'.\- ]*[A-Z])$")


# ---------------------------------------------------------------------------
# Reading positions from the PDF (PDFKit)
# ---------------------------------------------------------------------------

def _open_page(pdf: str):
    from Foundation import NSURL
    from Quartz import PDFDocument
    doc = PDFDocument.alloc().initWithURL_(NSURL.fileURLWithPath_(pdf))
    if doc is None or doc.pageCount() == 0:
        raise ValueError(f"Cannot read PDF {pdf}")
    return doc, doc.pageAtIndex_(0)


def _page_lines(page) -> list[tuple[float, float, float, str]]:
    """Text lines as (x, y, width, text), y from the top of the page."""
    box = page.boundsForBox_(0)
    height = box.size.height
    out = []
    for sel in page.selectionForRect_(box).selectionsByLine():
        r = sel.boundsForPage_(page)
        text = (sel.string() or "").strip()
        if text:
            out.append((r.origin.x, height - r.origin.y - r.size.height, r.size.width, text))
    return sorted(out, key=lambda t: (round(t[1]), t[0]))


def _cell(page, x0: float, x1: float, y: float) -> str:
    from Foundation import NSMakeRect
    height = page.boundsForBox_(0).size.height
    sel = page.selectionForRect_(NSMakeRect(x0, height - y - 8, x1 - x0, 6))
    return re.sub(r"\s+", " ", (sel.string() or "") if sel else "").strip()


def _column_starts(doc, page, header: tuple[float, float, float, str]) -> list[float]:
    """x of each header word, found by searching the header's words at the header's y."""
    hx, hy, hw, text = header
    height = page.boundsForBox_(0).size.height
    starts = []
    for word in dict.fromkeys(text.split()):
        for sel in doc.findString_withOptions_(word, 0) or []:
            r = sel.boundsForPage_(page)
            y = height - r.origin.y - r.size.height
            if abs(y - hy) < 3 and hx - 2 <= r.origin.x <= hx + hw:
                starts.append(r.origin.x)
    starts = sorted(set(round(s, 1) for s in starts))
    if len(starts) != len(text.split()):
        raise ValueError(f"Header columns not found for {text!r}")
    return starts


def _table_rows(doc, page, lines, header) -> list[dict]:
    """Cells of each row under a table header, keyed by column index."""
    hx, hy, hw, text = header
    starts = _column_starts(doc, page, header)
    ends = [s - 3 for s in starts[1:]] + [hx + hw + 15]
    rows = []
    for x, y, w, t in lines:
        if y <= hy + 5 or x + w < hx or x > hx + hw or not _PERIOD_ROW.match(t):
            continue
        cells = [_cell(page, s - 3, e, y) for s, e in zip(starts, ends)]
        if text == PENALTY_HEADER:
            cells = penalty_cells(t, cells)
        rows.append(cells)
    return rows


def penalty_cells(line: str, cells: list[str]) -> list[str]:
    """The code is not aligned with its header column, so PER, NO., MIN. and the code come from
    the row text; only the OFF, START and ON times are taken by position."""
    m = _PENALTY_TEXT.match(line)
    if not m:
        return cells
    return [m.group(1), m.group(2), m.group(3), m.group(4), ""] + cells[5:8]


def read_gamesheet_pdf(pdf: str) -> dict:
    doc, page = _open_page(pdf)
    lines = _page_lines(page)
    pen_headers = sorted((l for l in lines if l[3] == PENALTY_HEADER), key=lambda l: l[0])
    goal_headers = sorted((l for l in lines if l[3] == SCORING_HEADER), key=lambda l: l[0])
    if len(pen_headers) != 2 or len(goal_headers) != 2:
        raise ValueError(f"{pdf} is not a GameSheet scoresheet (table headers not found)")

    def label_below(label: str) -> str:
        for x, y, w, t in lines:
            if t == label:
                below = [l for l in lines if abs(l[0] - x) < 10 and 5 < l[1] - y < 20]
                return below[0][3] if below else ""
        return ""

    def roster(no_header) -> dict[str, dict]:
        nx, ny = no_header[0], no_header[1]
        end = min((l[1] for l in lines if l[3] == "Head Coach" and abs(l[0] - nx) < 10), default=1e9)
        out = {}
        for x, y, w, t in lines:
            m = _ROSTER_ROW.match(t)
            if m and ny < y < end and abs(x - nx) < 15:
                out[m.group(1)] = {"name": m.group(3).title(), "goalie": bool(m.group(2))}
        return out

    no_headers = sorted((l for l in lines if l[3] == "NO."), key=lambda l: l[0])[:2]
    final = next((re.search(r"HOME (\d+) VISITOR (\d+)", l[3]) for l in lines
                  if re.search(r"HOME \d+ VISITOR \d+", l[3])), None)
    date = next((l[3][5:].strip() for l in lines if l[3].startswith("Date ")), "")
    tables = {
        "scoring": {side: _table_rows(doc, page, lines, h) for side, h in zip(("home", "away"), goal_headers)},
        "penalties": {side: _table_rows(doc, page, lines, h) for side, h in zip(("home", "away"), pen_headers)},
    }
    rosters = {side: roster(h) for side, h in zip(("home", "away"), no_headers)}
    meta = {"photo": pdf, "date": date, "teams": {"home": label_below("HOME"), "away": label_below("VISITOR")},
            "final_score": {"home": int(final.group(1)), "away": int(final.group(2))} if final else None}
    return sheet_from_tables(meta, rosters, tables)


# ---------------------------------------------------------------------------
# Turning table cells into the Game Sheet (pure; tested)
# ---------------------------------------------------------------------------

def goal_row(cells: list[str]) -> dict:
    """[PER, TIME, G, A, A] -> the scoring row keys scoresheet.check_goals reads."""
    per, time, scorer, *assists = (cells + [""] * 5)[:5]
    return {"per": per, "time": time, "scorer": scorer, "assist": "-".join(a for a in assists if a), "type": ""}


def penalty_row(cells: list[str]) -> dict:
    """[PER, NO., MIN., Code, Infraction, OFF, START, ON] -> the penalty row keys check_penalties reads."""
    per, player, minutes, code, infraction, off, start, on = (cells + [""] * 8)[:8]
    if not on:                       # some exports leave ON empty and put the end time in OFF
        off, on = start, off
    return {"per": per, "player": player, "minutes": minutes, "infraction": infraction or code,
            "off": off or start, "start": start or off, "on": on}


def check_final_score(final: dict | None, goals: dict) -> list[str]:
    if not final:
        return ["final score not found"]
    return [f"{side}: final score {final[side]}, goal rows count {len(goals[side])}"
            for side in ("home", "away") if final[side] != len(goals[side])]


def _in_box(pen: dict, per: str, t: int) -> bool:
    """The player is in the box at clock time t (the clock counts down; a 10-min misconduct is not short-handed)."""
    start, on = parse_time(pen["start"]), parse_time(pen["on"])
    return (pen["per"] == per and pen["minutes"] != "10" and start is not None and on is not None
            and start > t >= on)


def infer_goal_types(goals: dict, penalties: dict) -> None:
    """GameSheet has no type column: PP / SH / ES from the players in the box at the goal time."""
    other = {"home": "away", "away": "home"}
    for side in ("home", "away"):
        for g in goals[side]:
            if g["type"] or g["time_s"] is None:
                continue
            own = sum(_in_box(p, g["per"], g["time_s"]) for p in penalties[side])
            opp = sum(_in_box(p, g["per"], g["time_s"]) for p in penalties[other[side]])
            g["type"] = "PP" if opp > own else "SH" if own > opp else "ES"


def sheet_from_tables(meta: dict, rosters: dict, tables: dict) -> dict:
    names = {side: {n: r["name"] for n, r in rosters[side].items()} for side in ("home", "away")}
    goals = {side: check_goals([goal_row(c) for c in tables["scoring"][side]], names[side])
             for side in ("home", "away")}
    other = {"home": "away", "away": "home"}
    penalties = {side: check_penalties([penalty_row(c) for c in tables["penalties"][side]], names[side],
                                       goals[other[side]])
                 for side in ("home", "away")}
    infer_goal_types(goals, penalties)
    return {
        "source": "gamesheet_pdf",
        "photo": meta["photo"],
        "date": meta.get("date", ""),
        "teams": meta["teams"],
        "rosters": names,
        "goals": goals,
        "penalties": penalties,
        "score_by_periods": {},
        "final_score": meta.get("final_score"),
        "flags": check_final_score(meta.get("final_score"), goals),
    }
