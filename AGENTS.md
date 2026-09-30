# Hockey Highlight Generator — Project Conventions

## Project purpose

Analyzes GoPro MP4 chapters (two cameras, one per net) from amateur hockey games.
Detects high-action moments using optical flow (Farneback), assembles a pre-cut
multicam candidate reel in DaVinci Resolve, and outputs timeline markers.
Audio is explicitly disabled — no crowd noise in amateur games.

## Shell alias

`hockeydetect` → `run_detect.sh` at the project root. **Do not move run_detect.sh.**
The alias is defined on the user's machine and points to this file by absolute path.

`hockeyrecap` → `run_recap.sh` at the project root (runs `v3/scripts/hockeyrecap.py`
with `.venv/bin/python`). **Do not move run_recap.sh.** Same alias convention.

## Active code

- **`v2/scripts/`** — detection engine (optical flow + rolling threshold). Active, do not break.
- **`v3/scripts/`** — V3 chapter discovery (`discover.py`) and timecode sync (`gopro_meta.py`). Complete.
- **`v3/scripts/hockeyrecap.py`** — `hockeyrecap run|status|rerun --from|check` CLI.
  Stage runner in `recap_runner.py`, option layers (League, Team Config in
  `~/hockey/`, per-game file, CLI flags) in `recap_options.py`, Recording check
  in `recap_check.py`. Stages call the existing scripts as subprocesses.
- **`v3/resolve_scripts/`** — V3 Resolve reel assembly (`compile_reel.py`). Complete.
- **`v1/`** — read-only archive. Do not modify.

V3 Phase 1 is complete and merged. See `docs/specs/` and `docs/plans/` for specs and plans.

## Key design decisions

- **Audio disabled by default** (`--audio_weight 0` in run_detect.sh): GoPro rink
  audio has no crowd noise. Even goal celebrations are inconsistent (quiet during
  blowouts). Do not re-enable audio weighting without a good reason.
- **No new major dependencies in V2**: Core stack is OpenCV, NumPy, SciPy, ffmpeg/ffprobe.
  matplotlib is allowed but must be lazy-imported (only under `--debug_plot`).
  PyTorch is accepted in V3 (for HockeyAI/YOLOv8 in Phase 2).
- **signals.py is a pure extraction module**: It returns `(net_flow, slot_flow,
  audio_rms)` as separate float32 arrays. Fusion weights belong in detect_events.py,
  not here.
- **sys.path injection**: Scripts use `sys.path.insert(0, os.path.dirname(__file__))`
  for sibling imports — the scripts are not a package.
- **Output schema is frozen**: `markers.csv` columns (Frame, Name, Note, Color,
  Duration) must not change — downstream Resolve scripts depend on them.

## V3 pipeline

Phase 1 (complete): Chapter-aware detection + Resolve reel assembly.
- Spec: `docs/specs/001-v3-highlight-pipeline-2026-04-25.md`
- Plan: `docs/plans/2026-04-25-v3-highlight-pipeline.md`
- New scripts: `v3/scripts/discover.py`, `v3/scripts/gopro_meta.py`,
  `v3/resolve_scripts/compile_reel.py`
- Modified: `v2/scripts/signals.py` (concat manifest support), `run_detect.sh`

Phase 2 (planned): HockeyAI (YOLOv8) object detection + ML re-ranker for
goal/penalty/non-scoring classification. See spec 001 Phase 2 section.

## Game folder structure (V3)

A Game Folder is either flat (the user's normal layout: both cameras' card
contents plus the Scoresheet photo, named `opponent_MMDDYYYY`) or has `cam1/`
and `cam2/` subfolders. `discover.py` sorts files into cameras by the serial in
each file's `CASN` atom, groups chapters into Recordings by GoPro file number,
and skips black Recordings (lens covered).

```
game_folder/
  GX01xxxx.MP4 ...   ← raw GoPro chapters from both cameras (or cam1/ cam2/)
  chapters.json      ← cameras, Recordings, serials, excluded, missing, flags (discover.py)
                       cam3+ only when more than 2 cameras (hockeyrecap mode)
  sync_info.json     ← offset from timecode, checked against rink audio (gopro_meta.py)
  cam1_concat.txt    ← ffconcat manifest; offsets as '# seek' / '# recording' comments
  cam2_concat.txt
  rois.json          ← automatic (auto_roi.py) or from roi_picker.py
  rois_auto.json     ← goal box + confidence per camera; rois_preview.png
  events.csv / markers.csv
  recap_options.json ← per-game options (hockeyrecap); inferred values marked
  recap_status.json  ← stage states, fingerprints, flags (hockeyrecap status)
  recap.log          ← full log of all stage output
  recording_check.json ← hockeyrecap check report
  recap_previous/    ← copies of outputs from older tools, before first overwrite
  .recap_cache/signals/ ← flow signals per Recording (.npz); old keys are not pruned
```

`discover.py` ignores old output videos (`cam1.mp4`, `*recap*.mp4`,
`*_overlay.mp4`). The Scoresheet reader ignores non-sheet images and outputs
(`scoresheet.is_non_sheet_file`).

Never use ffmpeg concat `inpoint` for sync offsets: on GoPro HEVC it applies
only about 1/3 of the offset. Seek with `-ss` before the concat input.

## ROI convention

`rois.json` is per Game Folder (camera angle varies). Contains:
```json
{
  "camera_1": {"net": [x,y,w,h], "slot": [x,y,w,h]},
  "camera_2": {"net": [x,y,w,h], "slot": [x,y,w,h]}
}
```
Coordinates are in 1280×720 analysis frames. `auto_roi.py` derives them from
the HockeyAI goal box on a median background (optional stack:
`requirements-ml.txt`); `roi_picker.py` is the fallback. Do not hardcode ROI values.

## Testing

Tests live in `tests/`. Run with `pytest` from the project root (requires `.venv` active).
No test infrastructure exists yet in V1/V2 — V3 adds it.

## Worktrees

Feature branches use `.worktrees/` (project-local, git-ignored).

<!-- BEGIN BEADS CODEX SETUP: generated by bd setup codex -->
## Beads Issue Tracker

Use Beads (`bd`) for durable task tracking in repositories that include it. Use the `beads` skill at `.agents/skills/beads/SKILL.md` (project install) or `~/.agents/skills/beads/SKILL.md` (global install) for Beads workflow guidance, then use the `bd` CLI for issue operations.

### Quick Reference

```bash
bd ready                # Find available work
bd show <id>            # View issue details
bd update <id> --claim  # Claim work
bd close <id>           # Complete work
bd prime                # Refresh Beads context
```

### Rules

- Use `bd` for all task tracking; do not create markdown TODO lists.
- Run `bd prime` when Beads context is missing or stale. Codex 0.129.0+ can load Beads context automatically through native hooks; use `/hooks` to inspect or toggle them.
- Keep persistent project memory in Beads via `bd remember`; do not create ad hoc memory files.

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.
<!-- END BEADS CODEX SETUP -->
