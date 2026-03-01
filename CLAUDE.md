# CLAUDE.md — Hockey Highlight Generator

## Project purpose

Analyzes two synced GoPro MP4s (cam1, cam2) from amateur hockey games and outputs
timeline markers for DaVinci Resolve. Detection is optical flow based (Farneback);
audio is explicitly disabled (no crowd noise in amateur games — blowouts especially).

## Shell alias

`hockeydetect` → `run_detect.sh` at the project root. **Do not move run_detect.sh.**
The alias is defined on the user's machine and points to this file by absolute path.

## Active code: v2/

All active detection logic lives in `v2/scripts/`. The `v1/` directory is a read-only
archive — do not modify it. Scripts that were unchanged from V1 (roi_picker, edl,
resolve scripts) are copied into v2/ so v2/ is self-contained.

## Key design decisions

- **Audio disabled by default** (`--audio_weight 0` in run_detect.sh): GoPro rink
  audio has no crowd noise. Even goal celebrations are inconsistent (quiet during
  blowouts). Do not re-enable audio weighting without a good reason.
- **No new major dependencies**: Core stack is OpenCV, NumPy, SciPy, ffmpeg/ffprobe.
  matplotlib is allowed but must be lazy-imported (only under `--debug_plot`).
  PyTorch, librosa, ollama, faiss are explicitly excluded until V3.
- **signals.py is a pure extraction module**: It returns `(net_flow, slot_flow,
  audio_rms)` as separate float32 arrays. Fusion weights belong in detect_events.py,
  not here.
- **sys.path injection**: `v2/scripts/detect_events.py` imports signals via
  `sys.path.insert(0, os.path.dirname(__file__))` — the scripts are not a package.
- **Output schema is frozen**: `markers.csv` columns (Frame, Name, Note, Color,
  Duration) must not change — downstream Resolve scripts depend on them.

## V3 plan (not yet built)

See `C:\Users\troyj\.claude\plans\nifty-splashing-puppy.md` for full context.
Short version:
1. **Option B** (recommended first): small manual label set (5-10 games, ~15 min/game)
   → `sklearn.LogisticRegression` re-ranker on V2 feature vectors.
   Files: `v3/scripts/label_events.py`, `train_reranker.py`, `rerank.py`.
2. **Option C** (later): VLM keyframe scoring (moondream2 via ollama). The 100
   existing highlight clips provide few-shot keyframes for prompting.
3. **Option D** (future, complex): PU learning via cut detection + per-segment audio
   alignment of highlight clips to original game footage (~50+ segments per clip).

## Game footage facts

- Two GoPro cameras: `*cam1*.mp4` and `*cam2*.mp4` per game folder
- ~45 min of raw footage → ~4-5 min of highlights (~10% kept)
- ~100 completed highlight videos exist as reference (positive examples only)
- Highlight clips: hard cuts every 4-5 seconds, no transitions, no text overlays
- No EDL/marker files saved from prior editing sessions

## ROI convention

`rois.json` is per-game-folder (camera angle varies). Contains:
```json
{
  "camera_1": {"net": [x,y,w,h], "slot": [x,y,w,h]},
  "camera_2": {"net": [x,y,w,h], "slot": [x,y,w,h]}
}
```
Generated interactively by `roi_picker.py` on first run. Do not hardcode ROI values.
