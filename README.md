# Hockey Highlight Generator

Analyzes GoPro MP4 chapters from amateur hockey games (two cameras, one per net),
detects high-action moments using computer vision, and assembles a pre-cut multicam
candidate reel in DaVinci Resolve.

## Quick start

```bash
# Folder mode (V3) — pass a game folder containing cam1/ and cam2/ subfolders
hockeydetect /path/to/game-folder

# File mode (V2) — pass two MP4s directly
hockeydetect cam1.mp4 cam2.mp4
```

The `hockeydetect` alias points to `run_detect.sh` at the project root.
That file is the live entry point — do not move it.

---

## V3 workflow (folder mode)

**Game folder layout:**

```
game_folder/
  cam1/              ← raw GoPro chapters for camera 1 (behind one net)
  cam2/              ← raw GoPro chapters for camera 2 (behind other net)
```

**What `hockeydetect game_folder` does:**

1. **Chapter discovery** (`v3/scripts/discover.py`) — scans `cam1/` and `cam2/`, sorts chapters alphabetically, writes `chapters.json`.
2. **Timecode sync** (`v3/scripts/gopro_meta.py`) — extracts GoPro timecodes via ffprobe, computes camera alignment offset, writes `sync_info.json` and ffmpeg concat manifests (`cam1_concat.txt`, `cam2_concat.txt`).
3. **Detection** (`v2/scripts/detect_events.py`) — optical flow analysis on the concat manifests, writes `events.csv` and `markers.csv`.

**Outputs per game folder:**

| File | Description |
|------|-------------|
| `chapters.json` | Sorted chapter paths for each camera |
| `sync_info.json` | Camera alignment offset and sync method |
| `cam1_concat.txt` / `cam2_concat.txt` | ffmpeg concat manifests (with inpoint for earlier camera) |
| `events.csv` | All candidate windows sorted by score |
| `markers.csv` | Resolve marker import CSV (frozen schema) |

**Resolve reel assembly** (`v3/resolve_scripts/compile_reel.py`):

After detection, run this script from inside DaVinci Resolve:
**Workspace → Scripts → compile_reel**

It will:
1. Prompt for the game folder.
2. Import all GoPro chapters into a named Media Pool bin.
3. Create a multicam clip from the two camera sets.
4. Place every detected event on Track 1 (pre-cut with 3 s preroll / 2 s postroll), with the primary camera angle pre-selected.

---

## V2 file mode

Pass two MP4 files directly (single-chapter games or pre-exported clips):

```bash
hockeydetect cam1.mp4 cam2.mp4
```

### Detection tuning

```bash
# More sensitive (more events, more false positives)
python v2/scripts/detect_events.py cam1.mp4 cam2.mp4 rois.json \
  --thresh_pct 88 --cooldown_s 8 --verbose

# Debug plot — saves events_debug.png next to events.csv
python v2/scripts/detect_events.py cam1.mp4 cam2.mp4 rois.json \
  --debug_plot --verbose

# Disable audio weighting (default for GoPro rink audio)
python v2/scripts/detect_events.py cam1.mp4 cam2.mp4 rois.json \
  --audio_weight 0 --verbose
```

### Signal fusion

```
flow_component  = 0.7 * net_flow + 0.3 * slot_flow   (normalized to P99)
fused           = --flow_weight * flow + --audio_weight * audio
```

Default weights: `--flow_weight 0.8 --audio_weight 0` (audio disabled — GoPro rink audio has no useful crowd noise).

---

## Repository structure

```
project-root/
├── run_detect.sh              # Live entry point (alias target)
├── README.md
│
├── v3/                        # V3 pipeline (Phase 1 complete)
│   ├── scripts/
│   │   ├── discover.py        # Chapter discovery — scans cam1/ cam2/, writes chapters.json
│   │   └── gopro_meta.py      # Timecode extraction, sync computation, concat manifests
│   └── resolve_scripts/
│       └── compile_reel.py    # Resolve: import chapters → multicam → place events on Track 1
│
├── v2/                        # Detection engine (active — do not break)
│   ├── scripts/
│   │   ├── signals.py         # Signal extraction: optical flow + audio RMS; accepts concat manifests
│   │   ├── detect_events.py   # Peak finding, window scoring, output writing
│   │   ├── roi_picker.py      # Interactive OpenCV ROI selection
│   │   ├── edl.py             # markers.csv → EDL
│   │   └── timeline_convert.py# markers.csv → FCPXML
│   └── resolve_scripts/
│       ├── expand_markers.py  # Resolve: expand point markers → ranged + set angles
│       └── import_markers_resolve.py  # Resolve: import markers.fcpxml onto timeline
│
├── tests/                     # pytest suite (requires .venv active)
│   ├── conftest.py
│   ├── test_discover.py
│   ├── test_gopro_meta.py
│   ├── test_signals_concat.py
│   ├── test_run_detect_folder_mode.py
│   └── test_compile_reel_logic.py
│
└── v1/                        # Archive — read-only
```

---

## Dependencies

- Python 3.10+
- OpenCV (`cv2`) — optical flow
- NumPy, SciPy — signal processing
- ffmpeg / ffprobe — frame extraction and timecode probing (must be on PATH)
- matplotlib — optional, only with `--debug_plot`

```bash
pip install opencv-python numpy scipy
pip install matplotlib  # optional
```

---

## Testing

```bash
source .venv/bin/activate
pytest
```
