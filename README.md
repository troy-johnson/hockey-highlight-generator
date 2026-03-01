# Hockey Highlight Generator

Analyzes two synced GoPro MP4s, detects high-action moments using computer
vision + audio, and outputs timeline markers for DaVinci Resolve.

## Quick start

```bash
# Run from the game footage folder (auto-detects *cam1*.mp4 / *cam2*.mp4)
hockeydetect /path/to/game-folder

# Or pass files directly
hockeydetect cam1.mp4 cam2.mp4
```

The `hockeydetect` alias points to `run_detect.sh` at the project root.
That file is the live entry point — do not move it.

---

## Repository structure

```
project-root/
├── run_detect.sh              # Live entry point (alias target). Points to v2/scripts/.
├── README.md
│
├── v2/                        # Active detection engine
│   ├── scripts/
│   │   ├── signals.py         # Signal extraction: optical flow (Farneback) + audio RMS
│   │   ├── detect_events.py   # Peak finding, window scoring, output writing
│   │   ├── roi_picker.py      # Interactive OpenCV ROI selection (unchanged from v1)
│   │   ├── edl.py             # markers.csv → EDL (unchanged from v1)
│   │   └── timeline_convert.py# markers.csv → FCPXML (completed from v1)
│   └── resolve_scripts/
│       ├── expand_markers.py  # Resolve: expand point markers → ranged + set angles
│       └── import_markers_resolve.py  # Resolve: import markers.fcpxml onto timeline
│
└── v1/                        # Archive of the original V1 scripts
    ├── scripts/
    │   ├── detect_events.py   # V1: frame-difference motion energy
    │   ├── roi_picker.py
    │   ├── edl.py
    │   └── timeline_convert.py
    ├── resolve_scripts/
    │   ├── expand_markers.py
    │   └── import_markers_resolve.py
    └── run_detect.sh          # V1 version for reference
```

---

## Outputs (per game folder)

| File | Description |
|------|-------------|
| `events.csv` | Debug: all candidate windows sorted by score |
| `markers.csv` | Resolve marker import CSV |
| `markers.fcpxml` | Final Cut Pro XML (also importable by Resolve) |
| `markers.edl` | CMX 3600 EDL |

---

## V2 detection engine

### What changed from V1

| V1 | V2 |
|----|----|
| Frame-differencing in ROIs | Farneback optical flow in ROIs |
| Fixed global percentile threshold | Rolling percentile per 5-min segment + global P70 floor |
| No audio | Audio RMS fused into signal (tunable weight) |
| Hardcoded scoring weights | `--w_attack` / `--w_rebound` / `--w_drop` CLI args |
| No debug tooling | `--debug_plot` saves a PNG of the fused signal + peaks |

### How signals are fused

```
flow_component  = 0.7 * net_flow + 0.3 * slot_flow   (normalized to P99)
audio_component = audio_rms                            (normalized to P99)
fused           = --flow_weight * flow + --audio_weight * audio
```

Default weights: `--flow_weight 0.8 --audio_weight 0.2`.
Set `--audio_weight 0` to disable audio entirely (e.g. if the video has no useful crowd noise).

### Rolling threshold

The detector computes the `--thresh_pct` percentile within each non-overlapping
5-minute segment of the signal, then takes the maximum of that and the global
70th-percentile floor. This prevents the threshold from collapsing to near-zero
during quiet stretches while still adapting across game periods.

### Tuning guide

```bash
# Default run (good starting point)
hockeydetect /path/to/game

# More sensitive (catches more events, more false positives)
python v2/scripts/detect_events.py cam1.mp4 cam2.mp4 rois.json \
  --thresh_pct 88 --cooldown_s 8 --verbose

# Debug plot to inspect the signal
python v2/scripts/detect_events.py cam1.mp4 cam2.mp4 rois.json \
  --debug_plot --out_csv /tmp/events.csv --verbose

# Disable audio (e.g. GoPro wind noise)
python v2/scripts/detect_events.py cam1.mp4 cam2.mp4 rois.json \
  --audio_weight 0 --verbose
```

*(The debug PNG is saved next to `events.csv` as `events_debug.png`.)*

---

## Dependencies

- Python 3.10+
- OpenCV (`cv2`) — optical flow
- NumPy, SciPy — signal processing
- ffmpeg / ffprobe — frame extraction and audio decoding (must be on PATH)
- matplotlib — only required with `--debug_plot`

```bash
pip install opencv-python numpy scipy
# matplotlib is optional:
pip install matplotlib
```

---

## DaVinci Resolve integration

1. Run `hockeydetect` to generate `markers.fcpxml` in the game folder.
2. In Resolve: **Workspace → Scripts → import_markers_resolve** — pick the game folder.
3. Markers are copied onto the active timeline.
4. Optionally run **expand_markers** to convert point markers into ranged clips and set multicam angles.
