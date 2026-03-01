#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Activate venv ----
if [ -f "$REPO_DIR/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$REPO_DIR/.venv/bin/activate"
else
  echo "[WARN] No .venv found at $REPO_DIR/.venv (continuing with system python)"
fi

# --- helpers ---
strip_quotes() {
  local p="$1"
  p="${p%/}"
  p="${p%\"}"
  p="${p#\"}"
  echo "$p"
}

usage() {
  echo "Usage:"
  echo "  run_detect.sh                # uses current folder as project folder"
  echo "  run_detect.sh <project_dir>  # folder mode"
  echo "  run_detect.sh <cam1.mp4> <cam2.mp4>  # file mode"
}

# ---- Parse args ----
MODE="folder"
PROJECT_DIR=""
CAM1=""
CAM2=""

if [ "$#" -eq 0 ]; then
  PROJECT_DIR="$(pwd)"
elif [ "$#" -eq 1 ]; then
  PROJECT_DIR="$(strip_quotes "$1")"
elif [ "$#" -ge 2 ]; then
  # If first arg is a directory, treat as folder mode and ignore rest
  if [ -d "$1" ]; then
    PROJECT_DIR="$(strip_quotes "$1")"
  else
    MODE="files"
    CAM1="$(strip_quotes "$1")"
    CAM2="$(strip_quotes "$2")"
  fi
else
  usage
  exit 1
fi

# ---- Folder mode ----
if [ "$MODE" = "folder" ]; then
  if [ ! -d "$PROJECT_DIR" ]; then
    echo "[ERROR] Folder does not exist: $PROJECT_DIR"
    exit 1
  fi

  # Auto-detect cameras
  CAM1=$(ls "$PROJECT_DIR"/*cam1*.mp4 2>/dev/null | head -n 1 || true)
  CAM2=$(ls "$PROJECT_DIR"/*cam2*.mp4 2>/dev/null | head -n 1 || true)

  if [ -z "$CAM1" ] || [ -z "$CAM2" ]; then
    echo "[ERROR] Could not find *cam1*.mp4 and *cam2*.mp4 in:"
    echo "        $PROJECT_DIR"
    exit 1
  fi

  ROIS="$PROJECT_DIR/rois.json"
  OUT_CSV="$PROJECT_DIR/events.csv"
  OUT_MARKERS="$PROJECT_DIR/markers.csv"
  OUT_FCPXML="$PROJECT_DIR/markers.fcpxml"
  OUT_EDL="$PROJECT_DIR/markers.edl"

  echo ""
  echo "Project folder: $PROJECT_DIR"
  echo "Cam1: $CAM1"
  echo "Cam2: $CAM2"

  # Run ROI picker if missing
  if [ ! -f "$ROIS" ]; then
    echo ""
    echo "[INFO] rois.json not found. Launching ROI picker..."
    python "$REPO_DIR/scripts/roi_picker.py" \
      "$CAM1" \
      "$CAM2" \
      --out "$ROIS"

    if [ ! -f "$ROIS" ]; then
      echo "[ERROR] ROI picker did not create: $ROIS"
      exit 1
    fi
  fi
fi

# ---- File mode ----
if [ "$MODE" = "files" ]; then
  if [ ! -f "$CAM1" ] || [ ! -f "$CAM2" ]; then
    echo "[ERROR] Missing cam files:"
    echo "  cam1: $CAM1"
    echo "  cam2: $CAM2"
    exit 1
  fi

  PROJECT_DIR="$(cd "$(dirname "$CAM1")" && pwd)"
  ROIS="$PROJECT_DIR/rois.json"
  OUT_CSV="$PROJECT_DIR/events.csv"
  OUT_MARKERS="$PROJECT_DIR/markers.csv"
  OUT_FCPXML="$PROJECT_DIR/markers.fcpxml"
  OUT_EDL="$PROJECT_DIR/markers.edl"

  echo ""
  echo "File mode"
  echo "Output folder: $PROJECT_DIR"
  echo "Cam1: $CAM1"
  echo "Cam2: $CAM2"

  if [ ! -f "$ROIS" ]; then
    echo ""
    echo "[INFO] rois.json not found in output folder. Launching ROI picker..."
    python "$REPO_DIR/scripts/roi_picker.py" \
      "$CAM1" \
      "$CAM2" \
      --out "$ROIS"

    if [ ! -f "$ROIS" ]; then
      echo "[ERROR] ROI picker did not create: $ROIS"
      exit 1
    fi
  fi
fi

echo ""
echo "[INFO] Running detection..."

python "$REPO_DIR/scripts/detect_events.py" \
  "$CAM1" \
  "$CAM2" \
  "$ROIS" \
  --fps 12 \
  --width 1280 \
  --thresh_pct 95 \
  --min_sep_s 12 \
  --merge_gap_s 0 \
  --max_win_s 20 \
  --max_big_win_s 32 \
  --big_color_min Orange \
  --replay_markers \
  --replay_offset_s 1.0 \
  --cooldown_s 12 \
  --out_csv "$OUT_CSV" \
  --out_markers "$OUT_MARKERS" \
  --verbose

echo ""
echo "[INFO] Converting markers.csv -> markers.fcpxml (timeline fps = 60)..."
python "$REPO_DIR/scripts/timeline_convert.py" \
  "$OUT_MARKERS" \
  --fps 60 \
  --out "$OUT_FCPXML" \
  --project-name "AI Markers"

echo ""
echo "[INFO] Converting markers.csv -> markers.edl (fps = 60)..."
python "$REPO_DIR/scripts/edl.py" \
  "$OUT_MARKERS" \
  --fps 60 \
  --out "$OUT_EDL"

echo ""
echo "Done."
echo "Outputs:"
echo "  $OUT_CSV"
echo "  $OUT_MARKERS"
echo "  $OUT_FCPXML"
echo "  $OUT_EDL"