#!/bin/bash
# run_recap.sh — `hockeyrecap` command (V3 stage runner).
# This file lives at the project root so a `hockeyrecap` shell alias can point
# to it by absolute path, the same way `hockeydetect` points to run_detect.sh.
#
#   hockeyrecap run <game_folder> [<game_folder> ...]
#   hockeyrecap status <game_folder>
#   hockeyrecap rerun --from <stage> <game_folder>
#   hockeyrecap check <game_folder>
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO_DIR/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "[WARN] No .venv found at $REPO_DIR/.venv (continuing with system python)" >&2
  PYTHON="python3"
fi

exec "$PYTHON" "$REPO_DIR/v3/scripts/hockeyrecap.py" "$@"
