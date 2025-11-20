#!/bin/zsh
BASE_DIR="$HOME/Desktop/DSDN"
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline.log"

source "$BASE_DIR/.venv/bin/activate"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Start daily pipeline" >> "$LOG_FILE"
cd "$BASE_DIR" || exit 1
python3 daily_pipeline.py >> "$LOG_FILE" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done" >> "$LOG_FILE"
