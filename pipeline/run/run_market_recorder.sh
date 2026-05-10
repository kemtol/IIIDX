#!/bin/bash
PROJECT_DIR="/home/kemal/idx"
RECORDER_DIR="$PROJECT_DIR/pipeline/fetch/discovery/ws_recorder"
LOG_DIR="$PROJECT_DIR/_LOG"
DATA_DIR="$PROJECT_DIR/data/Level_0_Raw"
STAMP=$(date +%Y%m%d)

JSONL_PATH="$DATA_DIR/ws_captures/ws_trend_$STAMP.jsonl"
DB_PATH="$DATA_DIR/ws_trend_1m.duckdb"

cd "$RECORDER_DIR" || exit 1

# 1. Start the Go Recorder in background
go run main.go >> "$LOG_DIR/recorder_$STAMP.log" 2>&1 &
RECORDER_PID=$!

echo "[$(date)] Recorder started with PID $RECORDER_PID"

# 2. Sync Loop: Every 5 minutes
# Runs as long as the recorder is alive
while ps -p $RECORDER_PID > /dev/null; do
    sleep 300
    /home/kemal/.venv/bin/python3 sync_to_duckdb.py "$JSONL_PATH" "$DB_PATH" >> "$LOG_DIR/recorder_sync.log" 2>&1
done
