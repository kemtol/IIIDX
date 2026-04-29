#!/usr/bin/env bash
# run_inference_bpjs.sh
# Daily inference runner for BPJS intraday strategy.
# Cron: 08:30 WIB Mon-Fri (before open)
#
# NOTE: BPJS is currently PAUSED. Uncomment when strategy is reactivated.
#
# Sequence:
#   1. fetch.py           — update inference DB with today's features
#   2. run.py --variant v16b — score and print picks
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
INFERENCE_DIR="$REPO_ROOT/idx/inferences/bpjs"
PY_BIN="${PY_BIN:-$REPO_ROOT/.venv/bin/python3}"
LOG_DIR="$REPO_ROOT/idx/_LOG"
LOG_FILE="$LOG_DIR/inference_bpjs_$(date +%Y%m%d).log"

mkdir -p "$LOG_DIR"

# PAUSED — uncomment to activate
# echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting BPJS inference" | tee -a "$LOG_FILE"
# cd "$INFERENCE_DIR"
# "$PY_BIN" fetch.py              2>&1 | tee -a "$LOG_FILE"
# "$PY_BIN" run.py --variant v16b --log-picks 2>&1 | tee -a "$LOG_FILE"
# echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done" | tee -a "$LOG_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] BPJS inference is PAUSED — skipping" | tee -a "$LOG_FILE"
