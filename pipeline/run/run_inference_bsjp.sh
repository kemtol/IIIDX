#!/usr/bin/env bash
# run_inference_bsjp.sh
# Daily inference runner for BSJP overnight strategy.
# Cron: 15:10 WIB Mon-Fri
#
# Sequence:
#   1. fetch_ohlcv_ipot.py — real-time 1-min bars → ipot_ohlcv_1h.parquet (no yfinance delay)
#   2. python/fetch.py            — L1 + L2 → upsert inference DB
#   3. python/run.py --variant v15 — standard picks
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
INFERENCE_DIR="$REPO_ROOT/idx/inferences/bsjp/python"
PY_BIN="${PY_BIN:-$REPO_ROOT/.venv/bin/python3}"
LOG_DIR="$REPO_ROOT/idx/_LOG"
LOG_FILE="$LOG_DIR/inference_bsjp_$(date +%Y%m%d).log"

mkdir -p "$LOG_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting BSJP inference" | tee -a "$LOG_FILE"

cd "$INFERENCE_DIR"

# Step 1: Fetch real-time OHLCV from IPOT (avoids yfinance delay for close price)
"$PY_BIN" "$REPO_ROOT/idx/pipeline/fetch/fetch_ohlcv_ipot.py" 2>&1 | tee -a "$LOG_FILE"

# Step 2: Fetch / update inference DB (L1 + L2, once, shared across variants)
"$PY_BIN" fetch.py 2>&1 | tee -a "$LOG_FILE"

# Step 3: Score each variant
"$PY_BIN" run.py --variant v15 --log-picks 2>&1 | tee -a "$LOG_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done" | tee -a "$LOG_FILE"
