#!/usr/bin/env bash
# run_pnl_recap_bsjp.sh
# Daily BSJP walk-forward PnL recap - cron @ 16:00 WIB Mon-Fri.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IDX_DIR="$REPO_ROOT/idx"
LOG_DIR="$IDX_DIR/_LOG"
LOG_FILE="$LOG_DIR/pnl_recap_bsjp_$(date +%Y%m%d).log"
PY_BIN="${PY_BIN:-$REPO_ROOT/.venv/bin/python}"

if [[ ! -x "$PY_BIN" ]]; then
  PY_BIN="python3"
fi

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

cd "$IDX_DIR"

log "=== BSJP PnL RECAP START ==="
EXTRA_ARGS=()
if [[ "${BSJP_PNL_RECAP_BOOTSTRAP_MISSING:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--bootstrap-missing)
fi
if [[ "${BSJP_PNL_RECAP_BOOTSTRAP_FETCH:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--bootstrap-fetch-missing-features)
fi
"$PY_BIN" "$IDX_DIR/pipeline/run/bsjp_pnl_recap.py" --send --no-print "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
log "=== BSJP PnL RECAP END ==="
