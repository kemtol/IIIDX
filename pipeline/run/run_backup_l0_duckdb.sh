#!/usr/bin/env bash
# run_backup_l0_duckdb.sh
# Daily cron — backup all existing L0 DuckDB files via tier-aware retention.
# No-op on non-Sunday (tiers_for_date returns []).
# Per PRD 0003 §6.1.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IDX_DIR="$REPO_ROOT/idx"
LOG_DIR="$IDX_DIR/_LOG"
mkdir -p "$LOG_DIR"

PY_BIN="${PY_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PY_BIN" ]]; then
  echo "[error] Python venv not found at: $PY_BIN" >&2
  exit 1
fi

LOG_FILE="$LOG_DIR/duckdb_backup_$(date +%Y%m%d).log"

cd "$IDX_DIR"
{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] L0 DuckDB backup start"
  "$PY_BIN" -m pipeline.storage.backup
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] L0 DuckDB backup done"
} 2>&1 | tee -a "$LOG_FILE"
