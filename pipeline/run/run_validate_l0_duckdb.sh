#!/usr/bin/env bash
# run_validate_l0_duckdb.sh
# Daily cron — for each L0 source in stage 1+ (duckdb_write enabled),
# compare parquet vs DuckDB and write _LOG/duckdb_canary_<source>_<date>.jsonc.
# Stage-0 sources are skipped.
# Per PRD 0003 §5.3.
#
# Exits non-zero if any source's validation fails (caller can alert).
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

LOG_FILE="$LOG_DIR/duckdb_validate_$(date +%Y%m%d).log"

cd "$IDX_DIR"
{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] L0 DuckDB validate start"
  "$PY_BIN" -m pipeline.storage.validate
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] L0 DuckDB validate done"
} 2>&1 | tee -a "$LOG_FILE"
