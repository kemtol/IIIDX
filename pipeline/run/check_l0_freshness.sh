#!/usr/bin/env bash
# Check L0 data freshness — warns if stale. No fetch.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IDX_DIR="$REPO_ROOT/idx"
LOG_DIR="$IDX_DIR/_LOG"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/check_freshness.log"

echo "=== $(date) ===" >> "$LOG"

check_parquet() {
  local label=$1; local path=$2; local date_col=$3
  if [[ ! -f "$path" ]]; then
    echo "[STALE] $label: file not found $path" >> "$LOG"
    return
  fi
  local latest
  latest=$("$REPO_ROOT/.venv/bin/python" -c "
import pyarrow.parquet as pq
t = pq.read_table('$path', columns=['$date_col'])
dates = t.column('$date_col').to_pylist()
print(max(dates))
" 2>/dev/null) || latest="ERR"
  echo "[OK] $label: max=$latest" >> "$LOG"
}

check_parquet "yfinance_1h"   "$IDX_DIR/data/Level_0_Raw/yfinance_1h.parquet"          datetime
check_parquet "yfinance_daily" "$IDX_DIR/data/Level_0_Raw/yfinance_daily.parquet"       date
check_parquet "broksum"        "$IDX_DIR/data/Level_0_Raw/broksum_bybroker.parquet"      date
check_parquet "global_indices" "$IDX_DIR/data/Level_0_Raw/global_indices.parquet"        date
