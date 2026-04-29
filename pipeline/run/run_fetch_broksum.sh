#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IDX_DIR="$REPO_ROOT/idx"
LOG_DIR="$IDX_DIR/_LOG"
mkdir -p "$LOG_DIR"

PY_BIN="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PY_BIN" ]]; then
  echo "[error] Python venv not found at: $PY_BIN"
  echo "Please create/install venv first."
  exit 1
fi

cd "$REPO_ROOT"
"$PY_BIN" "$IDX_DIR/pipeline/fetch/fetch_broksum_ipot.py" \
  --all-brokers \
  --days 1 \
  --repair-days 14 \
  --update-mode append \
  --max-concurrent 8 \
  --day-sleep 0.05 \
  --output "$IDX_DIR/data/Level_0_Raw/broksum_bybroker.parquet" \
  "$@"
