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

cd "$IDX_DIR"
"$PY_BIN" "$IDX_DIR/pipeline/fetch/fetch_yfinance.py" \
  --data-dir "$IDX_DIR/data/Level_0_Raw" \
  --master-path "$IDX_DIR/data/Level_0_Raw/master_emiten.parquet" \
  "$@"
