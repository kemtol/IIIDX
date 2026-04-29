#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IDX_DIR="$REPO_ROOT/idx"
GEN_SCRIPT="$IDX_DIR/pipeline/feature/generate_broksum_datamart.py"

PY_BIN="${PY_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PY_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PY_BIN="$(command -v python3)"
  else
    echo "[error] python interpreter not found (.venv/bin/python or python3)."
    exit 1
  fi
fi

cd "$REPO_ROOT"
echo "[run] $PY_BIN $GEN_SCRIPT $*"
"$PY_BIN" "$GEN_SCRIPT" "$@"
