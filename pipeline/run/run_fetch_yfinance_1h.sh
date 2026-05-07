#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IDX_DIR="$REPO_ROOT/idx"

bash "$IDX_DIR/pipeline/run/run_fetch_yfinance.sh" --intervals 1h "$@"
