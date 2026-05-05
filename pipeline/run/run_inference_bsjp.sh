#!/usr/bin/env bash
# run_inference_bsjp.sh
# Daily production inference — cron @ 15:15 WIB Mon-Fri
# Sequence:
#   1. Quick preflight check (L0, DuckDB, model)
#   2. If L0 stale → auto-repair (yf 1h fetch) + recheck
#   3. bsjp fetch (compute features from L0)
#   4. bsjp predict v19d + v20 ARA policy
#   5. bsjp predict v15 (baseline)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IDX_DIR="$REPO_ROOT/idx"
BSJP="$IDX_DIR/inferences/bsjp/golang/bsjp"
LOG_DIR="$IDX_DIR/_LOG"
LOG_FILE="$LOG_DIR/inference_bsjp_$(date +%Y%m%d).log"
TODAY="$(date +%Y-%m-%d)"

# Auto-detect latest date with 1h bars (skip if market hasn't opened yet)
LATEST_1H=$(python3 -c "
import pandas as pd
df = pd.read_parquet('$IDX_DIR/data/Level_0_Raw/yfinance_1h.parquet')
df = df[df['datetime'] >= pd.Timestamp('$TODAY') - pd.Timedelta(days=7)]
print(df['datetime'].max().strftime('%Y-%m-%d') if len(df) > 0 else '$TODAY')
" 2>/dev/null || echo "$TODAY")
INFERENCE_DATE="$LATEST_1H"

# Load secrets
ENV_FILE="$HOME/.bsjp_notify.env"
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"

mkdir -p "$LOG_DIR"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "BSJP Inference START"

cd "$IDX_DIR"

# Build binary if missing
if [[ ! -x "$BSJP" ]]; then
  log "Building bsjp..."
  cd "$IDX_DIR/inferences/bsjp/golang"
  GOTOOLCHAIN=local go build -o bsjp ./cmd/bsjp/ 2>&1 | tee -a "$LOG_FILE"
  cd "$IDX_DIR"
fi

# ═══════════════════════════════════════════════════════════════
# Step 1: Preflight check
# ═══════════════════════════════════════════════════════════════
for attempt in 1 2 3 4; do
  log "Preflight attempt #$attempt"
  OUTPUT=$("$BSJP" check --verbose 2>&1) || true
  
  # Check for yf_1h staleness
  STALE_YF1H=$(echo "$OUTPUT" | grep "❌.*yf_1h" || true)
  STALE_GLOBAL=$(echo "$OUTPUT" | grep "❌.*global" || true)
  
  if [[ -n "$STALE_YF1H" ]]; then
    log "yf_1h stale → fetching..."
    bash "$IDX_DIR/pipeline/run/run_fetch_yfinance.sh" 2>&1 | tee -a "$LOG_FILE" &
    wait
    sleep 10
    continue
  fi
  
  if [[ -n "$STALE_GLOBAL" ]]; then
    log "global indices stale → fetching..."
    PY_BIN="${PY_BIN:-$REPO_ROOT/.venv/bin/python}"
    "$PY_BIN" "$IDX_DIR/pipeline/fetch/fetch_global_indices_simple.py" \
      --output "$IDX_DIR/data/Level_0_Raw/global_indices.parquet" 2>&1 | tee -a "$LOG_FILE" &
    wait
    sleep 5
    continue
  fi
  
  # All green or only non-critical issues
  CRITICAL=$(echo "$OUTPUT" | grep "❌" | grep -v "global\|yf_1h" || true)
  if [[ -z "$CRITICAL" ]]; then
    log "Preflight: READY"
    break
  fi
  
  log "Preflight: $CRITICAL"
  sleep 60
done

# ═══════════════════════════════════════════════════════════════
# Step 2: Fetch features from L0
# ═══════════════════════════════════════════════════════════════
log "Computing features for $INFERENCE_DATE"
"$BSJP" fetch --date "$INFERENCE_DATE" --force 2>&1 | tee -a "$LOG_FILE"

# ═══════════════════════════════════════════════════════════════
# Step 3: Predict
# ═══════════════════════════════════════════════════════════════
log "Scoring v19d_close10_preclose14_orb_md100_l21.5"
"$BSJP" predict --variant v19d_close10_preclose14_orb_md100_l21.5 --log 2>&1 | tee -a "$LOG_FILE"

# ── Telegram: signal summary ──
if [[ -n "${BSJP_TELEGRAM_TOKEN:-}" && -n "${BSJP_TELEGRAM_CHAT_ID:-}" ]]; then
  V19D_PICKS=$("$BSJP" predict --variant v19d_close10_preclose14_orb_md100_l21.5 --date "$INFERENCE_DATE" 2>&1 \
    | grep -E '^  #1 ' | sed 's/^  //' || echo "(no picks)")

  SIGNAL_MSG="📡 <b>BSJP Signal — ${INFERENCE_DATE}</b>

<b>v19d + v20 ARA:</b>
${V19D_PICKS}"

  curl -s -X POST "https://api.telegram.org/bot$BSJP_TELEGRAM_TOKEN/sendMessage" \
    -d "chat_id=$BSJP_TELEGRAM_CHAT_ID" \
    --data-urlencode "text=$SIGNAL_MSG" \
    -d "parse_mode=HTML" > /dev/null 2>&1 || true
fi

log "Done"
