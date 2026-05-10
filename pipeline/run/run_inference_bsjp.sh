#!/usr/bin/env bash
# run_inference_bsjp.sh
# Daily production inference — cron @ 15:42 WIB Mon-Fri
# Sequence:
#   1. Quick preflight check (L0, DuckDB, model)
#   2. If L0 stale → auto-repair (yf 1h fetch) + recheck
#   3. bsjp fetch (compute features from L0)
#   4. bsjp predict v19d + v20 ARA policy
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IDX_DIR="$REPO_ROOT/idx"
BSJP="$IDX_DIR/inferences/bsjp/golang/bsjp"
LOG_DIR="$IDX_DIR/_LOG"
LOG_FILE="$LOG_DIR/inference_bsjp_$(date +%Y%m%d).log"
TODAY="$(date +%Y-%m-%d)"
CUTOFF_HHMM="${BSJP_SIGNAL_CUTOFF_HHMM:-1555}"

# Auto-detect latest date with 1h bars (skip if market hasn't opened yet)
LATEST_1H=$(python3 -c "
import pandas as pd
df = pd.read_parquet('$IDX_DIR/data/Level_0_Raw/yfinance_1h.parquet')
dt = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('Asia/Jakarta')
df = df.assign(_dt_wib=dt)
df = df[df['_dt_wib'] >= pd.Timestamp('$TODAY', tz='Asia/Jakarta') - pd.Timedelta(days=7)]
print(df['_dt_wib'].max().strftime('%Y-%m-%d') if len(df) > 0 else '$TODAY')
" 2>/dev/null || echo "$TODAY")
INFERENCE_DATE="$LATEST_1H"

# Load secrets
ENV_FILE="$HOME/.bsjp_notify.env"
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"

mkdir -p "$LOG_DIR"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
now_hhmm() { date '+%H%M'; }

send_telegram() {
  local msg="$1"
  if [[ -n "${BSJP_TELEGRAM_TOKEN:-}" && -n "${BSJP_TELEGRAM_CHAT_ID:-}" ]]; then
    curl -s -X POST "https://api.telegram.org/bot$BSJP_TELEGRAM_TOKEN/sendMessage" \
      -d "chat_id=$BSJP_TELEGRAM_CHAT_ID" \
      --data-urlencode "text=$msg" \
      -d "parse_mode=HTML" > /dev/null 2>&1 || true
  fi
}

if [[ "$(now_hhmm)" > "$CUTOFF_HHMM" ]]; then
  msg="⛔ <b>BSJP Signal skipped — ${TODAY}</b>

Reason: inference started after cutoff ${CUTOFF_HHMM} WIB.
No live signal will be published."
  log "Cutoff passed before start; skipping live signal"
  send_telegram "$msg"
  exit 1
fi

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
    log "yf_1h stale → fetching 1h only..."
    bash "$IDX_DIR/pipeline/run/run_fetch_yfinance_1h.sh" 2>&1 | tee -a "$LOG_FILE" &
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
  
  # DB_State is recomputed below after the latest L0 fetch.
  CRITICAL=$(echo "$OUTPUT" | grep "❌" | grep -v "global\|yf_1h\|DB_State" || true)
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

POST_OUTPUT=$("$BSJP" check --verbose 2>&1) || true
log "$POST_OUTPUT"
if ! echo "$POST_OUTPUT" | grep -q "DB_State.*quality=OK"; then
  msg="⛔ <b>BSJP Signal skipped — ${INFERENCE_DATE}</b>

Reason: DB_State is not executable after fetch.
No live signal was published."
  log "DB_State not OK after fetch; aborting"
  send_telegram "$msg"
  exit 1
fi

if [[ "$(now_hhmm)" > "$CUTOFF_HHMM" ]]; then
  msg="⛔ <b>BSJP Signal skipped — ${INFERENCE_DATE}</b>

Reason: inference completed after cutoff ${CUTOFF_HHMM} WIB.
No live signal will be published."
  log "Cutoff passed after feature computation; aborting live signal"
  send_telegram "$msg"
  exit 1
fi

# ═══════════════════════════════════════════════════════════════
# Step 3: Predict
# ═══════════════════════════════════════════════════════════════
log "Scoring v24b_final_production"
"$BSJP" predict --variant v24b_final_production --date "$INFERENCE_DATE" --log 2>&1 | tee -a "$LOG_FILE"

# ── Telegram: signal summary ──
if [[ "$(now_hhmm)" > "$CUTOFF_HHMM" ]]; then
  log "Cutoff passed after scoring; not sending live Telegram signal"
else
  V24B_PICKS=$("$BSJP" predict --variant v24b_final_production --date "$INFERENCE_DATE" 2>&1 \
    | grep -E '^  #[12] ' | sed 's/^  //' || echo "(no picks)")

  SIGNAL_MSG="📡 <b>BSJP Signal — ${INFERENCE_DATE}</b>

<b>v24b (Inventory Decay):</b>
${V24B_PICKS}"

  send_telegram "$SIGNAL_MSG"
fi

log "Done"
