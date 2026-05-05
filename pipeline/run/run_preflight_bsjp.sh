#!/usr/bin/env bash
# run_preflight_bsjp.sh
# Pre-flight readiness check — cron @ 14:00 WIB Mon-Fri
# Loops until all-clear or timeout (60 min). Notifies Telegram + Discord each cycle.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IDX_DIR="$REPO_ROOT/idx"
BSJP="$IDX_DIR/inferences/bsjp/golang/bsjp"
LOG_DIR="$IDX_DIR/_LOG"
LOG_FILE="$LOG_DIR/preflight_bsjp_$(date +%Y%m%d).log"
mkdir -p "$LOG_DIR"

# ── Config ──
# Load secrets from env file (gitignored)
ENV_FILE="$HOME/.bsjp_notify.env"
if [[ -f "$ENV_FILE" ]]; then
  source "$ENV_FILE"
fi

MAX_RETRIES=12       # 60 min / 5 min = 12 iterations
SLEEP_SEC=300         # 5 minutes
PREVIOUS_STATE=""

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "BSJP Preflight START"

# Build binary if missing
if [[ ! -x "$BSJP" ]]; then
  log "Building bsjp..."
  cd "$IDX_DIR/inferences/bsjp/golang"
  GOTOOLCHAIN=local go build -o bsjp ./cmd/bsjp/
  cd "$IDX_DIR"
fi

# ── Main loop ──
for ((i=1; i<=MAX_RETRIES; i++)); do
  TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

  # Run check (always with telegram+discord on first and changed runs)
  TG_FLAG="--telegram"
  OUTPUT=$("$BSJP" check --verbose $TG_FLAG 2>&1) || CHECK_FAILED=1
  if [[ -z "${CHECK_FAILED:-}" ]]; then
    CHECK_FAILED=0
  fi

  # Compute current state hash (all non-empty lines)
  CURRENT_STATE=$(echo "$OUTPUT" | grep -v '^$' | sort)

  # Determine if state changed
  if [[ "$CURRENT_STATE" != "$PREVIOUS_STATE" ]]; then
    # Send notification on first run or state change
    if [[ -z "$PREVIOUS_STATE" ]]; then
      log "Check #$i — initial"
    else
      log "Check #$i — status changed"
    fi
    echo "$OUTPUT" | tee -a "$LOG_FILE"
    PREVIOUS_STATE="$CURRENT_STATE"
  else
    log "Check #$i — no change, skipping notify"
    echo "$OUTPUT" | grep -E '^(✅|❌)' | tee -a "$LOG_FILE"
  fi

  # If all green, done
  if [[ "$CHECK_FAILED" -eq 0 ]]; then
    log "ALL GREEN. Inference on schedule."
    
    # Send final all-clear
    FINAL_MSG="✅ BSJP PREFLIGHT ALL-CLEAR — Inference running now"
    if [[ -n "${BSJP_TELEGRAM_TOKEN:-}" && -n "${BSJP_TELEGRAM_CHAT_ID:-}" ]]; then
      curl -s -X POST "https://api.telegram.org/bot$BSJP_TELEGRAM_TOKEN/sendMessage" \
        -d "chat_id=$BSJP_TELEGRAM_CHAT_ID" \
        -d "text=$FINAL_MSG" \
        -d "parse_mode=Markdown" > /dev/null 2>&1 || true
    fi
    exit 0
  fi

  # Auto-repair: trigger Python fetchers for stale L0 files
  STALE_GLOBAL=$(echo "$OUTPUT" | grep "❌.*global" || true)
  STALE_YF1H=$(echo "$OUTPUT" | grep "❌.*yf_1h" || true)

  if [[ -n "$STALE_YF1H" ]]; then
    log "Auto-repair: fetching yfinance 1h..."
    bash "$IDX_DIR/pipeline/run/run_fetch_yfinance.sh" &
  fi

  if [[ -n "$STALE_GLOBAL" ]]; then
    log "Auto-repair: fetching global indices..."
    PY_BIN="${PY_BIN:-$REPO_ROOT/.venv/bin/python}"
    "$PY_BIN" "$IDX_DIR/pipeline/fetch/fetch_global_indices_simple.py" \
      --output "$IDX_DIR/data/Level_0_Raw/global_indices.parquet" &
  fi

  log "Check #$i done. Sleeping ${SLEEP_SEC}s..."
  sleep "$SLEEP_SEC"
  unset CHECK_FAILED
done

# ── Timeout ──
log "TIMEOUT after $(($MAX_RETRIES * $SLEEP_SEC / 60)) minutes. Inference CANCELLED."
TIMEOUT_MSG="🚨 BSJP PREFLIGHT TIMEOUT — Inference CANCELLED after $(($MAX_RETRIES * $SLEEP_SEC / 60)) min"
if [[ -n "${BSJP_TELEGRAM_TOKEN:-}" && -n "${BSJP_TELEGRAM_CHAT_ID:-}" ]]; then
  curl -s -X POST "https://api.telegram.org/bot$BSJP_TELEGRAM_TOKEN/sendMessage" \
    -d "chat_id=$BSJP_TELEGRAM_CHAT_ID" \
    -d "text=$TIMEOUT_MSG" \
    -d "parse_mode=Markdown" > /dev/null 2>&1 || true
fi
exit 1
