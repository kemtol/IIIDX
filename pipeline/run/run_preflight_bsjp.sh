#!/usr/bin/env bash
# run_preflight_bsjp.sh
# Autonomous Pre-flight Operator — Heartbeat, Repair, and Pre-cook.
# Designed to run hourly (e.g., via cron) to ensure 15:00 WIB readiness.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IDX_DIR="$REPO_ROOT/idx"
BSJP="$IDX_DIR/inferences/bsjp/golang/bsjp"
LOG_DIR="$IDX_DIR/_LOG"
LOG_FILE="$LOG_DIR/preflight_bsjp_$(date +%Y%m%d).log"
LOCK_FILE="/tmp/bsjp_preflight.lock"
mkdir -p "$LOG_DIR"

# -- Config --
# Load secrets from env file (gitignored)
ENV_FILE="$HOME/.bsjp_notify.env"
if [[ -f "$ENV_FILE" ]]; then
  source "$ENV_FILE"
fi

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# 1. Concurrency Protection
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
  log "Another instance is running. Exiting."
  exit 0
fi

log "=== BSJP HEARTBEAT CYCLE START ==="

# 2. Build binary if missing
if [[ ! -x "$BSJP" ]]; then
  log "Building bsjp binary..."
  cd "$IDX_DIR/inferences/bsjp/golang"
  GOTOOLCHAIN=local go build -o bsjp ./cmd/bsjp/
  cd "$IDX_DIR"
fi

# 3. Step 1: Check Readiness (L0 and DB)
CHECK_OUTPUT=$("$BSJP" check --verbose 2>&1) || CHECK_FAILED=1
if [[ -z "${CHECK_FAILED:-}" ]]; then
  CHECK_FAILED=0
fi

# Detect Weekend from output
if echo "$CHECK_OUTPUT" | grep -q "Weekend"; then
  log "🛌 Weekend detected. Heartbeat only, skipping repairs."
  # Send Telegram and exit
  if [[ -n "${BSJP_TELEGRAM_TOKEN:-}" && -n "${BSJP_TELEGRAM_CHAT_ID:-}" ]]; then
    curl -s -X POST "https://api.telegram.org/bot$BSJP_TELEGRAM_TOKEN/sendMessage" \
      -d "chat_id=$BSJP_TELEGRAM_CHAT_ID" \
      -d "text=$(echo -e "🛌 *BSJP RELAXING* [$(date +%H:%M)]\nMarket is closed. See you Monday!")" \
      -d "parse_mode=Markdown" > /dev/null 2>&1 || true
  fi
  exit 0
fi

# 4. Step 2: Smart Auto-Repair (Level 0)
# If L0 files are stale, trigger repairs in SYNC mode.
STALE_GLOBAL=$(echo "$CHECK_OUTPUT" | grep "❌.*global" || true)
STALE_YF1H=$(echo "$CHECK_OUTPUT" | grep "❌.*yf_1h" || true)
STALE_BROKSUM=$(echo "$CHECK_OUTPUT" | grep "❌.*broksum" || true)

REPAIR_TRIGGERED=0

if [[ -n "$STALE_BROKSUM" ]]; then
  log "🛠️ Repairing L0: broksum is stale. Running fetch_broksum..."
  REPAIR_TRIGGERED=1
  bash "$IDX_DIR/pipeline/run/run_fetch_broksum.sh" || log "🚨 broksum repair failed"
fi

if [[ -n "$STALE_YF1H" ]]; then
  log "🛠️ Repairing L0: yf_1h is stale. Running download..."
  REPAIR_TRIGGERED=1
  # Use Go download for speed/parity
  "$BSJP" download --date today || log "🚨 yf_1h download failed"
fi

if [[ -n "$STALE_GLOBAL" ]]; then
  log "🛠️ Repairing L0: global is stale. Running fetcher..."
  REPAIR_TRIGGERED=1
  PY_BIN="${PY_BIN:-$REPO_ROOT/.venv/bin/python}"
  "$PY_BIN" "$IDX_DIR/pipeline/fetch/fetch_global_indices_simple.py" \
    --output "$IDX_DIR/data/Level_0_Raw/global_indices.parquet" || log "🚨 global repair failed"
fi

# 5. Step 3: Pre-cooking (Feature Engine)
# If L0 is now likely OK (or was OK), trigger 'bsjp fetch' to update DuckDB.
# We do this even if CHECK_FAILED=0 to keep DuckDB "fresh" with latest intraday.
STALE_DB=$(echo "$CHECK_OUTPUT" | grep "❌.*DuckDB" || true)

if [[ "$CHECK_FAILED" -eq 0 ]] || [[ "$REPAIR_TRIGGERED" -eq 1 ]] || [[ -n "$STALE_DB" ]]; then
  log "🍳 Pre-cooking features for today into DuckDB..."
  FETCH_START=$(date +%s)
  if "$BSJP" fetch --date today; then
    FETCH_END=$(date +%s)
    log "✅ Pre-cook finished in $(($FETCH_END - $FETCH_START))s"
    COOK_SUCCESS=1
  else
    log "🚨 Pre-cook FAILED"
    COOK_SUCCESS=0
  fi
fi

# 6. Step 4: Final Verification & Notification
# Re-run check to get final status for notification
FINAL_OUTPUT=$("$BSJP" check --verbose 2>&1) || FINAL_FAILED=1
if [[ -z "${FINAL_FAILED:-}" ]]; then
  FINAL_FAILED=0
fi

# Notification Logic
SEND_NOTIFY=1
CURRENT_HOUR=$(date +%H)

if [[ "$FINAL_FAILED" -eq 0 ]]; then
  ICON="💓"
  STATUS_MSG="PULSE"
  READINESS="READY"
else
  ICON="⚠️"
  STATUS_MSG="REPAIR_NEEDED"
  READINESS="STALE"
fi

# Build Telegram/Discord message
MSG="$ICON *BSJP $STATUS_MSG [$(date +%H:%M)]*\n\n"
MSG+="$(echo "$FINAL_OUTPUT" | sed 's/$/\\n/')\n"

if [[ "$FINAL_FAILED" -eq 0 ]]; then
  MSG+="\n✅ *System is cooked and ready.*"
else
  MSG+="\n🚨 *Issues persist. Manual check required.*"
fi

# Send to Telegram
if [[ -n "${BSJP_TELEGRAM_TOKEN:-}" && -n "${BSJP_TELEGRAM_CHAT_ID:-}" ]]; then
  log "Sending Heartbeat to Telegram..."
  curl -s -X POST "https://api.telegram.org/bot$BSJP_TELEGRAM_TOKEN/sendMessage" \
    -d "chat_id=$BSJP_TELEGRAM_CHAT_ID" \
    -d "text=$(echo -e "$MSG")" \
    -d "parse_mode=Markdown" > /dev/null 2>&1 || true
fi

# Send to Discord
if [[ -n "${BSJP_DISCORD_TOKEN:-}" && -n "${BSJP_DISCORD_CHANNEL:-}" ]]; then
  log "Sending Heartbeat to Discord..."
  # (Simpler version for shell-based discord notify)
  curl -s -X POST "https://discord.com/api/webhooks/${BSJP_DISCORD_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "{\"content\": \"$ICON **BSJP $STATUS_MSG** [$(date +%H:%M)]\\nReadiness: $READINESS\"}" > /dev/null 2>&1 || true
fi

log "=== BSJP HEARTBEAT CYCLE END (Success=$((1-FINAL_FAILED))) ==="
