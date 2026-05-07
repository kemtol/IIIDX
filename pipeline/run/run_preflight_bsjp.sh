#!/usr/bin/env bash
# run_preflight_bsjp.sh
# Autonomous Pre-flight Operator — Heartbeat, Repair, and Pre-cook.
# Designed to be called frequently by cron; interval is state-driven.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IDX_DIR="$REPO_ROOT/idx"
BSJP="$IDX_DIR/inferences/bsjp/golang/bsjp"
LOG_DIR="$IDX_DIR/_LOG"
STATE_DIR="$IDX_DIR/_STATE"
STATE_FILE="$STATE_DIR/bsjp_heartbeat.json"
HEARTBEAT="$IDX_DIR/pipeline/run/bsjp_heartbeat.py"
LOG_FILE="$LOG_DIR/preflight_bsjp_$(date +%Y%m%d).log"
LOCK_FILE="/tmp/bsjp_preflight.lock"
mkdir -p "$LOG_DIR" "$STATE_DIR"

PY_BIN="${PY_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PY_BIN" ]]; then
  PY_BIN="python3"
fi

# -- Config --
# Load secrets from env file (gitignored)
ENV_FILE="$HOME/.bsjp_notify.env"
if [[ -f "$ENV_FILE" ]]; then
  source "$ENV_FILE"
fi

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

should_send_discord() {
  local hhmm minute
  hhmm="$(date '+%H%M')"
  minute="$(date '+%M')"

  # Discord is intentionally quieter than Telegram:
  # every 15 minutes, from 08:00 through 16:00 WIB only.
  if [[ "$hhmm" < "0800" || "$hhmm" > "1600" ]]; then
    return 1
  fi
  if (( 10#$minute % 15 != 0 )); then
    return 1
  fi
  return 0
}

# 1. Concurrency Protection
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
  log "Another instance is running. Exiting."
  exit 0
fi

log "=== BSJP HEARTBEAT CYCLE START ==="

# 1.1 Telegram command intake + interval gate
"$PY_BIN" "$HEARTBEAT" process-telegram --state-path "$STATE_FILE" 2>>"$LOG_FILE" || true
FORCE_FLAG=()
if [[ "${BSJP_HEARTBEAT_FORCE:-0}" == "1" ]]; then
  FORCE_FLAG=(--force)
fi
if ! NEXT_DUE=$("$PY_BIN" "$HEARTBEAT" should-send --state-path "$STATE_FILE" "${FORCE_FLAG[@]}" 2>>"$LOG_FILE"); then
  log "Heartbeat not due yet; next eligible send at ${NEXT_DUE:-unknown}."
  exit 0
fi

# 2. Build binary if missing
if [[ ! -x "$BSJP" ]]; then
  log "Building bsjp binary..."
  cd "$IDX_DIR/inferences/bsjp/golang"
  GOTOOLCHAIN=local go build -o bsjp ./cmd/bsjp/
  cd "$IDX_DIR"
fi

# 3. Step 1: Check Readiness (L0 and DB)
CHECK_FAILED=0
CHECK_OUTPUT=$("$BSJP" check --verbose 2>&1) || CHECK_FAILED=1

# Detect Weekend from output
if echo "$CHECK_OUTPUT" | grep -q "Weekend"; then
  log "Weekend detected. Heartbeat only, skipping repairs."
  FINAL_FILE=$(mktemp)
  MSG_FILE=$(mktemp)
  printf '%s\n' "$CHECK_OUTPUT" > "$FINAL_FILE"
  "$PY_BIN" "$HEARTBEAT" format \
    --state-path "$STATE_FILE" \
    --check-output-file "$FINAL_FILE" \
    --message-file "$MSG_FILE" \
    --failed "$CHECK_FAILED" \
    --cook-success "" \
    --cook-seconds "" 2>>"$LOG_FILE"
  "$PY_BIN" "$HEARTBEAT" send-telegram --message-file "$MSG_FILE" 2>>"$LOG_FILE" || true
  if should_send_discord; then
    "$PY_BIN" "$HEARTBEAT" send-discord --message-file "$MSG_FILE" 2>>"$LOG_FILE" || true
  fi
  "$PY_BIN" "$HEARTBEAT" mark-sent --state-path "$STATE_FILE" 2>>"$LOG_FILE" || true
  rm -f "$FINAL_FILE" "$MSG_FILE"
  exit 0
fi

# 4. Step 2: Smart Auto-Repair (Level 0)
# If L0 files are stale, trigger repairs in SYNC mode.
STALE_GLOBAL=$(echo "$CHECK_OUTPUT" | grep "❌.*global" || true)
STALE_YF1H=$(echo "$CHECK_OUTPUT" | grep "❌.*yf_1h" || true)
STALE_BROKSUM=$(echo "$CHECK_OUTPUT" | grep "❌.*broksum" || true)
NEED_T1=$(echo "$CHECK_OUTPUT" | sed -n 's/.*D-Day: [0-9-]* | T-1: \([0-9-]*\).*/\1/p' | head -1)

REPAIR_TRIGGERED=0

if [[ -n "$STALE_BROKSUM" ]]; then
  if [[ -n "$NEED_T1" ]]; then
    log "🛠️ Repairing L0: broksum is stale. Fetching explicit T-1=$NEED_T1..."
  else
    log "🛠️ Repairing L0: broksum is stale. Running fetch_broksum..."
  fi
  REPAIR_TRIGGERED=1
  if [[ -n "$NEED_T1" ]]; then
    bash "$IDX_DIR/pipeline/run/run_fetch_broksum.sh" \
      --from-date "$NEED_T1" \
      --to-date "$NEED_T1" \
      --source-mode fetch \
      --repair-days 0 \
      --disable-resume-state || log "🚨 broksum repair failed"
  else
    bash "$IDX_DIR/pipeline/run/run_fetch_broksum.sh" || log "🚨 broksum repair failed"
  fi
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
  "$PY_BIN" "$IDX_DIR/pipeline/fetch/fetch_global_indices_simple.py" \
    --output "$IDX_DIR/data/Level_0_Raw/global_indices.parquet" || log "🚨 global repair failed"
fi

# 5. Step 3: Pre-cooking (Feature Engine)
# If L0 is now likely OK (or was OK), trigger 'bsjp fetch' to update DuckDB.
# We do this even if CHECK_FAILED=0 to keep DuckDB "fresh" with latest intraday.
STALE_DB=$(echo "$CHECK_OUTPUT" | grep -E "❌.*(DuckDB|DB_|entry_price)" || true)

COOK_SUCCESS=""
COOK_SECONDS=""
if [[ "$CHECK_FAILED" -eq 0 ]] || [[ "$REPAIR_TRIGGERED" -eq 1 ]] || [[ -n "$STALE_DB" ]]; then
  log "🍳 Pre-cooking features for today into DuckDB..."
  FETCH_START=$(date +%s)
  if "$BSJP" fetch --date today; then
    FETCH_END=$(date +%s)
    log "✅ Pre-cook finished in $(($FETCH_END - $FETCH_START))s"
    COOK_SUCCESS=1
    COOK_SECONDS=$(($FETCH_END - $FETCH_START))
  else
    log "🚨 Pre-cook FAILED"
    COOK_SUCCESS=0
  fi
fi

# 6. Step 4: Final Verification & Notification
# Re-run check to get final status for notification
FINAL_FAILED=0
FINAL_OUTPUT=$("$BSJP" check --verbose 2>&1) || FINAL_FAILED=1

# Notification Logic
if [[ "$FINAL_FAILED" -eq 0 ]]; then
  ICON="💓"
  STATUS_MSG="PULSE"
  READINESS="READY"
else
  ICON="⚠️"
  STATUS_MSG="REPAIR_NEEDED"
  READINESS="STALE"
fi

FINAL_FILE=$(mktemp)
MSG_FILE=$(mktemp)
printf '%s\n' "$FINAL_OUTPUT" > "$FINAL_FILE"
"$PY_BIN" "$HEARTBEAT" format \
  --state-path "$STATE_FILE" \
  --check-output-file "$FINAL_FILE" \
  --message-file "$MSG_FILE" \
  --failed "$FINAL_FAILED" \
  --cook-success "$COOK_SUCCESS" \
  --cook-seconds "$COOK_SECONDS" 2>>"$LOG_FILE"

# Send to Telegram
if [[ -n "${BSJP_TELEGRAM_TOKEN:-}" && -n "${BSJP_TELEGRAM_CHAT_ID:-}" ]]; then
  log "Sending Heartbeat to Telegram..."
  "$PY_BIN" "$HEARTBEAT" send-telegram --message-file "$MSG_FILE" 2>>"$LOG_FILE" || true
fi

# Send to Discord
if should_send_discord && [[ -n "${BSJP_DISCORD_WEBHOOK_URL:-}" || -n "${BSJP_DISCORD_TOKEN:-}" ]]; then
  log "Sending Heartbeat to Discord..."
  "$PY_BIN" "$HEARTBEAT" send-discord --message-file "$MSG_FILE" 2>>"$LOG_FILE" || true
fi

"$PY_BIN" "$HEARTBEAT" mark-sent --state-path "$STATE_FILE" 2>>"$LOG_FILE" || true
rm -f "$FINAL_FILE" "$MSG_FILE"

log "=== BSJP HEARTBEAT CYCLE END (Success=$((1-FINAL_FAILED))) ==="
