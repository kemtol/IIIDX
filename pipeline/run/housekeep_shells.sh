#!/usr/bin/env bash
set -euo pipefail

# Keep service folder clean by archiving non-recurring shell scripts.
# Recurring scripts should stay in machinelearning/idx/service/heartbeat.

SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BAK_DIR="$SERVICE_DIR/_BAK"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$BAK_DIR"

mapfile -t CANDIDATES < <(
  find "$SERVICE_DIR" -maxdepth 1 -type f \( -name "*.sh" -o -name "*.bash" \) | sort
)

if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  echo "[housekeeping] no shell files to move in: $SERVICE_DIR"
  exit 0
fi

echo "[housekeeping] archiving shell files to: $BAK_DIR"
for file in "${CANDIDATES[@]}"; do
  base="$(basename "$file")"
  dest="$BAK_DIR/${base}.bak_${TS}"
  mv "$file" "$dest"
  echo "  moved: $file -> $dest"
done

echo "[housekeeping] done."
