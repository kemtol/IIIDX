#!/usr/bin/env bash
# deploy_inference.sh
# Rsync inferences/ and pipeline/run/run_inference_*.sh to VPS.
# Usage: bash pipeline/run/deploy_inference.sh
set -euo pipefail

VPS="root@213.199.49.18"
REMOTE_ROOT="/root/MMMACHINE/idx"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Deploying inferences/ → $VPS:$REMOTE_ROOT"

rsync -avz --delete \
  "$LOCAL_ROOT/inferences/" \
  "$VPS:$REMOTE_ROOT/inferences/"

rsync -avz \
  "$LOCAL_ROOT/pipeline/run/run_inference_bsjp.sh" \
  "$LOCAL_ROOT/pipeline/run/run_inference_bpjs.sh" \
  "$VPS:$REMOTE_ROOT/pipeline/run/"

echo "Done."
