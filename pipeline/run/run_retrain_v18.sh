#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(dirname "$(realpath "$0")")
REPO=$(realpath "$SCRIPT_DIR/../..")
cd "$REPO/edges/bsjp_overnight_sl2/scripts"

MODULES="$REPO/data/Level_1_Features/modules"
OUTPUT="$REPO/model/BSJP/bsjp_v18_market_100d_fix"

echo "Training v18 with --feature-modules-dir..."
python3 train_lightgbm.py \
  --output-dir "$OUTPUT" \
  --feature-modules-dir "$MODULES" \
  --feature-prune-top-n 0 \
  --tp-pct 0.01 \
  --sl-pct -0.02 \
  --oot-valid-days 100

echo ""
echo "=== Tree count ==="
grep -c "^Tree=" "$OUTPUT/model_lightgbm_opening_tp3.txt"
echo "=== Feature idx ==="
grep "max_feature_idx" "$OUTPUT/model_lightgbm_opening_tp3.txt"
echo "=== AUC ==="
python3 -c "
import csv
with open('$OUTPUT/walkforward_metrics.csv') as f:
    rows = list(csv.DictReader(f))
    aucs = [float(r['auc']) for r in rows]
    print(f'Mean AUC: {sum(aucs)/len(aucs):.4f} ({len(aucs)} folds)')
"
ls -la "$OUTPUT/model_lightgbm_opening_tp3.txt"
