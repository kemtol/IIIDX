#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(dirname "$(realpath "$0")")
REPO=$(realpath "$SCRIPT_DIR/../..")
cd "$REPO/edges/bsjp_overnight_sl2/scripts"
echo "[1/2] generate_datamart..."
python generate_datamart.py --modules-dir "$REPO/data/Level_1_Features/modules"
echo "[2/2] train_lightgbm..."
TP=0.01 SL=-0.02 OOT=100 TOPN=0
V="v18_market_100d_fix"
python train_lightgbm.py --output-dir "$REPO/model/BSJP/$V" --feature-modules-dir "$REPO/data/Level_1_Features/modules" --feature-prune-top-n $TOPN --tp-pct $TP --sl-pct $SL --oot-valid-days $OOT
echo "done"
ls -la "$REPO/model/BSJP/$V/model_lightgbm_opening_tp3.txt"
grep -c "^Tree=" "$REPO/model/BSJP/$V/model_lightgbm_opening_tp3.txt"
