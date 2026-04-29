#!/bin/bash
# =============================================================================
# Grid Search Phase 2: Fine Grid
# =============================================================================
# Narrows the search around the best-performing region from Phase 1.
# Tests fine-grained combinations of min_data_in_leaf × lambda_l1/l2.
# Also tests min_gain_to_split variations for top candidates.
#
# Phase 1 winners: md=100-200, lam=0.75-2.0
#
# Usage:
#   cd /home-ssd/mkemalw/Projects/MMMACHINE/idx
#   bash edges/bsjp_overnight_sl2/scripts/grid_search_fine.sh
#
# Output: model/BSJP/bsjp_grid_fine/md{min_data}_lam{lambda}_gain{gain}/
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV_DIR="$(cd "$PROJECT_ROOT/.." && pwd)/.venv"
PYTHON="$VENV_DIR/bin/python"
TRAIN_SCRIPT="$SCRIPT_DIR/train_lightgbm.py"

# --- Grid Parameters ---
MIN_DATA_VALUES=(80 100 150 200)
LAMBDA_VALUES=(0.75 1.0 1.5 2.0)
MIN_GAIN_VALUES=(0.02 0.05 0.1)

# --- Shared Arguments ---
DATAMART="$PROJECT_ROOT/data/Level_2_Datamart/training_datamart_bsjp_close10_v10.parquet"
BASE_OUTPUT="$PROJECT_ROOT/model/BSJP/bsjp_grid_fine"
FEATURE_IMPORTANCE="$PROJECT_ROOT/model/BSJP/bsjp_v9d_classifier_ref/feature_importance.csv"

mkdir -p "$BASE_OUTPUT"

LOG_FILE="$BASE_OUTPUT/grid_search_fine.log"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Grid Search Fine - $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Grid: min_data_in_leaf = ${MIN_DATA_VALUES[*]}" | tee -a "$LOG_FILE"
echo "      lambda_l1/l2     = ${LAMBDA_VALUES[*]}" | tee -a "$LOG_FILE"
echo "      min_gain_to_split = ${MIN_GAIN_VALUES[*]}" | tee -a "$LOG_FILE"
echo "Datamart: $DATAMART" | tee -a "$LOG_FILE"
TOTAL=$(( ${#MIN_DATA_VALUES[@]} * ${#LAMBDA_VALUES[@]} * ${#MIN_GAIN_VALUES[@]} ))
echo "Total combos: $TOTAL (full factorial)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

COUNT=0
FAILED=0

# Helper function to format lambda for directory names
fmt_lam() {
    echo "$1" | sed 's/\./_/'
}

for md in "${MIN_DATA_VALUES[@]}"; do
    for lam in "${LAMBDA_VALUES[@]}"; do
        for gain in "${MIN_GAIN_VALUES[@]}"; do
            COUNT=$((COUNT + 1))
            
            MD_PAD=$(printf "%03d" "$md")
            LAM_STR=$(fmt_lam "$lam")
            GAIN_STR=$(fmt_lam "$gain")
            
            # Only run gain variations for promising parameter regions
            # Phase 1 showed gain=0.05 works well, we test 0.02 and 0.1 mainly around best area
            # To keep total manageable, we test gain variations only for md=100,200 × lam=1.0,2.0
            # For other combos, fix gain=0.05
            if [ "$gain" != "0.05" ]; then
                # Only test gain variations for the most promising md/lam combos
                if [ "$md" -ne 100 ] && [ "$md" -ne 200 ]; then
                    continue
                fi
                if [ "$lam" != "1.0" ] && [ "$lam" != "1.5" ] && [ "$lam" != "2.0" ]; then
                    continue
                fi
            fi
            
            OUTPUT_DIR="$BASE_OUTPUT/md${MD_PAD}_lam${LAM_STR}_gain${GAIN_STR}"
            
            echo "" | tee -a "$LOG_FILE"
            echo "[$COUNT/$TOTAL] md=$md, lam=$lam, gain=$gain, output=$OUTPUT_DIR" | tee -a "$LOG_FILE"
            echo "------------------------------------------------" | tee -a "$LOG_FILE"
            
            if [ -f "$OUTPUT_DIR/metrics.json" ]; then
                echo "[SKIP] $OUTPUT_DIR already exists" | tee -a "$LOG_FILE"
                continue
            fi
            
            START_TS=$(date +%s)
            set +e
            $PYTHON "$TRAIN_SCRIPT" \
                --training-path "$DATAMART" \
                --output-dir "$OUTPUT_DIR" \
                --feature-importance-path "$FEATURE_IMPORTANCE" \
                --feature-prune-top-n 0 \
                --tp-pct 0.01 --sl-pct -0.02 \
                --oot-valid-days 100 \
                --rank-weights "0.6,0.3,0.1" \
                --min-data-in-leaf "$md" \
                --lambda-l1 "$lam" \
                --lambda-l2 "$lam" \
                --min-gain-to-split "$gain" \
                --num-leaves 63 --max-depth 6 \
                --learning-rate 0.02 \
                --feature-fraction 0.6 --bagging-fraction 0.8 --bagging-freq 1 \
                --scale-pos-weight -1.0 \
                --oversample-ratio 0.0 \
                --early-stopping-rounds 200 \
                --n-estimators 4000 \
                --topk-list "5,10,20" \
                2>&1 | tee -a "$LOG_FILE"
            
            EXIT_CODE=$?
            set -e
            END_TS=$(date +%s)
            DURATION=$((END_TS - START_TS))
            
            if [ $EXIT_CODE -eq 0 ]; then
                echo "[DONE] md=$md, lam=$lam, gain=$gain, duration=${DURATION}s" | tee -a "$LOG_FILE"
            else
                FAILED=$((FAILED + 1))
                echo "[FAIL] md=$md, lam=$lam, gain=$gain (exit=$EXIT_CODE, duration=${DURATION}s)" | tee -a "$LOG_FILE"
            fi
        done
    done
done

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Grid Search Fine Complete - $(date)" | tee -a "$LOG_FILE"
echo "Total attempted: $COUNT, Failed: $FAILED" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# Collect and display results
COLLECT_SCRIPT="$BASE_OUTPUT/collect_results.py"
cat > "$COLLECT_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
"""Collect fine grid search results and print summary + Phase 1 vs Phase 2 comparison."""
import json
import math
import sys
from pathlib import Path

base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

results = []
for d in sorted(base.iterdir()):
    if not d.is_dir():
        continue
    metrics_file = d / "metrics.json"
    if not metrics_file.exists():
        continue
    try:
        m = json.loads(metrics_file.read_text())
        status = m.get("status", "?")
        oot_valid = m.get("metrics", {}).get("oot_valid", {})
        gap = m.get("metrics", {}).get("overfit_gap", {})
        portfolio = m.get("metrics", {}).get("portfolio_oot", {})
        model_info = m.get("model", {})
        model_params = model_info.get("params", {})
        
        cum_ret_ratio = portfolio.get("cumulative_net_return", -1)
        max_dd_ratio = portfolio.get("max_drawdown", -1)
        cum_ret_pct = round(cum_ret_ratio * 100, 2) if cum_ret_ratio != -1 else -1
        max_dd_pct = round(max_dd_ratio * 100, 2) if max_dd_ratio != -1 else -1
        
        mean_daily_ret = portfolio.get("mean_daily_net_return", 0)
        vol_daily = portfolio.get("volatility_daily", 0)
        if vol_daily and vol_daily > 0 and mean_daily_ret != 0:
            sharpe = round((mean_daily_ret / vol_daily) * math.sqrt(252), 3)
        else:
            sharpe = -1.0
        
        name = d.name
        parts = name.split("_")
        md = int(parts[0].replace("md", "").lstrip("0") or "0") if len(parts) > 0 else -1
        lam_str = parts[1].replace("lam", "").replace("_", ".") if len(parts) > 1 else "?"
        gain_str = parts[2].replace("gain", "").replace("_", ".") if len(parts) > 2 else "?"

        results.append({
            "dir": name,
            "min_data": model_params.get("min_data_in_leaf", md),
            "lambda": model_params.get("lambda_l1", lam_str),
            "min_gain": model_params.get("min_gain_to_split", gain_str),
            "status": status,
            "auc_oot": oot_valid.get("auc", -1),
            "aucpr_oot": oot_valid.get("aucpr", -1),
            "overfit_gap_auc": gap.get("auc_train_minus_oot", -1),
            "overfit_gap_aucpr": gap.get("aucpr_train_minus_oot", -1),
            "cum_return_pct": cum_ret_pct,
            "max_dd_pct": max_dd_pct,
            "sharpe": sharpe,
            "best_iteration": model_info.get("best_iteration", -1),
            "trading_days": portfolio.get("trading_days", -1),
        })
    except Exception as e:
        print(f"[WARN] Failed to parse {metrics_file}: {e}", file=sys.stderr)

if not results:
    print("No results found.")
    sys.exit(0)

# Sort by cumulative return descending
results.sort(key=lambda r: r["cum_return_pct"], reverse=True)

print(f"\n{'Dir':<32} {'md':>4} {'lam':>5} {'gain':>5} {'Status':>8} {'AUC_OOT':>8} {'CumRet%':>9} "
      f"{'MaxDD%':>8} {'Sharpe':>7} {'Gap_AUC':>8} {'BestIter':>8}")
print("-" * 115)
for r in results:
    print(f"{r['dir']:<32} {r['min_data']:>4} {str(r['lambda']):>5} {str(r['min_gain']):>5} {r['status']:>8} "
          f"{r['auc_oot']:>8.4f} {r['cum_return_pct']:>9.2f} {r['max_dd_pct']:>8.2f} "
          f"{r['sharpe']:>7.3f} {r['overfit_gap_auc']:>8.4f} {r['best_iteration']:>8}")

# Summary statistics
print("\n=== TOP 5 BY CUMULATIVE RETURN ===")
for i, r in enumerate(results[:5], 1):
    print(f"  {i}. {r['dir']}: CumRet={r['cum_return_pct']:.2f}%, DD={r['max_dd_pct']:.2f}%, "
          f"Sharpe={r['sharpe']:.3f}, AUC={r['auc_oot']:.4f}, Gap={r['overfit_gap_auc']:.4f}")

# Best by risk-adjusted
by_sharpe = sorted(results, key=lambda r: r["sharpe"], reverse=True)
print("\n=== TOP 5 BY SHARPE RATIO ===")
for i, r in enumerate(by_sharpe[:5], 1):
    print(f"  {i}. {r['dir']}: Sharp={r['sharpe']:.3f}, CumRet={r['cum_return_pct']:.2f}%, "
          f"DD={r['max_dd_pct']:.2f}%, AUC={r['auc_oot']:.4f}, Gap={r['overfit_gap_auc']:.4f}")

# Best by overfit gap
by_gap = sorted(results, key=lambda r: r["overfit_gap_auc"])
print("\n=== TOP 5 BY LOWEST OVERFIT GAP ===")
for i, r in enumerate(by_gap[:5], 1):
    print(f"  {i}. {r['dir']}: Gap={r['overfit_gap_auc']:.4f}, CumRet={r['cum_return_pct']:.2f}%, "
          f"DD={r['max_dd_pct']:.2f}%, Sharpe={r['sharpe']:.3f}")

print(f"\nTotal completed: {len(results)}")
print(f"Total failed: {sum(1 for r in results if r['status'] != 'PASS')}")
PYEOF

# Run the collection script
$PYTHON "$COLLECT_SCRIPT" "$BASE_OUTPUT" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Results collected. See: $LOG_FILE" | tee -a "$LOG_FILE"
