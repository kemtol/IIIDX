#!/bin/bash
# =============================================================================
# Grid Search Phase 1: Coarse Grid
# =============================================================================
# Tests 16 combinations of min_data_in_leaf × lambda_l1/l2
# Fixes: min_gain_to_split=0.05 (middle-ground)
# Uses close10 objective datamart (best performing objective)
#
# Usage:
#   cd /home-ssd/mkemalw/Projects/MMMACHINE/idx
#   bash edges/bsjp_overnight_sl2/scripts/grid_search_coarse.sh
#
# Output: model/BSJP/bsjp_grid_coarse/md{min_data}_lam{lambda}/
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# Virtual environment is at MMMACHINE/.venv, one level above idx/
VENV_DIR="$(cd "$PROJECT_ROOT/.." && pwd)/.venv"
PYTHON="$VENV_DIR/bin/python"
TRAIN_SCRIPT="$SCRIPT_DIR/train_lightgbm.py"

# --- Grid Parameters ---
MIN_DATA_VALUES=(50 100 200 500)
LAMBDA_VALUES=(0.1 0.5 1.0 2.0)
MIN_GAIN=0.05  # fixed for coarse grid

# --- Shared Arguments ---
DATAMART="$PROJECT_ROOT/data/Level_2_Datamart/training_datamart_bsjp_close10_v10.parquet"
BASE_OUTPUT="$PROJECT_ROOT/model/BSJP/bsjp_grid_coarse"
FEATURE_IMPORTANCE="$PROJECT_ROOT/model/BSJP/bsjp_v9d_classifier_ref/feature_importance.csv"

# Ensure base output dir exists
mkdir -p "$BASE_OUTPUT"

# Log file
LOG_FILE="$BASE_OUTPUT/grid_search_coarse.log"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Grid Search Coarse - $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Grid: min_data_in_leaf = ${MIN_DATA_VALUES[*]}" | tee -a "$LOG_FILE"
echo "      lambda_l1/l2     = ${LAMBDA_VALUES[*]}" | tee -a "$LOG_FILE"
echo "      min_gain_to_split = $MIN_GAIN (fixed)" | tee -a "$LOG_FILE"
echo "Datamart: $DATAMART" | tee -a "$LOG_FILE"
echo "Total combos: $(( ${#MIN_DATA_VALUES[@]} * ${#LAMBDA_VALUES[@]} ))" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

TOTAL=$(( ${#MIN_DATA_VALUES[@]} * ${#LAMBDA_VALUES[@]} ))
COUNT=0
FAILED=0

for md in "${MIN_DATA_VALUES[@]}"; do
    for lam in "${LAMBDA_VALUES[@]}"; do
        COUNT=$((COUNT + 1))
        
        # Output dir name: pad md to 3 digits, use underscore for lambda decimal
        LAM_STR="${lam/./_}"
        MD_PAD=$(printf "%03d" "$md")
        OUTPUT_DIR="$BASE_OUTPUT/md${MD_PAD}_lam${LAM_STR}"
        
        echo "" | tee -a "$LOG_FILE"
        echo "[$COUNT/$TOTAL] md=$md, lambda=$lam, output=$OUTPUT_DIR" | tee -a "$LOG_FILE"
        echo "------------------------------------------------" | tee -a "$LOG_FILE"
        
        # Skip if already completed (check for metrics.json)
        if [ -f "$OUTPUT_DIR/metrics.json" ]; then
            EXISTING_STATUS=$(grep -o '"status":"[^"]*"' "$OUTPUT_DIR/metrics.json" | head -1 || echo "unknown")
            echo "[SKIP] $OUTPUT_DIR already exists (status=$EXISTING_STATUS)" | tee -a "$LOG_FILE"
            continue
        fi
        
        # Run training
        START_TS=$(date +%s)
        set +e  # allow failure
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
            --min-gain-to-split "$MIN_GAIN" \
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
            echo "[DONE] md=$md, lambda=$lam, duration=${DURATION}s" | tee -a "$LOG_FILE"
        else
            FAILED=$((FAILED + 1))
            echo "[FAIL] md=$md, lambda=$lam (exit=$EXIT_CODE, duration=${DURATION}s)" | tee -a "$LOG_FILE"
        fi
    done
done

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Grid Search Complete - $(date)" | tee -a "$LOG_FILE"
echo "Total: $TOTAL, Failed: $FAILED" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# Collect results summary
echo "" | tee -a "$LOG_FILE"
echo "=== RESULTS SUMMARY ===" | tee -a "$LOG_FILE"
COLLECT_SCRIPT="$BASE_OUTPUT/collect_results.py"
cat > "$COLLECT_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
"""Collect grid search results from metrics.json files and print summary table."""
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
        
        # Portfolio values are ratios (e.g., 7.36 = +636%), convert to percentages
        cum_ret_ratio = portfolio.get("cumulative_net_return", -1)
        max_dd_ratio = portfolio.get("max_drawdown", -1)
        cum_ret_pct = round(cum_ret_ratio * 100, 2) if cum_ret_ratio != -1 else -1
        max_dd_pct = round(max_dd_ratio * 100, 2) if max_dd_ratio != -1 else -1
        
        # Compute Sharpe ratio from daily values
        mean_daily_ret = portfolio.get("mean_daily_net_return", 0)
        vol_daily = portfolio.get("volatility_daily", 0)
        if vol_daily and vol_daily > 0 and mean_daily_ret != 0:
            sharpe = round((mean_daily_ret / vol_daily) * math.sqrt(252), 3)
        else:
            sharpe = -1.0
        
        # Parse dir name for display
        name = d.name
        parts = name.split("_")
        md = int(parts[0].replace("md", "").lstrip("0") or "0") if len(parts) > 0 else -1
        lam_str = parts[1].replace("lam", "").replace("_", ".") if len(parts) > 1 else "?"

        results.append({
            "dir": name,
            "min_data": model_params.get("min_data_in_leaf", md),
            "lambda": model_params.get("lambda_l1", lam_str),
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

print(f"\n{'Dir':<30} {'md':>5} {'lam':>6} {'Status':>10} {'AUC_OOT':>8} {'AUCPR':>8} "
      f"{'Gap_AUC':>8} {'Gap_PR':>8} {'CumRet%':>9} {'MaxDD%':>8} {'Sharpe':>7} "
      f"{'BestIter':>8} {'Trades':>7}")
print("-" * 130)
for r in results:
    print(f"{r['dir']:<30} {r['min_data']:>5} {str(r['lambda']):>6} {r['status']:>10} "
          f"{r['auc_oot']:>8.4f} {r['aucpr_oot']:>8.4f} "
          f"{r['overfit_gap_auc']:>8.4f} {r['overfit_gap_aucpr']:>8.4f} "
          f"{r['cum_return_pct']:>9.2f} {r['max_dd_pct']:>8.2f} {r['sharpe']:>7.3f} "
          f"{r['best_iteration']:>8} {r['trading_days']:>7}")

# Summary statistics
print("\n=== SUMMARY STATISTICS ===")
print(f"Total completed: {len(results)}")
if results:
    best = results[0]
    print(f"Best by CumReturn: {best['dir']} (md={best['min_data']}, lam={best['lambda']}, "
          f"cum_ret={best['cum_return_pct']:.2f}%, dd={best['max_dd_pct']:.2f}%, sharpe={best['sharpe']:.3f})")
    
    # Best by Sharpe
    by_sharpe = sorted(results, key=lambda r: r["sharpe"], reverse=True)
    best_s = by_sharpe[0]
    print(f"Best by Sharpe: {best_s['dir']} (md={best_s['min_data']}, lam={best_s['lambda']}, "
          f"sharpe={best_s['sharpe']:.3f}, cum_ret={best_s['cum_return_pct']:.2f}%)")
    
    # Best by overfit gap (lowest gap)
    by_gap = sorted(results, key=lambda r: r["overfit_gap_auc"])
    best_g = by_gap[0]
    print(f"Best by OverfitGap: {best_g['dir']} (md={best_g['min_data']}, lam={best_g['lambda']}, "
          f"gap={best_g['overfit_gap_auc']:.4f}, cum_ret={best_g['cum_return_pct']:.2f}%)")
    
    # Best by AUC OOT
    by_auc = sorted(results, key=lambda r: r["auc_oot"], reverse=True)
    best_a = by_auc[0]
    print(f"Best by AUC OOT: {best_a['dir']} (md={best_a['min_data']}, lam={best_a['lambda']}, "
          f"auc={best_a['auc_oot']:.4f}, cum_ret={best_a['cum_return_pct']:.2f}%)")
PYEOF

# Run the collection script
$PYTHON "$COLLECT_SCRIPT" "$BASE_OUTPUT" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Results collected. See: $LOG_FILE" | tee -a "$LOG_FILE"
