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
