#!/usr/bin/env python3
"""
Ablation Study for BSJP v23 features.
Trains the model multiple times while excluding specific feature families
to understand which ones provide the most signal.
"""

import argparse
from pathlib import Path
import subprocess
import pandas as pd
import json
import sys

# Ensure we can import from the same directory
sys.path.append(str(Path(__file__).resolve().parent))

from _v23_policy_tools import (
    Policy,
    simulate_policy,
    load_predictions,
)

IDX_DIR = Path(__file__).resolve().parents[3]

# Mapping of families to column prefixes or specific lists
FAMILIES = {
    "broker": ["flow_", "ctx_", "tfl_", "mg_", "xc_", "sq_", "yp_", "pd_", "total_retail_", "localfund_", "bandar_", "cvd_"],
    "macro": ["nasdaq_", "nikkei_", "vix_", "usdidr_", "ihsg_"],
    "pre14": ["pre14_", "intensity_", "vol_ratio_", "close_to_vwap", "open_pm_to_vwap_am", "vwap_trend", "last_hour_above_vwap", "close_drive", "vol_above_vwap_pct"],
    "history": ["overnight_", "gapdown_", "gap_", "was_ara_", "ara_count_", "consecutive_ara_", "noara_streak_", "days_since_last_ara", "last_ara_return", "max_return_"],
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=IDX_DIR / "model/BSJP/v23b_ablation")
    parser.add_argument("--training-path", type=Path, default=IDX_DIR / "data/Level_2_Datamart/training_datamart_bsjp_close10_rebuild_v18like.parquet")
    parser.add_argument("--modules-dir", type=Path, default=IDX_DIR / "data/Level_1_Features/modules")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()

def get_all_feature_cols(modules_dir: Path) -> list[str]:
    all_cols = []
    for f in modules_dir.glob("*_features.parquet"):
        df = pd.read_parquet(f, columns=[]) # Fast read schema
        df = pd.read_parquet(f).head(0)
        cols = [c for c in df.columns if c not in ['date', 'ticker']]
        all_cols.extend(cols)
    return sorted(list(set(all_cols)))

def main():
    args = parse_args()
    all_features = get_all_feature_cols(args.modules_dir)
    print(f"Total features detected: {len(all_features)}")
    
    scenarios = ["baseline"] + list(FAMILIES.keys())
    
    results = []
    policy = Policy(max_positions=2, conviction_top_k=2, max_weight=0.25, max_pre14_market_cost_est=0.03, adaptive_quantile=0.85)

    for scenario in scenarios:
        scenario_dir = args.output_dir / scenario
        
        blacklist = []
        if scenario != "baseline":
            prefixes = FAMILIES[scenario]
            blacklist = [c for c in all_features if any(c.startswith(p) for p in prefixes)]
            print(f"\nScenario: No-{scenario} (Blacklisting {len(blacklist)} features)")
        else:
            print(f"\nScenario: Baseline")
            
        cmd = [
            sys.executable, str(IDX_DIR / "edges/bsjp_overnight_sl2/scripts/train_lightgbm.py"),
            "--training-path", str(args.training_path),
            "--output-dir", str(scenario_dir),
            "--feature-modules-dir", str(args.modules_dir),
            "--oot-valid-days", "100",
            "--min-data-in-leaf", "100",
            "--lambda-l1", "1.0",
            "--lambda-l2", "1.5",
            "--feature-prune-top-n", "0",
            "--tp-pct", "0.01",
            "--sl-pct", "-0.02",
            "--feature-blacklist", ",".join(blacklist)
        ]
        
        if args.run:
            if not (scenario_dir / "metrics.json").exists():
                print(f"  Running training...")
                subprocess.run(cmd, check=True)
            
            # Evaluate policy
            pred = load_predictions(model_dir=scenario_dir, modules_dir=args.modules_dir)
            _, _, summary = simulate_policy(pred, policy)
            summary["scenario"] = scenario
            
            # Load LGB metrics
            with open(scenario_dir / "metrics.json") as f:
                m = json.load(f)
                summary["oot_auc"] = m.get("metrics", {}).get("oot_valid", {}).get("auc", 0.0)
                summary["trees"] = m.get("model_info", {}).get("tree_count", 0)
            
            results.append(summary)
            auc_val = summary.get('oot_auc', 0.0)
            cum_ret = summary.get('cumulative_net_return', 0.0)
            max_dd = summary.get('max_drawdown', 0.0)
            print(f"  Result: AUC={auc_val:.4f}, CumNet={cum_ret:.2%}, MaxDD={max_dd:.2%}")
        else:
            print(f"  [Dry Run] {' '.join(cmd)}")

    if results:
        res_df = pd.DataFrame(results)
        print("\nAblation Study Results:")
        print(res_df[['scenario', 'trees', 'oot_auc', 'cumulative_net_return', 'max_drawdown', 'trading_days']])
        res_df.to_csv(args.output_dir / "ablation_summary.csv", index=False)

if __name__ == '__main__':
    main()
