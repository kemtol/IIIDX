#!/usr/bin/env python3
"""
Rolling Retrain Validation for BSJP v23 policies.
This script coordinates executing `train_lightgbm.py` iteratively over a moving 
OOT window to validate stability across time, preventing single-window overfitting.
"""

import argparse
from pathlib import Path
import subprocess
import shutil
import json
import pandas as pd
from datetime import datetime
import sys

# Ensure we can import from the same directory
sys.path.append(str(Path(__file__).resolve().parent))

from _v23_policy_tools import (
    Policy,
    simulate_policy,
    load_predictions,
)

IDX_DIR = Path(__file__).resolve().parents[3]

def generate_windows(trading_days: list[datetime], num_windows: int, step_days: int, oot_days: int) -> list[dict]:
    """
    Generate date bounds for rolling validation windows.
    trading_days must be sorted in ascending order.
    """
    trading_days = sorted(trading_days)
    total_days = len(trading_days)
    
    # Ensure we have enough days for the furthest window's OOT + at least 1 day of train
    max_shift = (num_windows - 1) * step_days
    required_days = max_shift + oot_days + 1
    
    if total_days < required_days:
        raise ValueError(f"Insufficient trading days. Need at least {required_days}, got {total_days}")
    
    windows = []
    for i in range(num_windows):
        shift = i * step_days
        
        oot_end_idx = total_days - 1 - shift
        oot_start_idx = oot_end_idx - oot_days + 1
        train_end_idx = oot_start_idx - 1
        
        window = {
            "window_idx": i,
            "train_end_date": trading_days[train_end_idx],
            "oot_start_date": trading_days[oot_start_idx],
            "oot_end_date": trading_days[oot_end_idx],
            "oot_dates": trading_days[oot_start_idx : oot_end_idx + 1]
        }
        windows.append(window)
        
    return windows

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-output-dir", type=Path, default=IDX_DIR / "model/BSJP/v23b_rolling")
    parser.add_argument("--windows", type=int, default=5, help="Number of rolling windows to run")
    parser.add_argument("--step-days", type=int, default=20, help="Trading days to shift window")
    parser.add_argument("--base-oot-days", type=int, default=100, help="OOT days per window")
    parser.add_argument("--training-path", type=Path, default=IDX_DIR / "data/Level_2_Datamart/training_datamart_bsjp_close10_rebuild_v18like.parquet")
    parser.add_argument("--modules-dir", type=Path, default=IDX_DIR / "data/Level_1_Features/modules")
    parser.add_argument("--run", action="store_true", help="Actually run the trainings")
    return parser.parse_args()

def get_trading_days(training_path: Path) -> list[datetime]:
    df = pd.read_parquet(training_path, columns=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    dates = df["date"].dropna().unique()
    return sorted(pd.to_datetime(dates).tolist())

def main():
    args = parse_args()
    
    trading_days = get_trading_days(args.training_path)
    print(f"Loaded {len(trading_days)} trading days from datamart.")
    
    windows = generate_windows(
        trading_days=trading_days,
        num_windows=args.windows,
        step_days=args.step_days,
        oot_days=args.base_oot_days
    )
    
    print(f"Rolling validation plan: {args.windows} windows, shifted by {args.step_days} days.")
    
    if args.run:
        args.base_output_dir.mkdir(parents=True, exist_ok=True)
    
    for w in windows:
        window_dir = args.base_output_dir / f"window_{w['window_idx']}"
        max_date_str = w['oot_end_date'].strftime('%Y-%m-%d')
        oot_start_str = w['oot_start_date'].strftime('%Y-%m-%d')
        
        print(f"\n=== Window {w['window_idx']} ===")
        print(f"  OOT: {oot_start_str} to {max_date_str}")
        
        cmd = [
            sys.executable, str(IDX_DIR / "edges/bsjp_overnight_sl2/scripts/train_lightgbm.py"),
            "--training-path", str(args.training_path),
            "--output-dir", str(window_dir),
            "--feature-modules-dir", str(args.modules_dir),
            "--max-date", max_date_str,
            "--oot-valid-days", str(args.base_oot_days),
            "--min-data-in-leaf", "100",
            "--lambda-l1", "1.0",
            "--lambda-l2", "1.5",
            "--feature-prune-top-n", "0",
            "--tp-pct", "0.01",
            "--sl-pct", "-0.02"
        ]
        
        if args.run:
            if (window_dir / "metrics.json").exists():
                print(f"  Window {w['window_idx']} already trained. Skipping.")
            else:
                print(f"  Running: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
        else:
            print(f"  [Dry Run] {' '.join(cmd)}")
            
    if args.run:
        print("\n=== Aggregating Rolling Validation Results ===")
        metrics = []
        policy = Policy(max_positions=2, conviction_top_k=2, max_weight=0.25, max_pre14_market_cost_est=0.03, adaptive_quantile=0.85)
        for w in windows:
            window_dir = args.base_output_dir / f"window_{w['window_idx']}"
            if not window_dir.exists():
                continue
            # Note: load_predictions needs modules_dir and looks for valid_predictions.parquet in model_dir
            pred = load_predictions(model_dir=window_dir, modules_dir=args.modules_dir)
            daily, _, summary = simulate_policy(pred, policy)
            summary["window_idx"] = w['window_idx']
            summary["oot_start"] = w['oot_start_date'].strftime('%Y-%m-%d')
            summary["oot_end"] = w['oot_end_date'].strftime('%Y-%m-%d')
            metrics.append(summary)
            
        if metrics:
            res = pd.DataFrame(metrics)
            print("\nPer-Window Results:")
            print(res[['window_idx', 'oot_start', 'oot_end', 'days', 'trading_days', 'cumulative_net_return', 'max_drawdown', 'mean_daily_net_return']])
            print("\nAggregate Metrics (Median across windows):")
            print(res[['cumulative_net_return', 'max_drawdown', 'mean_daily_net_return']].median())
            
            res.to_csv(args.base_output_dir / "rolling_validation_summary.csv", index=False)

if __name__ == '__main__':
    main()
