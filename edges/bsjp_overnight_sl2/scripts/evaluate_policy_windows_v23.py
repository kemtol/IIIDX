#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _v23_policy_tools import (
    DEFAULT_LOG_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODULES_DIR,
    Policy,
    load_predictions,
    simulate_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate v23 policies across rolling OOT subwindows.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--modules-dir", type=Path, default=DEFAULT_MODULES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_LOG_DIR)
    return parser.parse_args()


def window_metrics(daily: pd.DataFrame, window: int, step: int) -> pd.DataFrame:
    rows = []
    daily = daily.sort_values("date").reset_index(drop=True)
    for start in range(0, max(len(daily) - window + 1, 0), step):
        sub = daily.iloc[start : start + window].copy()
        eq = (1.0 + sub["net_return"]).cumprod()
        dd = eq / eq.cummax() - 1.0
        rows.append(
            {
                "window": window,
                "start_date": sub["date"].iloc[0],
                "end_date": sub["date"].iloc[-1],
                "days": len(sub),
                "trading_days": int((sub["positions"] > 0).sum()),
                "cum_net": float(eq.iloc[-1] - 1.0),
                "mean_daily_net": float(sub["net_return"].mean()),
                "max_drawdown": float(dd.min()),
                "win_rate": float((sub["net_return"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred = load_predictions(args.model_dir, args.modules_dir)

    policies = {
        "baseline_k3_w25": Policy(max_positions=3, conviction_top_k=3, max_weight=0.25),
        "quickwin_k2_w25": Policy(max_positions=2, conviction_top_k=2, max_weight=0.25),
        "lowerdd_k2_w20_q90": Policy(max_positions=2, conviction_top_k=2, max_weight=0.20, adaptive_quantile=0.90),
    }
    daily_frames = []
    summary_rows = []
    window_frames = []

    for name, policy in policies.items():
        daily, trades, summary = simulate_policy(pred, policy)
        daily = daily.assign(policy=name)
        daily_frames.append(daily)
        summary_rows.append({"policy": name, **summary})
        for window, step in [(7, 1), (20, 5), (30, 5), (60, 10)]:
            w = window_metrics(daily, window=window, step=step)
            w["policy"] = name
            window_frames.append(w)

    daily_all = pd.concat(daily_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    windows = pd.concat(window_frames, ignore_index=True)

    window_summary = (
        windows.groupby(["policy", "window"], as_index=False)
        .agg(
            windows=("cum_net", "count"),
            positive_rate=("cum_net", lambda s: float((s > 0).mean())),
            median_cum_net=("cum_net", "median"),
            worst_cum_net=("cum_net", "min"),
            median_maxdd=("max_drawdown", "median"),
            worst_maxdd=("max_drawdown", "min"),
            median_trading_days=("trading_days", "median"),
        )
        .sort_values(["window", "median_cum_net"], ascending=[True, False])
    )

    prefix = args.output_dir / args.model_dir.name
    daily_path = prefix.with_name(f"{args.model_dir.name}_policy_daily_compare_20260509.csv")
    summary_path = prefix.with_name(f"{args.model_dir.name}_policy_summary_compare_20260509.csv")
    windows_path = prefix.with_name(f"{args.model_dir.name}_policy_rolling_windows_20260509.csv")
    window_summary_path = prefix.with_name(f"{args.model_dir.name}_policy_rolling_window_summary_20260509.csv")
    report_path = prefix.with_name(f"{args.model_dir.name}_policy_rolling_report_20260509.json")

    daily_all.to_csv(daily_path, index=False)
    summary.to_csv(summary_path, index=False)
    windows.to_csv(windows_path, index=False)
    window_summary.to_csv(window_summary_path, index=False)

    report = {
        "model_dir": str(args.model_dir),
        "policies": {name: policy.__dict__ for name, policy in policies.items()},
        "summary": summary.to_dict("records"),
        "window_summary": window_summary.to_dict("records"),
        "artifacts": {
            "daily": str(daily_path),
            "summary": str(summary_path),
            "windows": str(windows_path),
            "window_summary": str(window_summary_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(summary.to_string(index=False))
    print()
    print(window_summary.to_string(index=False))


if __name__ == "__main__":
    main()
