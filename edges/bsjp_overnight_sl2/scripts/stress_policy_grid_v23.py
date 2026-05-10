#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _v23_policy_tools import (
    DEFAULT_LOG_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODULES_DIR,
    Policy,
    load_predictions,
    simulate_policy,
    summarize_window,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Policy stress grid for v23 BSJP candidate.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--modules-dir", type=Path, default=DEFAULT_MODULES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--micro-grid", action="store_true", help="Run a hand-picked small policy set.")
    parser.add_argument("--quick-grid", action="store_true", help="Run a smaller first-pass grid for fast iteration.")
    parser.add_argument("--max-rows", type=int, default=0, help="Debug limit for grid rows; 0 = full grid.")
    return parser.parse_args()


def tick_thresholds(pred: pd.DataFrame) -> dict[str, float | None]:
    x = pd.to_numeric(pred.get("pre14_tick_pct"), errors="coerce")
    return {
        "none": None,
        "p90": float(x.quantile(0.90)),
        "p95": float(x.quantile(0.95)),
    }


def robust_score(row: pd.Series, baseline_active_days: int) -> float:
    active_ratio = row["trading_days"] / max(baseline_active_days, 1)
    penalty = 0.0
    if active_ratio < 0.60:
        penalty += (0.60 - active_ratio) * 2.0
    if row["max_drawdown"] < -0.20:
        penalty += abs(row["max_drawdown"] + 0.20) * 2.0
    if row["worst_30d_cum"] < 0:
        penalty += abs(row["worst_30d_cum"])
    return float(row["median_30d_cum"] + row["cumulative_net_return"] * 0.10 + row["max_drawdown"] - penalty)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred = load_predictions(args.model_dir, args.modules_dir)
    baseline_policy = Policy()
    baseline_daily, _, baseline_summary = simulate_policy(pred, baseline_policy)
    official = pd.read_parquet(args.model_dir / "portfolio_daily.parquet")
    official["date"] = pd.to_datetime(official["date"]).dt.normalize()
    check = baseline_daily.merge(official[["date", "positions", "net_return"]], on="date", suffixes=("_script", "_official"))
    max_abs_diff = float((check["net_return_script"] - check["net_return_official"]).abs().max())
    pos_mismatch = int((check["positions_script"] != check["positions_official"]).sum())
    if max_abs_diff > 1e-10 or pos_mismatch:
        raise SystemExit(
            f"Baseline policy mismatch: max_abs_diff={max_abs_diff}, pos_mismatch={pos_mismatch}"
        )

    tick_map = tick_thresholds(pred)
    if args.micro_grid:
        grid = [
            (3, 0.25, 0.030, 0.85, 0.000, "none", False),  # baseline
            (3, 0.20, 0.030, 0.85, 0.000, "none", False),
            (2, 0.25, 0.030, 0.85, 0.000, "none", False),
            (2, 0.20, 0.030, 0.85, 0.000, "none", False),
            (3, 0.25, 0.025, 0.85, 0.000, "none", False),
            (3, 0.20, 0.025, 0.85, 0.000, "none", False),
            (3, 0.25, 0.030, 0.90, 0.000, "none", False),
            (3, 0.25, 0.030, 0.85, 0.005, "none", False),
            (3, 0.25, 0.030, 0.85, 0.000, "p95", False),
            (3, 0.25, 0.030, 0.85, 0.000, "none", True),
            (2, 0.20, 0.025, 0.90, 0.000, "none", False),
            (2, 0.20, 0.025, 0.85, 0.005, "none", False),
            (2, 0.20, 0.030, 0.90, 0.000, "p95", False),
            (3, 0.20, 0.025, 0.90, 0.005, "p95", False),
            (3, 0.25, 0.025, 0.90, 0.005, "p95", False),
            (2, 0.25, 0.025, 0.85, 0.000, "none", True),
        ]
    elif args.quick_grid:
        grid_values = ([2, 3], [0.20, 0.25], [0.025, 0.030], [0.85, 0.90], [0.0, 0.005], ["none", "p95"], [False, True])
        grid = list(itertools.product(*grid_values))
    else:
        grid_values = ([1, 2, 3], [0.15, 0.20, 0.25], [0.020, 0.025, 0.030], [0.80, 0.85, 0.90], [0.0, 0.005, 0.010], ["none", "p90", "p95"], [False, True])
        grid = list(itertools.product(*grid_values))
    if args.max_rows > 0:
        grid = grid[: args.max_rows]

    rows = []
    for i, (k, max_weight, max_cost, quantile, score_gap, tick_name, no_ara_like) in enumerate(grid, start=1):
        policy = Policy(
            max_positions=k,
            conviction_top_k=k,
            max_weight=max_weight,
            max_pre14_market_cost_est=max_cost,
            adaptive_quantile=quantile,
            score_gap_min=score_gap,
            max_pre14_tick_pct=tick_map[tick_name],
            exclude_pre14_ara_like=no_ara_like,
        )
        daily, trades, summary = simulate_policy(pred, policy)
        windows30 = []
        windows60 = []
        for end in range(30, len(daily) + 1, 10):
            windows30.append(summarize_window(daily.iloc[:end], 30)["cum_net"])
        for end in range(60, len(daily) + 1, 10):
            windows60.append(summarize_window(daily.iloc[:end], 60)["cum_net"])
        row = {
            "grid_id": i,
            "k": k,
            "max_weight": max_weight,
            "max_pre14_cost": max_cost,
            "adaptive_quantile": quantile,
            "score_gap_min": score_gap,
            "tick_veto": tick_name,
            "max_pre14_tick_pct": tick_map[tick_name],
            "exclude_pre14_ara_like": no_ara_like,
            "days": summary["days"],
            "trading_days": summary["trading_days"],
            "active_day_ratio": summary["trading_days"] / max(baseline_summary["trading_days"], 1),
            "mean_daily_net_return": summary["mean_daily_net_return"],
            "cumulative_net_return": summary["cumulative_net_return"],
            "max_drawdown": summary["max_drawdown"],
            "win_rate_days": summary["win_rate_days"],
            "volatility_daily": summary["volatility_daily"],
            "net_expectancy_per_trade": summary["net_expectancy_per_trade"],
            "median_30d_cum": float(np.nanmedian(windows30)) if windows30 else np.nan,
            "worst_30d_cum": float(np.nanmin(windows30)) if windows30 else np.nan,
            "median_60d_cum": float(np.nanmedian(windows60)) if windows60 else np.nan,
            "worst_60d_cum": float(np.nanmin(windows60)) if windows60 else np.nan,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out["robust_score"] = out.apply(lambda r: robust_score(r, baseline_summary["trading_days"]), axis=1)
    out = out.sort_values(["robust_score", "max_drawdown", "cumulative_net_return"], ascending=[False, False, False])

    suffix = "policy_stress_micro_grid" if args.micro_grid else ("policy_stress_quick_grid" if args.quick_grid else "policy_stress_grid")
    out_path = args.output_dir / f"{args.model_dir.name}_{suffix}_20260509.csv"
    summary_path = args.output_dir / f"{args.model_dir.name}_{suffix}_summary_20260509.json"
    out.to_csv(out_path, index=False)
    top = out.head(20).copy()
    report = {
        "model_dir": str(args.model_dir),
        "baseline_summary": baseline_summary,
        "baseline_match": {"max_abs_net_return_diff": max_abs_diff, "position_mismatch_days": pos_mismatch},
        "grid_rows": int(len(out)),
        "output_path": str(out_path),
        "top20": top.to_dict("records"),
    }
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["grid_rows", "output_path", "baseline_match"]}, indent=2))
    print(top.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
