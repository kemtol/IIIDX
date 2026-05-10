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
    add_risk_bands,
    load_predictions,
    simulate_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Worst-day attribution for v23 BSJP candidate.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--modules-dir", type=Path, default=DEFAULT_MODULES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--policy-name", type=str, default="baseline")
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--max-weight", type=float, default=0.25)
    parser.add_argument("--max-pre14-cost", type=float, default=0.030)
    parser.add_argument("--adaptive-quantile", type=float, default=0.85)
    parser.add_argument("--score-gap-min", type=float, default=0.0)
    parser.add_argument("--max-pre14-tick-pct", type=float, default=-1.0)
    parser.add_argument("--exclude-pre14-ara-like", action="store_true")
    parser.add_argument("--loss-threshold", type=float, default=-0.02)
    parser.add_argument("--bottom-n", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred = load_predictions(args.model_dir, args.modules_dir)
    policy = Policy(
        max_positions=args.max_positions,
        conviction_top_k=args.max_positions,
        max_weight=args.max_weight,
        max_pre14_market_cost_est=args.max_pre14_cost,
        adaptive_quantile=args.adaptive_quantile,
        score_gap_min=args.score_gap_min,
        max_pre14_tick_pct=(None if args.max_pre14_tick_pct < 0 else args.max_pre14_tick_pct),
        exclude_pre14_ara_like=args.exclude_pre14_ara_like,
    )
    daily, trades, summary = simulate_policy(pred, policy)

    max_abs_diff = None
    pos_mismatch = None
    if args.policy_name == "baseline":
        official = pd.read_parquet(args.model_dir / "portfolio_daily.parquet")
        official["date"] = pd.to_datetime(official["date"]).dt.normalize()
        check = daily.merge(official[["date", "positions", "net_return"]], on="date", suffixes=("_script", "_official"))
        max_abs_diff = float((check["net_return_script"] - check["net_return_official"]).abs().max())
        pos_mismatch = int((check["positions_script"] != check["positions_official"]).sum())
        if max_abs_diff > 1e-10 or pos_mismatch:
            raise SystemExit(
                f"Baseline policy mismatch: max_abs_diff={max_abs_diff}, pos_mismatch={pos_mismatch}"
            )

    worst_dates = set(daily.loc[daily["net_return"] <= args.loss_threshold, "date"])
    bottom_dates = set(daily.nsmallest(args.bottom_n, "net_return")["date"])
    selected_dates = sorted(worst_dates | bottom_dates)

    daily_out = daily[daily["date"].isin(selected_dates)].sort_values("net_return").copy()
    trades_out = trades[trades["date"].isin(selected_dates)].copy()
    trades_out = add_risk_bands(trades_out)

    key_cols = [
        "date",
        "ticker",
        "pred_proba",
        "pred_rank",
        "score",
        "weight",
        "trade_gross_return",
        "trade_net_return",
        "weighted_net",
        "entry_price",
        "exit_price",
        "pre14_market_cost_est",
        "pre14_tick_pct",
        "pre14_spread_cost_est",
        "pre14_range_pct",
        "pre14_return_from_prev_close",
        "pre14_ara_distance_pct",
        "ara_bucket",
        "price_tier",
        "vix_regime",
        "ihsg_prev_return",
        "usdidr_5d_return",
    ]
    key_cols = [c for c in key_cols if c in trades_out.columns]
    trades_out = trades_out.sort_values(["date", "weighted_net"], ascending=[True, True])

    by_bucket = (
        trades_out.groupby("ara_bucket", dropna=False)
        .agg(
            trades=("ticker", "count"),
            mean_trade_net=("trade_net_return", "mean"),
            weighted_net=("weighted_net", "sum"),
            mean_pred=("pred_proba", "mean"),
        )
        .reset_index()
        .sort_values("weighted_net")
    )
    by_price = (
        trades_out.groupby("price_tier", dropna=False)
        .agg(
            trades=("ticker", "count"),
            mean_trade_net=("trade_net_return", "mean"),
            weighted_net=("weighted_net", "sum"),
        )
        .reset_index()
        .sort_values("weighted_net")
    )
    by_cost = (
        trades_out.groupby("pre14_market_cost_est_band", dropna=False)
        .agg(
            trades=("ticker", "count"),
            mean_trade_net=("trade_net_return", "mean"),
            weighted_net=("weighted_net", "sum"),
        )
        .reset_index()
        .sort_values("weighted_net")
        if "pre14_market_cost_est_band" in trades_out.columns
        else pd.DataFrame()
    )

    prefix = args.output_dir / args.model_dir.name
    suffix = f"{args.model_dir.name}_{args.policy_name}"
    daily_path = prefix.with_name(f"{suffix}_worst_days_20260509.csv")
    trade_path = prefix.with_name(f"{suffix}_worst_day_pick_attribution_20260509.csv")
    bucket_path = prefix.with_name(f"{suffix}_worst_day_bucket_summary_20260509.csv")
    price_path = prefix.with_name(f"{suffix}_worst_day_price_summary_20260509.csv")
    cost_path = prefix.with_name(f"{suffix}_worst_day_cost_summary_20260509.csv")
    summary_path = prefix.with_name(f"{suffix}_worst_day_summary_20260509.json")

    daily_out.to_csv(daily_path, index=False)
    trades_out[key_cols].to_csv(trade_path, index=False)
    by_bucket.to_csv(bucket_path, index=False)
    by_price.to_csv(price_path, index=False)
    if not by_cost.empty:
        by_cost.to_csv(cost_path, index=False)

    report = {
        "model_dir": str(args.model_dir),
        "policy_name": args.policy_name,
        "policy": policy.__dict__,
        "summary": summary,
        "baseline_match": {"max_abs_net_return_diff": max_abs_diff, "position_mismatch_days": pos_mismatch},
        "loss_threshold": args.loss_threshold,
        "worst_day_count": int(len(daily_out)),
        "worst_dates": [str(pd.Timestamp(d).date()) for d in selected_dates],
        "worst_total_weighted_net": float(trades_out["weighted_net"].sum()) if not trades_out.empty else 0.0,
        "worst_bucket_summary_path": str(bucket_path),
        "worst_pick_attribution_path": str(trade_path),
    }
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
