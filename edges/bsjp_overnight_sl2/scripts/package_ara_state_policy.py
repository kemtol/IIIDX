#!/usr/bin/env python3
"""
Package BSJP ARA-state policy variants on top of an existing v19d model.

This is a policy-layer artifact builder, not a retraining script. It reuses the
existing v19d predictions, joins the latest preclose14 ARA-state module, applies
a frozen veto policy, and writes a model-like directory with metrics, daily PnL,
trades, policy summary, and charts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


IDX_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_MODEL = IDX_DIR / "model" / "BSJP" / "bsjp_v19d_close10_preclose14_orb_md100_l21.5"
DEFAULT_OUTPUT_DIR = IDX_DIR / "model" / "BSJP" / "bsjp_v20_ara_continuation_state_policy_clean"
DEFAULT_MODULE = IDX_DIR / "data" / "Level_1_Features" / "modules" / "preclose14_features.parquet"


TICK_MULTIPLIER = {
    (0, 200): (1, 2.5),
    (200, 500): (2, 2.0),
    (500, 2000): (5, 1.5),
    (2000, 5000): (10, 1.5),
    (5000, float("inf")): (25, 1.0),
}


POLICY_BUCKETS = {
    "clean": {"ara_touched_single_release", "near_ara_not_touched_0_3pct"},
    "aggressive": {
        "ara_touched_single_release",
        "near_ara_not_touched_0_3pct",
        "momentum_near_ara_3_8pct",
    },
    "single_release_only": {"ara_touched_single_release"},
    "all": None,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Package frozen ARA-state policy artifact.")
    p.add_argument("--source-model-dir", type=Path, default=DEFAULT_SOURCE_MODEL)
    p.add_argument("--preclose14-path", type=Path, default=DEFAULT_MODULE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--policy", choices=sorted(POLICY_BUCKETS), default="clean")
    return p.parse_args()


def estimate_spread_frac(price: float) -> float:
    if price <= 0 or pd.isna(price):
        return 0.0
    for (lo, hi), (tick, mult) in TICK_MULTIPLIER.items():
        if lo <= price < hi:
            return (tick * mult) / price
    return 0.0


def ara_bucket(row: pd.Series) -> str:
    touched = row.get("pre14_ara_touched", 0.0) >= 0.5
    released = row.get("pre14_ara_release_wick_count", 0.0) > 0
    locked = row.get("pre14_ara_locked_proxy", 0.0) >= 0.5
    dist = row.get("pre14_ara_distance_pct", np.nan)
    if touched and locked:
        return "ara_touched_locked_proxy"
    if touched and released:
        if row.get("pre14_ara_release_wick_count", 0.0) >= 2:
            return "ara_touched_repeated_release"
        return "ara_touched_single_release"
    if touched:
        return "ara_touched_other"
    if pd.notna(dist) and dist <= 0.03:
        return "near_ara_not_touched_0_3pct"
    if pd.notna(dist) and dist <= 0.08:
        return "momentum_near_ara_3_8pct"
    return "non_ara_far_gt8pct"


def load_scored(source_model_dir: Path, preclose14_path: Path) -> pd.DataFrame:
    pred_path = source_model_dir / "valid_predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not preclose14_path.exists():
        raise FileNotFoundError(preclose14_path)

    ara_cols = [
        "date",
        "ticker",
        "pre14_ara_touched",
        "pre14_ara_touch_hour",
        "pre14_ara_touched_bar_count",
        "pre14_ara_release_wick_count",
        "pre14_ara_release_wick_ratio",
        "pre14_ara_release_wick_depth_max",
        "pre14_ara_release_wick_depth_mean",
        "pre14_ara_close_at_ara_count",
        "pre14_ara_flat_ohlc_count",
        "pre14_ara_last_bar_touched",
        "pre14_ara_last_bar_release",
        "pre14_ara_last_bar_close_at_ara",
        "pre14_ara_last_bar_locked",
        "pre14_ara_last_bar_release_depth",
        "pre14_ara_locked_proxy",
        "pre14_ara_touched_released",
        "pre14_ara_distance_pct",
        "pre14_return_from_prev_close",
    ]
    pred = pd.read_parquet(pred_path)
    ara = pd.read_parquet(preclose14_path, columns=ara_cols)
    pred["date"] = pd.to_datetime(pred["date"]).dt.normalize()
    ara["date"] = pd.to_datetime(ara["date"]).dt.normalize()
    pred = pred.drop(columns=[c for c in ara_cols if c in pred.columns and c not in {"date", "ticker"}], errors="ignore")
    scored = pred.merge(ara, on=["date", "ticker"], how="left")
    for col in scored.columns:
        if col not in {"date", "ticker"}:
            scored[col] = pd.to_numeric(scored[col], errors="coerce")
    scored["ara_state_bucket"] = scored.apply(ara_bucket, axis=1)
    return scored


def reconstruct_base_trades(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct v19d base selections exactly enough to match official metrics."""
    daily_rows = []
    trade_rows = []
    capital = 10_000_000.0
    total_cost_frac = 0.004

    for d, day in scored.groupby("date", sort=True):
        day = day.copy()
        for col in ["pred_proba", "risk_norm", "entry_price", "exit_price", "overnight_return", "pre14_market_cost_est"]:
            day[col] = pd.to_numeric(day[col], errors="coerce")
        day["risk_norm"] = day["risk_norm"].fillna(0.5).clip(0, 1)
        day = day.dropna(subset=["pred_proba", "entry_price", "exit_price", "overnight_return"])
        day = day[(day["entry_price"] > 0) & (day["entry_price"] >= 500) & (day["pre14_market_cost_est"] <= 0.030)].copy()
        if day.empty:
            daily_rows.append({"date": d, "positions": 0, "gross_return": 0.0, "net_return": 0.0})
            continue

        day = day.sort_values("pred_proba", ascending=False).reset_index(drop=True)
        day["pred_rank"] = np.arange(1, len(day) + 1)
        threshold = max(0.035, float(day["pred_proba"].quantile(0.85)))
        day = day[(day["pred_rank"] <= 3) & (day["pred_proba"] >= threshold)].copy()
        if day.empty:
            daily_rows.append({"date": d, "positions": 0, "gross_return": 0.0, "net_return": 0.0})
            continue
        day["score"] = np.maximum(day["pred_proba"] - threshold, 0.0) * (1.0 - day["risk_norm"])
        day = day[day["score"] > 0].sort_values("score", ascending=False).head(3).copy()
        if day.empty:
            daily_rows.append({"date": d, "positions": 0, "gross_return": 0.0, "net_return": 0.0})
            continue

        scores = day["score"].to_numpy(dtype=float)
        weights = scores / scores.sum()
        # Match train_lightgbm capped_weights projection with max weight 25%.
        for _ in range(10):
            over = weights > 0.25
            if not over.any():
                break
            excess = float((weights[over] - 0.25).sum())
            weights[over] = 0.25
            under = ~over
            under_sum = float(weights[under].sum())
            if under_sum <= 0:
                break
            weights[under] += weights[under] / under_sum * excess
        day["weight"] = np.clip(weights, 0.0, 0.25)
        day["allocation_idr"] = capital * day["weight"]
        day = day[day["allocation_idr"] >= 100_000].copy()
        if day.empty:
            daily_rows.append({"date": d, "positions": 0, "gross_return": 0.0, "net_return": 0.0})
            continue

        day["trade_gross_return"] = day["overnight_return"]
        day["spread_cost_frac"] = day["entry_price"].apply(estimate_spread_frac) + day["exit_price"].apply(estimate_spread_frac)
        day["trade_net_return"] = day["trade_gross_return"] - total_cost_frac - day["spread_cost_frac"]
        day["weighted_gross"] = day["weight"] * day["trade_gross_return"]
        day["weighted_net"] = day["weight"] * day["trade_net_return"]
        daily_rows.append(
            {
                "date": d,
                "positions": int(len(day)),
                "gross_return": float(day["weighted_gross"].sum()),
                "net_return": float(day["weighted_net"].sum()),
            }
        )
        trade_rows.append(day.assign(date=d))
        capital *= 1.0 + float(day["weighted_net"].sum())

    daily = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    trades = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    return daily, trades


def apply_policy(base_daily: pd.DataFrame, base_trades: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    allowed = POLICY_BUCKETS[policy]
    if allowed is None:
        trades = base_trades.copy()
    else:
        trades = base_trades[base_trades["ara_state_bucket"].isin(allowed)].copy()

    by_day = (
        trades.groupby("date")
        .agg(
            positions=("ticker", "size"),
            gross_return=("weighted_gross", "sum"),
            net_return=("weighted_net", "sum"),
        )
        .reset_index()
        if not trades.empty
        else pd.DataFrame(columns=["date", "positions", "gross_return", "net_return"])
    )
    daily = base_daily[["date"]].merge(by_day, on="date", how="left")
    daily[["positions", "gross_return", "net_return"]] = daily[["positions", "gross_return", "net_return"]].fillna(0.0)
    daily["positions"] = daily["positions"].astype(int)

    capital = 10_000_000.0
    starts = []
    ends = []
    for ret in daily["net_return"]:
        starts.append(capital)
        capital *= 1.0 + float(ret)
        ends.append(capital)
    daily["capital_start"] = starts
    daily["capital_end"] = ends
    return daily, trades


def summarize(daily: pd.DataFrame, trades: pd.DataFrame) -> dict:
    equity = (1.0 + daily["net_return"]).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    out = {
        "days": int(len(daily)),
        "trading_days": int((daily["positions"] > 0).sum()),
        "positions": int(daily["positions"].sum()),
        "mean_daily_net_return": float(daily["net_return"].mean()),
        "mean_daily_gross_return": float(daily["gross_return"].mean()),
        "net_expectancy_per_trade": float(trades["trade_net_return"].mean()) if not trades.empty else 0.0,
        "win_rate_days": float((daily["net_return"] > 0).mean()),
        "cumulative_net_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "volatility_daily": float(daily["net_return"].std(ddof=0)),
    }
    for window in [50, 20]:
        tail = daily.tail(window)
        out[f"cumulative_net_return_{window}d"] = float((1.0 + tail["net_return"]).prod() - 1.0)
        out[f"positions_{window}d"] = int(tail["positions"].sum())
        out[f"trading_days_{window}d"] = int((tail["positions"] > 0).sum())
    return out


def copy_source_artifacts(source: Path, output: Path) -> None:
    for name in ["model_lightgbm_opening_tp3.txt", "feature_importance.csv"]:
        src = source / name
        if src.exists():
            shutil.copy2(src, output / name)


def main() -> None:
    args = parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    scored = load_scored(args.source_model_dir, args.preclose14_path)
    base_daily, base_trades = reconstruct_base_trades(scored)
    daily, trades = apply_policy(base_daily, base_trades, args.policy)
    summary = summarize(daily, trades)

    daily.to_parquet(output / "portfolio_daily.parquet", index=False)
    trades.to_parquet(output / "policy_trades.parquet", index=False)
    scored.to_parquet(output / "valid_predictions_with_ara_state.parquet", index=False)

    rows = []
    for policy in POLICY_BUCKETS:
        d, t = apply_policy(base_daily, base_trades, policy)
        s = summarize(d, t)
        rows.append({"policy": policy, **s})
    policy_summary = pd.DataFrame(rows).sort_values("cumulative_net_return", ascending=False)
    policy_summary.to_csv(output / "policy_summary.csv", index=False)

    metrics = {
        "status": "RESEARCH_POLICY_LAYER",
        "variant": output.name,
        "source_model_dir": str(args.source_model_dir),
        "policy": {
            "name": args.policy,
            "allowed_buckets": sorted(POLICY_BUCKETS[args.policy]) if POLICY_BUCKETS[args.policy] is not None else "all",
            "mechanic": "veto_after_original_v19d_selection_no_reweight",
        },
        "metrics": {
            "portfolio_oot": summary,
        },
        "artifacts": {
            "portfolio_daily_path": str(output / "portfolio_daily.parquet"),
            "policy_trades_path": str(output / "policy_trades.parquet"),
            "policy_summary_path": str(output / "policy_summary.csv"),
            "valid_predictions_with_ara_state_path": str(output / "valid_predictions_with_ara_state.parquet"),
        },
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    copy_source_artifacts(args.source_model_dir, output)

    plot_dir = IDX_DIR / "pipeline" / "plot"
    if str(plot_dir) not in sys.path:
        sys.path.insert(0, str(plot_dir))
    from plot_pnl import generate_pnl_charts

    generate_pnl_charts(output)

    print(f"[write] {output}")
    print(json.dumps(metrics["metrics"]["portfolio_oot"], indent=2))
    print(policy_summary[["policy", "cumulative_net_return", "cumulative_net_return_50d", "cumulative_net_return_20d", "max_drawdown", "positions"]].to_string(index=False))


if __name__ == "__main__":
    main()
