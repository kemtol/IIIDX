#!/usr/bin/env python3
"""
Walk-forward validation for frozen BSJP ARA-state policy rules.

This script retrains v19d-style fold models on the pre-OOT period and applies
ARA-state policy vetoes on each validation fold. It is intentionally policy-only:
ARA-state columns are used for execution filtering, not as LightGBM features.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


IDX_DIR = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = IDX_DIR / "data" / "Level_2_Datamart" / "training_datamart_bsjp_close10_rebuild_v18like.parquet"
MODULES_DIR = IDX_DIR / "data" / "Level_1_Features" / "modules"
OUT_DIR = IDX_DIR / "_LOG"


def load_train_module():
    spec = importlib.util.spec_from_file_location("bsjp_train_lightgbm", SCRIPT_DIR / "train_lightgbm.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to import train_lightgbm.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate frozen ARA-state policies with walk-forward folds.")
    p.add_argument("--training-path", type=Path, default=TRAIN_PATH)
    p.add_argument("--feature-modules-dir", type=Path, default=MODULES_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--oot-valid-days", type=int, default=100)
    p.add_argument("--walkforward-folds", type=int, default=4)
    p.add_argument("--walkforward-valid-days", type=int, default=20)
    p.add_argument("--min-trading-days", type=int, default=120)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


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


def capped_weights(scores: np.ndarray, max_weight: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0 or scores.sum() <= 0:
        return np.zeros_like(scores)
    w = scores / scores.sum()
    max_weight = min(max_weight, 1.0)
    for _ in range(10):
        over = w > max_weight
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        under = ~over
        if not under.any():
            break
        under_sum = float(w[under].sum())
        if under_sum <= 0:
            break
        w[under] += w[under] / under_sum * excess
    return np.clip(w, 0.0, max_weight)


TICK_MULTIPLIER = {
    (0, 200): (1, 2.5),
    (200, 500): (2, 2.0),
    (500, 2000): (5, 1.5),
    (2000, 5000): (10, 1.5),
    (5000, float("inf")): (25, 1.0),
}


def estimate_spread_frac(price: float) -> float:
    if price <= 0 or pd.isna(price):
        return 0.0
    for (lo, hi), (tick, mult) in TICK_MULTIPLIER.items():
        if lo <= price < hi:
            return (tick * mult) / price
    return 0.0


def simulate_veto(scored: pd.DataFrame, variant: str, mask_func) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    capital = 10_000_000.0
    total_cost_frac = 0.004
    rows = []
    trade_rows = []

    for d, day in scored.groupby("date", sort=True):
        day = day.copy()
        for col in ["pred_proba", "risk_norm", "entry_price", "exit_price", "overnight_return", "pre14_market_cost_est"]:
            day[col] = pd.to_numeric(day[col], errors="coerce")
        day["risk_norm"] = day["risk_norm"].fillna(0.5).clip(0, 1)
        day = day.dropna(subset=["pred_proba", "entry_price", "exit_price", "overnight_return"])
        day = day[(day["entry_price"] > 0) & (day["entry_price"] >= 500) & (day["pre14_market_cost_est"] <= 0.030)].copy()
        if day.empty:
            rows.append({"variant": variant, "date": d, "positions": 0, "gross_return": 0.0, "net_return": 0.0})
            continue

        day = day.sort_values("pred_proba", ascending=False).reset_index(drop=True)
        day["pred_rank"] = np.arange(1, len(day) + 1)
        threshold = max(0.035, float(day["pred_proba"].quantile(0.85)))
        day = day[(day["pred_rank"] <= 3) & (day["pred_proba"] >= threshold)].copy()
        day["score"] = np.maximum(day["pred_proba"] - threshold, 0.0) * (1.0 - day["risk_norm"])
        day = day[day["score"] > 0].sort_values("score", ascending=False).head(3).copy()
        if not day.empty:
            day = day[mask_func(day)].copy()
        if day.empty:
            rows.append({"variant": variant, "date": d, "positions": 0, "gross_return": 0.0, "net_return": 0.0})
            continue

        day["weight"] = capped_weights(day["score"].to_numpy(), 0.25)
        day["allocation_idr"] = capital * day["weight"]
        day = day[day["allocation_idr"] >= 100_000].copy()
        if day.empty:
            rows.append({"variant": variant, "date": d, "positions": 0, "gross_return": 0.0, "net_return": 0.0})
            continue

        day["trade_gross_return"] = day["overnight_return"]
        day["spread_cost_frac"] = day["entry_price"].apply(estimate_spread_frac) + day["exit_price"].apply(estimate_spread_frac)
        day["trade_net_return"] = day["trade_gross_return"] - total_cost_frac - day["spread_cost_frac"]
        day["weighted_gross"] = day["weight"] * day["trade_gross_return"]
        day["weighted_net"] = day["weight"] * day["trade_net_return"]
        gross_ret = float(day["weighted_gross"].sum())
        net_ret = float(day["weighted_net"].sum())
        rows.append({"variant": variant, "date": d, "positions": int(len(day)), "gross_return": gross_ret, "net_return": net_ret})
        trade_rows.append(day.assign(variant=variant, date=d))
        capital *= 1.0 + net_ret

    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    trades = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    equity = (1.0 + daily["net_return"]).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    summary = {
        "variant": variant,
        "days": int(len(daily)),
        "trading_days": int((daily["positions"] > 0).sum()),
        "positions": int(daily["positions"].sum()),
        "cum_net": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "max_dd": float(drawdown.min()) if len(drawdown) else 0.0,
        "mean_daily_net": float(daily["net_return"].mean()) if len(daily) else 0.0,
        "net_expectancy_per_trade": float(trades["trade_net_return"].mean()) if not trades.empty else 0.0,
    }
    return daily, trades, summary


def main() -> None:
    args = parse_args()
    train = load_train_module()

    print(f"[load] training={args.training_path}")
    core = pd.read_parquet(args.training_path, columns=train.CORE_MODEL_COLS)
    df = train.load_features_from_modules(args.feature_modules_dir, core)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "ticker", train.TARGET_COL]).copy()
    df[train.TARGET_COL] = pd.to_numeric(df[train.TARGET_COL], errors="coerce").astype(int)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    df["ara_state_bucket"] = df.apply(ara_bucket, axis=1)

    unique_days = sorted(df["date"].unique().tolist())
    pre_oot_days = unique_days[:-args.oot_valid_days]
    pre_oot_df = df[df["date"].isin(pre_oot_days)].copy()

    # Match v19d model setup.
    train_args = argparse.Namespace(
        random_state=args.random_state,
        model_objective="binary",
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=-1,
        min_data_in_leaf=100,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l1=0.0,
        lambda_l2=1.5,
        min_gain_to_split=0.05,
        scale_pos_weight=-1.0,
        pos_weight_multiplier=1.0,
        oversample_ratio=0.0,
        early_stopping_rounds=200,
    )

    feature_blacklist = train.resolve_feature_blacklist(argparse.Namespace(feature_blacklist="", feature_blacklist_preset="preclose14"))
    feature_cols = train.choose_feature_columns(pre_oot_df)
    feature_cols = [c for c in feature_cols if c not in set(feature_blacklist)]
    x_pre = pre_oot_df[feature_cols]
    all_null = [c for c in feature_cols if x_pre[c].isna().all()]
    constant = [c for c in feature_cols if x_pre[c].nunique(dropna=True) <= 1]
    duplicate_cols = sorted(x_pre.columns[x_pre.T.duplicated()].tolist())
    drop_cols = sorted(set(all_null + constant + duplicate_cols))
    feature_cols = [c for c in feature_cols if c not in set(drop_cols)]
    print(f"[features] selected={len(feature_cols)} dropped={len(drop_cols)}")

    splits = train.build_walkforward_splits(
        days=pre_oot_days,
        folds=args.walkforward_folds,
        valid_days=args.walkforward_valid_days,
        min_train_days=args.min_trading_days,
    )
    print(f"[folds] {len(splits)}")

    preds = []
    fold_summaries = []
    for i, (tr_days, va_days) in enumerate(splits, start=1):
        tr = pre_oot_df[pre_oot_df["date"].isin(tr_days)].copy()
        va = pre_oot_df[pre_oot_df["date"].isin(va_days)].copy()
        model = train.train_lgbm(
            tr[feature_cols],
            tr[train.TARGET_COL],
            va[feature_cols],
            va[train.TARGET_COL],
            tr["date"],
            va["date"],
            train_args,
        )
        va_pred = va[
            [
                "date",
                "ticker",
                train.TARGET_COL,
                "entry_price",
                "exit_price",
                "overnight_return",
                "pre14_market_cost_est",
                "ara_state_bucket",
                "pre14_return_from_prev_close",
                "pre14_ara_distance_pct",
                "pre14_ara_release_wick_count",
                "pre14_ara_release_wick_depth_max",
                "pre14_ara_locked_proxy",
            ]
        ].copy()
        va_pred["pred_proba"] = train.predict_signal(model, va[feature_cols], train_args)
        va_pred["risk_norm"] = train.build_risk_proxy(va[feature_cols])
        va_pred["fold"] = i
        preds.append(va_pred)
        fold_summaries.append(
            {
                "fold": i,
                "train_days": len(tr_days),
                "valid_days": len(va_days),
                "valid_rows": len(va),
                "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
            }
        )
        print(f"[fold {i}] rows={len(va):,} best_iter={fold_summaries[-1]['best_iteration']}")

    scored = pd.concat(preds, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    variants = {
        "wf_all_base": lambda x: pd.Series(True, index=x.index),
        "wf_veto_single_release_only": lambda x: x["ara_state_bucket"].eq("ara_touched_single_release"),
        "wf_veto_single_plus_near_0_3": lambda x: x["ara_state_bucket"].isin(["ara_touched_single_release", "near_ara_not_touched_0_3pct"]),
        "wf_veto_single_near_momentum_0_8": lambda x: x["ara_state_bucket"].isin(["ara_touched_single_release", "near_ara_not_touched_0_3pct", "momentum_near_ara_3_8pct"]),
        "wf_veto_exclude_repeated_locked": lambda x: ~x["ara_state_bucket"].isin(["ara_touched_repeated_release", "ara_touched_locked_proxy"]),
        "wf_veto_repeated_only": lambda x: x["ara_state_bucket"].eq("ara_touched_repeated_release"),
        "wf_veto_far_nonara_only": lambda x: x["ara_state_bucket"].eq("non_ara_far_gt8pct"),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_daily = []
    all_trades = []
    summaries = []
    for name, mask in variants.items():
        daily, trades, summary = simulate_veto(scored, name, mask)
        all_daily.append(daily)
        if not trades.empty:
            all_trades.append(trades)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries).sort_values("cum_net", ascending=False)
    daily_df = pd.concat(all_daily, ignore_index=True)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    summary_path = args.out_dir / "pre14_ara_state_policy_walkforward_summary_20260501.csv"
    daily_path = args.out_dir / "pre14_ara_state_policy_walkforward_daily_20260501.csv"
    trades_path = args.out_dir / "pre14_ara_state_policy_walkforward_trades_20260501.csv"
    pred_path = args.out_dir / "pre14_ara_state_policy_walkforward_predictions_20260501.parquet"
    fold_path = args.out_dir / "pre14_ara_state_policy_walkforward_folds_20260501.csv"

    summary_df.to_csv(summary_path, index=False)
    daily_df.to_csv(daily_path, index=False)
    if not trades_df.empty:
        trades_df.to_csv(trades_path, index=False)
    scored.to_parquet(pred_path, index=False)
    pd.DataFrame(fold_summaries).to_csv(fold_path, index=False)

    print(f"[write] {summary_path}")
    print(f"[write] {daily_path}")
    print(f"[write] {trades_path}")
    print(f"[write] {pred_path}")
    show = summary_df.copy()
    for c in ["cum_net", "max_dd", "mean_daily_net", "net_expectancy_per_trade"]:
        show[c] = (show[c] * 100).round(2)
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
