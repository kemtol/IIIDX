#!/usr/bin/env python3
"""
Diagnose ARA-history buckets before model training.

This script is read-only with respect to feature/data inputs. It joins the
close10 core labels, preclose14 ARA-state features, and T-1 ARA-history module,
then writes bucket summaries for pre-OOT and OOT review.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


IDX_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = IDX_DIR / "data"
MODULES_DIR = DATA_DIR / "Level_1_Features" / "modules"
LOG_DIR = IDX_DIR / "_LOG"

DEFAULT_TRAINING = DATA_DIR / "Level_2_Datamart" / "training_datamart_bsjp_close10_rebuild_v18like.parquet"
DEFAULT_PRE14 = MODULES_DIR / "preclose14_features.parquet"
DEFAULT_ARA_HISTORY = MODULES_DIR / "ara_history_features.parquet"

ROUNDTRIP_COST = 0.004

CORE_COLS = [
    "date",
    "ticker",
    "label_tp",
    "entry_price",
    "exit_price",
    "overnight_return",
]

PRE14_COLS = [
    "date",
    "ticker",
    "pre14_return_from_prev_close",
    "pre14_ara_distance_pct",
    "pre14_ara_touched",
    "pre14_ara_release_wick_count",
    "pre14_ara_locked_proxy",
    "pre14_ara_touched_bar_count",
    "pre14_ara_last_bar_locked",
    "pre14_ara_last_bar_release",
    "pre14_ara_release_wick_depth_max",
    "pre14_market_cost_est",
    "pre14_turnover_until14",
]

ARA_HISTORY_COLS = [
    "date",
    "ticker",
    "was_ara_tminus1",
    "ara_count_5d",
    "ara_count_20d",
    "ara_count_60d",
    "consecutive_ara_streak_tminus1",
    "noara_streak_tminus1",
    "days_since_last_ara",
    "last_ara_return",
    "max_return_5d_tminus1",
    "max_return_20d_tminus1",
]


def normalize_ticker_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.upper()
        .str.strip()
        .str.replace(".JK", "", regex=False)
    )


def read_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    available = set(pd.read_parquet(path).columns)
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return pd.read_parquet(path, columns=columns)


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize().dt.tz_localize(None)
    out["ticker"] = normalize_ticker_series(out["ticker"])
    return out.dropna(subset=["date", "ticker"])


def assign_buckets(df: pd.DataFrame) -> pd.Series:
    ara20 = pd.to_numeric(df["ara_count_20d"], errors="coerce")
    if "pre14_ara_touched" in df.columns:
        touched = pd.to_numeric(df["pre14_ara_touched"], errors="coerce").fillna(0.0)
    elif "ara_state_bucket" in df.columns:
        touched = df["ara_state_bucket"].astype(str).str.startswith("ara_touched").astype("float64")
    else:
        touched = pd.Series(0.0, index=df.index)
    release_count = pd.to_numeric(df["pre14_ara_release_wick_count"], errors="coerce").fillna(0.0)
    locked = pd.to_numeric(df["pre14_ara_locked_proxy"], errors="coerce").fillna(0.0)
    distance = pd.to_numeric(df["pre14_ara_distance_pct"], errors="coerce")

    virgin = ara20.fillna(0.0) == 0.0
    repeated = ara20.fillna(0.0) > 0.0
    near_0_3 = distance.between(0.0, 0.03, inclusive="both")
    near_3_8 = (distance > 0.03) & (distance <= 0.08)
    far = distance > 0.08

    bucket = pd.Series("other", index=df.index, dtype="object")
    rules = [
        ("far_nonara", (touched < 0.5) & far),
        ("momentum_near_3_8_not_touched", (touched < 0.5) & near_3_8),
        ("recent_near_not_touched", repeated & (touched < 0.5) & near_0_3),
        ("repeated_touched", repeated & (touched >= 0.5)),
        ("virgin_locked_no_release", virgin & (touched >= 0.5) & (locked >= 0.5) & (release_count == 0.0)),
        ("virgin_touched_single_release", virgin & (touched >= 0.5) & (release_count == 1.0)),
        ("virgin_near_not_touched", virgin & (touched < 0.5) & near_0_3),
    ]
    for name, mask in rules:
        bucket = bucket.mask(mask, name)
    return bucket


def add_splits(df: pd.DataFrame, oot_days: int) -> pd.DataFrame:
    out = df.copy()
    days = sorted(out["date"].dropna().unique().tolist())
    if len(days) <= oot_days:
        raise ValueError(f"Need more than {oot_days} days, got {len(days)}")
    oot_set = set(days[-oot_days:])
    out["split"] = np.where(out["date"].isin(oot_set), "oot", "pre_oot")
    return out


def summarize(group: pd.DataFrame) -> pd.Series:
    net = pd.to_numeric(group["net_return"], errors="coerce")
    gross = pd.to_numeric(group["overnight_return"], errors="coerce")
    label = pd.to_numeric(group["label_tp"], errors="coerce")
    return pd.Series(
        {
            "rows": int(len(group)),
            "days": int(group["date"].nunique()),
            "tickers": int(group["ticker"].nunique()),
            "tp_rate": float(label.mean()),
            "mean_gross_return": float(gross.mean()),
            "mean_net_return": float(net.mean()),
            "median_net_return": float(net.median()),
            "p10_net_return": float(net.quantile(0.10)),
            "p90_net_return": float(net.quantile(0.90)),
            "hit_gt_1pct": float((net > 0.01).mean()),
            "hit_gt_3pct": float((net > 0.03).mean()),
            "loss_lt_minus2pct": float((net < -0.02).mean()),
            "avg_pre14_return": float(pd.to_numeric(group["pre14_return_from_prev_close"], errors="coerce").mean()),
            "avg_ara_distance": float(pd.to_numeric(group["pre14_ara_distance_pct"], errors="coerce").mean()),
            "avg_market_cost": float(pd.to_numeric(group["pre14_market_cost_est"], errors="coerce").mean()),
            "avg_turnover_until14": float(pd.to_numeric(group["pre14_turnover_until14"], errors="coerce").mean()),
        }
    )


def build_diagnostic(training_path: Path, pre14_path: Path, ara_history_path: Path, oot_days: int) -> pd.DataFrame:
    core = normalize_keys(read_parquet_columns(training_path, CORE_COLS))
    pre14 = normalize_keys(read_parquet_columns(pre14_path, PRE14_COLS))
    ara = normalize_keys(read_parquet_columns(ara_history_path, ARA_HISTORY_COLS))

    df = core.merge(pre14, on=["date", "ticker"], how="left", validate="one_to_one")
    df = df.merge(ara, on=["date", "ticker"], how="left", validate="one_to_one")

    numeric_cols = [c for c in df.columns if c not in {"date", "ticker"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["net_return"] = df["overnight_return"] - ROUNDTRIP_COST
    df["bucket"] = assign_buckets(df)
    df = add_splits(df, oot_days=oot_days)
    return df


def build_diagnostic_from_predictions(prediction_paths: list[Path], ara_history_path: Path) -> pd.DataFrame:
    frames = []
    required_base = ["date", "ticker", "label_tp", "entry_price", "exit_price", "overnight_return"]
    for path in prediction_paths:
        frame = pd.read_parquet(path)
        missing = sorted(set(required_base) - set(frame.columns))
        if missing:
            raise ValueError(f"{path} missing required prediction columns: {missing}")
        frame = normalize_keys(frame)
        frame["_source"] = path.name
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["date", "ticker", "_source"]).drop_duplicates(["date", "ticker"], keep="last")

    ara = normalize_keys(read_parquet_columns(ara_history_path, ARA_HISTORY_COLS))
    df = df.merge(ara, on=["date", "ticker"], how="left", validate="one_to_one")

    for col in [c for c in df.columns if c not in {"date", "ticker", "ara_state_bucket", "_source"}]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["net_return"] = df["overnight_return"] - ROUNDTRIP_COST
    df["bucket"] = assign_buckets(df)
    df["split"] = np.where(df["_source"].str.contains("walkforward"), "pre_oot_wf", "oot")
    return df


def make_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        df.groupby(["split", "bucket"], sort=False)
        .apply(summarize, include_groups=False)
        .reset_index()
        .sort_values(["split", "mean_net_return"], ascending=[True, False])
    )

    all_summary = (
        df.assign(split="all")
        .groupby(["split", "bucket"], sort=False)
        .apply(summarize, include_groups=False)
        .reset_index()
        .sort_values(["split", "mean_net_return"], ascending=[True, False])
    )
    summary = pd.concat([all_summary, summary], ignore_index=True)

    daily = (
        df.groupby(["split", "date", "bucket"], sort=False)
        .agg(
            n=("ticker", "size"),
            mean_net_return=("net_return", "mean"),
            tp_rate=("label_tp", "mean"),
            best_net_return=("net_return", "max"),
            worst_net_return=("net_return", "min"),
        )
        .reset_index()
        .sort_values(["split", "date", "bucket"])
    )

    examples = (
        df.sort_values(["split", "bucket", "pre14_return_from_prev_close", "overnight_return"], ascending=[True, True, False, False])
        .groupby(["split", "bucket"], sort=False)
        .head(20)
        .sort_values(["split", "bucket", "pre14_return_from_prev_close"], ascending=[True, True, False])
    )
    example_cols = [
        "split",
        "bucket",
        "date",
        "ticker",
        "label_tp",
        "overnight_return",
        "net_return",
        "pre14_return_from_prev_close",
        "pre14_ara_distance_pct",
        "pre14_ara_touched",
        "pre14_ara_release_wick_count",
        "pre14_ara_locked_proxy",
        "ara_count_5d",
        "ara_count_20d",
        "ara_count_60d",
        "consecutive_ara_streak_tminus1",
        "days_since_last_ara",
    ]
    examples = examples[[c for c in example_cols if c in examples.columns]].copy()
    return summary, daily, examples


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(description="Diagnose ARA-history bucket returns before training.")
    parser.add_argument("--training-path", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--pre14-path", type=Path, default=DEFAULT_PRE14)
    parser.add_argument("--ara-history-path", type=Path, default=DEFAULT_ARA_HISTORY)
    parser.add_argument(
        "--prediction-path",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional joined prediction parquet with labels/returns and pre14 ARA-state columns. "
            "May be passed multiple times. When set, --training-path and --pre14-path are ignored."
        ),
    )
    parser.add_argument("--oot-days", type=int, default=100)
    parser.add_argument("--output-prefix", type=Path, default=LOG_DIR / f"ara_history_bucket_{stamp}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prediction_path:
        df = build_diagnostic_from_predictions(args.prediction_path, args.ara_history_path)
    else:
        df = build_diagnostic(args.training_path, args.pre14_path, args.ara_history_path, args.oot_days)
    summary, daily, examples = make_outputs(df)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    daily_path = args.output_prefix.with_name(args.output_prefix.name + "_daily.csv")
    examples_path = args.output_prefix.with_name(args.output_prefix.name + "_examples.csv")
    summary.to_csv(summary_path, index=False)
    daily.to_csv(daily_path, index=False)
    examples.to_csv(examples_path, index=False)

    print(f"[Join] rows={len(df):,}, days={df['date'].nunique():,}, tickers={df['ticker'].nunique():,}")
    print(f"[Coverage] pre14_ara_distance_pct={df['pre14_ara_distance_pct'].notna().mean():.4f}, ara_count_20d={df['ara_count_20d'].notna().mean():.4f}")
    print(f"[Write] {summary_path}")
    print(f"[Write] {daily_path}")
    print(f"[Write] {examples_path}")
    display_cols = ["split", "bucket", "rows", "days", "tp_rate", "mean_net_return", "p10_net_return", "loss_lt_minus2pct", "avg_ara_distance"]
    print(summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
