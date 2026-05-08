#!/usr/bin/env python3
"""
Generate T-1 safe daily ARA-history features for BSJP.

The output is a feature module joined at training time on (date, ticker):

  data/Level_1_Features/modules/ara_history_features.parquet

All feature values on row date D are computed only from daily bars strictly
before D. The ARA flag for D is computed internally as history for later rows,
but is never exposed on row D.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


IDX_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = IDX_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "Level_0_Raw"
MODULES_DIR = DATA_DIR / "Level_1_Features" / "modules"

DEFAULT_YF_DAILY = RAW_DATA_DIR / "yfinance_daily.parquet"
DEFAULT_OUTPUT = MODULES_DIR / "ara_history_features.parquet"


def normalize_ticker_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.upper()
        .str.strip()
        .str.replace(".JK", "", regex=False)
    )


def idx_ara_limit_pct(reference_price: pd.Series) -> pd.Series:
    ref = pd.to_numeric(reference_price, errors="coerce")
    out = pd.Series(np.nan, index=ref.index, dtype="float64")
    out = out.mask(ref > 0, 0.35)
    out = out.mask(ref > 200, 0.25)
    out = out.mask(ref > 5000, 0.20)
    return out


def _streak_features_one_ticker(group: pd.DataFrame) -> pd.DataFrame:
    """Return row-D features using only rows before D for one ticker."""
    is_ara = group["_is_ara"].to_numpy(dtype=bool)
    daily_ret = group["_daily_return"].to_numpy(dtype=float)

    ara_streak_prev: list[float] = []
    noara_streak_prev: list[float] = []
    days_since_prev: list[float] = []
    last_ara_ret_prev: list[float] = []

    ara_streak = 0
    noara_streak = 0
    days_since: float = np.nan
    last_ara_ret: float = np.nan

    for flag, ret in zip(is_ara, daily_ret):
        ara_streak_prev.append(float(ara_streak))
        noara_streak_prev.append(float(noara_streak))
        days_since_prev.append(float(days_since) if not np.isnan(days_since) else np.nan)
        last_ara_ret_prev.append(float(last_ara_ret) if not np.isnan(last_ara_ret) else np.nan)

        if flag:
            ara_streak += 1
            noara_streak = 0
            days_since = 0.0
            last_ara_ret = ret
        else:
            ara_streak = 0
            noara_streak += 1
            if not np.isnan(days_since):
                days_since += 1.0

    return pd.DataFrame(
        {
            "consecutive_ara_streak_tminus1": ara_streak_prev,
            "noara_streak_tminus1": noara_streak_prev,
            "days_since_last_ara": days_since_prev,
            "last_ara_return": last_ara_ret_prev,
        },
        index=group.index,
    )


def build_ara_history_features(daily: pd.DataFrame, ara_buffer_pct: float = 0.005) -> pd.DataFrame:
    required = {"date", "ticker", "close"}
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"daily input missing required columns: {missing}")

    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize().dt.tz_localize(None)
    df["ticker"] = normalize_ticker_series(df["ticker"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "close"])
    df = df[df["close"] > 0].copy()
    df = df.sort_values(["ticker", "date"]).drop_duplicates(["date", "ticker"], keep="last")

    grp = df.groupby("ticker", sort=False)
    df["_prev_close"] = grp["close"].shift(1)
    df["_daily_return"] = (df["close"] / df["_prev_close"].replace(0, np.nan)) - 1.0
    df["_ara_limit_pct"] = idx_ara_limit_pct(df["_prev_close"])
    df["_is_ara"] = (
        df["_prev_close"].notna()
        & df["_daily_return"].notna()
        & df["_ara_limit_pct"].notna()
        & (df["_daily_return"] >= (df["_ara_limit_pct"] - ara_buffer_pct))
    )

    shifted_ara = grp["_is_ara"].shift(1).astype("float64")
    df["was_ara_tminus1"] = shifted_ara
    for window in (3, 5, 10, 20, 60):
        df[f"ara_count_{window}d"] = (
            shifted_ara.groupby(df["ticker"], sort=False)
            .rolling(window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

    shifted_return = grp["_daily_return"].shift(1)
    for window in (5, 20):
        df[f"max_return_{window}d_tminus1"] = (
            shifted_return.groupby(df["ticker"], sort=False)
            .rolling(window, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
        )

    streak = (
        df.groupby("ticker", group_keys=False, sort=False)
        .apply(_streak_features_one_ticker, include_groups=False)
        .sort_index()
    )
    df = pd.concat([df, streak], axis=1)

    feature_cols = [
        "was_ara_tminus1",
        "ara_count_3d",
        "ara_count_5d",
        "ara_count_10d",
        "ara_count_20d",
        "ara_count_60d",
        "consecutive_ara_streak_tminus1",
        "noara_streak_tminus1",
        "days_since_last_ara",
        "last_ara_return",
        "max_return_5d_tminus1",
        "max_return_20d_tminus1",
    ]

    out = df[["date", "ticker"] + feature_cols].copy()
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate T-1 safe ARA-history feature module.")
    parser.add_argument("--yf-daily-path", type=Path, default=DEFAULT_YF_DAILY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ara-buffer-pct", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    daily = pd.read_parquet(args.yf_daily_path)
    features = build_ara_history_features(daily, ara_buffer_pct=args.ara_buffer_pct)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)
    print(
        f"[Done] wrote {args.output} rows={len(features):,} cols={len(features.columns)} "
        f"date={features['date'].min().date()}->{features['date'].max().date()} "
        f"tickers={features['ticker'].nunique():,}"
    )


if __name__ == "__main__":
    main()
