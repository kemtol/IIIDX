#!/usr/bin/env python3
"""
Level 1 Feature Enrichment: Broker Baseline Metrics
Generate broksum_datamart.parquet with rolling MA features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_broksum_data(path: Path) -> pd.DataFrame:
    """Load broksum_bybroker.parquet and normalize schema."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    # Handle both 'stock_code' and 'ticker' column names
    ticker_col = "stock_code" if "stock_code" in df.columns else "ticker"
    df["ticker"] = df[ticker_col].astype(str).str.upper().str.strip()
    df["broker"] = df["broker"].astype(str).str.upper().str.strip()
    # Calculate total_net_buy from buy_val - sell_val
    if "total_net_buy" not in df.columns:
        df["total_net_buy"] = df["buy_val"] - df["sell_val"]
    return df.dropna(subset=["date", "ticker", "broker"])


def calculate_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate rolling MA and z-scores for broker net buy."""
    df = df.sort_values(["broker", "ticker", "date"]).reset_index(drop=True)

    windows = [5, 20, 60, 120, 240]

    for w in windows:
        # Rolling mean (shift 1 for no-lookahead)
        df[f"total_net_buy_ma_{w}"] = df.groupby(["broker", "ticker"], sort=False)["total_net_buy"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=w//2).mean()
        )
        # Rolling std
        df[f"total_net_buy_std_{w}"] = df.groupby(["broker", "ticker"], sort=False)["total_net_buy"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=w//2).std()
        )
        # Z-score
        df[f"total_net_buy_z_{w}"] = (
            (df["total_net_buy"] - df[f"total_net_buy_ma_{w}"]) / df[f"total_net_buy_std_{w}"].replace(0, np.nan)
        )
        # Velocity (MA short / MA long)
        if w == 20:
            df[f"total_net_buy_velocity_{w}"] = df[f"total_net_buy_ma_5"] / df[f"total_net_buy_ma_{w}"].replace(0, np.nan)

    return df


def add_market_coverage_flags(df: pd.DataFrame, yf_daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add flags for yfinance data availability."""
    df["has_yf_daily"] = 1  # Assume available if processing
    df["has_yf_1h"] = 1
    df["has_yf_4h"] = 1
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Path to broksum_bybroker.parquet")
    parser.add_argument("--output", type=Path, required=True, help="Output path for broksum_datamart.parquet")
    args = parser.parse_args()

    print(f"[Load] Reading {args.input}")
    df = load_broksum_data(args.input)
    print(f"[Load] Rows: {len(df):,}, Brokers: {df['broker'].nunique()}, Tickers: {df['ticker'].nunique()}")

    print("[Enrich] Calculating rolling features...")
    df = calculate_rolling_features(df)

    print("[Enrich] Adding coverage flags...")
    df = add_market_coverage_flags(df)

    # Downcast numeric columns
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype("float32")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"[Done] Output: {args.output}")
    print(f"[Done] Rows: {len(df):,}, Cols: {len(df.columns)}")


if __name__ == "__main__":
    main()
