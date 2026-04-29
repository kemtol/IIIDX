"""
generate_vwap_features.py — VWAP-based features from yfinance 1h bars.

Features per (date, ticker):
    close_to_vwap        (close_eod - vwap_full) / vwap_full
    open_pm_to_vwap_am   (close_first_pm_bar - vwap_morning) / vwap_morning
    vwap_trend           (vwap_full - vwap_morning) / vwap_morning
    last_hour_above_vwap 1 if penultimate bar close > cumvwap at that bar
    close_drive          (close_eod - close_first_pm_bar) / vwap_full
    vol_above_vwap_pct   sum(vol of bars where close > cumvwap) / total_vol

Morning session = hours [9, 10, 11]
Afternoon open  = first bar with hour >= 13

Input:  data/Level_0_Raw/yfinance_1h.parquet
Output: data/Level_1_Features/vwap_features.parquet

Usage
-----
    python generate_vwap_features.py
    python generate_vwap_features.py --date-from 2026-01-01
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT   = Path(__file__).resolve().parents[2]
INPUT_PATH  = REPO_ROOT / "data/Level_0_Raw/yfinance_1h.parquet"
OUTPUT_PATH = REPO_ROOT / "data/Level_1_Features/vwap_features.parquet"

MORNING_HOURS = {9, 10, 11}


def load_data(date_from: str | None) -> pd.DataFrame:
    df = pd.read_parquet(INPUT_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"]     = df["datetime"].dt.date
    df["hour"]     = df["datetime"].dt.hour
    df["ticker"]   = df["ticker"].str.replace(".JK", "", regex=False)
    if date_from:
        cutoff = pd.Timestamp(date_from).date()
        df = df[df["date"] >= cutoff]
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date", "hour"]).copy()

    df["tp"]     = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_vol"] = df["tp"] * df["volume"]

    g = df.groupby(["ticker", "date"])

    # Cumulative VWAP per bar (used for last_hour_above_vwap + vol_above_vwap_pct)
    cumvol         = g["volume"].cumsum()
    cumtp_vol      = g["tp_vol"].cumsum()
    df["cumvwap"]  = cumtp_vol / cumvol.replace(0, np.nan)

    # ── Full-day aggregates ────────────────────────────────────────────────────
    day = g.agg(
        total_tp_vol = ("tp_vol",  "sum"),
        total_vol    = ("volume",  "sum"),
        close_eod    = ("close",   "last"),
    ).reset_index()
    day["vwap_full"] = day["total_tp_vol"] / day["total_vol"].replace(0, np.nan)

    # ── Morning VWAP (hours 9–11) ──────────────────────────────────────────────
    am = df[df["hour"].isin(MORNING_HOURS)].groupby(["ticker", "date"]).agg(
        am_tp_vol = ("tp_vol", "sum"),
        am_vol    = ("volume", "sum"),
    ).reset_index()
    am["vwap_morning"] = am["am_tp_vol"] / am["am_vol"].replace(0, np.nan)

    # ── First afternoon bar (hour >= 13) ──────────────────────────────────────
    pm_first = (
        df[df["hour"] >= 13]
        .sort_values("hour")
        .groupby(["ticker", "date"])
        .first()
        .reset_index()[["ticker", "date", "close"]]
        .rename(columns={"close": "close_pm_open"})
    )

    # ── Penultimate bar (rank 1 from last = 0) ────────────────────────────────
    df["rank_from_last"] = g.cumcount(ascending=False)
    penult = (
        df[df["rank_from_last"] == 1][["ticker", "date", "close", "cumvwap"]]
        .rename(columns={"close": "close_penult", "cumvwap": "cumvwap_penult"})
    )

    # ── Volume above cumVWAP ──────────────────────────────────────────────────
    df["vol_above"] = (df["close"] > df["cumvwap"]).astype(float) * df["volume"]
    vol_agg = g["vol_above"].sum().reset_index()

    # ── Merge ─────────────────────────────────────────────────────────────────
    out = (
        day[["date", "ticker", "vwap_full", "close_eod", "total_vol"]]
        .merge(am[["date", "ticker", "vwap_morning"]], on=["date", "ticker"], how="left")
        .merge(pm_first,                               on=["date", "ticker"], how="left")
        .merge(penult,                                 on=["date", "ticker"], how="left")
        .merge(vol_agg,                                on=["date", "ticker"], how="left")
    )

    # ── Final features ────────────────────────────────────────────────────────
    vf = out["vwap_full"]
    vm = out["vwap_morning"]

    out["close_to_vwap"]        = (out["close_eod"]    - vf) / vf
    out["open_pm_to_vwap_am"]   = (out["close_pm_open"] - vm) / vm
    out["vwap_trend"]           = (vf - vm) / vm
    out["last_hour_above_vwap"] = (out["close_penult"] > out["cumvwap_penult"]).astype("Int8")
    out["close_drive"]          = (out["close_eod"] - out["close_pm_open"]) / vf
    out["vol_above_vwap_pct"]   = out["vol_above"] / out["total_vol"].replace(0, np.nan)

    feature_cols = [
        "date", "ticker",
        "close_to_vwap", "open_pm_to_vwap_am", "vwap_trend",
        "last_hour_above_vwap", "close_drive", "vol_above_vwap_pct",
    ]
    return out[feature_cols].reset_index(drop=True)


def upsert_parquet(new_df: pd.DataFrame, date_from: str | None) -> None:
    if OUTPUT_PATH.exists() and date_from:
        existing = pd.read_parquet(OUTPUT_PATH)
        existing["date"] = pd.to_datetime(existing["date"]).dt.date
        cutoff = pd.Timestamp(date_from).date()
        existing = existing[existing["date"] < cutoff]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date-from", default=None, help="Process only dates >= this (YYYY-MM-DD)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[VWAP] Loading yfinance_1h{'  (full)' if not args.date_from else f'  from {args.date_from}'}")
    df = load_data(args.date_from)
    print(f"[VWAP] {len(df):,} bars | {df['ticker'].nunique()} tickers | {df['date'].nunique()} dates")

    print("[VWAP] Computing features...")
    features = compute_features(df)
    print(f"[VWAP] {len(features):,} rows computed")

    upsert_parquet(features, args.date_from)
    print(f"[VWAP] Saved → {OUTPUT_PATH.name}")

    print("\nSample (BBRI, last 3 rows):")
    sample = features[features["ticker"] == "BBRI"].tail(3)
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
