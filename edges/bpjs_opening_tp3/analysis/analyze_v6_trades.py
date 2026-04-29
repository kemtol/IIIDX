#!/usr/bin/env python3
"""Analyze executed trades from V6 adaptive-threshold predictions.

Reconstructs execution logic:
- daily_p95 = quantile(pred_proba, 0.95)
- effective_threshold = max(0.035, daily_p95)
- execute when daily_rank <= 3 and pred_proba >= effective_threshold

Then compares max spike vs 10:00 close behavior.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
V6_PRED_PATH = REPO_ROOT / "machinelearning" / "idx" / "model" / "BPJS" / "bpjs_lgbm_opening_tp3_v6" / "valid_predictions.parquet"
TRAINING_PATH = REPO_ROOT / "machinelearning" / "idx" / "data" / "training" / "training_datamart_opening_tp3.parquet"

CONVICTION_FLOOR = 0.035
CONVICTION_TOP_K = 3


def reconstruct_v6_execution(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "ticker", "pred_proba"]).copy()

    df["daily_rank"] = (
        df.groupby("date", sort=False)["pred_proba"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    daily_p95 = df.groupby("date", sort=False)["pred_proba"].quantile(0.95).rename("daily_p95")
    df = df.merge(daily_p95, on="date", how="left")
    df["effective_threshold"] = np.maximum(CONVICTION_FLOOR, df["daily_p95"])

    executed = df[
        (df["daily_rank"] <= CONVICTION_TOP_K)
        & (df["pred_proba"] >= df["effective_threshold"])
    ].copy()
    return executed


def add_trade_anatomy(executed: pd.DataFrame, training_df: pd.DataFrame) -> pd.DataFrame:
    needed = ["date", "ticker", "entry_price_opening", "high_to_cutoff", "close_to_cutoff"]
    px = training_df[needed].copy()
    px["date"] = pd.to_datetime(px["date"], errors="coerce").dt.normalize()

    out = executed.merge(px, on=["date", "ticker"], how="left", validate="m:1")

    out["max_spike_pct"] = (out["high_to_cutoff"] - out["entry_price_opening"]) / out["entry_price_opening"] * 100.0
    out["close_return_pct"] = (out["close_to_cutoff"] - out["entry_price_opening"]) / out["entry_price_opening"] * 100.0
    out["dump_from_high_pct"] = (out["close_to_cutoff"] - out["high_to_cutoff"]) / out["high_to_cutoff"] * 100.0

    out = out.sort_values(["date", "pred_proba"], ascending=[True, False]).reset_index(drop=True)
    return out


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "nan"
    return f"{x:.4f}%"


def main() -> None:
    if not V6_PRED_PATH.exists():
        raise FileNotFoundError(f"Missing predictions file: {V6_PRED_PATH}")
    if not TRAINING_PATH.exists():
        raise FileNotFoundError(f"Missing training file: {TRAINING_PATH}")

    pred = pd.read_parquet(V6_PRED_PATH)
    training = pd.read_parquet(TRAINING_PATH)

    executed = reconstruct_v6_execution(pred)
    analyzed = add_trade_anatomy(executed, training)

    avg_max_spike = float(analyzed["max_spike_pct"].mean()) if not analyzed.empty else float("nan")
    avg_close_return = float(analyzed["close_return_pct"].mean()) if not analyzed.empty else float("nan")
    avg_dump_from_high = float(analyzed["dump_from_high_pct"].mean()) if not analyzed.empty else float("nan")

    print(f"Predictions path: {V6_PRED_PATH}")
    print(f"Training path:    {TRAINING_PATH}")
    print(f"Total executed trades: {len(analyzed)}")
    print("Averages:")
    print(f"- max_spike_pct:      {fmt_pct(avg_max_spike)}")
    print(f"- close_return_pct:   {fmt_pct(avg_close_return)}")
    print(f"- dump_from_high_pct: {fmt_pct(avg_dump_from_high)}")

    cols = [
        "date",
        "ticker",
        "pred_proba",
        "entry_price_opening",
        "max_spike_pct",
        "close_return_pct",
        "dump_from_high_pct",
    ]
    out = analyzed[cols].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["pred_proba"] = out["pred_proba"].map(lambda v: f"{v:.6f}")
    for c in ["max_spike_pct", "close_return_pct", "dump_from_high_pct"]:
        out[c] = out[c].map(lambda v: f"{v:.4f}")

    print("\nExecuted Trades Table:")
    try:
        print(out.to_markdown(index=False))
    except Exception:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
