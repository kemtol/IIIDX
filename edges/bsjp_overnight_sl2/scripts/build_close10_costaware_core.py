#!/usr/bin/env python3
"""
Build a thin BSJP close10 core datamart with cost-aware labels.

Important:
- Keep ``overnight_return`` as gross close10 return for portfolio simulation.
- Replace only training labels using market net outcome:
  gross_return - fee - estimated entry/exit bid-ask spread.

This avoids double-counting costs when train_lightgbm.py later runs with
``--execution-model market``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


IDX_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = IDX_DIR / "data"
DATAMART_DIR = DATA_DIR / "Level_2_Datamart"

DEFAULT_INPUT = DATAMART_DIR / "training_datamart_bsjp_close10_rebuild_v18like.parquet"
DEFAULT_OUTPUT = DATAMART_DIR / "training_datamart_bsjp_close10_costaware.parquet"

ROUNDTRIP_COST_FRAC = 0.004
SL_THRESHOLD = -0.02


def idx_tick_size(price: float) -> float:
    if price < 200:
        return 1.0
    if price < 500:
        return 2.0
    if price < 2000:
        return 5.0
    if price < 5000:
        return 10.0
    return 25.0


def estimate_spread_frac(price: float) -> float:
    if price <= 0:
        return 0.0
    tick = idx_tick_size(price)
    if price < 200:
        mult = 2.5
    elif price < 500:
        mult = 2.0
    elif price < 5000:
        mult = 1.5
    else:
        mult = 1.0
    return (tick * mult) / price


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build close10 cost-aware core datamart.")
    p.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cols = [
        "date",
        "ticker",
        "entry_price",
        "exit_price",
        "overnight_return",
        "label_tp",
        "label_sl2",
        "label_name",
    ]
    print(f"[costaware] input={args.input_path}")
    df = pd.read_parquet(args.input_path, columns=cols)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize().astype("datetime64[ns]")
    df["ticker"] = df["ticker"].astype(str)
    for c in ["entry_price", "exit_price", "overnight_return"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["entry_price", "exit_price", "overnight_return"]).copy()
    df = df[(df["entry_price"] > 0) & (df["exit_price"] > 0)].copy()

    entry_spread = df["entry_price"].apply(estimate_spread_frac)
    exit_spread = df["exit_price"].apply(estimate_spread_frac)
    market_cost = ROUNDTRIP_COST_FRAC + entry_spread + exit_spread
    net_return_market = df["overnight_return"] - market_cost

    df["label_tp"] = (net_return_market > 0).astype("int8")
    df["label_sl2"] = (net_return_market < SL_THRESHOLD).astype("int8")
    df["label_name"] = "bsjp_close10_market_costaware"

    summary = {
        "rows": int(len(df)),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "gross_mean": float(df["overnight_return"].mean()),
        "gross_tp_rate_old_0p4pct": float((df["overnight_return"] > ROUNDTRIP_COST_FRAC).mean()),
        "market_cost_mean": float(market_cost.mean()),
        "market_cost_p50": float(market_cost.quantile(0.5)),
        "market_cost_p90": float(market_cost.quantile(0.9)),
        "net_mean": float(net_return_market.mean()),
        "costaware_tp_rate": float(df["label_tp"].mean()),
        "costaware_sl2_rate": float(df["label_sl2"].mean()),
    }
    print("[costaware] summary=")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if args.dry_run:
        print("[costaware] dry-run, not writing")
        return

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_path, index=False)
    print(f"[costaware] wrote {args.output_path}")


if __name__ == "__main__":
    main()
