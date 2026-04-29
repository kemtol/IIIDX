#!/usr/bin/env python3
"""
Build a BSJP datamart variant with objective aligned to close(T) -> close at ~10:00 on T+1.

Why this script:
- Keep feature set exactly the same as the existing overnight datamart.
- Replace only outcome columns (exit_price / overnight_return / labels) so training
  and backtest become objective-aligned for 10:00 exit simulation.

Output remains schema-compatible with train_lightgbm.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


IDX_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = IDX_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "Level_0_Raw"
DATAMART_DIR = DATA_DIR / "Level_2_Datamart"

DEFAULT_BASE_DATAMART = DATAMART_DIR / "training_datamart_bsjp_overnight.parquet"
DEFAULT_YF_1H = RAW_DATA_DIR / "yfinance_1h.parquet"
DEFAULT_OUTPUT = DATAMART_DIR / "training_datamart_bsjp_close10_v10.parquet"

ROUNDTRIP_COST = 0.004
SL_THRESHOLD = -0.02


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create close10-objective BSJP datamart variant.")
    p.add_argument("--base-datamart-path", type=Path, default=DEFAULT_BASE_DATAMART)
    p.add_argument("--yf-1h-path", type=Path, default=DEFAULT_YF_1H)
    p.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--target-hour", type=int, default=10, help="Target hour for cutoff bar close.")
    p.add_argument(
        "--prefer-hours",
        type=str,
        default="10,9,11",
        help="Fallback hour priority when target-hour bar is missing (comma-separated).",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _parse_hours(text: str) -> list[int]:
    out: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("prefer-hours cannot be empty")
    return out


def _load_exit_date_map(yf_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild mapping (trade_date T, ticker) -> exit_date T+1 using same logic as build_label:
    - exit reference row is hour-9 open on date D
    - trade_date for that row is previous available D per ticker (shift(1))
    """
    df = yf_1h.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.normalize().astype("datetime64[ns]")
    df["hour"] = df["datetime"].dt.hour
    df["ticker"] = df["ticker"].astype(str).str.replace(r"\.JK$", "", regex=True)

    exit9 = (
        df[df["hour"] == 9]
        .groupby(["date", "ticker"], sort=False)["open"]
        .first()
        .reset_index()
        .rename(columns={"date": "exit_date", "open": "exit_open_9"})
    )
    exit9 = exit9.sort_values(["ticker", "exit_date"]).reset_index(drop=True)
    exit9["trade_date"] = exit9.groupby("ticker", sort=False)["exit_date"].shift(1)
    exit9 = exit9.dropna(subset=["trade_date"]).copy()
    exit9["trade_date"] = pd.to_datetime(exit9["trade_date"]).astype("datetime64[ns]")
    return exit9[["trade_date", "ticker", "exit_date", "exit_open_9"]]


def _load_cutoff_close_map(yf_1h: pd.DataFrame, target_hour: int, prefer_hours: list[int]) -> pd.DataFrame:
    """
    Build (exit_date, ticker) -> cutoff close using deterministic fallback:
    prefer exact target-hour first, then prefer-hours order, then nearest hour.
    """
    df = yf_1h.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["exit_date"] = df["datetime"].dt.normalize().astype("datetime64[ns]")
    df["hour"] = df["datetime"].dt.hour
    df["ticker"] = df["ticker"].astype(str).str.replace(r"\.JK$", "", regex=True)

    # Keep regular market hours only.
    df = df[df["hour"].between(9, 16)].copy()

    # Priority by explicit preferred hours order, then by distance to target hour.
    priority_map = {h: i for i, h in enumerate(prefer_hours)}
    df["priority"] = df["hour"].map(priority_map).fillna(999).astype(int)
    df["distance"] = (df["hour"] - int(target_hour)).abs()

    chosen = (
        df.sort_values(["ticker", "exit_date", "priority", "distance", "datetime"])
        .drop_duplicates(["ticker", "exit_date"], keep="first")
        .rename(columns={"close": "exit_cutoff_close", "hour": "exit_cutoff_hour"})
    )
    return chosen[["exit_date", "ticker", "exit_cutoff_close", "exit_cutoff_hour"]]


def main() -> None:
    args = parse_args()
    prefer_hours = _parse_hours(args.prefer_hours)

    print(f"[Init] base_datamart={args.base_datamart_path}")
    print(f"[Init] yf_1h={args.yf_1h_path}")
    print(f"[Init] output={args.output_path}")
    print(f"[Init] target_hour={args.target_hour}, prefer_hours={prefer_hours}")

    base = pd.read_parquet(args.base_datamart_path)
    base["date"] = pd.to_datetime(base["date"]).dt.normalize().astype("datetime64[ns]")
    base["ticker"] = base["ticker"].astype(str)
    print(f"[Load] base_rows={len(base):,}, base_cols={len(base.columns)}")

    required = {"date", "ticker", "entry_price", "exit_price", "overnight_return", "label_tp", "label_sl2", "label_name"}
    missing = sorted(required - set(base.columns))
    if missing:
        raise ValueError(f"Base datamart missing required columns: {missing}")

    yf_1h = pd.read_parquet(args.yf_1h_path, columns=["datetime", "ticker", "open", "close"])
    print(f"[Load] yf_1h_rows={len(yf_1h):,}")

    exit_map = _load_exit_date_map(yf_1h)
    cutoff_map = _load_cutoff_close_map(yf_1h, target_hour=args.target_hour, prefer_hours=prefer_hours)

    df = base.merge(
        exit_map.rename(columns={"trade_date": "date"}),
        on=["date", "ticker"],
        how="left",
    )
    exit_cov = float(df["exit_date"].notna().mean())
    print(f"[Map] exit_date coverage={exit_cov:.4%}")

    df = df.merge(cutoff_map, on=["exit_date", "ticker"], how="left")
    cutoff_cov = float(df["exit_cutoff_close"].notna().mean())
    print(f"[Map] cutoff_close coverage={cutoff_cov:.4%}")

    # Replace outcomes with close10 objective; keep feature columns untouched.
    entry = pd.to_numeric(df["entry_price"], errors="coerce")
    exit_close = pd.to_numeric(df["exit_cutoff_close"], errors="coerce")
    close10_ret = (exit_close - entry) / entry.replace(0, np.nan)

    valid_mask = entry.notna() & (entry > 0) & exit_close.notna() & np.isfinite(close10_ret)
    dropped = int((~valid_mask).sum())
    if dropped > 0:
        print(f"[Clean] dropping {dropped:,} rows without valid close10 objective.")
    out = df[valid_mask].copy()

    out["exit_price"] = exit_close[valid_mask].astype("float64")
    out["overnight_return"] = close10_ret[valid_mask].astype("float64")
    out["label_tp"] = (out["overnight_return"] > ROUNDTRIP_COST).astype("int8")
    out["label_sl2"] = (out["overnight_return"] < SL_THRESHOLD).astype("int8")
    out["label_name"] = "bsjp_close10_v10"

    # Drop helper columns to avoid any future-leakage path via numeric helpers.
    out = out.drop(columns=["exit_date", "exit_open_9", "exit_cutoff_close", "exit_cutoff_hour"], errors="ignore")

    print(
        "[Out] rows={rows:,}, cols={cols}, tp_rate={tp:.2%}, ret_mean={ret:.4%}, ret_std={std:.4%}".format(
            rows=len(out),
            cols=len(out.columns),
            tp=float(out["label_tp"].mean()),
            ret=float(out["overnight_return"].mean()),
            std=float(out["overnight_return"].std(ddof=0)),
        )
    )

    # Minimal parity checks vs base schema.
    old_cols = list(base.columns)
    new_cols = list(out.columns)
    missing_old = sorted(set(old_cols) - set(new_cols))
    added_new = sorted(set(new_cols) - set(old_cols))
    print(f"[Schema] missing_old={len(missing_old)}, added_new={len(added_new)}")
    if missing_old:
        raise ValueError(f"Output missing original columns: {missing_old}")

    if args.dry_run:
        summary = {
            "rows": int(len(out)),
            "cols": int(len(out.columns)),
            "tp_rate": float(out["label_tp"].mean()),
            "mean_return": float(out["overnight_return"].mean()),
            "std_return": float(out["overnight_return"].std(ddof=0)),
            "missing_old_columns": missing_old,
            "added_new_columns": added_new,
        }
        print("[DryRun] summary=" + json.dumps(summary, indent=2))
        return

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output_path, index=False)
    print(f"[Done] wrote {args.output_path}")


if __name__ == "__main__":
    main()
