#!/usr/bin/env python3
"""
Enrich master_broker.parquet with normalized broker-character composition.

Adds:
  - foreignfund_%
  - localfund_%
  - retail_%

Mode:
  - empty (default): keep composition columns empty (NaN), ready for manual/LLM inputs
  - heuristic: dominant character based on `category` gets 60%, others 20%
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASE_DATA_DIR = Path(__file__).parent.parent / "data"
LEVEL0_DATA_DIR = BASE_DATA_DIR / "Level_0_Raw"
DEFAULT_INPUT = (LEVEL0_DATA_DIR if LEVEL0_DATA_DIR.exists() else BASE_DATA_DIR) / "master_broker.parquet"


def category_to_mix(category: str) -> tuple[int, int, int]:
    """
    Return (foreignfund_%, localfund_%, retail_%) composition.
    """
    cat = str(category or "").strip().lower()

    if "foreign" in cat:
        return (60, 20, 20)
    if "local" in cat:
        return (20, 60, 20)
    if "retail" in cat:
        return (20, 20, 60)

    # Unknown category fallback: neutral split.
    return (34, 33, 33)


def enrich_master_broker(input_path: Path, output_path: Path, mode: str = "empty") -> pd.DataFrame:
    df = pd.read_parquet(input_path).copy()

    required_cols = {"broker_code", "broker_name", "category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in master broker file: {sorted(missing)}")

    if mode == "empty":
        for c in ["foreignfund_%", "localfund_%", "retail_%"]:
            df[c] = pd.Series([float("nan")] * len(df), dtype="float64")
    elif mode == "heuristic":
        mixes = df["category"].apply(category_to_mix)
        df["foreignfund_%"] = mixes.map(lambda x: x[0]).astype("int64")
        df["localfund_%"] = mixes.map(lambda x: x[1]).astype("int64")
        df["retail_%"] = mixes.map(lambda x: x[2]).astype("int64")
        df["mix_total_%"] = df[["foreignfund_%", "localfund_%", "retail_%"]].sum(axis=1).astype("int64")

        if not (df["mix_total_%"] == 100).all():
            bad = df.loc[df["mix_total_%"] != 100, ["broker_code", "mix_total_%"]]
            raise ValueError(f"Found rows with mix total != 100:\n{bad.to_string(index=False)}")

        df = df.drop(columns=["mix_total_%"])
    else:
        raise ValueError("mode must be either 'empty' or 'heuristic'")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich master_broker.parquet with composition columns")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input master_broker.parquet path")
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUT, help="Output parquet path")
    parser.add_argument(
        "--mode",
        choices=["empty", "heuristic"],
        default="empty",
        help="empty: create columns with NaN; heuristic: auto-fill 60/20/20 profile from category",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = enrich_master_broker(args.input, args.output, mode=args.mode)

    print(f"[Done] Wrote: {args.output.resolve()}")
    print(f"[Done] Mode: {args.mode}")
    print(f"[Done] Rows: {len(df)}")
    print(f"[Done] Columns: {df.columns.tolist()}")
    print("\n[Sample]")
    print(
        df.loc[:, ["broker_code", "broker_name", "category", "foreignfund_%", "localfund_%", "retail_%"]]
        .head(12)
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
