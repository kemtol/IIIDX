#!/usr/bin/env python3
"""Build a P0-clean BSJP v25 training datamart from the locked audit artifact.

This script is intentionally conservative:
- it never overwrites the locked source unless explicitly pointed there;
- it removes feature families that are not proven available at 15:30 decision
  time;
- it drops low-coverage trading days and duplicate (date, ticker) rows;
- it writes an audit manifest beside the output parquet.

The goal is a clean training input for the next active-model retrain, not a
research benchmark that preserves every experimental v25 column.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data/Level_2_Datamart/training_datamart_bsjp_v25_locked.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "data/Level_2_Datamart/training_datamart_bsjp_v25_clean.parquet"
DEFAULT_MASTER = REPO_ROOT / "data/Level_0_Raw/master_emiten.parquet"
DEFAULT_MODULES_DIR = REPO_ROOT / "data/Level_1_Features/modules"

KEY_COLS = ["date", "ticker"]
OUTCOME_COLS = [
    "entry_price",
    "exit_price",
    "exit_date",
    "overnight_return",
    "label_tp",
    "label_sl2",
    "label_name",
]
UNSAFE_PREFIXES = ("f2_", "forensic_", "v25_sector_")
UNSAFE_EXACT = {"sector_ticker_count"}
QUICK_T1_SOURCES = {
    "forensic_v2_features.parquet": {
        "f2_inventory_decay": "f2_inventory_decay_t1",
    },
    "forensic_features.parquet": {
        "forensic_inventory_10d": "forensic_inventory_10d_t1",
    },
    "sector_features.parquet": {
        "v25_sector_turnover_share": "v25_sector_turnover_share_t1",
        "v25_sector_flow_share": "v25_sector_flow_share_t1",
    },
}
ROUNDTRIP_COST = 0.004
SL_THRESHOLD = -0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build P0-clean v25 BSJP datamart.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--master-path", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--modules-dir", type=Path, default=DEFAULT_MODULES_DIR)
    parser.add_argument("--min-date-coverage", type=int, default=300)
    parser.add_argument(
        "--add-t1-quick-features",
        action="store_true",
        help="Add shifted T-1 quick-win versions of selected dropped v25 feature families.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow replacing an existing output parquet.",
    )
    return parser.parse_args()


def normalize_ticker_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"\.JK$", "", regex=True)
    )


def as_jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def load_master_active(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_parquet(path)
    if "ticker" not in df.columns:
        return set()
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.upper().eq("ACTIVE")]
    return set(normalize_ticker_series(df["ticker"]).dropna())


def unsafe_columns(columns: list[str]) -> list[str]:
    out: list[str] = []
    for col in columns:
        if col in KEY_COLS or col in OUTCOME_COLS:
            continue
        if col.endswith("_t1"):
            continue
        if col in UNSAFE_EXACT or col.startswith(UNSAFE_PREFIXES):
            out.append(col)
    return out


def load_t1_quick_features(modules_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load selected unsafe features from source modules and shift by ticker.

    The shift makes date T use the source value from the previous available
    trading row for the same ticker. Source columns keep their original formula;
    output columns are suffixed `_t1` to make availability explicit.
    """
    pieces: list[pd.DataFrame] = []
    audit: dict[str, Any] = {}

    for filename, mapping in QUICK_T1_SOURCES.items():
        path = modules_dir / filename
        if not path.exists():
            audit[filename] = {"status": "missing", "path": str(path)}
            continue
        columns = ["date", "ticker"] + list(mapping)
        src = pd.read_parquet(path, columns=columns)
        src["date"] = pd.to_datetime(src["date"]).dt.normalize()
        src["ticker"] = normalize_ticker_series(src["ticker"])
        duplicate_count = int(src.duplicated(KEY_COLS).sum())
        src = src.drop_duplicates(KEY_COLS, keep="first").sort_values(["ticker", "date"]).copy()

        out = src[KEY_COLS].copy()
        for old_col, new_col in mapping.items():
            out[new_col] = pd.to_numeric(src[old_col], errors="coerce")
            out[new_col] = out.groupby(src["ticker"], sort=False)[new_col].shift(1)
        pieces.append(out)
        audit[filename] = {
            "status": "loaded",
            "path": str(path),
            "source_rows": int(len(src)),
            "source_duplicate_keys_removed": duplicate_count,
            "output_columns": list(mapping.values()),
        }

    if not pieces:
        return pd.DataFrame(columns=KEY_COLS), audit

    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.merge(piece, on=KEY_COLS, how="outer")
    merged = merged.drop_duplicates(KEY_COLS, keep="first")
    return merged, audit


def numeric_inf_counts(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            n = int(np.isinf(pd.to_numeric(df[col], errors="coerce").to_numpy()).sum())
            if n:
                counts[col] = n
    return counts


def validate_formulas(df: pd.DataFrame) -> dict[str, Any]:
    entry = pd.to_numeric(df["entry_price"], errors="coerce")
    exitp = pd.to_numeric(df["exit_price"], errors="coerce")
    ret = pd.to_numeric(df["overnight_return"], errors="coerce")
    calc_ret = (exitp - entry) / entry.replace(0, np.nan)
    ret_diff = (calc_ret - ret).abs()
    tp_calc = (ret > ROUNDTRIP_COST).astype("int8")
    sl_calc = (ret < SL_THRESHOLD).astype("int8")

    result = {
        "invalid_entry_price_count": int(entry.isna().sum() + (entry <= 0).sum()),
        "invalid_exit_price_count": int(exitp.isna().sum() + (exitp <= 0).sum()),
        "return_formula_mismatch_gt_1e_9": int((ret_diff > 1e-9).sum()),
        "return_formula_max_abs_diff": as_jsonable(ret_diff.max()),
        "label_tp_mismatch": int((df["label_tp"].astype("int8") != tp_calc).sum()),
        "label_sl2_mismatch": int((df["label_sl2"].astype("int8") != sl_calc).sum()),
        "exit_not_after_entry_count": int((pd.to_datetime(df["exit_date"]) <= df["date"]).sum()),
    }

    if {"pre14_ara_limit_pct", "pre14_return_from_prev_close", "pre14_ara_distance_pct"}.issubset(df.columns):
        dist_diff = (
            pd.to_numeric(df["pre14_ara_limit_pct"], errors="coerce")
            - pd.to_numeric(df["pre14_return_from_prev_close"], errors="coerce")
            - pd.to_numeric(df["pre14_ara_distance_pct"], errors="coerce")
        ).abs()
        result["pre14_ara_distance_identity_mismatch_gt_1e_6"] = int((dist_diff > 1e-6).sum())
        result["pre14_ara_distance_identity_max_abs_diff"] = as_jsonable(dist_diff.max())

    if {"pre14_market_cost_est", "pre14_spread_cost_est"}.issubset(df.columns):
        cost_diff = (
            pd.to_numeric(df["pre14_market_cost_est"], errors="coerce")
            - (ROUNDTRIP_COST + pd.to_numeric(df["pre14_spread_cost_est"], errors="coerce"))
        ).abs()
        result["pre14_market_cost_identity_mismatch_gt_1e_6"] = int((cost_diff > 1e-6).sum())
        result["pre14_market_cost_identity_max_abs_diff"] = as_jsonable(cost_diff.max())

    return result


def coverage_summary(df: pd.DataFrame, active_universe: set[str]) -> dict[str, Any]:
    by_date = df.groupby("date")["ticker"].nunique().sort_index()
    by_ticker = df.groupby("ticker")["date"].nunique().sort_values()
    latest = df["date"].max()
    latest_tickers = set(df.loc[df["date"].eq(latest), "ticker"])
    return {
        "rows": int(len(df)),
        "date_min": str(df["date"].min().date()) if len(df) else None,
        "date_max": str(df["date"].max().date()) if len(df) else None,
        "n_dates": int(df["date"].nunique()),
        "n_tickers": int(df["ticker"].nunique()),
        "rows_per_date": {
            "min": as_jsonable(by_date.min()),
            "p05": as_jsonable(by_date.quantile(0.05)),
            "median": as_jsonable(by_date.median()),
            "p95": as_jsonable(by_date.quantile(0.95)),
            "max": as_jsonable(by_date.max()),
        },
        "ticker_days": {
            "min": as_jsonable(by_ticker.min()),
            "p05": as_jsonable(by_ticker.quantile(0.05)),
            "median": as_jsonable(by_ticker.median()),
            "p95": as_jsonable(by_ticker.quantile(0.95)),
            "max": as_jsonable(by_ticker.max()),
        },
        "latest_date": str(latest.date()) if pd.notna(latest) else None,
        "latest_date_tickers": int(len(latest_tickers)),
        "latest_date_missing_vs_master_count": int(len(active_universe - latest_tickers)) if active_universe else None,
        "lowest_20_dates_by_ticker_count": {
            str(k.date()): int(v) for k, v in by_date.sort_values().head(20).items()
        },
    }


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.output.exists() and not args.allow_overwrite and not args.dry_run:
        raise FileExistsError(f"Output exists; pass --allow-overwrite to replace: {args.output}")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Refusing to write clean output over source input")

    active_universe = load_master_active(args.master_path)
    source = pd.read_parquet(args.input)
    source["date"] = pd.to_datetime(source["date"]).dt.normalize()
    source["ticker"] = normalize_ticker_series(source["ticker"])

    audit: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "input": str(args.input),
        "output": str(args.output),
        "parameters": {
            "min_date_coverage": args.min_date_coverage,
            "unsafe_prefixes": list(UNSAFE_PREFIXES),
            "unsafe_exact": sorted(UNSAFE_EXACT),
            "add_t1_quick_features": bool(args.add_t1_quick_features),
        },
        "before": coverage_summary(source, active_universe),
        "corrections": {},
        "after": {},
        "gates": {},
    }

    work = source.copy()

    duplicate_mask = work.duplicated(KEY_COLS, keep=False)
    duplicate_rows = work.loc[duplicate_mask, KEY_COLS + ["entry_price", "exit_price", "overnight_return"]]
    before_rows = len(work)
    work = work.drop_duplicates(KEY_COLS, keep="first").copy()
    audit["corrections"]["duplicate_key_rows_removed"] = int(before_rows - len(work))
    audit["corrections"]["duplicate_key_examples"] = duplicate_rows.head(20).assign(
        date=lambda x: x["date"].dt.strftime("%Y-%m-%d")
    ).to_dict("records")

    by_date = work.groupby("date")["ticker"].nunique()
    bad_dates = by_date[by_date < args.min_date_coverage].sort_index()
    low_coverage_rows_removed = int(work["date"].isin(bad_dates.index).sum())
    work = work[~work["date"].isin(bad_dates.index)].copy()
    audit["corrections"]["low_coverage_dates_dropped"] = {
        str(k.date()): int(v) for k, v in bad_dates.items()
    }
    audit["corrections"]["low_coverage_rows_removed"] = low_coverage_rows_removed

    drop_cols = unsafe_columns(list(work.columns))
    work = work.drop(columns=drop_cols)
    audit["corrections"]["unsafe_columns_dropped"] = drop_cols
    audit["corrections"]["unsafe_columns_dropped_count"] = len(drop_cols)

    if args.add_t1_quick_features:
        t1_features, t1_audit = load_t1_quick_features(args.modules_dir)
        audit["corrections"]["t1_quick_feature_sources"] = t1_audit
        if t1_features.empty:
            audit["corrections"]["t1_quick_features_added"] = []
        else:
            before_cols = set(work.columns)
            work = work.merge(t1_features, on=KEY_COLS, how="left")
            added = [c for c in work.columns if c not in before_cols]
            audit["corrections"]["t1_quick_features_added"] = added
            audit["corrections"]["t1_quick_feature_nan"] = {
                c: {"count": int(work[c].isna().sum()), "pct": float(work[c].isna().mean())}
                for c in added
            }
    else:
        audit["corrections"]["t1_quick_features_added"] = []

    work = work.sort_values(KEY_COLS).reset_index(drop=True)
    audit["after"] = coverage_summary(work, active_universe)

    formulas = validate_formulas(work)
    inf_counts = numeric_inf_counts(work)
    remaining_unsafe = unsafe_columns(list(work.columns))
    duplicate_after = int(work.duplicated(KEY_COLS).sum())
    nan_counts = work.isna().sum().sort_values(ascending=False)
    audit["gates"] = {
        "duplicate_date_ticker_after": duplicate_after,
        "remaining_unsafe_columns": remaining_unsafe,
        "numeric_inf_counts": inf_counts,
        "formula_checks": formulas,
        "columns_with_nan": int((nan_counts > 0).sum()),
        "top_nan_columns": {
            str(k): {"count": int(v), "pct": float(v / len(work))}
            for k, v in nan_counts[nan_counts > 0].head(30).items()
        },
    }

    hard_failures: list[str] = []
    if duplicate_after:
        hard_failures.append("duplicate date/ticker keys remain")
    if remaining_unsafe:
        hard_failures.append("unsafe v25 columns remain")
    if inf_counts:
        hard_failures.append("numeric inf values remain")
    for key, value in formulas.items():
        if key.endswith("_count") or key.endswith("_mismatch") or key.endswith("_mismatch_gt_1e_9") or key.endswith("_mismatch_gt_1e_6"):
            if isinstance(value, int) and value:
                hard_failures.append(f"formula gate failed: {key}={value}")
    audit["gates"]["hard_failures"] = hard_failures
    audit["status"] = "FAIL" if hard_failures else "PASS"

    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        work.to_parquet(tmp, index=False)
        tmp.replace(args.output)
        audit_path = args.output.with_suffix(args.output.suffix + ".audit.json")
        audit_path.write_text(json.dumps(audit, indent=2, default=as_jsonable) + "\n")
    else:
        audit_path = None

    print(json.dumps({
        "status": audit["status"],
        "input_rows": audit["before"]["rows"],
        "output_rows": audit["after"]["rows"],
        "output_columns": len(work.columns),
        "duplicate_rows_removed": audit["corrections"]["duplicate_key_rows_removed"],
        "low_coverage_dates_dropped": audit["corrections"]["low_coverage_dates_dropped"],
        "unsafe_columns_dropped_count": len(drop_cols),
        "hard_failures": hard_failures,
        "output": str(args.output),
        "audit": str(audit_path) if audit_path else None,
    }, indent=2))

    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
