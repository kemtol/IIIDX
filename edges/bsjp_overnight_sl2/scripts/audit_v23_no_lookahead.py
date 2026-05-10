#!/usr/bin/env python3
"""
No-lookahead audit for BSJP v23-style training artifacts.

This is a read-only diagnostic. It does not train and does not modify feature
modules. Outputs a JSON report plus small CSVs under _LOG/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


IDX_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = IDX_DIR / "model/BSJP/v23b_t1audit_clean"
DEFAULT_MODULES_DIR = IDX_DIR / "data/Level_1_Features/modules"
DEFAULT_YF_1H = IDX_DIR / "data/Level_0_Raw/yfinance_1h.parquet"
DEFAULT_YF_DAILY = IDX_DIR / "data/Level_0_Raw/yfinance_daily.parquet"
DEFAULT_GLOBAL = IDX_DIR / "data/Level_0_Raw/global_indices.parquet"
DEFAULT_L1_BROKSUM = IDX_DIR / "data/Level_1_Features/broksum_datamart.parquet"
DEFAULT_LOG_DIR = IDX_DIR / "_LOG"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cmp_counts(a: pd.Series, b: pd.Series, rtol: float = 1e-6, atol: float = 1e-6) -> dict[str, int]:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if int(mask.sum()) == 0:
        return {"match": 0, "comparable": 0}
    match = np.isclose(aa[mask], bb[mask], rtol=rtol, atol=atol)
    return {"match": int(match.sum()), "comparable": int(mask.sum())}


def _normalize_ticker(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.JK$", "", regex=True).str.upper().str.strip()


def audit_feature_set(model_dir: Path) -> dict[str, Any]:
    fi = pd.read_csv(model_dir / "feature_importance.csv")
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    features = set(fi["feature"].astype(str))

    outcome_tokens = [
        "label",
        "exit",
        "overnight_return",
        "entry_price_opening",
        "high_to_cutoff",
        "low_to_cutoff",
        "close_to_cutoff",
        "entry_datetime",
        "trade_date",
        "target",
    ]
    blacklist = set(metrics["features"].get("feature_blacklist", []))
    return {
        "feature_count": int(len(features)),
        "nonzero_gain_count": int((fi["importance_gain"] > 0).sum()),
        "exact_blacklist_present": sorted(features & blacklist),
        "outcome_like_present": sorted(
            [f for f in features if any(tok in f.lower() for tok in outcome_tokens)]
        ),
        "policy_only_unblocked": metrics["features"].get("policy_only_columns_unblocked", []),
        "best_iteration": int(metrics["model"]["best_iteration"]),
        "oot_auc": float(metrics["metrics"]["oot_valid"]["auc"]),
        "oot_aucpr": float(metrics["metrics"]["oot_valid"]["aucpr"]),
        "oot_cum_net": float(metrics["metrics"]["portfolio_oot"]["cumulative_net_return"]),
        "oot_maxdd": float(metrics["metrics"]["portfolio_oot"]["max_drawdown"]),
        "overfit_auc_gap": float(metrics["metrics"]["overfit_gap"]["auc_train_minus_oot"]),
        "overfit_flags": {
            "best_iteration_le_10": bool(metrics["model"]["best_iteration"] <= 10),
            "auc_gap_gt_0_05": bool(metrics["metrics"]["overfit_gap"]["auc_train_minus_oot"] > 0.05),
        },
    }


def audit_pre14_cutoff(
    model_dir: Path,
    modules_dir: Path,
    yf_1h_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    pre14_mod = _load_module(
        IDX_DIR / "edges/bsjp_overnight_sl2/scripts/generate_preclose14_features.py",
        "bsjp_pre14_audit_mod",
    )

    raw = pd.read_parquet(yf_1h_path)
    normalized = pre14_mod.normalize_yf_1h_session_time(raw)
    selected_bars = normalized[normalized["hour"].isin(pre14_mod.PRECLOSE_HOURS)].copy()
    session_after_14 = normalized[normalized["hour"] > 14].copy()

    module = pd.read_parquet(modules_dir / "preclose14_features.parquet")
    module["date"] = pd.to_datetime(module["date"]).dt.normalize()
    module["ticker"] = _normalize_ticker(module["ticker"])

    fi = pd.read_csv(model_dir / "feature_importance.csv")
    top_pre14_cols = [
        c
        for c in fi.loc[fi["feature"].str.startswith("pre14_"), "feature"].head(40).tolist()
        if c in module.columns
    ]
    fresh = pre14_mod.build_preclose14_features(raw)
    fresh["date"] = pd.to_datetime(fresh["date"]).dt.normalize()
    fresh["ticker"] = _normalize_ticker(fresh["ticker"])
    compare_cols = ["date", "ticker"] + top_pre14_cols
    joined = module[compare_cols].merge(
        fresh[compare_cols],
        on=["date", "ticker"],
        how="inner",
        suffixes=("_module", "_fresh"),
    )
    rows = []
    for col in top_pre14_cols:
        counts = _cmp_counts(joined[f"{col}_module"], joined[f"{col}_fresh"])
        rows.append({"feature": col, **counts})
    comp = pd.DataFrame(rows)
    comp_path = output_dir / "v23b_pre14_reconstruction_audit_20260509.csv"
    comp.to_csv(comp_path, index=False)

    # prev_close must be previous available session close, not current-day last.
    daily_last = (
        normalized.dropna(subset=["close"])
        .sort_values(["ticker", "date", "datetime"])
        .groupby(["ticker", "date"], sort=False)["close"]
        .last()
        .reset_index(name="current_session_last_close")
        .sort_values(["ticker", "date"])
    )
    daily_last["expected_prev_close"] = daily_last.groupby("ticker", sort=False)[
        "current_session_last_close"
    ].shift(1)
    prev_join = module[["date", "ticker", "pre14_prev_close"]].merge(
        daily_last, on=["date", "ticker"], how="inner"
    )
    prev_expected = _cmp_counts(prev_join["pre14_prev_close"], prev_join["expected_prev_close"])
    prev_sameday = _cmp_counts(prev_join["pre14_prev_close"], prev_join["current_session_last_close"])

    bad_reconstruction = comp[comp["match"] != comp["comparable"]]
    return {
        "preclose_hours_allowed": sorted(int(x) for x in pre14_mod.PRECLOSE_HOURS),
        "selected_bar_max_hour": int(selected_bars["hour"].max()) if not selected_bars.empty else None,
        "selected_bars_after_14_count": int((selected_bars["hour"] > 14).sum()),
        "session_rows_after_14_available_but_not_selected": int(len(session_after_14)),
        "top_pre14_reconstruction_csv": str(comp_path.relative_to(IDX_DIR)),
        "top_pre14_reconstruction_failed_features": bad_reconstruction.to_dict("records"),
        "pre14_prev_close_expected_prev_match": prev_expected,
        "pre14_prev_close_current_day_match": prev_sameday,
    }


def audit_macro(modules_dir: Path, global_path: Path) -> dict[str, Any]:
    bpjs_mod = _load_module(
        IDX_DIR / "edges/bpjs_opening_tp3/scripts/generate_datamart.py",
        "bpjs_datamart_audit_mod",
    )
    module = pd.read_parquet(modules_dir / "global_indices_features.parquet")
    module["date"] = pd.to_datetime(module["date"]).dt.normalize()
    fresh = bpjs_mod.load_global_indices(global_path)
    fresh["date"] = pd.to_datetime(fresh["date"]).dt.normalize()
    cols = [c for c in module.columns if c != "date" and c in fresh.columns]
    joined = module.merge(fresh, on="date", how="inner", suffixes=("_module", "_fresh"))
    exact = {}
    for col in cols:
        exact[col] = _cmp_counts(joined[f"{col}_module"], joined[f"{col}_fresh"])

    raw = pd.read_parquet(global_path)
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")

    same_day_checks = {}
    symbol_map = {
        "ihsg_prev_close": "^JKSE",
        "usdidr_prev_close": "IDR=X",
        "vix_prev_close": "^VIX",
    }
    for feat, symbol in symbol_map.items():
        same = raw[raw["symbol"] == symbol][["date", "close"]].rename(columns={"close": "same_day_close"})
        tmp = module[["date", feat]].merge(same, on="date", how="inner")
        same_day_checks[feat] = _cmp_counts(tmp[feat], tmp["same_day_close"])

    return {
        "exact_rebuild_counts": exact,
        "same_day_close_match_counts": same_day_checks,
    }


def audit_broker_shift(modules_dir: Path, l1_path: Path, output_dir: Path) -> dict[str, Any]:
    check_cols = [
        "yf_daily_high",
        "yf_daily_close",
        "yf_daily_range_pct",
        "flow_total_net_buy",
        "ctx_broker_market_share",
        "tfl_net_buy_z_20",
    ]
    l1 = pd.read_parquet(l1_path, columns=["date", "ticker", "broker", *check_cols])
    l1["date"] = pd.to_datetime(l1["date"]).dt.normalize()
    l1["ticker"] = _normalize_ticker(l1["ticker"])
    l1["broker"] = l1["broker"].astype(str).str.upper().str.strip()
    l1 = l1.sort_values(["broker", "ticker", "date"])

    raw_agg = (
        l1.groupby(["date", "ticker"], sort=False)[check_cols]
        .mean()
        .reset_index()
        .rename(columns={c: f"{c}_raw_mean" for c in check_cols})
    )
    shifted = l1.copy()
    shifted[check_cols] = shifted.groupby(["broker", "ticker"], sort=False)[check_cols].shift(1)
    shifted_agg = (
        shifted.groupby(["date", "ticker"], sort=False)[check_cols]
        .mean()
        .reset_index()
        .rename(columns={c: f"{c}_shift_mean" for c in check_cols})
    )

    module_cols = [
        "date",
        "ticker",
        "yf_daily_high_mean",
        "yf_daily_close_mean",
        "yf_daily_range_pct_mean",
        "flow_total_net_buy_mean",
        "ctx_broker_market_share_mean",
        "tfl_net_buy_z_20_mean",
    ]
    module = pd.read_parquet(modules_dir / "broker_aggregate_features.parquet", columns=module_cols)
    module["date"] = pd.to_datetime(module["date"]).dt.normalize()
    module["ticker"] = _normalize_ticker(module["ticker"])
    base = module.merge(raw_agg, on=["date", "ticker"], how="left").merge(
        shifted_agg, on=["date", "ticker"], how="left"
    )

    pairs = [
        ("yf_daily_high_mean", "yf_daily_high_raw_mean", "yf_daily_high_shift_mean"),
        ("yf_daily_close_mean", "yf_daily_close_raw_mean", "yf_daily_close_shift_mean"),
        ("yf_daily_range_pct_mean", "yf_daily_range_pct_raw_mean", "yf_daily_range_pct_shift_mean"),
        ("flow_total_net_buy_mean", "flow_total_net_buy_raw_mean", "flow_total_net_buy_shift_mean"),
        ("ctx_broker_market_share_mean", "ctx_broker_market_share_raw_mean", "ctx_broker_market_share_shift_mean"),
        ("tfl_net_buy_z_20_mean", "tfl_net_buy_z_20_raw_mean", "tfl_net_buy_z_20_shift_mean"),
    ]
    rows = []
    for feat, raw_col, shift_col in pairs:
        raw_counts = _cmp_counts(base[feat], base[raw_col])
        shift_counts = _cmp_counts(base[feat], base[shift_col])
        rows.append(
            {
                "feature": feat,
                "raw_same_day_match": raw_counts["match"],
                "raw_same_day_comparable": raw_counts["comparable"],
                "shifted_match": shift_counts["match"],
                "shifted_comparable": shift_counts["comparable"],
            }
        )
    out = pd.DataFrame(rows)
    path = output_dir / "v23b_broker_shift_audit_20260509.csv"
    out.to_csv(path, index=False)
    failed = out[out["shifted_match"] != out["shifted_comparable"]]
    return {
        "broker_shift_csv": str(path.relative_to(IDX_DIR)),
        "failed_shift_features": failed.to_dict("records"),
        "rows": out.to_dict("records"),
    }


def audit_ara_history(modules_dir: Path, yf_daily_path: Path) -> dict[str, Any]:
    ara_mod = _load_module(
        IDX_DIR / "edges/bsjp_overnight_sl2/scripts/generate_ara_history_features.py",
        "ara_history_audit_mod",
    )
    module = pd.read_parquet(modules_dir / "ara_history_features.parquet")
    module["date"] = pd.to_datetime(module["date"]).dt.normalize()
    module["ticker"] = _normalize_ticker(module["ticker"])
    fresh = ara_mod.build_ara_history_features(pd.read_parquet(yf_daily_path))
    fresh["date"] = pd.to_datetime(fresh["date"]).dt.normalize()
    fresh["ticker"] = _normalize_ticker(fresh["ticker"])
    cols = [
        "was_ara_tminus1",
        "ara_count_5d",
        "ara_count_20d",
        "days_since_last_ara",
        "last_ara_return",
        "max_return_5d_tminus1",
        "max_return_20d_tminus1",
    ]
    joined = module[["date", "ticker", *cols]].merge(
        fresh[["date", "ticker", *cols]],
        on=["date", "ticker"],
        how="inner",
        suffixes=("_module", "_fresh"),
    )
    counts = {col: _cmp_counts(joined[f"{col}_module"], joined[f"{col}_fresh"]) for col in cols}
    return {"exact_rebuild_counts": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit v23 no-lookahead invariants.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--modules-dir", type=Path, default=DEFAULT_MODULES_DIR)
    parser.add_argument("--yf-1h-path", type=Path, default=DEFAULT_YF_1H)
    parser.add_argument("--yf-daily-path", type=Path, default=DEFAULT_YF_DAILY)
    parser.add_argument("--global-path", type=Path, default=DEFAULT_GLOBAL)
    parser.add_argument("--l1-broksum-path", type=Path, default=DEFAULT_L1_BROKSUM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    args.model_dir = args.model_dir.resolve()
    args.modules_dir = args.modules_dir.resolve()
    args.yf_1h_path = args.yf_1h_path.resolve()
    args.yf_daily_path = args.yf_daily_path.resolve()
    args.global_path = args.global_path.resolve()
    args.l1_broksum_path = args.l1_broksum_path.resolve()
    args.output_dir = args.output_dir.resolve()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model_dir": str(args.model_dir.relative_to(IDX_DIR)),
        "feature_set": audit_feature_set(args.model_dir),
        "pre14_cutoff": audit_pre14_cutoff(args.model_dir, args.modules_dir, args.yf_1h_path, args.output_dir),
        "macro_timing": audit_macro(args.modules_dir, args.global_path),
        "broker_shift": audit_broker_shift(args.modules_dir, args.l1_broksum_path, args.output_dir),
        "ara_history": audit_ara_history(args.modules_dir, args.yf_daily_path),
    }
    report["failures"] = {
        "feature_blacklist_present": bool(report["feature_set"]["exact_blacklist_present"]),
        "outcome_like_present": bool(report["feature_set"]["outcome_like_present"]),
        "policy_only_unblocked": bool(report["feature_set"]["policy_only_unblocked"]),
        "pre14_reconstruction_failed": bool(report["pre14_cutoff"]["top_pre14_reconstruction_failed_features"]),
        "pre14_selected_after_14": report["pre14_cutoff"]["selected_bars_after_14_count"] > 0,
        "broker_shift_failed": bool(report["broker_shift"]["failed_shift_features"]),
    }
    out_path = args.output_dir / f"{args.model_dir.name}_no_lookahead_audit_20260509.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["failures"], indent=2))
    print(f"[Done] {out_path}")


if __name__ == "__main__":
    main()
