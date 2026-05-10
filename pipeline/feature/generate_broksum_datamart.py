#!/usr/bin/env python3
"""
Level 1 datamart generator for broker summary features.

Orchestrator flow:
1. Refresh Level 0 sources (broksum + yfinance 1h/4h + yfinance daily).
2. Build broker-centric features from broksum panel.
3. Build market context features from yfinance datasets.
4. Merge all features into one parquet output for downstream layers.

Design notes:
- Feature temporal baselines use shift(1) to avoid look-ahead.
- Refresh step is enabled by default; disable with --skip-refresh-level0.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


MACHINELEARNING_DIR = Path(__file__).resolve().parents[2]
# Support both layouts:
# - legacy:   <repo>/machinelearning/service/...
# - current:  <repo>/machinelearning/idx/service/...
PROJECT_ROOT = MACHINELEARNING_DIR.parent
if not (PROJECT_ROOT / ".venv").exists() and (PROJECT_ROOT.parent / ".venv").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
DATA_DIR = MACHINELEARNING_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "Level_0_Raw"
if not RAW_DATA_DIR.exists():
    RAW_DATA_DIR = DATA_DIR
FETCH_DIR = MACHINELEARNING_DIR / "pipeline" / "fetch"

DEFAULT_BROKSUM_INPUT = RAW_DATA_DIR / "broksum_bybroker.parquet"
DEFAULT_OUTPUT = DATA_DIR / "Level_1_Features" / "broksum_datamart.parquet"
DEFAULT_YF_DAILY = RAW_DATA_DIR / "yfinance_daily.parquet"
DEFAULT_YF_1H = RAW_DATA_DIR / "yfinance_1h.parquet"
DEFAULT_YF_4H = RAW_DATA_DIR / "yfinance_4h.parquet"
DEFAULT_MASTER_EMITEN = RAW_DATA_DIR / "master_emiten.parquet"
DEFAULT_MASTER_BROKER = RAW_DATA_DIR / "master_broker.parquet"

FETCH_BROKSUM_SCRIPT = FETCH_DIR / "fetch_broksum_ipot.py"
FETCH_YFINANCE_SCRIPT = FETCH_DIR / "fetch_yfinance.py"
FETCH_YFINANCE_DAILY_SCRIPT = FETCH_DIR / "fetch_yfinance_daily.py"
DEFAULT_VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

KEY_COLUMNS = ["date", "broker", "ticker"]
MIN_REQUIRED_COLUMNS = {"date", "broker", "ticker", "total_net_buy"}


def parse_args() -> argparse.Namespace:
    default_python_bin = str(DEFAULT_VENV_PYTHON) if DEFAULT_VENV_PYTHON.exists() else sys.executable

    parser = argparse.ArgumentParser(
        description="Generate Level 1 broksum datamart with automatic Level 0 refresh."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_BROKSUM_INPUT,
        help=f"Input broksum parquet. Default: {DEFAULT_BROKSUM_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output datamart parquet. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--yfinance-daily-path",
        type=Path,
        default=DEFAULT_YF_DAILY,
        help=f"Daily yfinance parquet path. Default: {DEFAULT_YF_DAILY}",
    )
    parser.add_argument(
        "--yfinance-1h-path",
        type=Path,
        default=DEFAULT_YF_1H,
        help=f"1h yfinance parquet path. Default: {DEFAULT_YF_1H}",
    )
    parser.add_argument(
        "--yfinance-4h-path",
        type=Path,
        default=DEFAULT_YF_4H,
        help=f"4h yfinance parquet path. Default: {DEFAULT_YF_4H}",
    )
    parser.add_argument(
        "--master-emiten-path",
        type=Path,
        default=DEFAULT_MASTER_EMITEN,
        help=f"Master emiten parquet path. Default: {DEFAULT_MASTER_EMITEN}",
    )
    parser.add_argument(
        "--master-broker-path",
        type=Path,
        default=DEFAULT_MASTER_BROKER,
        help=f"Master broker parquet path for broker_type enrichment. Default: {DEFAULT_MASTER_BROKER}",
    )
    parser.add_argument("--date-from", type=str, default=None, help="Inclusive date filter (YYYY-MM-DD).")
    parser.add_argument("--date-to", type=str, default=None, help="Inclusive date filter (YYYY-MM-DD).")
    parser.add_argument(
        "--windows",
        type=str,
        default="5,20,60",
        help="Comma-separated rolling windows. Example: 5,20,60",
    )
    parser.add_argument(
        "--warmup-bdays-multiplier",
        type=int,
        default=5,
        help=(
            "When --date-from is set, include warmup history before it for rolling features. "
            "Warmup size = max(window) * multiplier business days."
        ),
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=0,
        help="Use first N rows after normalization for fast testing. 0 means full data.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run pipeline without writing output parquet.")

    parser.add_argument(
        "--refresh-level0",
        dest="refresh_level0",
        action="store_true",
        default=True,
        help="Refresh Level 0 sources before feature generation (default).",
    )
    parser.add_argument(
        "--skip-refresh-level0",
        dest="refresh_level0",
        action="store_false",
        help="Skip Level 0 refresh and use existing parquet snapshots.",
    )
    parser.add_argument(
        "--strict-refresh",
        action="store_true",
        help="Fail fast if any refresh step fails.",
    )
    parser.add_argument(
        "--refresh-broksum",
        dest="refresh_broksum",
        action="store_true",
        default=True,
        help="Run broksum fetch refresh step (default).",
    )
    parser.add_argument(
        "--skip-refresh-broksum",
        dest="refresh_broksum",
        action="store_false",
        help="Skip broksum fetch refresh step.",
    )
    parser.add_argument(
        "--refresh-yf-intraday",
        dest="refresh_yf_intraday",
        action="store_true",
        default=True,
        help="Run yfinance intraday (1h/4h) refresh step (default).",
    )
    parser.add_argument(
        "--skip-refresh-yf-intraday",
        dest="refresh_yf_intraday",
        action="store_false",
        help="Skip yfinance intraday (1h/4h) refresh step.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=default_python_bin,
        help=f"Python binary for calling refresh scripts. Default: {default_python_bin}",
    )

    parser.add_argument("--refresh-broksum-days", type=int, default=1, help="Recent weekday count for broksum update.")
    parser.add_argument(
        "--refresh-broksum-repair-days",
        type=int,
        default=14,
        help="Repair window in append mode for broksum update.",
    )
    parser.add_argument(
        "--refresh-broksum-max-concurrent",
        type=int,
        default=8,
        help="Max broker concurrency for broksum refresh.",
    )
    parser.add_argument(
        "--refresh-broksum-day-sleep",
        type=float,
        default=0.05,
        help="Inter-request sleep for broksum refresh.",
    )

    parser.add_argument(
        "--refresh-yf-intervals",
        type=str,
        default="1h,4h",
        help="Intervals passed to fetch_yfinance.py",
    )
    parser.add_argument(
        "--refresh-yf-limit-tickers",
        type=int,
        default=0,
        help="Optional ticker limit for yfinance refresh (0 = all).",
    )
    parser.add_argument(
        "--refresh-yf-pause-seconds",
        type=float,
        default=0.2,
        help="Pause between ticker requests in intraday yfinance refresh.",
    )

    parser.add_argument(
        "--refresh-yf-daily",
        dest="refresh_yf_daily",
        action="store_true",
        default=True,
        help="Refresh yfinance_daily.parquet using incremental 1d pulls (default).",
    )
    parser.add_argument(
        "--skip-refresh-yf-daily",
        dest="refresh_yf_daily",
        action="store_false",
        help="Skip daily yfinance refresh.",
    )
    parser.add_argument(
        "--refresh-yf-daily-min-fetch-days",
        type=int,
        default=14,
        help="Minimum daily yfinance fetch period in days.",
    )
    parser.add_argument(
        "--refresh-yf-daily-buffer-days",
        type=int,
        default=5,
        help="Extra daily fetch overlap to repair late bars.",
    )
    parser.add_argument(
        "--refresh-yf-daily-max-fetch-days",
        type=int,
        default=1825,
        help="Maximum daily yfinance fetch period in days.",
    )
    parser.add_argument(
        "--refresh-yf-daily-pause-seconds",
        type=float,
        default=0.05,
        help="Pause between ticker requests in daily yfinance refresh.",
    )
    parser.add_argument(
        "--refresh-yf-daily-limit-tickers",
        type=int,
        default=0,
        help="Optional ticker limit for daily yfinance refresh (0 = all active tickers).",
    )
    return parser.parse_args()


def parse_windows(windows_text: str) -> list[int]:
    windows = []
    for token in windows_text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"Rolling window must be > 0. Found: {value}")
        windows.append(value)
    if not windows:
        raise ValueError("No valid rolling windows parsed. Example: --windows 5,20,60")
    return sorted(set(windows))


def derive_feature_date_from(
    date_from: str | None,
    windows: list[int],
    warmup_bdays_multiplier: int,
) -> str | None:
    if not date_from:
        return None
    if warmup_bdays_multiplier <= 0 or not windows:
        return date_from

    base = pd.to_datetime(date_from)
    warmup_bdays = max(windows) * warmup_bdays_multiplier
    return (base - BDay(warmup_bdays)).strftime("%Y-%m-%d")


def print_window_readiness(panel: pd.DataFrame, windows: list[int]) -> None:
    if panel.empty:
        return

    counts = panel.groupby(["broker", "ticker"], sort=False).size()
    if counts.empty:
        return

    total_pairs = len(counts)
    max_obs = int(counts.max())
    for window in windows:
        eligible = int((counts >= window).sum())
        pct = (eligible / total_pairs) * 100.0 if total_pairs else 0.0
        print(
            f"[WindowReady] w={window} -> eligible_pairs={eligible:,}/{total_pairs:,} "
            f"({pct:.2f}%), max_obs_per_pair={max_obs}"
        )


def normalize_ticker_series(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.upper().str.strip()
    out = out.str.replace(r"[^A-Z0-9.]", "", regex=True)
    out = out.str.replace(r"\.JK$", "", regex=True)
    return out


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return num / den


def run_subprocess_command(cmd: list[str], *, label: str, strict: bool) -> bool:
    print(f"[Refresh] {label}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    if result.returncode == 0:
        print(f"[Refresh] {label}: success")
        return True

    msg = f"[Refresh] {label}: failed (exit={result.returncode})"
    if strict:
        raise RuntimeError(msg)
    print(f"{msg} -> continue with existing snapshot")
    return False


def refresh_broksum_source(args: argparse.Namespace) -> bool:
    cmd = [
        args.python_bin,
        str(FETCH_BROKSUM_SCRIPT),
        "--all-brokers",
        "--days",
        str(args.refresh_broksum_days),
        "--repair-days",
        str(args.refresh_broksum_repair_days),
        "--update-mode",
        "append",
        "--source-mode",
        "auto",
        "--max-concurrent",
        str(args.refresh_broksum_max_concurrent),
        "--day-sleep",
        str(args.refresh_broksum_day_sleep),
        "--output",
        str(args.input),
        "--parquet-store",
        str(args.input),
    ]
    return run_subprocess_command(cmd, label="broksum", strict=args.strict_refresh)


def refresh_yfinance_intraday_source(args: argparse.Namespace) -> bool:
    cmd = [
        args.python_bin,
        str(FETCH_YFINANCE_SCRIPT),
        "--intervals",
        args.refresh_yf_intervals,
        "--pause-seconds",
        str(args.refresh_yf_pause_seconds),
        "--data-dir",
        str(RAW_DATA_DIR),
        "--master-path",
        str(args.master_emiten_path),
    ]
    if args.refresh_yf_limit_tickers > 0:
        cmd += ["--limit-tickers", str(args.refresh_yf_limit_tickers)]
    return run_subprocess_command(cmd, label="yfinance_intraday", strict=args.strict_refresh)


def refresh_yfinance_daily_source(args: argparse.Namespace) -> bool:
    cmd = [
        args.python_bin,
        str(FETCH_YFINANCE_DAILY_SCRIPT),
        "--master-path",
        str(args.master_emiten_path),
        "--output",
        str(args.yfinance_daily_path),
        "--pause-seconds",
        str(args.refresh_yf_daily_pause_seconds),
        "--min-fetch-days",
        str(args.refresh_yf_daily_min_fetch_days),
        "--buffer-days",
        str(args.refresh_yf_daily_buffer_days),
        "--max-fetch-days",
        str(args.refresh_yf_daily_max_fetch_days),
    ]
    if args.refresh_yf_daily_limit_tickers > 0:
        cmd += ["--limit-tickers", str(args.refresh_yf_daily_limit_tickers)]
    return run_subprocess_command(cmd, label="yfinance_daily", strict=args.strict_refresh)


def refresh_level0_sources(args: argparse.Namespace) -> None:
    if not args.refresh_level0:
        print("[Refresh] Skipped (--skip-refresh-level0).")
        return

    if args.refresh_broksum:
        refresh_broksum_source(args)
    else:
        print("[Refresh] broksum skipped (--skip-refresh-broksum).")

    if args.refresh_yf_intraday:
        refresh_yfinance_intraday_source(args)
    else:
        print("[Refresh] yfinance_intraday skipped (--skip-refresh-yf-intraday).")

    if args.refresh_yf_daily:
        refresh_yfinance_daily_source(args)
    else:
        print("[Refresh] yfinance_daily skipped (--skip-refresh-yf-daily).")


def print_source_snapshot(
    broksum_path: Path,
    yf_daily_path: Path,
    yf_1h_path: Path,
    yf_4h_path: Path,
) -> None:
    def max_time(path: Path, column: str) -> str:
        if not path.exists() or path.stat().st_size == 0:
            return "missing"
        try:
            s = pd.read_parquet(path, columns=[column])[column]
            t = pd.to_datetime(s, errors="coerce").dropna()
            if t.empty:
                return "n/a"
            return str(t.max())
        except Exception as exc:  # noqa: BLE001
            return f"error:{type(exc).__name__}"

    print("[SourceSnapshot]")
    print(f"  broksum_bybroker max(date): {max_time(broksum_path, 'date')}")
    print(f"  yfinance_daily   max(date): {max_time(yf_daily_path, 'date')}")
    print(f"  yfinance_1h  max(datetime): {max_time(yf_1h_path, 'datetime')}")
    print(f"  yfinance_4h  max(datetime): {max_time(yf_4h_path, 'datetime')}")


def build_parquet_filters(column: str, date_from: str | pd.Timestamp | None, date_to: str | pd.Timestamp | None, target_is_string: bool = False):
    filters = []
    f_from = pd.to_datetime(date_from) if date_from is not None else None
    f_to = pd.to_datetime(date_to) if date_to is not None else None

    if column == "datetime":
        if f_from is not None and f_from.tzinfo is None:
            f_from = f_from.tz_localize("UTC")
        if f_to is not None and f_to.tzinfo is None:
            f_to = f_to.tz_localize("UTC")
    
    if target_is_string:
        if f_from is not None: f_from = f_from.strftime("%Y-%m-%d")
        if f_to is not None: f_to = f_to.strftime("%Y-%m-%d")

    if f_from is not None:
        filters.append((column, ">=", f_from))
    if f_to is not None:
        filters.append((column, "<=", f_to))
    return filters or None


def normalize_schema(raw_df: pd.DataFrame) -> pd.DataFrame:
    alias_map = {
        "stock_code": "ticker",
        "net_val": "total_net_buy",
        "buy_val": "gross_buy",
        "sell_val": "gross_sell",
        "total_val": "gross_turnover",
        "buy_vol": "buy_volume",
        "sell_vol": "sell_volume",
        "net_vol": "net_volume",
        "breadth": "market_breadth",
    }

    normalized = raw_df.rename(columns=alias_map).copy()
    missing = [col for col in MIN_REQUIRED_COLUMNS if col not in normalized.columns]
    if missing:
        raise ValueError(
            f"Input missing required columns after alias mapping: {missing}. "
            f"Available={list(raw_df.columns)}"
        )

    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
    normalized["broker"] = normalized["broker"].astype(str).str.strip().str.upper()
    normalized["ticker"] = normalize_ticker_series(normalized["ticker"])

    numeric_columns = [
        "total_net_buy",
        "gross_buy",
        "gross_sell",
        "gross_turnover",
        "buy_volume",
        "sell_volume",
        "net_volume",
        "buy_freq",
        "sell_freq",
        "avg_buy_price",
        "avg_sell_price",
        "market_breadth",
    ]
    for col in numeric_columns:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    normalized = normalized.dropna(subset=["date", "broker", "ticker"])
    normalized = normalized.drop_duplicates(subset=KEY_COLUMNS, keep="last")
    normalized = normalized.sort_values(KEY_COLUMNS).reset_index(drop=True)
    return normalized


def enrich_broker_metadata(panel: pd.DataFrame, master_broker_path: Path) -> pd.DataFrame:
    """Replace broker_type='Unknown' with derived type from master_broker composition."""
    if not master_broker_path.exists():
        print(f"[BrokerMeta] master_broker not found at {master_broker_path}, skipping enrichment")
        return panel

    available_cols = pd.read_parquet(master_broker_path, columns=["broker_code"]).columns.tolist()
    load_cols = ["broker_code", "foreignfund_%", "localfund_%", "retail_%"]
    if "is_foreign" in available_cols:
        load_cols.append("is_foreign")
    mb = pd.read_parquet(master_broker_path, columns=load_cols)
    mb["broker_type"] = mb[["foreignfund_%", "localfund_%", "retail_%"]].idxmax(axis=1).map({
        "foreignfund_%": "Foreign",
        "localfund_%": "LocalFund",
        "retail_%": "Retail",
    })
    if "is_foreign" not in mb.columns:
        mb["is_foreign"] = (mb["broker_type"] == "Foreign").astype("int8")
    mb = mb.rename(columns={"broker_code": "broker"})[["broker", "broker_type", "is_foreign"]]

    out = panel.drop(columns=["broker_type"], errors="ignore")
    out = out.merge(mb, on="broker", how="left")
    out["broker_type"] = out["broker_type"].fillna("Unknown")
    out["is_foreign"] = out["is_foreign"].fillna(0).astype("int8")

    n_enriched = (out["broker_type"] != "Unknown").sum()
    print(f"[BrokerMeta] enriched {n_enriched:,}/{len(out):,} rows — types: {out['broker_type'].value_counts().to_dict()}")
    return out


def apply_date_filter(panel: pd.DataFrame, date_from: str | None, date_to: str | None) -> pd.DataFrame:
    out = panel
    if date_from:
        out = out[out["date"] >= pd.to_datetime(date_from)]
    if date_to:
        out = out[out["date"] <= pd.to_datetime(date_to)]
    return out.reset_index(drop=True)


def build_base_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()

    if "gross_turnover" not in df.columns:
        if {"gross_buy", "gross_sell"}.issubset(df.columns):
            df["gross_turnover"] = df["gross_buy"].fillna(0) + df["gross_sell"].fillna(0)
        else:
            df["gross_turnover"] = np.nan

    if "buy_volume" in df.columns and "sell_volume" in df.columns and "net_volume" not in df.columns:
        df["net_volume"] = df["buy_volume"].fillna(0) - df["sell_volume"].fillna(0)

    if {"gross_buy", "gross_sell"}.issubset(df.columns):
        buy = df["gross_buy"].fillna(0)
        sell = df["gross_sell"].fillna(0)
        df["is_buy_only"] = ((buy > 0) & (sell == 0)).astype("int8")
        df["is_sell_only"] = ((sell > 0) & (buy == 0)).astype("int8")

    df["abs_net_buy"] = df["total_net_buy"].abs()
    df["net_flow_ratio"] = safe_div(df["total_net_buy"], df["gross_turnover"])

    if {"gross_buy", "gross_sell"}.issubset(df.columns):
        ratio = safe_div(df["gross_buy"], df["gross_sell"])
        ratio = ratio.where(df["is_buy_only"] == 0, 99.0)
        ratio = ratio.where(df["is_sell_only"] == 0, 0.01)
        df["buy_sell_val_ratio"] = ratio
    if {"buy_volume", "sell_volume"}.issubset(df.columns):
        vratio = safe_div(df["buy_volume"], df["sell_volume"])
        if "is_buy_only" in df.columns:
            vratio = vratio.where(df["is_buy_only"] == 0, 99.0)
            vratio = vratio.where(df["is_sell_only"] == 0, 0.01)
        df["buy_sell_vol_ratio"] = vratio
    if {"buy_freq", "sell_freq"}.issubset(df.columns):
        fratio = safe_div(df["buy_freq"], df["sell_freq"])
        if "is_buy_only" in df.columns:
            fratio = fratio.where(df["is_buy_only"] == 0, 99.0)
            fratio = fratio.where(df["is_sell_only"] == 0, 0.01)
        df["buy_sell_freq_ratio"] = fratio
        df["total_trades"] = df["buy_freq"].fillna(0) + df["sell_freq"].fillna(0)
        df["net_buy_per_trade"] = safe_div(df["total_net_buy"], df["total_trades"])

    df["churn_ratio"] = safe_div(df["gross_turnover"], df["abs_net_buy"])

    if {"avg_buy_price", "avg_sell_price"}.issubset(df.columns):
        df["avg_spread_price"] = df["avg_sell_price"] - df["avg_buy_price"]
        df["avg_spread_pct"] = safe_div(df["avg_spread_price"], df["avg_buy_price"])
    return df


def build_context_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()

    broker_day_abs_flow = df.groupby(["date", "broker"], sort=False)["abs_net_buy"].transform("sum")
    ticker_day_abs_flow = df.groupby(["date", "ticker"], sort=False)["abs_net_buy"].transform("sum")
    market_day_abs_flow = df.groupby("date", sort=False)["abs_net_buy"].transform("sum")

    df["broker_day_abs_flow"] = broker_day_abs_flow
    df["ticker_day_abs_flow"] = ticker_day_abs_flow
    df["market_day_abs_flow"] = market_day_abs_flow

    df["broker_ticker_specificity"] = safe_div(df["abs_net_buy"], broker_day_abs_flow)
    df["broker_market_share"] = safe_div(broker_day_abs_flow, market_day_abs_flow)
    df["ticker_market_share"] = safe_div(ticker_day_abs_flow, market_day_abs_flow)

    df["broker_net_buy_rank"] = (
        df.groupby(["date", "ticker"], sort=False)["total_net_buy"]
        .rank(method="average", ascending=True, pct=True)
    )
    return df


def rolling_group_stat(
    series: pd.Series,
    group_keys: list[pd.Series],
    window: int,
    stat: str,
) -> pd.Series:
    grouped = series.groupby(group_keys, sort=False).rolling(window=window, min_periods=window)
    if stat == "mean":
        out = grouped.mean()
    elif stat == "std":
        out = grouped.std()
    else:
        raise ValueError(f"Unsupported rolling stat: {stat}")
    return out.reset_index(level=list(range(len(group_keys))), drop=True)


def build_temporal_features(panel: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = panel.sort_values(["broker", "ticker", "date"]).copy()
    broker_key = df["broker"]
    ticker_key = df["ticker"]

    base_temporal_cols = [col for col in ["total_net_buy", "net_flow_ratio", "churn_ratio"] if col in df.columns]
    for col in base_temporal_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in base_temporal_cols:
        shifted = df.groupby(["broker", "ticker"], sort=False)[col].shift(1)
        df[f"{col}_lag1"] = shifted

        for window in windows:
            mean_col = f"{col}_ma_{window}"
            df[mean_col] = rolling_group_stat(shifted, [broker_key, ticker_key], window, "mean")

            if col == "total_net_buy":
                std_col = f"{col}_std_{window}"
                z_col = f"{col}_z_{window}"
                velocity_col = f"{col}_velocity_{window}"
                df[std_col] = rolling_group_stat(shifted, [broker_key, ticker_key], window, "std")
                df[z_col] = safe_div(df[col] - df[mean_col], df[std_col]).clip(-10, 10)
                df[velocity_col] = safe_div(df[col], df[mean_col])

    sign_shifted = df.groupby(["broker", "ticker"], sort=False)["total_net_buy"].shift(1).gt(0).astype(float)
    df["net_buy_sign_consistency_5"] = rolling_group_stat(sign_shifted, [broker_key, ticker_key], 5, "mean")

    return df.sort_values(KEY_COLUMNS).reset_index(drop=True)


def build_activity_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add active_days_in_5 and days_since_last_active per broker×ticker."""
    df = panel.sort_values(["broker", "ticker", "date"]).copy()
    broker_key = df["broker"]
    ticker_key = df["ticker"]

    active_shifted = df.groupby(["broker", "ticker"], sort=False)["total_net_buy"].transform(
        lambda x: (x.abs() > 0).shift(1).astype(float)
    )
    df["active_days_in_5"] = rolling_group_stat(active_shifted, [broker_key, ticker_key], 5, "mean")
    df["active_days_in_5"] = (df["active_days_in_5"] * 5).round().astype("Int8")

    def _days_since(x: pd.Series) -> pd.Series:
        shifted = (x.abs() > 0).shift(1)
        result, counter = [], np.nan
        for is_act in shifted:
            if pd.isna(is_act):
                result.append(np.nan)
            elif is_act:
                counter = 0
                result.append(0.0)
            else:
                counter = counter + 1 if not pd.isna(counter) else np.nan
                result.append(counter)
        return pd.Series(result, index=x.index, dtype=float)

    df["days_since_last_active"] = (
        df.groupby(["broker", "ticker"], sort=False)["total_net_buy"].transform(_days_since)
    )

    return df.sort_values(KEY_COLUMNS).reset_index(drop=True)


def build_price_context_features(datamart: pd.DataFrame) -> pd.DataFrame:
    """Add broker price aggression features — requires yf_daily already merged."""
    df = datamart.copy()

    if {"avg_buy_price", "yf_daily_low", "yf_daily_high"}.issubset(df.columns):
        day_range = df["yf_daily_high"] - df["yf_daily_low"]
        df["avg_buy_price_in_range"] = safe_div(
            df["avg_buy_price"] - df["yf_daily_low"], day_range
        ).clip(0, 1)

    if {"avg_sell_price", "yf_daily_low", "yf_daily_high"}.issubset(df.columns):
        day_range = df["yf_daily_high"] - df["yf_daily_low"]
        df["avg_sell_price_in_range"] = safe_div(
            df["avg_sell_price"] - df["yf_daily_low"], day_range
        ).clip(0, 1)

    if {"buy_volume", "yf_daily_volume"}.issubset(df.columns):
        df["buy_volume_in_total_volume"] = safe_div(
            df["buy_volume"], df["yf_daily_volume"]
        ).clip(0, 1)

    n = len(df)
    for col in ["avg_buy_price_in_range", "avg_sell_price_in_range", "buy_volume_in_total_volume"]:
        if col in df.columns:
            valid = df[col].notna().sum()
            print(f"[PriceCtx] {col}: {valid:,}/{n:,} valid ({100*valid/n:.1f}%)")

    return df


def rolling_by_ticker(series: pd.Series, ticker_key: pd.Series, window: int, stat: str) -> pd.Series:
    grouped = series.groupby(ticker_key, sort=False).rolling(window=window, min_periods=window)
    if stat == "mean":
        out = grouped.mean()
    elif stat == "std":
        out = grouped.std()
    else:
        raise ValueError(f"Unsupported stat: {stat}")
    return out.reset_index(level=0, drop=True)


def load_yfinance_daily_features(
    path: Path,
    windows: list[int],
    panel_tickers: set[str],
    date_min: pd.Timestamp,
    date_max: pd.Timestamp,
) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        print(f"[Market] yfinance_daily missing/empty: {path}")
        return pd.DataFrame(columns=["date", "ticker"])

    cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    filters = build_parquet_filters("date", date_min, date_max)
    df = pd.read_parquet(path, columns=cols, filters=filters)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["ticker"] = normalize_ticker_series(df["ticker"])
    df = df.dropna(subset=["date", "ticker"])

    if panel_tickers:
        df = df[df["ticker"].isin(panel_tickers)]
    df = df[(df["date"] >= date_min) & (df["date"] <= date_max)]
    if df.empty:
        print("[Market] yfinance_daily has no matching rows for panel range.")
        return pd.DataFrame(columns=["date", "ticker"])

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    ticker_key = df["ticker"]

    close_lag1 = df.groupby("ticker", sort=False)["close"].shift(1)
    volume_lag1 = df.groupby("ticker", sort=False)["volume"].shift(1)
    range_pct_lag1 = df.groupby("ticker", sort=False).apply(
        lambda g: safe_div(g["high"] - g["low"], g["open"]).shift(1)
    ).reset_index(level=0, drop=True)
    out = df.copy()
    out["yf_daily_open"] = out["open"]
    out["yf_daily_high"] = out["high"]
    out["yf_daily_low"] = out["low"]
    out["yf_daily_close"] = out["close"]
    out["yf_daily_volume"] = out["volume"]
    out["yf_daily_prev_close"] = close_lag1
    out["yf_daily_ret_oc"] = safe_div(out["close"] - out["open"], out["open"])
    out["yf_daily_ret_cc"] = safe_div(out["close"] - close_lag1, close_lag1)
    out["yf_daily_gap_open_prev_close"] = safe_div(out["open"] - close_lag1, close_lag1)
    out["yf_daily_range_pct"] = safe_div(out["high"] - out["low"], out["open"])
    out["yf_daily_turnover"] = out["close"] * out["volume"]

    for window in windows:
        close_ma = rolling_by_ticker(close_lag1, ticker_key, window, "mean")
        close_std = rolling_by_ticker(close_lag1, ticker_key, window, "std")
        vol_ma = rolling_by_ticker(volume_lag1, ticker_key, window, "mean")
        range_ma = rolling_by_ticker(range_pct_lag1, ticker_key, window, "mean")
        range_std = rolling_by_ticker(range_pct_lag1, ticker_key, window, "std")
        out[f"yf_daily_close_ma_{window}"] = close_ma
        out[f"yf_daily_close_z_{window}"] = safe_div(out["close"] - close_ma, close_std).clip(-10, 10)
        out[f"yf_daily_volume_ma_{window}"] = vol_ma
        out[f"yf_daily_volume_velocity_{window}"] = safe_div(out["volume"], vol_ma)
        out[f"yf_daily_range_ma_{window}"] = range_ma
        out[f"yf_daily_range_z_{window}"] = safe_div(out["yf_daily_range_pct"] - range_ma, range_std).clip(-10, 10)

    keep_cols = [
        "date",
        "ticker",
        "yf_daily_open",
        "yf_daily_high",
        "yf_daily_low",
        "yf_daily_close",
        "yf_daily_volume",
        "yf_daily_prev_close",
        "yf_daily_ret_oc",
        "yf_daily_ret_cc",
        "yf_daily_gap_open_prev_close",
        "yf_daily_range_pct",
        "yf_daily_turnover",
    ] + [c for c in out.columns if c.startswith("yf_daily_close_ma_") or c.startswith("yf_daily_close_z_")] + [
        c for c in out.columns if c.startswith("yf_daily_volume_ma_") or c.startswith("yf_daily_volume_velocity_")
    ] + [c for c in out.columns if c.startswith("yf_daily_range_ma_") or c.startswith("yf_daily_range_z_")]
    return out[keep_cols].drop_duplicates(subset=["date", "ticker"], keep="last")


def load_yfinance_intraday_features(
    path: Path,
    prefix: str,
    panel_tickers: set[str],
    date_min: pd.Timestamp,
    date_max: pd.Timestamp,
) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        print(f"[Market] {prefix} missing/empty: {path}")
        return pd.DataFrame(columns=["date", "ticker"])

    cols = ["datetime", "ticker", "open", "high", "low", "close", "volume"]
    # Include the full last day; intraday timestamps are not normalized.
    filters = build_parquet_filters("datetime", date_min, date_max + pd.Timedelta(days=1))
    df = pd.read_parquet(path, columns=cols, filters=filters)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["date"] = df["datetime"].dt.normalize()
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    
    df["ticker"] = normalize_ticker_series(df["ticker"])
    df = df.dropna(subset=["datetime", "date", "ticker"])

    if panel_tickers:
        df = df[df["ticker"].isin(panel_tickers)]
    df = df[(df["date"] >= date_min) & (df["date"] <= date_max)]
    if df.empty:
        print(f"[Market] {prefix} has no matching rows for panel range.")
        return pd.DataFrame(columns=["date", "ticker"])

    df = df.sort_values(["ticker", "date", "datetime"]).reset_index(drop=True)
    keys = ["date", "ticker"]

    agg = df.groupby(keys, sort=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        bar_count=("datetime", "count"),
    )

    bars = df[keys + ["close"]].copy()
    bars["bar_pos"] = bars.groupby(keys, sort=False).cumcount()
    first_close = bars[bars["bar_pos"] == 0][keys + ["close"]].rename(columns={"close": "first_close"})
    second_close = bars[bars["bar_pos"] == 1][keys + ["close"]].rename(columns={"close": "second_close"})

    agg = agg.reset_index()
    agg = agg.merge(first_close, on=keys, how="left")
    agg = agg.merge(second_close, on=keys, how="left")

    agg["ret_oc"] = safe_div(agg["close"] - agg["open"], agg["open"])
    agg["range_pct"] = safe_div(agg["high"] - agg["low"], agg["open"])
    agg["first_bar_ret"] = safe_div(agg["first_close"] - agg["open"], agg["open"])
    agg["two_bar_ret"] = safe_div(agg["second_close"] - agg["open"], agg["open"])
    agg["turnover"] = agg["close"] * agg["volume"]
    agg = agg.drop(columns=["first_close", "second_close"])

    rename = {col: f"{prefix}_{col}" for col in agg.columns if col not in keys}
    out = agg.rename(columns=rename)
    return out.drop_duplicates(subset=["date", "ticker"], keep="last")


def merge_market_features(
    panel: pd.DataFrame,
    yf_daily: pd.DataFrame,
    yf_1h: pd.DataFrame,
    yf_4h: pd.DataFrame,
) -> pd.DataFrame:
    out = panel.copy()

    if not yf_daily.empty:
        out = out.merge(yf_daily, on=["date", "ticker"], how="left")
    if not yf_1h.empty:
        out = out.merge(yf_1h, on=["date", "ticker"], how="left")
    if not yf_4h.empty:
        out = out.merge(yf_4h, on=["date", "ticker"], how="left")

    if {"total_net_buy", "yf_daily_turnover"}.issubset(out.columns):
        out["net_buy_to_yf_daily_turnover"] = safe_div(out["total_net_buy"], out["yf_daily_turnover"])
        out["abs_net_buy_to_yf_daily_turnover"] = safe_div(out["abs_net_buy"], out["yf_daily_turnover"])

    if {"total_net_buy", "yf_1h_turnover"}.issubset(out.columns):
        out["net_buy_to_yf_1h_turnover"] = safe_div(out["total_net_buy"], out["yf_1h_turnover"])

    if {"total_net_buy", "yf_4h_turnover"}.issubset(out.columns):
        out["net_buy_to_yf_4h_turnover"] = safe_div(out["total_net_buy"], out["yf_4h_turnover"])

    if "yf_daily_close" in out.columns:
        out["has_yf_daily"] = out["yf_daily_close"].notna().astype("int8")
    if "yf_1h_close" in out.columns:
        out["has_yf_1h"] = out["yf_1h_close"].notna().astype("int8")
    if "yf_4h_close" in out.columns:
        out["has_yf_4h"] = out["yf_4h_close"].notna().astype("int8")

    return out


def apply_column_prefixes(df: pd.DataFrame) -> pd.DataFrame:
    """Rename L1 columns with family prefixes. Keys and yf_* columns are unchanged."""
    rename = {}
    for col in df.columns:
        if col in ("broker", "ticker", "date", "scraped_at"):
            continue
        elif col.startswith("yf_") or col.startswith("has_yf_"):
            continue
        elif col in ("broker_type", "is_foreign", "is_buy_only", "is_sell_only"):
            rename[col] = f"brkm_{col}"
        elif col in (
            "gross_buy", "gross_sell", "total_net_buy", "gross_turnover",
            "buy_volume", "sell_volume", "net_volume", "buy_freq", "sell_freq",
            "avg_buy_price", "avg_sell_price", "market_breadth",
            "abs_net_buy", "net_flow_ratio", "buy_sell_val_ratio", "buy_sell_vol_ratio",
            "buy_sell_freq_ratio", "total_trades", "net_buy_per_trade", "churn_ratio",
            "avg_spread_price", "avg_spread_pct",
        ):
            rename[col] = f"flow_{col}"
        elif col in (
            "broker_day_abs_flow", "ticker_day_abs_flow", "market_day_abs_flow",
            "broker_ticker_specificity", "broker_market_share", "ticker_market_share",
            "broker_net_buy_rank",
        ):
            rename[col] = f"ctx_{col}"
        elif col.startswith("total_net_buy_"):
            rename[col] = f"tfl_{col.replace('total_net_buy_', 'net_buy_')}"
        elif col.startswith("net_flow_ratio_") or col.startswith("churn_ratio_"):
            rename[col] = f"tfl_{col}"
        elif col == "net_buy_sign_consistency_5":
            rename[col] = "tfl_net_buy_sign_consistency_5"
        elif col in ("active_days_in_5", "days_since_last_active"):
            rename[col] = f"act_{col}"
        elif col in ("avg_buy_price_in_range", "avg_sell_price_in_range", "buy_volume_in_total_volume"):
            rename[col] = f"pc_{col}"
        elif col.startswith("net_buy_to_yf_") or col.startswith("abs_net_buy_to_yf_"):
            rename[col] = f"flow_{col}"

    df = df.rename(columns=rename)
    print(f"[Prefix] Applied prefixes to {len(rename)} columns — families: "
          + ", ".join(sorted({v.split('_')[0] for v in rename.values()})))
    return df


def finalize_datamart(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out = out.drop_duplicates(subset=KEY_COLUMNS, keep="last")
    out = out.sort_values(KEY_COLUMNS).reset_index(drop=True)
    return out


def print_run_summary(stage: str, df: pd.DataFrame) -> None:
    min_date = df["date"].min()
    max_date = df["date"].max()
    trading_days = df["date"].nunique()
    print(
        f"[{stage}] rows={len(df):,}, cols={len(df.columns)}, "
        f"date_range={min_date.date() if pd.notna(min_date) else None} -> "
        f"{max_date.date() if pd.notna(max_date) else None}, "
        f"trading_days={trading_days}"
    )


def print_market_coverage(datamart: pd.DataFrame) -> None:
    for flag in ["has_yf_daily", "has_yf_1h", "has_yf_4h"]:
        if flag in datamart.columns:
            pct = float(datamart[flag].mean() * 100.0)
            print(f"[Coverage] {flag}={pct:.2f}%")


def main() -> None:
    args = parse_args()
    windows = parse_windows(args.windows)
    output_date_from = args.date_from
    output_date_to = args.date_to
    feature_date_from = derive_feature_date_from(
        date_from=output_date_from,
        windows=windows,
        warmup_bdays_multiplier=args.warmup_bdays_multiplier,
    )
    feature_date_to = output_date_to

    print(f"[Init] Input broksum: {args.input}")
    print(f"[Init] Output datamart: {args.output}")
    print(f"[Init] Windows: {windows}")
    if output_date_from or output_date_to:
        print(f"[Init] Output date filter: from={output_date_from}, to={output_date_to}")
        print(
            "[Init] Feature-build range (with warmup): "
            f"from={feature_date_from}, to={feature_date_to}"
        )

    refresh_level0_sources(args)
    print_source_snapshot(
        broksum_path=args.input,
        yf_daily_path=args.yfinance_daily_path,
        yf_1h_path=args.yfinance_1h_path,
        yf_4h_path=args.yfinance_4h_path,
    )

    if not args.input.exists():
        raise FileNotFoundError(f"Input parquet not found: {args.input}")

    raw_filters = build_parquet_filters("date", feature_date_from, feature_date_to, target_is_string=True)
    if raw_filters:
        print(f"[Load] Applying raw broksum parquet filters: {raw_filters}")
    raw_df = pd.read_parquet(args.input, filters=raw_filters)
    print(f"[Load] raw_rows={len(raw_df):,}, raw_cols={len(raw_df.columns)}")

    panel = normalize_schema(raw_df)
    panel = enrich_broker_metadata(panel, args.master_broker_path)
    panel = apply_date_filter(panel, feature_date_from, feature_date_to)

    if args.sample_rows and args.sample_rows > 0:
        panel = panel.head(args.sample_rows).copy()
        print(f"[Sample] Using first {len(panel):,} rows after normalization.")

    if panel.empty:
        raise RuntimeError("No data left after normalization/date filter.")

    print_run_summary("Stage0-Normalized", panel)
    print_window_readiness(panel, windows)

    datamart = build_base_features(panel)
    print_run_summary("Stage1-BaseFeatures", datamart)

    datamart = build_context_features(datamart)
    print_run_summary("Stage2-ContextFeatures", datamart)

    datamart = build_temporal_features(datamart, windows=windows)
    print_run_summary("Stage3-TemporalFeatures", datamart)

    datamart = build_activity_features(datamart)
    print_run_summary("Stage4-ActivityFeatures", datamart)

    panel_tickers = set(datamart["ticker"].astype(str).unique().tolist())
    date_min = datamart["date"].min()
    date_max = datamart["date"].max()

    yf_daily = load_yfinance_daily_features(
        path=args.yfinance_daily_path,
        windows=windows,
        panel_tickers=panel_tickers,
        date_min=date_min,
        date_max=date_max,
    )
    print(f"[Market] yfinance_daily rows matched: {len(yf_daily):,}")

    yf_1h = load_yfinance_intraday_features(
        path=args.yfinance_1h_path,
        prefix="yf_1h",
        panel_tickers=panel_tickers,
        date_min=date_min,
        date_max=date_max,
    )
    print(f"[Market] yfinance_1h rows matched: {len(yf_1h):,}")

    yf_4h = load_yfinance_intraday_features(
        path=args.yfinance_4h_path,
        prefix="yf_4h",
        panel_tickers=panel_tickers,
        date_min=date_min,
        date_max=date_max,
    )
    print(f"[Market] yfinance_4h rows matched: {len(yf_4h):,}")

    datamart = merge_market_features(datamart, yf_daily, yf_1h, yf_4h)
    datamart = build_price_context_features(datamart)
    datamart = apply_column_prefixes(datamart)
    datamart = finalize_datamart(datamart)
    print_run_summary("Stage5-MarketMerged", datamart)
    print_market_coverage(datamart)

    if output_date_from or output_date_to:
        datamart = apply_date_filter(datamart, output_date_from, output_date_to)
        if datamart.empty:
            raise RuntimeError("No datamart rows left after output date filter.")
        print_run_summary("Stage6-OutputDateFiltered", datamart)

    if args.dry_run:
        print("[DryRun] Datamart not written. Preview first 40 columns:")
        print(datamart.columns.tolist()[:40])
        print(f"[DryRun] Total columns: {len(datamart.columns)}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Incremental mode: upsert new dates into existing parquet (never overwrite history).
    #
    # PERFORMANCE WARNING:
    # This remains a single-file parquet upsert. It must read existing L1 and
    # rewrite the merged file, so it is not the production <15 minute
    # fetch-to-signal fast path. Production inference should consume prebuilt
    # fresh L1/L2 vectors, or move this layer to partitioned parquet/DuckDB.
    if (output_date_from or output_date_to) and args.output.exists():
        print(
            "[PerfWarning] Single-file L1 upsert reads and rewrites existing output. "
            "Do not use this as the default production signal path."
        )
        existing = pd.read_parquet(args.output)
        new_dates = set(datamart["date"].unique())
        existing = existing[~existing["date"].isin(new_dates)]
        datamart = pd.concat([existing, datamart], ignore_index=True)
        datamart = datamart.sort_values(["date", "ticker", "broker"]).reset_index(drop=True)
        print(f"[Done] Upserted {len(new_dates)} date(s) into existing parquet → {len(datamart):,} total rows")
    else:
        print(f"[Done] Full write → {len(datamart):,} rows")

    datamart.to_parquet(args.output, index=False)
    print(f"[Done] Wrote datamart to: {args.output}")


if __name__ == "__main__":
    main()
