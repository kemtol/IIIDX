#!/usr/bin/env python3
"""
Generate training datamart for BPJS objective (MVP):

- Entry: opening price at 09:00 on T+1 (derived from 1h candle ending 10:00)
- Exit reference: close price at 10:00 on T+1
- Label (V11 hybrid): V9-style learnable label with high positive rate (~18%)
  label_tp = 1 only when:
  - high reaches +2% threshold AND low stays above -3.5% (not stopped out)

Primary outputs:
1) Label table per trade day:
   idx/data/Level_2_Datamart/label_bpjs_intraday.parquet
2) Training datamart joined to feature day T:
   idx/data/Level_2_Datamart/training_datamart_bpjs_intraday.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import gc

def load_features_optimized(file_path):
    print(f"[RAM Optimization] Loading features from {file_path}...")
    
    # Load all columns (no filtering) to ensure all engineered features are retained
    df = pd.read_parquet(file_path)

    # Downcasting (Float64 -> Float32) - Mengurangi RAM 50%
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    
    print(f"[RAM Optimization] Memory usage reduced to: {df.memory_usage().sum() / 1024**2:.2f} MB")
    gc.collect() # Paksa RAM bersih-bersih
    return df

# edges/bpjs_opening_tp3/scripts/<file>.py → parents[3] = idx/
IDX_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = IDX_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "Level_0_Raw"
FEATURES_DIR = DATA_DIR / "Level_1_Features"
DATAMART_DIR = DATA_DIR / "Level_2_Datamart"

# Ensure directories exist (ignore if symlink already exists)
try:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass
try:
    DATAMART_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

DEFAULT_FEATURES = FEATURES_DIR / "broksum_datamart.parquet"
DEFAULT_YF_1H = RAW_DATA_DIR / "yfinance_1h.parquet"
DEFAULT_YF_DAILY = RAW_DATA_DIR / "yfinance_daily.parquet"
DEFAULT_MASTER_BROKER = RAW_DATA_DIR / "master_broker.parquet"
DEFAULT_GLOBAL_INDICES = RAW_DATA_DIR / "global_indices.parquet"
DEFAULT_BROKSUM_BYBROKER = RAW_DATA_DIR / "broksum_bybroker.parquet"
STOCKBIT_BROKER_CODE = "XL"
DEFAULT_LABEL_OUTPUT = DATAMART_DIR / "label_bpjs_intraday.parquet"
DEFAULT_TRAIN_OUTPUT = DATAMART_DIR / "training_datamart_bpjs_intraday.parquet"
DEFAULT_TRAIN_OUTPUT_RANK = DATAMART_DIR / "training_datamart_bpjs_rank.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build training datamart using opening-to-10:00 target label."
    )
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES, help="Level 1 feature parquet path.")
    parser.add_argument("--yfinance-1h-path", type=Path, default=DEFAULT_YF_1H, help="yfinance 1h parquet path.")
    parser.add_argument("--master-broker-path", type=Path, default=DEFAULT_MASTER_BROKER, help="master broker parquet path.")
    parser.add_argument("--label-output", type=Path, default=DEFAULT_LABEL_OUTPUT, help="Output label parquet.")
    parser.add_argument("--training-output", type=Path, default=DEFAULT_TRAIN_OUTPUT, help="Output training parquet.")
    parser.add_argument("--target-pct", type=float, default=0.020, help="Spike threshold (e.g., 0.020 = 2.0%%).")
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=-0.035,
        help="Stop-loss threshold as return (e.g., -0.035 = -3.5%%).",
    )
    parser.add_argument("--cutoff-time", type=str, default="10:00", help="Label window end time (HH:MM).")
    parser.add_argument(
        "--label-mode",
        type=str,
        default="fixed_tp",
        choices=["fixed_tp", "relative_rank"],
        help=(
            "Label definition: 'fixed_tp' = hit 2%% TP zone (current), "
            "'relative_rank' = close_return@cutoff in top-25%% cross-section per day."
        ),
    )
    parser.add_argument("--focus-broker", type=str, default="MG", help="Broker focus feature (default MG).")
    parser.add_argument("--bandar-localfund-min", type=float, default=60.0, help="Threshold localfund_%% for bandar bucket.")
    parser.add_argument("--yfinance-daily-path", type=Path, default=DEFAULT_YF_DAILY, help="yfinance daily parquet path for price proximity.")
    parser.add_argument("--global-indices-path", type=Path, default=DEFAULT_GLOBAL_INDICES, help="Global indices parquet path for macro sentiment.")
    parser.add_argument("--broksum-bybroker-path", type=Path, default=DEFAULT_BROKSUM_BYBROKER, help="Raw broksum by-broker parquet path for Stockbit (XL) features.")
    parser.add_argument("--strategy-mode", type=str, default="bpjs", choices=["bpjs", "bsjp"], help="Strategy mode: 'bpjs' (Beli Pagi Jual Sore) or 'bsjp' (Beli Sore Jual Pagi).")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing parquet outputs.")
    return parser.parse_args()


def normalize_ticker_series(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.upper().str.strip()
    out = out.str.replace(r"[^A-Z0-9.]", "", regex=True)
    out = out.str.replace(r"\.JK$", "", regex=True)
    return out


def parse_cutoff_time(cutoff: str) -> tuple[int, int]:
    hh, mm = cutoff.split(":")
    return int(hh), int(mm)


def ensure_exists(path: Path, name: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{name} missing or empty: {path}")


def load_master_broker(master_path: Path, bandar_localfund_min: float) -> tuple[set[str], set[str]]:
    if not master_path.exists() or master_path.stat().st_size == 0:
        return set(), set()

    mb = pd.read_parquet(master_path)
    if "broker_code" not in mb.columns:
        return set(), set()

    mb["broker_code"] = mb["broker_code"].astype(str).str.upper().str.strip()
    if "status" in mb.columns:
        mb = mb[mb["status"].astype(str).str.upper().eq("ACTIVE")]

    local_fund_brokers: set[str] = set()
    if "category" in mb.columns:
        local_fund_brokers = set(
            mb.loc[mb["category"].astype(str).str.lower().eq("local fund"), "broker_code"].dropna().tolist()
        )

    bandar_brokers: set[str] = set()
    if "localfund_%" in mb.columns:
        bandar_brokers = set(
            mb.loc[pd.to_numeric(mb["localfund_%"], errors="coerce").fillna(0) >= bandar_localfund_min, "broker_code"]
            .dropna()
            .tolist()
        )

    return local_fund_brokers, bandar_brokers


def build_label_table(
    yf_1h: pd.DataFrame,
    target_pct: float,
    stop_loss_pct: float,
    cutoff_hh: int,
    cutoff_mm: int,
    label_mode: str = "fixed_tp",
) -> pd.DataFrame:
    need = {"datetime", "ticker", "open", "high", "low", "close"}
    missing = [c for c in need if c not in yf_1h.columns]
    if missing:
        raise ValueError(f"yfinance_1h missing columns: {missing}")

    df = yf_1h.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "ticker", "open", "high", "low", "close"])
    df["ticker"] = normalize_ticker_series(df["ticker"])
    df["date"] = df["datetime"].dt.normalize()
    df["hm"] = df["datetime"].dt.hour * 100 + df["datetime"].dt.minute
    cutoff_hm = cutoff_hh * 100 + cutoff_mm

    df = df.sort_values(["ticker", "date", "datetime"]).reset_index(drop=True)

    # Use the 1h candle ending at cutoff time.
    # For Indonesian session this means:
    # - candle timestamp 10:00 represents the 09:00->10:00 interval,
    # - entry uses candle open (09:00),
    # - exit uses candle close (10:00).
    window = df[df["hm"] == cutoff_hm].copy()
    if window.empty:
        return pd.DataFrame()

    labels = (
        window.sort_values(["ticker", "date", "datetime"])
        .drop_duplicates(subset=["ticker", "date"], keep="last")
        .rename(
            columns={
                "date": "trade_date",
                "open": "entry_price_opening",
                "high": "high_to_cutoff",
                "low": "low_to_cutoff",
                "close": "close_to_cutoff",
            }
        )
        .copy()
    )
    if labels.empty:
        return labels

    labels["entry_datetime"] = labels["datetime"] - pd.Timedelta(hours=1)
    labels["entry_hm"] = labels["entry_datetime"].dt.hour * 100 + labels["entry_datetime"].dt.minute
    labels["bars_until_cutoff"] = 1

    labels["entry_price_opening"] = pd.to_numeric(labels["entry_price_opening"], errors="coerce")
    labels["high_to_cutoff"] = pd.to_numeric(labels["high_to_cutoff"], errors="coerce")
    labels["low_to_cutoff"] = pd.to_numeric(labels["low_to_cutoff"], errors="coerce")
    labels["close_to_cutoff"] = pd.to_numeric(labels["close_to_cutoff"], errors="coerce")

    base = labels["entry_price_opening"].replace(0, np.nan)
    labels["max_return_to_cutoff"] = (labels["high_to_cutoff"] / base) - 1.0
    labels["min_return_to_cutoff"] = (labels["low_to_cutoff"] / base) - 1.0
    labels["close_return_to_cutoff"] = (labels["close_to_cutoff"] / base) - 1.0
    # =============================================================================
    # V11 HYBRID LABEL: Learnable label (V9 style) for training
    # Positive if: NOT stopped out (-3.5%) AND reaches TP zone (2%)
    # This gives ~18% positive rate (learnable) while allowing Aggressive Scalper execution
    # =============================================================================
    labels["label_tp"] = (
        (labels["min_return_to_cutoff"] > -0.035) &  # NOT stopped out
        (labels["max_return_to_cutoff"] >= 0.020)     # Reaches 2% TP zone
    ).astype(int)
    labels["label_sl3"] = (labels["min_return_to_cutoff"] <= stop_loss_pct).astype(int)

    if label_mode == "relative_rank":
        # Override label_tp: top-25% close return per day (cross-sectional rank)
        # Model belajar "siapa yang outperform hari ini" bukan "siapa yang hit threshold"
        labels["label_tp"] = (
            labels.groupby("trade_date")["close_return_to_cutoff"]
            .transform(lambda x: (x >= x.quantile(0.75)).astype(int))
        )

    labels["trade_date"] = pd.to_datetime(labels["trade_date"], errors="coerce").dt.normalize()
    labels = labels.dropna(subset=["ticker", "trade_date", "entry_price_opening", "close_to_cutoff"]).copy()
    labels = labels.drop(columns=["datetime", "date", "hm"], errors="ignore")

    labels = labels.sort_values(["ticker", "trade_date"]).reset_index(drop=True)
    labels["feature_date"] = labels.groupby("ticker", sort=False)["trade_date"].shift(1)
    labels = labels.dropna(subset=["feature_date"]).copy()
    labels["feature_date"] = pd.to_datetime(labels["feature_date"], errors="coerce").dt.normalize()
    labels["trade_date"] = pd.to_datetime(labels["trade_date"], errors="coerce").dt.normalize()
    labels = labels.dropna(subset=["feature_date", "trade_date"])
    return labels


def build_label_bsjp(
    yf_daily: pd.DataFrame,
    target_pct: float,
    stop_loss_pct: float,
) -> pd.DataFrame:
    """
    BSJP (Beli Sore Jual Pagi) Label Generation:
    - Entry: Close price at day T (last price of trading day, ~16:00 WIB)
    - Exit: Open price at day T+1 (09:00 WIB next day)
    - Label: Whether next_day_open >= entry_price * (1 + target_pct)

    Args:
        yf_daily: Daily OHLCV data with columns [date, ticker, open, high, low, close]
        target_pct: Target profit percentage (e.g., 0.03 for +3%)
        stop_loss_pct: Stop loss percentage (e.g., -0.03 for -3%)

    Returns:
        Label DataFrame with columns for entry/exit prices and binary labels
    """
    need = {"date", "ticker", "open", "high", "low", "close"}
    missing = [c for c in need if c not in yf_daily.columns]
    if missing:
        raise ValueError(f"yfinance_daily missing columns: {missing}")

    df = yf_daily.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "ticker", "open", "high", "low", "close"])
    df["ticker"] = normalize_ticker_series(df["ticker"])

    # Sort for proper shift operations
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # BSJP Entry: Close price at day T
    df["entry_price"] = pd.to_numeric(df["close"], errors="coerce")

    # BSJP Exit: Next day open (shift -1 to get tomorrow's open today)
    df["exit_price"] = df.groupby("ticker", sort=False)["open"].shift(-1)

    # Calculate returns
    base = df["entry_price"].replace(0, np.nan)
    df["close_return"] = (df["exit_price"] - base) / base  # (next_open - close_T) / close_T

    # BSJP Label: Did we hit target on next day open?
    # Note: For BSJP, we sell at next day open, so we check if open >= target
    df["label_tp"] = (df["close_return"] >= target_pct).astype(int)
    df["label_sl3"] = (df["close_return"] <= stop_loss_pct).astype(int)

    # Create trade_date and feature_date mapping
    df["trade_date"] = df.groupby("ticker", sort=False)["date"].shift(-1)  # Exit date = next day
    df["feature_date"] = df["date"]  # Feature date = entry day

    # Filter out rows where we don't have next day data
    df = df.dropna(subset=["exit_price", "trade_date", "feature_date"]).copy()

    # Rename columns for consistency with BPJS labeling
    labels = df.rename(
        columns={
            "entry_price": "entry_price_close",
            "exit_price": "exit_price_next_open",
            "close_return": "close_return_next_day",
        }
    ).copy()

    # Add BSJP-specific metrics
    labels["intraday_range_pct"] = (labels["high"] - labels["low"]) / labels["entry_price_close"]
    labels["close_to_high_pct"] = (labels["close"] - labels["low"]) / (labels["high"] - labels["low"]).replace(0, np.nan)

    # Select final columns
    out_cols = [
        "feature_date", "trade_date", "ticker",
        "entry_price_close", "exit_price_next_open", "close_return_next_day",
        "label_tp", "label_sl3",
        "intraday_range_pct", "close_to_high_pct"
    ]
    labels = labels[out_cols].copy()

    # Downcast numeric columns
    for col in ["entry_price_close", "exit_price_next_open", "close_return_next_day",
                "intraday_range_pct", "close_to_high_pct"]:
        labels[col] = pd.to_numeric(labels[col], errors="coerce").astype("float32")
    for col in ["label_tp", "label_sl3"]:
        labels[col] = pd.to_numeric(labels[col], errors="coerce").astype("int8")

    return labels


def _choose_numeric_feature_columns(df: pd.DataFrame, include_extra: Iterable[str] | None = None) -> list[str]:
    """
    Keep MVP aggregation compact for speed.
    We only aggregate broker signal columns that are likely useful for first model iteration.
    Column names reflect L1 prefix schema (flow_, tfl_, ctx_, etc.)
    """
    block = {"date", "ticker", "broker", "scraped_at", "brkm_broker_type"}
    preferred_exact = {
        # flow_ family
        "flow_total_net_buy",
        "flow_abs_net_buy",
        "flow_gross_turnover",
        "flow_buy_freq",
        "flow_sell_freq",
        "flow_net_flow_ratio",
        "flow_churn_ratio",
        "flow_total_trades",
        "flow_net_buy_per_trade",
        # ctx_ family
        "ctx_broker_ticker_specificity",
        "ctx_broker_market_share",
        "ctx_ticker_market_share",
        "ctx_broker_net_buy_rank",
        # coverage flags (no prefix)
        "has_yf_daily",
        "has_yf_1h",
        "has_yf_4h",
    }
    preferred_prefixes = (
        "tfl_net_buy_z_",
        "tfl_net_buy_velocity_",
        "tfl_net_buy_ma_",
        "tfl_net_flow_ratio_ma_",
        "tfl_churn_ratio_ma_",
        "yf_daily_",
    )

    cols: list[str] = []
    for c in df.columns:
        if c in block:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if c in preferred_exact or any(c.startswith(pref) for pref in preferred_prefixes):
            cols.append(c)

    if include_extra:
        for c in include_extra:
            if c in df.columns and c not in cols and pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    return cols


def _extract_foreign_flow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Foreign Flow Family features separately from broker logic.
    Handles cases where foreign data is:
    1. Separate rows with broker='FOREIGN'
    2. Has foreign_buy and foreign_sell columns
    3. Has gross_buy/gross_sell with foreign_type indicator
    """
    foreign_cols = ["date", "ticker"]
    foreign_data = pd.DataFrame()

    # Case 1: Dedicated foreign rows (broker="FOREIGN")
    if "broker" in df.columns:
        foreign_mask = df["broker"].str.upper().isin(["FOREIGN", "ASING", "FRGN"])
        if foreign_mask.any():
            foreign_data = df[foreign_mask].copy()
            if "flow_total_net_buy" in foreign_data.columns:
                foreign_data["foreign_net_buy"] = pd.to_numeric(foreign_data["flow_total_net_buy"], errors="coerce")
            if "flow_gross_buy" in foreign_data.columns and "flow_gross_sell" in foreign_data.columns:
                foreign_data["foreign_buy"] = pd.to_numeric(foreign_data["flow_gross_buy"], errors="coerce")
                foreign_data["foreign_sell"] = pd.to_numeric(foreign_data["flow_gross_sell"], errors="coerce")
                foreign_data["foreign_net_buy"] = foreign_data["foreign_buy"] - foreign_data["foreign_sell"]

    # Case 2: Dedicated foreign columns (legacy fallback)
    if foreign_data.empty:
        if "foreign_buy" in df.columns and "foreign_sell" in df.columns:
            foreign_agg = df.groupby(["date", "ticker"], sort=False).agg(
                foreign_buy=("foreign_buy", "sum"),
                foreign_sell=("foreign_sell", "sum"),
            ).reset_index()
            foreign_agg["foreign_net_buy"] = foreign_agg["foreign_buy"] - foreign_agg["foreign_sell"]
            foreign_data = foreign_agg
        elif "foreign_net_buy" in df.columns:
            foreign_data = df.groupby(["date", "ticker"], sort=False).agg(
                foreign_net_buy=("foreign_net_buy", "sum"),
            ).reset_index()

    if foreign_data.empty or "foreign_net_buy" not in foreign_data.columns:
        return pd.DataFrame(columns=foreign_cols + ["foreign_net_buy", "foreign_participation_ratio", "foreign_net_buy_ma5", "foreign_net_buy_z20"])

    # Aggregate by date/ticker
    foreign_agg = foreign_data.groupby(["date", "ticker"], sort=False).agg(
        foreign_net_buy=("foreign_net_buy", "sum"),
    ).reset_index()

    # Calculate foreign participation ratio if turnover available
    if "flow_gross_turnover" in df.columns:
        turnover = df.groupby(["date", "ticker"], sort=False)["flow_gross_turnover"].sum().rename("total_turnover")
        foreign_agg = foreign_agg.merge(turnover.reset_index(), on=["date", "ticker"], how="left")
        foreign_abs = foreign_data.groupby(["date", "ticker"], sort=False).apply(
            lambda x: (x.get("foreign_buy", 0).sum() + x.get("foreign_sell", 0).sum())
            if "foreign_buy" in x.columns else x["foreign_net_buy"].abs().sum()
        ).rename("foreign_abs_flow").reset_index()
        foreign_agg = foreign_agg.merge(foreign_abs, on=["date", "ticker"], how="left")
        foreign_agg["foreign_participation_ratio"] = foreign_agg["foreign_abs_flow"] / foreign_agg["total_turnover"].replace(0, np.nan)
    else:
        foreign_agg["foreign_participation_ratio"] = np.nan

    # Calculate rolling features with shift(1) for no-lookahead
    foreign_agg = foreign_agg.sort_values(["ticker", "date"])
    foreign_agg["foreign_net_buy_ma5"] = foreign_agg.groupby("ticker", sort=False)["foreign_net_buy"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=3).mean()
    )
    foreign_agg["foreign_net_buy_std20"] = foreign_agg.groupby("ticker", sort=False)["foreign_net_buy"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).std()
    )
    foreign_agg["foreign_net_buy_z20"] = (
        (foreign_agg["foreign_net_buy"] - foreign_agg["foreign_net_buy_ma5"]) / foreign_agg["foreign_net_buy_std20"].replace(0, np.nan)
    )

    # Select and downcast
    result = foreign_agg[["date", "ticker", "foreign_net_buy", "foreign_participation_ratio", "foreign_net_buy_ma5", "foreign_net_buy_z20"]].copy()
    for col in ["foreign_net_buy", "foreign_participation_ratio", "foreign_net_buy_ma5", "foreign_net_buy_z20"]:
        result[col] = pd.to_numeric(result[col], errors="coerce").astype("float32")

    return result


def build_feature_aggregate(
    features: pd.DataFrame,
    focus_broker: str,
    local_fund_brokers: set[str],
    bandar_brokers: set[str],
) -> pd.DataFrame:
    need = {"date", "ticker", "broker", "flow_total_net_buy"}
    missing = [c for c in need if c not in features.columns]
    if missing:
        raise ValueError(f"features missing columns: {missing}")

    df = features.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["ticker"] = normalize_ticker_series(df["ticker"])
    df["broker"] = df["broker"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["date", "ticker", "broker"])

    # T-1 safe L1 shift (audit 2026-05-09): L1 rows mix broker activity and
    # daily/1h/4h OHLCV values for the row date. Broker data is published
    # EOD post-decision, and final daily high/low/close are not available at
    # the pre14 decision point. Shift all current-row L1 numeric families per
    # (broker, ticker) before any date/ticker aggregation.
    df = df.sort_values(["broker", "ticker", "date"])
    _grp = df.groupby(["broker", "ticker"], sort=False)
    _CURRENT_ROW_PREFIXES = (
        "flow_",
        "ctx_",
        "tfl_",
        "yf_daily_",
        "yf_1h_",
        "yf_4h_",
        "pc_",
        "has_yf_",
    )
    _shift_cols = [
        c
        for c in df.columns
        if c not in {"date", "ticker", "broker"}
        and c.startswith(_CURRENT_ROW_PREFIXES)
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if _shift_cols:
        df[_shift_cols] = _grp[_shift_cols].shift(1)

    # Recompute z / velocity from shifted T-1 flow vs shifted historical
    # baseline. This avoids carrying any L1 z-score that used a same-day
    # numerator before the broad shift above.
    for _w in (5, 20, 60):
        _z = f"tfl_net_buy_z_{_w}"
        _v = f"tfl_net_buy_velocity_{_w}"
        _ma = f"tfl_net_buy_ma_{_w}"
        _std = f"tfl_net_buy_std_{_w}"
        _num = "flow_total_net_buy" if "flow_total_net_buy" in df.columns else "tfl_net_buy_lag1"
        if _num in df.columns and _ma in df.columns:
            if _z in df.columns and _std in df.columns:
                df[_z] = ((df[_num] - df[_ma]) / df[_std].replace(0, np.nan)).clip(-10, 10)
            if _v in df.columns:
                df[_v] = df[_num] / df[_ma].replace(0, np.nan)

    # Base aggregate (all brokers)
    numeric_cols = _choose_numeric_feature_columns(df)
    agg_map: dict[str, list[str]] = {}
    for c in numeric_cols:
        if c in {"has_yf_daily", "has_yf_1h", "has_yf_4h"}:
            agg_map[c] = ["max"]
        else:
            agg_map[c] = ["sum", "mean"]

    grouped = df.groupby(["date", "ticker"], sort=False).agg(agg_map)
    grouped.columns = [f"{col}_{fn}" for col, fn in grouped.columns]
    grouped = grouped.reset_index()

    # Broker counts and side counts
    df["_is_buyer"] = (df["flow_total_net_buy"] > 0).astype("int8")
    df["_is_seller"] = (df["flow_total_net_buy"] < 0).astype("int8")
    side = (
        df.groupby(["date", "ticker"], sort=False)
        .agg(
            broker_count=("broker", "nunique"),
            buyer_broker_count=("_is_buyer", "sum"),
            seller_broker_count=("_is_seller", "sum"),
        )
        .reset_index()
    )
    side["buyer_ratio"] = side["buyer_broker_count"] / side["broker_count"].replace(0, np.nan)
    side["seller_ratio"] = side["seller_broker_count"] / side["broker_count"].replace(0, np.nan)

    out = grouped.merge(side, on=["date", "ticker"], how="left")

    # V10: Aggregate ALL key brokers (MG, XC, SQ, YP, PD)
    key_brokers = ["MG", "XC", "SQ", "YP", "PD"]
    broker_cols_pref = [
        "flow_total_net_buy",
        "tfl_net_buy_z_20",
        "tfl_net_buy_velocity_20",
        "flow_net_flow_ratio",
        "flow_churn_ratio",
    ]

    for broker_code in key_brokers:
        b = broker_code.upper().strip()
        cols = [c for c in broker_cols_pref if c in df.columns]
        if cols:
            broker_df = df[df["broker"] == b][["date", "ticker"] + cols].copy()
            if not broker_df.empty:
                rename = {c: f"{b.lower()}_{c}" for c in cols}
                broker_df = broker_df.rename(columns=rename)
                broker_df[f"{b.lower()}_present"] = 1
                out = out.merge(broker_df, on=["date", "ticker"], how="left")
                out[f"{b.lower()}_present"] = out[f"{b.lower()}_present"].fillna(0).astype("int8")
                for c in cols:
                    out[f"{b.lower()}_{c}"] = pd.to_numeric(out[f"{b.lower()}_{c}"], errors="coerce").fillna(0.0)

    # BSJP: Retail Aggregation (Exit Liquidity) - sum of retail broker net buys
    retail_brokers = ["XC", "YP", "PD", "SQ"]
    retail_net_buy = (
        df[df["broker"].isin([b.upper() for b in retail_brokers])]
        .groupby(["date", "ticker"], sort=False)["flow_total_net_buy"]
        .sum()
        .rename("total_retail_net_buy")
        .reset_index()
    )
    if not retail_net_buy.empty:
        out = out.merge(retail_net_buy, on=["date", "ticker"], how="left")
        out["total_retail_net_buy"] = pd.to_numeric(out["total_retail_net_buy"], errors="coerce").fillna(0.0)

    # V13: Foreign Flow Family (handled separately from broker logic)
    # Foreign data typically comes as separate rows with broker="FOREIGN" or has foreign_buy/foreign_sell columns
    foreign_flow = _extract_foreign_flow(df)
    if not foreign_flow.empty:
        out = out.merge(foreign_flow, on=["date", "ticker"], how="left")
        print(f"[ForeignFlow] merged foreign_net_buy, foreign_participation_ratio, foreign_net_buy_ma5, foreign_net_buy_z20")

    # Local-fund and bandar buckets
    def bucket_features(codes: set[str], prefix: str) -> pd.DataFrame:
        if not codes:
            return pd.DataFrame(columns=["date", "ticker"])
        sub = df[df["broker"].isin(codes)].copy()
        if sub.empty:
            return pd.DataFrame(columns=["date", "ticker"])

        sub["flow_total_net_buy"] = pd.to_numeric(sub["flow_total_net_buy"], errors="coerce").fillna(0.0)
        if "flow_buy_freq" in sub.columns:
            sub["flow_buy_freq"] = pd.to_numeric(sub["flow_buy_freq"], errors="coerce").fillna(0.0)
        else:
            sub["flow_buy_freq"] = 0.0

        sub["_abs_netbuy"] = sub["flow_total_net_buy"].abs()
        sub["_is_buyer"]   = (sub["flow_total_net_buy"] > 0).astype("int8")
        sub["_is_seller"]  = (sub["flow_total_net_buy"] < 0).astype("int8")

        g = (
            sub.groupby(["date", "ticker"], sort=False)
            .agg(
                netbuy_sum=("flow_total_net_buy", "sum"),
                netbuy_mean=("flow_total_net_buy", "mean"),
                abs_netbuy_sum=("_abs_netbuy", "sum"),
                broker_count=("broker", "nunique"),
                buyer_broker_count=("_is_buyer", "sum"),
                seller_broker_count=("_is_seller", "sum"),
                buy_freq_sum=("flow_buy_freq", "sum"),
                buy_freq_mean=("flow_buy_freq", "mean"),
            )
            .reset_index()
        )

        # Concentration metrics on positive net-buy share (who dominates localfund flow).
        share_df = sub[["date", "ticker", "broker", "flow_total_net_buy"]].copy()
        share_df["pos_netbuy"] = share_df["flow_total_net_buy"].clip(lower=0)
        pos_sum = (
            share_df.groupby(["date", "ticker"], sort=False)["pos_netbuy"]
            .sum()
            .rename("pos_sum")
            .reset_index()
        )
        share_df = share_df.merge(pos_sum, on=["date", "ticker"], how="left")
        share_df = share_df[share_df["pos_sum"] > 0].copy()

        if not share_df.empty:
            share_df["share"] = share_df["pos_netbuy"] / share_df["pos_sum"]
            share_df["share_sq"] = share_df["share"] ** 2
            share_df["share_rank"] = share_df.groupby(["date", "ticker"])["share"].rank(
                ascending=False, method="first"
            )
            hhi = (
                share_df.groupby(["date", "ticker"], sort=False)["share_sq"]
                .sum()
                .rename("concentration_hhi")
                .reset_index()
            )
            top1 = (
                share_df.groupby(["date", "ticker"], sort=False)["share"]
                .max()
                .rename("top1_share")
                .reset_index()
            )
            top3 = (
                share_df[share_df["share_rank"] <= 3]
                .groupby(["date", "ticker"], sort=False)["share"]
                .sum()
                .rename("top3_share")
                .reset_index()
            )
            conc = hhi.merge(top1, on=["date", "ticker"], how="left").merge(
                top3, on=["date", "ticker"], how="left"
            )
            g = g.merge(conc, on=["date", "ticker"], how="left")

        g = g.rename(
            columns={
                "netbuy_sum": f"{prefix}_netbuy_sum",
                "netbuy_mean": f"{prefix}_netbuy_mean",
                "abs_netbuy_sum": f"{prefix}_abs_netbuy_sum",
                "broker_count": f"{prefix}_broker_count",
                "buyer_broker_count": f"{prefix}_buyer_broker_count",
                "seller_broker_count": f"{prefix}_seller_broker_count",
                "buy_freq_sum": f"{prefix}_buy_freq_sum",
                "buy_freq_mean": f"{prefix}_buy_freq_mean",
                "concentration_hhi": f"{prefix}_concentration_hhi",
                "top1_share": f"{prefix}_top1_share",
                "top3_share": f"{prefix}_top3_share",
            }
        )
        return g

    lf = bucket_features(local_fund_brokers, "localfund")
    bd = bucket_features(bandar_brokers, "bandar")
    if not lf.empty:
        out = out.merge(lf, on=["date", "ticker"], how="left")
    if not bd.empty:
        out = out.merge(bd, on=["date", "ticker"], how="left")

    # Derived localfund confluence/intensity features.
    if "localfund_broker_count" in out.columns:
        out["localfund_participation_ratio"] = out["localfund_broker_count"] / out["broker_count"].replace(0, np.nan)
    if {"localfund_buyer_broker_count", "localfund_broker_count"}.issubset(out.columns):
        out["localfund_buyer_ratio"] = out["localfund_buyer_broker_count"] / out["localfund_broker_count"].replace(
            0, np.nan
        )
    if {"localfund_netbuy_sum", "localfund_abs_netbuy_sum"}.issubset(out.columns):
        out["localfund_consensus_strength"] = (
            out["localfund_netbuy_sum"].abs() / out["localfund_abs_netbuy_sum"].replace(0, np.nan)
        )

    # Localfund streak + rolling z-score style features (per ticker, using only past via shift(1)).
    if "localfund_netbuy_sum" in out.columns:
        out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
        out["localfund_streak_buy_days"] = (
            out.groupby("ticker", sort=False)["localfund_netbuy_sum"]
            .transform(lambda s: s.gt(0).groupby(s.le(0).cumsum()).cumsum())
            .astype("int32")
        )

        out["localfund_netbuy_ma20"] = out.groupby("ticker", sort=False)["localfund_netbuy_sum"].transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).mean()
        )
        out["localfund_netbuy_std20"] = out.groupby("ticker", sort=False)["localfund_netbuy_sum"].transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).std()
        )
        out["localfund_netbuy_z20"] = (
            (out["localfund_netbuy_sum"] - out["localfund_netbuy_ma20"]) / out["localfund_netbuy_std20"].replace(0, np.nan)
        )

        out["localfund_netbuy_3d_sum"] = out.groupby("ticker", sort=False)["localfund_netbuy_sum"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).sum()
        )
        out["localfund_netbuy_7d_sum"] = out.groupby("ticker", sort=False)["localfund_netbuy_sum"].transform(
            lambda s: s.shift(1).rolling(7, min_periods=3).sum()
        )

    if "localfund_buy_freq_sum" in out.columns:
        out["localfund_buy_freq_ma30"] = out.groupby("ticker", sort=False)["localfund_buy_freq_sum"].transform(
            lambda s: s.shift(1).rolling(30, min_periods=10).mean()
        )
        out["localfund_buy_freq_std30"] = out.groupby("ticker", sort=False)["localfund_buy_freq_sum"].transform(
            lambda s: s.shift(1).rolling(30, min_periods=10).std()
        )
        out["localfund_buy_freq_z30"] = (
            (out["localfund_buy_freq_sum"] - out["localfund_buy_freq_ma30"])
            / out["localfund_buy_freq_std30"].replace(0, np.nan)
        )
        out["localfund_buy_freq_ratio_to_ma30"] = (
            out["localfund_buy_freq_sum"] / out["localfund_buy_freq_ma30"].replace(0, np.nan)
        )

    # Keep bucket numerics dense for training.
    for c in out.columns:
        if c.startswith("localfund_") or c.startswith("bandar_"):
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    for c in out.columns:
        if c.endswith("_broker_count"):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int32")

    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    return out


def build_volume_features(yf_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Build Volume Family features for BSJP strategy:
    - volume_ma20: rolling 20-day mean of daily volume
    - relative_volume: Current Volume / volume_ma20
    - volume_surge_closing: Volume of last hour vs average hourly volume
    """
    df = yf_1h.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "ticker", "volume"])
    df["ticker"] = df["ticker"].str.replace(r"\.JK$", "", regex=True)
    df["date"] = df["datetime"].dt.normalize()
    df["hour"] = df["datetime"].dt.hour
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

    # Calculate daily total volume per ticker
    daily_volume = df.groupby(["date", "ticker"], sort=False)["volume"].sum().reset_index()
    daily_volume.columns = ["date", "ticker", "daily_volume"]

    # Calculate volume_ma20 (rolling 20-day mean) - using shift(1) for no-lookahead
    daily_volume = daily_volume.sort_values(["ticker", "date"])
    daily_volume["volume_ma20"] = (
        daily_volume.groupby("ticker", sort=False)["daily_volume"]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    )
    daily_volume["relative_volume"] = daily_volume["daily_volume"] / daily_volume["volume_ma20"].replace(0, np.nan)

    # Calculate volume_surge_closing (last hour vs avg hourly volume)
    # Last hour in Indonesian market is typically 14:00-15:00 (hour 14)
    hourly_volume = df.groupby(["date", "ticker", "hour"], sort=False)["volume"].sum().reset_index()

    # Average hourly volume per day (excluding last hour for fair comparison)
    avg_hourly = (
        hourly_volume[hourly_volume["hour"] < 14]
        .groupby(["date", "ticker"], sort=False)["volume"]
        .mean()
        .rename("avg_hourly_volume")
        .reset_index()
    )

    # Last hour volume
    last_hour = (
        hourly_volume[hourly_volume["hour"] == 14]
        .groupby(["date", "ticker"], sort=False)["volume"]
        .sum()
        .rename("last_hour_volume")
        .reset_index()
    )

    volume_feats = daily_volume.merge(avg_hourly, on=["date", "ticker"], how="left")
    volume_feats = volume_feats.merge(last_hour, on=["date", "ticker"], how="left")
    volume_feats["volume_surge_closing"] = volume_feats["last_hour_volume"] / volume_feats["avg_hourly_volume"].replace(0, np.nan)

    # Select and downcast to float32
    out = volume_feats[["date", "ticker", "volume_ma20", "relative_volume", "volume_surge_closing"]].copy()
    for col in ["volume_ma20", "relative_volume", "volume_surge_closing"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    out["date"] = out["date"].astype("datetime64[ns]")

    return out


def load_global_indices(global_indices_path: Path) -> pd.DataFrame:
    """
    Load global indices + macro signals and compute per-date features.

    Columns produced:
      nasdaq_prev_return       — NASDAQ T-1 daily return
      nikkei_prev_return       — NIKKEI T-1 daily return
      vix_prev_close           — VIX closing level T-1 (fear gauge absolute level)
      vix_5d_avg               — VIX 5-day rolling average (regime context)
      usdidr_prev_close        — USD/IDR T-1 close (rupiah strength)
      usdidr_prev_return       — USD/IDR T-1 daily return (positive = rupiah weakening)
      usdidr_5d_return         — USD/IDR 5-day cumulative return (sustained EM risk-off)
      ihsg_prev_close          — IHSG T-1 close (Jakarta Composite, local market context)
      ihsg_prev_return         — IHSG T-1 daily return
      ihsg_ma_5                — IHSG 5-day SMA (T-1, short-term trend)
      ihsg_ma_20               — IHSG 20-day SMA (T-1, medium-term trend)
      ihsg_ma_100              — IHSG 100-day SMA (T-1, long-term trend)
      ihsg_ma_200              — IHSG 200-day SMA (T-1, primary trend / bull-bear)
      ihsg_close_ma5_ratio     — IHSG close / MA5 (position relative to short MA)
      ihsg_close_ma20_ratio    — IHSG close / MA20 (position relative to med MA)
    """
    empty_cols = [
        "date", "nasdaq_prev_return", "nikkei_prev_return",
        "vix_prev_close", "vix_5d_avg",
        "usdidr_prev_close", "usdidr_prev_return", "usdidr_5d_return",
        "ihsg_prev_close", "ihsg_prev_return",
        "ihsg_ma_5", "ihsg_ma_20", "ihsg_ma_100", "ihsg_ma_200",
        "ihsg_close_ma5_ratio", "ihsg_close_ma20_ratio",
    ]
    if not global_indices_path.exists():
        return pd.DataFrame(columns=empty_cols)

    gi = pd.read_parquet(global_indices_path)
    required = {"date", "symbol", "close"}
    if not required.issubset(gi.columns):
        return pd.DataFrame(columns=empty_cols)

    gi["date"] = pd.to_datetime(gi["date"], errors="coerce").dt.normalize()
    gi["close"] = pd.to_numeric(gi["close"], errors="coerce")
    gi = gi.dropna(subset=["date", "symbol", "close"])

    def _prev_return(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        d = df.sort_values("date").copy()
        d[col_name] = d["close"].pct_change().shift(1)
        return d[["date", col_name]]

    # NASDAQ
    nasdaq_feats = _prev_return(gi[gi["symbol"] == "^IXIC"].copy(), "nasdaq_prev_return")

    # NIKKEI
    nikkei_feats = _prev_return(gi[gi["symbol"] == "^N225"].copy(), "nikkei_prev_return")

    # VIX — absolute level matters more than return (>20 = elevated, >30 = fear)
    vix = gi[gi["symbol"] == "^VIX"].sort_values("date").copy()
    vix["vix_prev_close"] = vix["close"].shift(1)
    vix["vix_5d_avg"] = vix["close"].shift(1).rolling(5, min_periods=3).mean()
    vix_feats = vix[["date", "vix_prev_close", "vix_5d_avg"]].copy()

    # USD/IDR — positive return = rupiah weakening = risk-off for IDX
    usdidr = gi[gi["symbol"] == "IDR=X"].sort_values("date").copy()
    usdidr["usdidr_prev_close"] = usdidr["close"].shift(1)
    usdidr["usdidr_prev_return"] = usdidr["close"].pct_change().shift(1)
    usdidr["usdidr_5d_return"] = (
        usdidr["close"].shift(1) / usdidr["close"].shift(6) - 1
    )
    usdidr_feats = usdidr[["date", "usdidr_prev_close", "usdidr_prev_return", "usdidr_5d_return"]].copy()

    # IHSG — Jakarta Composite Index (^JKSE)
    # MA ratios signal whether market is above/below key trend lines
    ihsg = gi[gi["symbol"] == "^JKSE"].sort_values("date").copy()
    ihsg["ihsg_prev_close"] = ihsg["close"].shift(1)
    ihsg["ihsg_prev_return"] = ihsg["close"].pct_change().shift(1)
    ihsg["ihsg_ma_5"] = ihsg["close"].shift(1).rolling(5, min_periods=3).mean()
    ihsg["ihsg_ma_20"] = ihsg["close"].shift(1).rolling(20, min_periods=10).mean()
    ihsg["ihsg_ma_100"] = ihsg["close"].shift(1).rolling(100, min_periods=50).mean()
    ihsg["ihsg_ma_200"] = ihsg["close"].shift(1).rolling(200, min_periods=100).mean()
    ihsg["ihsg_close_ma5_ratio"] = ihsg["ihsg_prev_close"] / ihsg["ihsg_ma_5"]
    ihsg["ihsg_close_ma20_ratio"] = ihsg["ihsg_prev_close"] / ihsg["ihsg_ma_20"]
    ihsg_feats = ihsg[
        ["date", "ihsg_prev_close", "ihsg_prev_return",
         "ihsg_ma_5", "ihsg_ma_20", "ihsg_ma_100", "ihsg_ma_200",
         "ihsg_close_ma5_ratio", "ihsg_close_ma20_ratio"]
    ].copy()

    # Merge all on date
    out = nasdaq_feats
    for df in [nikkei_feats, vix_feats, usdidr_feats, ihsg_feats]:
        if not df.empty:
            out = out.merge(df, on="date", how="outer")

    float_cols = [c for c in out.columns if c != "date"]
    for col in float_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")

    return out.sort_values("date").reset_index(drop=True)


def build_cvd_features(features: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """
    Compute Cumulative Volume Delta (CVD) per ticker from broker-level L1 data.

    CVD = rolling sum of net_volume (buy_vol - sell_vol) across ALL brokers for a ticker.
    Captures market-wide accumulation/distribution pressure over N days.

    Columns produced (per window N):
      cvd_{N}d          — raw CVD in shares (rolling sum of ticker net volume)
      cvd_{N}d_norm     — CVD normalized by total volume (scale-invariant)
    """
    if windows is None:
        windows = [5, 10, 20]

    required = {"date", "ticker", "flow_net_volume"}
    if features.empty or not required.issubset(features.columns):
        cols = ["date", "ticker"] + [f"cvd_{w}d" for w in windows] + [f"cvd_{w}d_norm" for w in windows]
        return pd.DataFrame(columns=cols)

    df = features[["date", "ticker", "flow_net_volume", "flow_buy_volume", "flow_sell_volume"]].copy() \
        if "flow_buy_volume" in features.columns \
        else features[["date", "ticker", "flow_net_volume"]].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["flow_net_volume"] = pd.to_numeric(df["flow_net_volume"], errors="coerce").fillna(0)

    has_vol = "flow_buy_volume" in df.columns and "flow_sell_volume" in df.columns
    if has_vol:
        df["flow_buy_volume"]  = pd.to_numeric(df["flow_buy_volume"],  errors="coerce").fillna(0)
        df["flow_sell_volume"] = pd.to_numeric(df["flow_sell_volume"], errors="coerce").fillna(0)

    # Aggregate across brokers per ticker/date
    agg_dict = {"flow_net_volume": "sum"}
    if has_vol:
        agg_dict["flow_buy_volume"]  = "sum"
        agg_dict["flow_sell_volume"] = "sum"
    daily = df.groupby(["date", "ticker"]).agg(agg_dict).reset_index()

    if has_vol:
        daily["total_volume"] = daily["flow_buy_volume"] + daily["flow_sell_volume"]
    else:
        daily["total_volume"] = daily["flow_net_volume"].abs() * 2  # rough proxy

    # Compute rolling CVD per ticker
    out_frames = []
    for ticker, grp in daily.groupby("ticker"):
        grp = grp.sort_values("date").copy()
        result = grp[["date"]].copy()
        result["ticker"] = ticker
        for w in windows:
            cvd = grp["flow_net_volume"].rolling(w, min_periods=1).sum()
            total_vol = grp["total_volume"].rolling(w, min_periods=1).sum()
            result[f"cvd_{w}d"] = cvd.values.astype("float32")
            with np.errstate(divide="ignore", invalid="ignore"):
                result[f"cvd_{w}d_norm"] = np.where(
                    total_vol > 0, cvd / total_vol, np.nan
                ).astype("float32")
        out_frames.append(result)

    if not out_frames:
        cols = ["date", "ticker"] + [f"cvd_{w}d" for w in windows] + [f"cvd_{w}d_norm" for w in windows]
        return pd.DataFrame(columns=cols)

    return pd.concat(out_frames, ignore_index=True)


def build_stockbit_features(broksum_bybroker: pd.DataFrame, broker_code: str = "XL") -> pd.DataFrame:
    """
    Stockbit (XL) broker activity features — proxy for retail markup participation.

    Thesis: when Stockbit retail starts actively trading a stock, it signals early markup phase.
    We ride the momentum and exit before retail fully piles in.

    Features (all T-1 shifted, no lookahead):
      xl_buy_freq_ma5    — 5-day rolling mean of XL buy frequency (T-1)
      xl_sell_freq_ma5   — 5-day rolling mean of XL sell frequency (retail attention proxy)
      xl_buy_freq_ma20   — 20-day rolling mean (baseline activity level)
      xl_freq_surge      — today's (buy+sell) freq vs 20d baseline (is activity spiking?)
    """
    required = {"broker", "stock_code", "date", "buy_freq", "sell_freq"}
    if broksum_bybroker.empty or not required.issubset(broksum_bybroker.columns):
        return pd.DataFrame(columns=["date", "ticker", "xl_buy_freq_ma5", "xl_sell_freq_ma5",
                                     "xl_buy_freq_ma20", "xl_freq_surge"])

    df = broksum_bybroker[broksum_bybroker["broker"] == broker_code].copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "ticker", "xl_buy_freq_ma5", "xl_sell_freq_ma5",
                                     "xl_buy_freq_ma20", "xl_freq_surge"])

    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    df = df.sort_values(["stock_code", "date"])
    df["buy_freq"] = pd.to_numeric(df["buy_freq"], errors="coerce").fillna(0)
    df["sell_freq"] = pd.to_numeric(df["sell_freq"], errors="coerce").fillna(0)
    df["total_freq"] = df["buy_freq"] + df["sell_freq"]

    grp = df.groupby("stock_code", sort=False)
    df["xl_buy_freq_ma5"]  = grp["buy_freq"].transform(lambda s: s.shift(1).rolling(5,  min_periods=2).mean())
    df["xl_sell_freq_ma5"] = grp["sell_freq"].transform(lambda s: s.shift(1).rolling(5,  min_periods=2).mean())
    df["xl_buy_freq_ma20"] = grp["buy_freq"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    df["xl_total_ma20"]    = grp["total_freq"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    df["xl_freq_surge"]    = df["total_freq"].shift(1) / df["xl_total_ma20"].replace(0, np.nan)

    out = df[["date", "stock_code", "xl_buy_freq_ma5", "xl_sell_freq_ma5",
              "xl_buy_freq_ma20", "xl_freq_surge"]].copy()
    out.columns = ["date", "ticker", "xl_buy_freq_ma5", "xl_sell_freq_ma5",
                   "xl_buy_freq_ma20", "xl_freq_surge"]
    for col in ["xl_buy_freq_ma5", "xl_sell_freq_ma5", "xl_buy_freq_ma20", "xl_freq_surge"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")

    return out


def build_adx_features(yf_daily: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute ADX (Average Directional Index) and directional movement features per ticker.

    Returns date/ticker DataFrame with:
      adx_14    — trend strength (higher = stronger trend)
      plus_di   — +DI (bullish directional pressure)
      minus_di  — -DI (bearish directional pressure)
      di_diff   — +DI minus -DI (net directional bias)

    Uses Wilder's smoothing (EWM alpha=1/period).
    """
    required = {"date", "ticker", "high", "low", "close"}
    if yf_daily.empty or not required.issubset(yf_daily.columns):
        return pd.DataFrame(columns=["date", "ticker", "adx_14", "plus_di", "minus_di", "di_diff"])

    df = yf_daily.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for col in ["high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "high", "low", "close"])

    alpha = 1.0 / period

    def _adx_per_ticker(ticker_name: str, g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        high  = g["high"].values.astype(float)
        low   = g["low"].values.astype(float)
        close = g["close"].values.astype(float)
        n = len(high)

        tr   = np.full(n, np.nan)
        pdm  = np.full(n, np.nan)
        ndm  = np.full(n, np.nan)

        for i in range(1, n):
            tr[i]  = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            up   = high[i] - high[i-1]
            down = low[i-1] - low[i]
            pdm[i] = up   if (up > down and up > 0) else 0.0
            ndm[i] = down if (down > up and down > 0) else 0.0

        atr     = pd.Series(tr).ewm(alpha=alpha, min_periods=period, adjust=False).mean().values
        pdi_raw = pd.Series(pdm).ewm(alpha=alpha, min_periods=period, adjust=False).mean().values
        ndi_raw = pd.Series(ndm).ewm(alpha=alpha, min_periods=period, adjust=False).mean().values

        with np.errstate(divide="ignore", invalid="ignore"):
            pdi = np.where(atr > 0, 100.0 * pdi_raw / atr, np.nan)
            ndi = np.where(atr > 0, 100.0 * ndi_raw / atr, np.nan)
            dx  = np.where((pdi + ndi) > 0, 100.0 * np.abs(pdi - ndi) / (pdi + ndi), np.nan)

        adx = pd.Series(dx).ewm(alpha=alpha, min_periods=period, adjust=False).mean().values

        return pd.DataFrame({
            "date":     g["date"].values,
            "ticker":   ticker_name,
            "adx_14":   adx.astype("float32"),
            "plus_di":  pdi.astype("float32"),
            "minus_di": ndi.astype("float32"),
            "di_diff":  (pdi - ndi).astype("float32"),
        })

    result = pd.concat(
        [_adx_per_ticker(t, g) for t, g in df.groupby("ticker")],
        ignore_index=True,
    )
    return result.reset_index(drop=True)


def build_price_proximity(yf_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Build BSJP-specific price proximity feature:
    - proximity_to_high: (Close_T - Low_T) / (High_T - Low_T)
      Values closer to 1.0 indicate stock closed near its high (strong BSJP signal)
    """
    if yf_daily.empty or not {"date", "ticker", "high", "low", "close"}.issubset(yf_daily.columns):
        return pd.DataFrame(columns=["date", "ticker", "proximity_to_high"])

    df = yf_daily.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "high", "low", "close"])

    # Avoid division by zero
    range_val = df["high"] - df["low"]
    df["proximity_to_high"] = np.where(
        range_val > 0,
        (df["close"] - df["low"]) / range_val,
        0.5  # Neutral when no range
    )

    out = df[["date", "ticker", "proximity_to_high"]].copy()
    out["proximity_to_high"] = out["proximity_to_high"].astype("float32")
    return out


def build_hmm_regime(ihsg_daily: pd.DataFrame, n_states: int = 3) -> pd.DataFrame:
    """
    Build IHSG Market Regime features using Gaussian Mixture Hidden Markov Model.
    States: 0 (Bear), 1 (Sideways), 2 (Bull)
    Uses expanding window to avoid look-ahead bias.
    """
    if ihsg_daily.empty or "close" not in ihsg_daily.columns:
        return pd.DataFrame(columns=["date", "ihsg_regime", "ihsg_regime_confidence"])

    df = ihsg_daily.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")

    # Calculate log returns
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["log_return"])

    if len(df) < 60:  # Minimum data for HMM
        return pd.DataFrame(columns=["date", "ihsg_regime", "ihsg_regime_confidence"])

    try:
        from hmmlearn.hmm import GaussianHMM

        returns = df["log_return"].values.reshape(-1, 1)

        # Fit HMM on full data for state ordering (we'll use this to label states)
        hmm = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
        hmm.fit(returns)

        # Determine state labels based on mean returns: Bear < Sideways < Bull
        state_means = hmm.means_.flatten()
        state_order = np.argsort(state_means)  # 0=lowest mean (Bear), 2=highest mean (Bull)
        label_map = {state_order[i]: i for i in range(n_states)}  # Map to 0,1,2

        # Expanding window prediction (no look-ahead)
        regimes = []
        confidences = []

        min_window = 30
        for i in range(len(df)):
            if i < min_window:
                regimes.append(np.nan)
                confidences.append(np.nan)
                continue

            # Fit on expanding window up to i-1 (previous day only)
            train_data = returns[:i]
            hmm_online = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=50, random_state=42)
            hmm_online.fit(train_data)

            # Predict current state
            current_return = returns[i].reshape(1, -1)
            state_probs = hmm_online.predict_proba(current_return)[0]
            predicted_state = np.argmax(state_probs)

            # Map to canonical labels
            # Note: We need to map based on means from current model
            online_means = hmm_online.means_.flatten()
            online_order = np.argsort(online_means)
            online_label_map = {online_order[j]: j for j in range(n_states)}

            canonical_state = online_label_map[predicted_state]
            confidence = state_probs[predicted_state]

            regimes.append(canonical_state)
            confidences.append(confidence)

        df["ihsg_regime"] = regimes
        df["ihsg_regime_confidence"] = confidences

        # Downcast
        df["ihsg_regime"] = pd.to_numeric(df["ihsg_regime"], errors="coerce").astype("float32")
        df["ihsg_regime_confidence"] = pd.to_numeric(df["ihsg_regime_confidence"], errors="coerce").astype("float32")

        return df[["date", "ihsg_regime", "ihsg_regime_confidence"]].copy()

    except ImportError:
        # Fallback: Simple quantile-based regime if hmmlearn not available
        print("[HMM] hmmlearn not available, using quantile-based regime fallback")
        df["ihsg_regime"] = pd.qcut(df["log_return"], q=[0, 0.33, 0.67, 1.0], labels=[0, 1, 2]).astype("float32")
        df["ihsg_regime_confidence"] = 0.5  # Neutral confidence for fallback
        return df[["date", "ihsg_regime", "ihsg_regime_confidence"]].copy()


def build_skewness_kurtosis(yf_daily: pd.DataFrame, windows: list[int] = [20, 60]) -> pd.DataFrame:
    """
    Build rolling skewness and kurtosis features as proxy for fat-tail risk.
    Uses shift(1) for no-lookahead.
    """
    if yf_daily.empty or not {"date", "ticker", "close"}.issubset(yf_daily.columns):
        return pd.DataFrame(columns=["date", "ticker"])

    df = yf_daily.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "close"]).sort_values(["ticker", "date"])

    # Calculate log returns
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    for w in windows:
        # Skewness with shift(1) for no-lookahead
        df[f"return_skew_{w}"] = (
            df.groupby("ticker", sort=False)["log_return"]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=w//2).skew())
        )
        # Kurtosis with shift(1) for no-lookahead
        df[f"return_kurt_{w}"] = (
            df.groupby("ticker", sort=False)["log_return"]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=w//2).kurt())
        )

    # Select only new columns + keys
    cols = ["date", "ticker"] + [f"return_skew_{w}" for w in windows] + [f"return_kurt_{w}" for w in windows]
    out = df[cols].copy()

    # Downcast to float32
    for col in out.columns:
        if col not in ["date", "ticker"]:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")

    return out


def build_ticker_tp_rate(labels: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """
    Compute per-ticker historical BPJS TP hit rate over rolling windows.

    Uses shift(1) to avoid lookahead: on feature_date T-1, the rate reflects
    results from trade sessions before T.

    Columns produced (per window N):
      ticker_tp_rate_{N}d — rolling mean of label_tp over N prior trade sessions

    Correlation with label_tp: ~+0.22 (strongest individual feature after range_pct)
    """
    if windows is None:
        windows = [20, 60, 120]

    required = {"feature_date", "ticker", "label_tp"}
    out_cols = ["date", "ticker"] + [f"ticker_tp_rate_{w}d" for w in windows]
    if labels.empty or not required.issubset(labels.columns):
        return pd.DataFrame(columns=out_cols)

    df = labels[["feature_date", "ticker", "label_tp"]].copy()
    df["feature_date"] = pd.to_datetime(df["feature_date"], errors="coerce").dt.normalize()
    df["label_tp"] = pd.to_numeric(df["label_tp"], errors="coerce")
    df = df.dropna().sort_values(["ticker", "feature_date"])

    result_frames = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("feature_date").copy()
        row = grp[["feature_date"]].copy()
        row["ticker"] = ticker
        for w in windows:
            # shift(1): rate from the session BEFORE the current feature_date
            row[f"ticker_tp_rate_{w}d"] = (
                grp["label_tp"].shift(1).rolling(w, min_periods=max(5, w // 4)).mean()
            ).values.astype("float32")
        result_frames.append(row)

    if not result_frames:
        return pd.DataFrame(columns=out_cols)

    out = pd.concat(result_frames, ignore_index=True)
    out = out.rename(columns={"feature_date": "date"})
    return out


def build_opening_session_features(yf_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ticker opening session (09:00 bar) history features from yf_1h.

    These are the strongest new feature candidates for BPJS (corr 0.12-0.20):
      open_sess_range_ma_20d   — avg opening session hi-lo range last 20 days (+0.197)
      open_sess_ret_std_20d    — std of opening session returns last 20 days (+0.190)
      open_sess_tp2_rate_20d   — fraction of sessions with opening return >= 2% (+0.168)
      open_sess_ret_ma_20d     — avg opening session close return last 20 days (+0.119)
      prev_open_sess_ret       — yesterday's opening session return (+0.063)

    ~78% NaN coverage (illiquid stocks without 09:00 bars); LightGBM handles natively.
    """
    required = {"datetime", "ticker", "open", "high", "low", "close"}
    out_cols = ["date", "ticker", "open_sess_ret_ma_20d", "open_sess_ret_std_20d",
                "open_sess_range_ma_20d", "open_sess_tp2_rate_20d", "prev_open_sess_ret"]
    if yf_1h.empty or not required.issubset(yf_1h.columns):
        return pd.DataFrame(columns=out_cols)

    df = yf_1h.copy()
    df["ticker"] = df["ticker"].str.replace(".JK", "", regex=False)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "ticker"])

    # Keep only 09:00 bars (opening session)
    open_bars = df[df["datetime"].dt.hour == 9].copy()
    open_bars["date"] = open_bars["datetime"].dt.normalize()
    for col in ["open", "high", "low", "close"]:
        open_bars[col] = pd.to_numeric(open_bars[col], errors="coerce")
    open_bars = open_bars.dropna(subset=["date", "open", "high", "low", "close"])
    open_bars = open_bars[open_bars["open"] > 0].copy()

    open_bars["open_sess_ret"]   = (open_bars["close"] - open_bars["open"]) / open_bars["open"]
    open_bars["open_sess_range"] = (open_bars["high"]  - open_bars["low"])  / open_bars["open"]
    open_bars = open_bars[["date", "ticker", "open_sess_ret", "open_sess_range"]].sort_values(["ticker", "date"])

    result_frames = []
    for ticker, g in open_bars.groupby("ticker"):
        g = g.sort_values("date").copy()
        r = g["open_sess_ret"]
        row = g[["date"]].copy()
        row["ticker"] = ticker
        row["open_sess_ret_ma_20d"]   = r.rolling(20, min_periods=10).mean().astype("float32")
        row["open_sess_ret_std_20d"]  = r.rolling(20, min_periods=10).std().astype("float32")
        row["open_sess_range_ma_20d"] = g["open_sess_range"].rolling(20, min_periods=10).mean().astype("float32")
        row["open_sess_tp2_rate_20d"] = (r >= 0.02).rolling(20, min_periods=10).mean().astype("float32")
        row["prev_open_sess_ret"]     = r.shift(1).astype("float32")
        result_frames.append(row)

    if not result_frames:
        return pd.DataFrame(columns=out_cols)

    return pd.concat(result_frames, ignore_index=True)


def build_training_datamart(feature_agg: pd.DataFrame, labels: pd.DataFrame, target_pct: float, cutoff: str) -> pd.DataFrame:
    if feature_agg.empty or labels.empty:
        return pd.DataFrame()

    merged = feature_agg.merge(
        labels,
        left_on=["date", "ticker"],
        right_on=["feature_date", "ticker"],
        how="inner",
    )
    merged = merged.drop(columns=["feature_date"])
    merged["label_name"] = f"open_to_{cutoff.replace(':', '')}_tp{int(target_pct * 100)}"
    merged = merged.sort_values(["date", "ticker"]).reset_index(drop=True)
    
    # CRITICAL: Force garbage collection after heavy merge to free RAM
    gc.collect()
    
    return merged


def main() -> None:
    args = parse_args()
    cutoff_hh, cutoff_mm = parse_cutoff_time(args.cutoff_time)

    ensure_exists(args.features_path, "features_path")
    ensure_exists(args.yfinance_1h_path, "yfinance_1h_path")

    print(f"[Init] features={args.features_path}")
    print(f"[Init] yfinance_1h={args.yfinance_1h_path}")
    print(
        f"[Init] target_pct={args.target_pct:.4f}, stop_loss_pct={args.stop_loss_pct:.4f}, "
        f"cutoff={args.cutoff_time}, focus_broker={args.focus_broker}"
    )

    features = load_features_optimized(args.features_path)
    yf_1h = pd.read_parquet(args.yfinance_1h_path, columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])

    # BSJP: Load raw yfinance daily for IHSG index extraction (needs close only)
    yf_daily_raw = pd.DataFrame()
    if args.yfinance_daily_path.exists():
        yf_daily_raw = pd.read_parquet(args.yfinance_daily_path, columns=["date", "ticker", "close"])
        print(f"[Load] yf_daily_raw_rows={len(yf_daily_raw):,}")

    # Derive yf_daily from features (for price proximity and skewness/kurtosis)
    yf_daily = pd.DataFrame()
    if not features.empty and all(col in features.columns for col in ["yf_daily_high", "yf_daily_low", "yf_daily_close"]):
        yf_daily = (
            features[["date", "ticker", "yf_daily_high", "yf_daily_low", "yf_daily_close"]]
            .drop_duplicates(subset=["date", "ticker"])
            .rename(columns=lambda c: c.replace("yf_daily_", ""))
        )
        print(f"[Load] yf_daily_from_features_rows={len(yf_daily):,}")

    print(f"[Load] features_rows={len(features):,}, features_cols={len(features.columns)}")
    print(f"[Load] yf_1h_rows={len(yf_1h):,}")

    local_fund_brokers, bandar_brokers = load_master_broker(args.master_broker_path, args.bandar_localfund_min)
    print(f"[Master] localfund_brokers={len(local_fund_brokers)}, bandar_brokers={len(bandar_brokers)}")

    # Strategy-specific label generation
    print(f"[Strategy] mode={args.strategy_mode.upper()}")
    if args.strategy_mode == "bpjs":
        # BPJS: Beli Pagi Jual Sore (entry 09:00, exit 10:00 same day)
        labels = build_label_table(
            yf_1h,
            target_pct=args.target_pct,
            stop_loss_pct=args.stop_loss_pct,
            cutoff_hh=cutoff_hh,
            cutoff_mm=cutoff_mm,
            label_mode=args.label_mode,
        )
    else:  # bsjp
        # BSJP: Beli Sore Jual Pagi (entry close day T, exit open day T+1)
        if not args.yfinance_daily_path.exists():
            raise FileNotFoundError("BSJP mode requires --yfinance-daily-path")
        # Load full daily data for BSJP labeling
        yf_daily_full = pd.read_parquet(args.yfinance_daily_path, columns=["date", "ticker", "open", "high", "low", "close"])
        labels = build_label_bsjp(
            yf_daily_full,
            target_pct=args.target_pct,
            stop_loss_pct=args.stop_loss_pct,
        )

    if labels.empty:
        raise RuntimeError(f"Label table is empty for {args.strategy_mode}. Check market data coverage.")
    print(
        f"[Label] rows={len(labels):,}, trade_date={labels['trade_date'].min().date()} -> {labels['trade_date'].max().date()}, "
        f"tp_rate={(labels['label_tp'].mean() * 100):.2f}%"
    )

    feat_agg = build_feature_aggregate(
        features=features,
        focus_broker=args.focus_broker,
        local_fund_brokers=local_fund_brokers,
        bandar_brokers=bandar_brokers,
    )

    # BSJP: Merge Volume Family features
    volume_feats = build_volume_features(yf_1h)
    if not volume_feats.empty:
        feat_agg = feat_agg.merge(volume_feats, on=["date", "ticker"], how="left")
        print(f"[VolumeFamily] merged volume_ma20, relative_volume, volume_surge_closing")

    # BSJP: Merge Global Index Family features
    global_indices = load_global_indices(args.global_indices_path)
    if not global_indices.empty:
        feat_agg = feat_agg.merge(global_indices, on="date", how="left")
        print(f"[GlobalIndex] merged nasdaq_prev_return, nikkei_morning_return")

    # BPJS Option A: ADX + Directional Movement features (per ticker, from daily OHLC)
    adx_feats = build_adx_features(yf_daily)
    if not adx_feats.empty:
        feat_agg = feat_agg.merge(adx_feats, on=["date", "ticker"], how="left")
        print(f"[ADX] merged adx_14, plus_di, minus_di, di_diff")

    # CVD (Cumulative Volume Delta) — ticker-level net volume accumulation over 5/10/20 days
    cvd_feats = build_cvd_features(features)
    if not cvd_feats.empty:
        feat_agg = feat_agg.merge(cvd_feats, on=["date", "ticker"], how="left")
        cvd_cols = [c for c in cvd_feats.columns if c not in ("date", "ticker")]
        print(f"[CVD] merged {', '.join(cvd_cols)}")

    # Stockbit (XL) retail activity — "ride the retail wave, exit before they realize"
    broksum_bybroker = pd.DataFrame()
    if args.broksum_bybroker_path.exists():
        broksum_bybroker = pd.read_parquet(args.broksum_bybroker_path)
    xl_feats = build_stockbit_features(broksum_bybroker, broker_code=STOCKBIT_BROKER_CODE)
    if not xl_feats.empty:
        feat_agg = feat_agg.merge(xl_feats, on=["date", "ticker"], how="left")
        print(f"[Stockbit/XL] merged xl_buy_freq_ma5, xl_sell_freq_ma5, xl_buy_freq_ma20, xl_freq_surge")

    # BSJP: Merge Price Proximity features
    price_prox = build_price_proximity(yf_daily)
    if not price_prox.empty:
        feat_agg = feat_agg.merge(price_prox, on=["date", "ticker"], how="left")
        print(f"[PriceProximity] merged proximity_to_high")

    # V13: Merge IHSG Market Regime (HMM) features
    # Extract IHSG data from yf_daily_raw (ticker = "^JKSE" or similar)
    ihsg_daily = pd.DataFrame()
    if not yf_daily_raw.empty and "ticker" in yf_daily_raw.columns:
        ihsg_mask = yf_daily_raw["ticker"].str.upper().isin(["^JKSE", "JKSE", "IHSG", "IDX"])
        if ihsg_mask.any():
            ihsg_daily = yf_daily_raw[ihsg_mask].copy()
        else:
            # Try to get IHSG from features if available
            if "yf_daily_close" in feat_agg.columns:
                # Use aggregate as proxy if available
                pass

    if not ihsg_daily.empty:
        hmm_regime = build_hmm_regime(ihsg_daily, n_states=3)
        if not hmm_regime.empty:
            feat_agg = feat_agg.merge(hmm_regime, on="date", how="left")
            print(f"[HMMRegime] merged ihsg_regime, ihsg_regime_confidence")

    # V13: Merge Skewness & Kurtosis features
    skew_kurt = build_skewness_kurtosis(yf_daily, windows=[20, 60])
    if not skew_kurt.empty:
        feat_agg = feat_agg.merge(skew_kurt, on=["date", "ticker"], how="left")
        print(f"[SkewKurt] merged return_skew_20, return_kurt_20, return_skew_60, return_kurt_60")

    # Per-ticker historical BPJS TP rate (rolling 20/60/120d) — corr ~+0.22 with label_tp
    ticker_tp_rate = build_ticker_tp_rate(labels, windows=[20, 60, 120])
    if not ticker_tp_rate.empty:
        feat_agg = feat_agg.merge(ticker_tp_rate, on=["date", "ticker"], how="left")
        tp_rate_cols = [c for c in ticker_tp_rate.columns if c not in ("date", "ticker")]
        print(f"[TickerTPRate] merged {', '.join(tp_rate_cols)}")
        # Derived: is this stock's recent tp_rate improving vs historical?
        if "ticker_tp_rate_20d" in feat_agg.columns and "ticker_tp_rate_60d" in feat_agg.columns:
            feat_agg["tp_rate_trend"] = (
                feat_agg["ticker_tp_rate_20d"] - feat_agg["ticker_tp_rate_60d"]
            ).astype("float32")
            print(f"[TickerTPRate] derived tp_rate_trend (20d-60d)")

    # Opening session history features from yf_1h (corr 0.12-0.20 with label_tp)
    open_sess_feats = build_opening_session_features(yf_1h)
    if not open_sess_feats.empty:
        feat_agg = feat_agg.merge(open_sess_feats, on=["date", "ticker"], how="left")
        os_cols = [c for c in open_sess_feats.columns if c not in ("date", "ticker")]
        print(f"[OpenSessFeats] merged {', '.join(os_cols)}")

    # Ensure all new numeric columns are float32 for RAM optimization
    for col in feat_agg.columns:
        if pd.api.types.is_float_dtype(feat_agg[col]) and col not in ["date"]:
            feat_agg[col] = feat_agg[col].astype("float32")

    print(
        f"[FeatureAgg] rows={len(feat_agg):,}, cols={len(feat_agg.columns)}, "
        f"date={feat_agg['date'].min().date()} -> {feat_agg['date'].max().date()}"
    )

    train = build_training_datamart(
        feature_agg=feat_agg,
        labels=labels,
        target_pct=args.target_pct,
        cutoff=args.cutoff_time,
    )
    if train.empty:
        raise RuntimeError("Training datamart is empty after join. Check date/ticker overlap.")
    print(
        f"[Training] rows={len(train):,}, cols={len(train.columns)}, "
        f"date={train['date'].min().date()} -> {train['date'].max().date()}, "
        f"tp_rate={(train['label_tp'].mean() * 100):.2f}%"
    )

    if args.dry_run:
        print("[DryRun] Skip writing outputs.")
        return

    # Auto-switch output path for relative_rank to avoid overwriting fixed_tp datamart
    training_output = args.training_output
    if args.label_mode == "relative_rank" and training_output == DEFAULT_TRAIN_OUTPUT:
        training_output = DEFAULT_TRAIN_OUTPUT_RANK

    args.label_output.parent.mkdir(parents=True, exist_ok=True)
    training_output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(args.label_output, index=False)
    train.to_parquet(training_output, index=False)
    print(f"[Done] label_output={args.label_output}")
    print(f"[Done] training_output={training_output}")


if __name__ == "__main__":
    main()
