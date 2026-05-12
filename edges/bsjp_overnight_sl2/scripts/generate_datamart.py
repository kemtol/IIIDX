"""
BSJP Overnight SL2 — generate_datamart.py

Strategy: Beli close ~15:30-15:45, jual di T+1 pada jam exit.
Default (--exit-hour 9):  jual open 09:xx T+1 → overnight label
Variant (--exit-hour 10): jual open 10:xx T+1 → close10 label
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse shared feature builders from BPJS script
_BPJS_SCRIPTS = Path(__file__).resolve().parents[3] / "edges" / "bpjs_opening_tp3" / "scripts"
if str(_BPJS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_BPJS_SCRIPTS))
from generate_datamart import (  # noqa: E402
    build_feature_aggregate,
    load_global_indices,
    load_master_broker,
    normalize_ticker_series,
)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
IDX_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = IDX_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "Level_0_Raw"
FEATURES_DIR = DATA_DIR / "Level_1_Features"
DATAMART_DIR = DATA_DIR / "Level_2_Datamart"

DEFAULT_YF_1H          = RAW_DATA_DIR / "yfinance_1h.parquet"
DEFAULT_YF_DAILY       = RAW_DATA_DIR / "yfinance_daily.parquet"
DEFAULT_FEATURES       = FEATURES_DIR / "broksum_datamart.parquet"
DEFAULT_BROKSUM_RAW    = RAW_DATA_DIR / "broksum_bybroker.parquet"
DEFAULT_MASTER_BROKER  = RAW_DATA_DIR / "master_broker.parquet"
DEFAULT_GLOBAL_INDICES = RAW_DATA_DIR / "global_indices.parquet"
DEFAULT_VWAP_FEATURES  = FEATURES_DIR / "vwap_features.parquet"
DEFAULT_TRAIN_OUTPUT   = DATAMART_DIR / "training_datamart_bsjp_overnight.parquet"
DEFAULT_TRAIN_OUTPUT_CLOSE10 = DATAMART_DIR / "training_datamart_bsjp_close10.parquet"

MODULES_DIR = FEATURES_DIR / "modules"
STOCKBIT_BROKER_CODE = "XL"
ROUNDTRIP_COST = 0.004
SL_THRESHOLD   = -0.02
ENTRY_HOUR     = 15
EXIT_HOUR      = 10

UNIV_MIN_PRICE           = 50.0
UNIV_MIN_TURNOVER_IDR    = 500_000_000.0
UNIV_ARA_RETURN_THRESH   = 0.20

# ---------------------------------------------------------------------------
# Normalization & Helpers
# ---------------------------------------------------------------------------

def idx_tick_size(price: float) -> int:
    if price < 200: return 1
    if price < 500: return 2
    if price < 2000: return 5
    if price < 5000: return 10
    return 25

def idx_ara_limit_pct(reference_price: float) -> float:
    if reference_price <= 0 or pd.isna(reference_price):
        return np.nan
    if reference_price <= 200:
        return 0.35
    if reference_price <= 5000:
        return 0.25
    return 0.20

def normalize_yf_1h_session_time(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize mixed yfinance_1h timestamp regimes into exchange-session time.
    """
    df = raw.copy().reset_index(drop=True)
    df["_row_id"] = np.arange(len(df), dtype=np.int64)
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    if dt.dt.tz is not None:
        utc_plus7 = dt.dt.tz_convert("Asia/Jakarta").dt.tz_localize(None)
        legacy_plus14 = (dt + pd.Timedelta(hours=14)).dt.tz_localize(None)
    else:
        utc_plus7 = dt
        legacy_plus14 = dt + pd.Timedelta(hours=14)

    ticker = df["ticker"].astype(str).str.replace(r"\.JK$", "", regex=True)
    candidates = []
    for scheme_order, (scheme, local_dt) in enumerate(
        [("utc_plus7", utc_plus7), ("legacy_plus14", legacy_plus14)]
    ):
        cand = df[["_row_id", "ticker"]].copy()
        cand["date"] = local_dt.dt.normalize()
        cand["hour"] = local_dt.dt.hour
        cand["ticker"] = ticker
        valid_hour = cand["hour"].isin({9, 10, 11, 13, 14, 15, 16}).astype(int)
        score = valid_hour.groupby([cand["date"], cand["ticker"]]).transform("sum")
        cand["_score"] = score.astype(float) - (scheme_order * 0.0001)
        candidates.append(cand)

    best = pd.concat(candidates).sort_values("_score", ascending=False).drop_duplicates("_row_id")
    df = df.merge(best[["_row_id", "date", "hour", "ticker"]], on="_row_id", suffixes=("_raw", ""))
    return df.drop(columns=["_row_id"])

# ---------------------------------------------------------------------------
# Label & Filter Builders
# ---------------------------------------------------------------------------

def build_label(yf_1h: pd.DataFrame, exit_hour: int = EXIT_HOUR) -> pd.DataFrame:
    df = yf_1h.copy()
    entry = (
        df[df["hour"] == ENTRY_HOUR]
        .sort_values(["ticker", "date", "datetime"])
        .groupby(["ticker", "date"], sort=False)["close"]
        .last()
        .reset_index(name="entry_price")
    ).rename(columns={"date": "trade_date"})

    exit_ = (
        df[df["hour"] == exit_hour]
        .sort_values(["ticker", "date", "datetime"])
        .groupby(["ticker", "date"], sort=False)["open"]
        .first()
        .reset_index(name="exit_price")
    ).rename(columns={"date": "exit_date"})
    
    exit_ = exit_.sort_values(["ticker", "exit_date"])
    exit_["trade_date"] = exit_.groupby("ticker")["exit_date"].shift(1)
    exit_ = exit_.dropna(subset=["trade_date"])
    exit_["trade_date"] = exit_["trade_date"].dt.tz_localize(None).astype("datetime64[ns]")

    labels = entry.merge(exit_[["trade_date", "ticker", "exit_price", "exit_date"]], on=["trade_date", "ticker"], how="inner")
    labels = labels.dropna(subset=["entry_price", "exit_price"])
    labels = labels[labels["entry_price"] > 0]
    labels["overnight_return"] = (labels["exit_price"] - labels["entry_price"]) / labels["entry_price"]

    labels["_gap_days"] = (labels["exit_date"] - labels["trade_date"]).dt.days
    labels = labels[~(((labels["_gap_days"] > 3) & (labels["overnight_return"].abs() > 0.15)) | (labels["overnight_return"].abs() > 0.50))].copy()
    labels = labels.drop(columns=["_gap_days"])
    labels["label_tp"]  = (labels["overnight_return"] > ROUNDTRIP_COST).astype("int8")
    labels["label_sl2"] = (labels["overnight_return"] < SL_THRESHOLD).astype("int8")
    labels["label_name"] = "bsjp_overnight_sl2" if exit_hour == 9 else f"bsjp_close{exit_hour}_sl2"
    return labels.rename(columns={"trade_date": "date"})

def build_universe_filter(yf_1h: pd.DataFrame, min_price: float, min_turnover_idr: float, ara_threshold: float) -> pd.DataFrame:
    df = yf_1h.copy()
    g = df.groupby(["date", "ticker"])
    daily = g.agg(
        close_T=("close", "last"),
        turnover_T=("volume", lambda s: (s * df.loc[s.index, "close"]).sum())
    ).reset_index()
    daily = daily.sort_values(["ticker", "date"])
    daily["median_turnover_20d"] = daily.groupby("ticker")["turnover_T"].transform(lambda x: x.rolling(20, min_periods=1).median())
    daily["prev_close"] = daily.groupby("ticker")["close_T"].shift(1)
    daily["day_return"] = (daily["close_T"] - daily["prev_close"]) / daily["prev_close"]
    
    daily["is_gocap"] = daily["close_T"] < min_price
    daily["is_illiquid"] = daily["median_turnover_20d"] < min_turnover_idr
    daily["is_ara"] = daily["day_return"] >= ara_threshold
    daily["exclude"] = daily["is_gocap"] | daily["is_illiquid"] | daily["is_ara"]
    return daily

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yf-1h-path", type=Path, default=DEFAULT_YF_1H)
    parser.add_argument("--yf-daily-path", type=Path, default=DEFAULT_YF_DAILY)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--broksum-raw-path", type=Path, default=DEFAULT_BROKSUM_RAW)
    parser.add_argument("--master-broker-path", type=Path, default=DEFAULT_MASTER_BROKER)
    parser.add_argument("--global-indices-path", type=Path, default=DEFAULT_GLOBAL_INDICES)
    parser.add_argument("--vwap-features-path", type=Path, default=DEFAULT_VWAP_FEATURES)
    parser.add_argument("--focus-broker", default="MG")
    parser.add_argument("--bandar-localfund-min", type=float, default=60.0)
    parser.add_argument("--training-output", type=Path, default=None)
    parser.add_argument("--enable-universe-filter", action="store_true")
    parser.add_argument("--exit-hour", type=int, default=EXIT_HOUR)
    parser.add_argument("--date-from", type=str, default=None)
    parser.add_argument("--warmup-calendar-days", type=int, default=120)
    parser.add_argument("--features-warmup-calendar-days", type=int, default=120)
    parser.add_argument("--modules-dir", type=Path, default=MODULES_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.training_output is None:
        args.training_output = DEFAULT_TRAIN_OUTPUT if args.exit_hour == 9 else DEFAULT_TRAIN_OUTPUT_CLOSE10

    yf_1h = pd.read_parquet(args.yf_1h_path)
    yf_1h = normalize_yf_1h_session_time(yf_1h)
    
    date_from_ts = pd.Timestamp(args.date_from) if args.date_from else None
    if date_from_ts:
        warmup = date_from_ts - pd.Timedelta(days=args.warmup_calendar_days)
        yf_1h = yf_1h[yf_1h["date"] >= warmup].copy()

    labels = build_label(yf_1h, exit_hour=args.exit_hour)
    if args.enable_universe_filter:
        filt = build_universe_filter(yf_1h, args.universe_min_price, 500_000_000, 0.20)
        labels = labels.merge(filt[["date", "ticker", "exclude"]], on=["date", "ticker"], how="left")
        labels = labels[~labels["exclude"].fillna(True)].copy().drop(columns=["exclude"])

    features = pd.read_parquet(args.features_path)
    features["date"] = pd.to_datetime(features["date"])
    features["ticker"] = normalize_ticker_series(features["ticker"])
    
    # Merge all modules
    train = labels.copy()
    modules_enabled = bool(args.modules_dir) and args.modules_dir != Path(".")
    
    # Forensic V2
    f2_path = args.modules_dir / "forensic_v2_features.parquet"
    if f2_path.exists():
        f2 = pd.read_parquet(f2_path)
        f2["date"] = pd.to_datetime(f2["date"])
        train = train.merge(f2, on=["date", "ticker"], how="left")

    # Forensic V1
    f1_path = args.modules_dir / "forensic_features.parquet"
    if f1_path.exists():
        f1 = pd.read_parquet(f1_path)
        f1["date"] = pd.to_datetime(f1["date"])
        train = train.merge(f1, on=["date", "ticker"], how="left")

    # Sector
    sec_path = args.modules_dir / "sector_features.parquet"
    if sec_path.exists():
        sec = pd.read_parquet(sec_path)
        sec["date"] = pd.to_datetime(sec["date"])
        train = train.merge(sec, on=["date", "ticker"], how="left")

    # Preclose14
    p14_path = args.modules_dir / "preclose14_features.parquet"
    if p14_path.exists():
        p14 = pd.read_parquet(p14_path)
        p14["date"] = pd.to_datetime(p14["date"])
        train = train.merge(p14, on=["date", "ticker"], how="left")

    print(f"[Done] rows={len(train):,}, dates={train['date'].min()} -> {train['date'].max()}")
    if not args.dry_run:
        train.to_parquet(args.training_output, index=False)

if __name__ == "__main__":
    main()
