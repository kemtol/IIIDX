"""
BSJP Overnight SL2 — generate_datamart.py

Strategy: Beli close ~15:30-15:45, jual di T+1 pada jam exit.
Default (--exit-hour 9):  jual open 09:xx T+1 → overnight label
Variant (--exit-hour 10): jual open 10:xx T+1 → close10 label

Label: overnight_return = (open@{EXIT_HOUR}:xx_T+1 - close@15:xx_T) / close@15:xx_T
Positive trade: overnight_return > roundtrip_cost (0.4%)
SL classification: overnight_return < -2%

No lookahead rules:
- Entry price = close candle 15:xx hari T (available saat entry)
- Exit price = open candle {EXIT_HOUR}:xx hari T+1 (LABEL ONLY, tidak boleh jadi feature)
- Broksum features pakai shift(1) / T-1 untuk aman
- OHLCV features pakai data s.d. close hari T

Output (default):
  --exit-hour 9  → idx/data/Level_2_Datamart/training_datamart_bsjp_overnight.parquet
  --exit-hour 10 → idx/data/Level_2_Datamart/training_datamart_bsjp_close10.parquet
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
)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
# edges/bsjp_overnight_sl2/scripts/<file>.py → parents[3] = idx/
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
DEFAULT_TRAIN_OUTPUT   = DATAMART_DIR / "training_datamart_bsjp_overnight.parquet"  # exit-hour 9
DEFAULT_TRAIN_OUTPUT_CLOSE10 = DATAMART_DIR / "training_datamart_bsjp_close10.parquet"  # exit-hour 10

# Modular feature loading: each feature group written to separate parquet
MODULES_DIR = FEATURES_DIR / "modules"

STOCKBIT_BROKER_CODE = "XL"
ROUNDTRIP_COST = 0.004   # 0.4%
SL_THRESHOLD   = -0.02   # -2%
ENTRY_HOUR     = 15      # close candle jam 15:xx sebagai entry price proxy
EXIT_HOUR      = 10      # open candle jam 10:xx T+1 sebagai exit price proxy (default)

# Universe filter defaults (Tier 1 — minimize delisting risk)
UNIV_MIN_PRICE           = 50.0           # exclude gocap (< Rp 50)
UNIV_MIN_TURNOVER_IDR    = 500_000_000.0  # exclude illiquid (< Rp 500M median 20d)
UNIV_ARA_RETURN_THRESH   = 0.20           # exclude saham yang close T gap-up >= 20% (proxy ARA)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build BSJP overnight training datamart.")
    parser.add_argument("--yf-1h-path",          type=Path, default=DEFAULT_YF_1H)
    parser.add_argument("--yf-daily-path",        type=Path, default=DEFAULT_YF_DAILY)
    parser.add_argument("--features-path",        type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--broksum-raw-path",     type=Path, default=DEFAULT_BROKSUM_RAW)
    parser.add_argument("--master-broker-path",   type=Path, default=DEFAULT_MASTER_BROKER)
    parser.add_argument("--global-indices-path",  type=Path, default=DEFAULT_GLOBAL_INDICES)
    parser.add_argument("--vwap-features-path",   type=Path, default=DEFAULT_VWAP_FEATURES)
    parser.add_argument("--focus-broker",         type=str,  default="MG")
    parser.add_argument("--bandar-localfund-min", type=float, default=60.0)
    parser.add_argument("--training-output",      type=Path, default=None,
                        help="Output path for training datamart parquet. "
                             "Defaults to training_datamart_bsjp_overnight.parquet (exit-hour 9) "
                             "or training_datamart_bsjp_close10.parquet (exit-hour 10).")
    parser.add_argument("--min-history-days",     type=int,  default=120)
    parser.add_argument("--universe-min-price",   type=float, default=UNIV_MIN_PRICE,
                        help="Exclude stocks with close_T < this price (delisting risk)")
    parser.add_argument("--universe-min-turnover", type=float, default=UNIV_MIN_TURNOVER_IDR,
                        help="Exclude stocks with median 20d turnover < this IDR amount")
    parser.add_argument("--universe-ara-threshold", type=float, default=UNIV_ARA_RETURN_THRESH,
                        help="Exclude stocks with day_return T >= this (ARA proxy)")
    parser.add_argument("--enable-universe-filter", action="store_true",
                        help="Apply universe filter (gocap/illiquid/ARA exclusion) — OFF by default")
    parser.add_argument("--exit-hour",             type=int,  default=EXIT_HOUR,
                        help="Hour of T+1 candle used as exit price (open). "
                             "9 = overnight (default), 10 = close10 variant.")
    parser.add_argument("--dry-run",              action="store_true")
    parser.add_argument(
        "--date-from", type=str, default=None,
        help="Incremental mode: only output rows >= this date (YYYY-MM-DD). "
             "Inputs are filtered with a warmup window for rolling features. "
             "If output parquet exists, existing rows before this date are preserved.",
    )
    parser.add_argument(
        "--warmup-calendar-days", type=int, default=120,
        help="Calendar days of yf_1h data before --date-from for overnight/label rolling features (default: 120).",
    )
    parser.add_argument(
        "--features-warmup-calendar-days", type=int, default=120,
        help="Calendar days of L1 features/broksum before --date-from for broker rolling features (default: 120).",
    )
    parser.add_argument(
        "--modules-dir", type=Path, default=MODULES_DIR,
        help="Directory to write modular feature parquets (default: data/Level_1_Features/modules/). "
             "Set to empty path to disable module extraction.",
    )
    return parser


# ---------------------------------------------------------------------------
# Label builder
# ---------------------------------------------------------------------------
def build_label(yf_1h: pd.DataFrame, exit_hour: int = EXIT_HOUR) -> pd.DataFrame:
    """
    Build label table.

    Entry: close of hour-15 candle on day T (proxy for ~15:30-15:45 entry)
    Exit:  open of hour-{exit_hour} candle on day T+1
           exit_hour=9  → overnight (09:05 exit)
           exit_hour=10 → close10 (~10:00 exit, recommended — 91.6% hour-10 coverage vs 25.6% hour-9)

    Returns grain: trade_date (= T), ticker
    """
    df = yf_1h.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["ticker"]   = df["ticker"].str.replace(r"\.JK$", "", regex=True)
    # Convert to Jakarta time for correct hour extraction
    df["dt_wib"]   = df["datetime"].dt.tz_convert("Asia/Jakarta") if df["datetime"].dt.tz is not None else df["datetime"]
    df["date"]     = df["dt_wib"].dt.normalize().dt.tz_localize(None).astype("datetime64[ns]")
    df["hour"]     = df["dt_wib"].dt.hour

    # Entry price: close of 15:xx candle
    entry = (
        df[df["hour"] == ENTRY_HOUR]
        .groupby(["date", "ticker"], sort=False)["close"]
        .last()
        .reset_index()
        .rename(columns={"close": "entry_price", "date": "trade_date"})
    )

    # Exit price: open of {exit_hour}:xx candle T+1
    exit_ = (
        df[df["hour"] == exit_hour]
        .groupby(["date", "ticker"], sort=False)["open"]
        .first()
        .reset_index()
        .rename(columns={"open": "exit_price", "date": "exit_date"})
    )

    # Align T+1: sort per ticker, shift exit by -1 trading day
    exit_ = exit_.sort_values(["ticker", "exit_date"])
    exit_["trade_date"] = exit_.groupby("ticker")["exit_date"].shift(1)
    exit_ = exit_.dropna(subset=["trade_date"])
    exit_["trade_date"] = exit_["trade_date"].dt.tz_localize(None).astype("datetime64[ns]")

    labels = entry.merge(exit_[["trade_date", "ticker", "exit_price", "exit_date"]], on=["trade_date", "ticker"], how="inner")
    labels = labels.dropna(subset=["entry_price", "exit_price"])
    labels = labels[labels["entry_price"] > 0]

    labels["overnight_return"] = (labels["exit_price"] - labels["entry_price"]) / labels["entry_price"]

    # Guard: shift(1) can jump to a distant bar when T+1 has no {exit_hour} candle,
    # or a stock split creates a ~2x overnight gap.
    # Both produce fake extreme returns that distort model training.
    # Filter strategy:
    #   - gap > 5d & |ret| > 15%  → shift misalignment (e.g. CUAN 42d → +1292%)
    #   - |ret| > 50%              → stock split / data artifact (e.g. BBNI +100%)
    #     (IDX daily ARA max is 35%, so >50% is never a real trading return)
    labels["_gap_days"] = (labels["exit_date"] - labels["trade_date"]).dt.days
    labels = labels[
        ~(
            ((labels["_gap_days"] > 3) & (labels["overnight_return"].abs() > 0.15))
            | (labels["overnight_return"].abs() > 0.50)
        )
    ].copy()
    labels = labels.drop(columns=["_gap_days"])
    labels["label_tp"]  = (labels["overnight_return"] > ROUNDTRIP_COST).astype("int8")
    labels["label_sl2"] = (labels["overnight_return"] < SL_THRESHOLD).astype("int8")
    label_name = "bsjp_overnight_sl2" if exit_hour == 9 else f"bsjp_close{exit_hour}_sl2"
    labels["label_name"] = label_name

    labels = labels.rename(columns={"trade_date": "date"})
    return labels[["date", "ticker", "entry_price", "exit_price",
                   "overnight_return", "label_tp", "label_sl2", "label_name"]]


# ---------------------------------------------------------------------------
# Universe filter (delisting risk mitigation)
# ---------------------------------------------------------------------------
def build_universe_filter(
    yf_1h: pd.DataFrame,
    min_price: float,
    min_turnover_idr: float,
    ara_threshold: float,
) -> pd.DataFrame:
    """
    Tier 1 universe filter untuk minimize delisting / suspension tail risk.

    Exclusion flags per (date, ticker):
    - is_gocap: close_T < min_price (saham gocap — highest delisting risk)
    - is_illiquid: median turnover 20d < min_turnover_idr
    - is_ara: day_return T >= ara_threshold (proxy ARA — often pump/manipulation)

    Returns DataFrame with columns: date, ticker, close_T, day_return, turnover_med20,
    is_gocap, is_illiquid, is_ara, exclude
    """
    df = yf_1h.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["ticker"]   = df["ticker"].str.replace(r"\.JK$", "", regex=True)
    # Convert to Jakarta time for correct hour extraction
    df["dt_wib"]   = df["datetime"].dt.tz_convert("Asia/Jakarta") if df["datetime"].dt.tz is not None else df["datetime"]
    df["date"]     = df["dt_wib"].dt.normalize().dt.tz_localize(None).astype("datetime64[ns]")
    df["hour"]     = df["dt_wib"].dt.hour

    close_15 = (
        df[df["hour"] == 15].groupby(["date", "ticker"])["close"].last()
        .reset_index().rename(columns={"close": "close_T"})
    )
    vol_day = (
        df[df["hour"].between(9, 15)].groupby(["date", "ticker"])["volume"].sum()
        .reset_index().rename(columns={"volume": "day_volume"})
    )
    daily = close_15.merge(vol_day, on=["date", "ticker"], how="left")
    daily["turnover"] = daily["close_T"] * daily["day_volume"]
    daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)

    grp = daily.groupby("ticker", sort=False)
    daily["prev_close"]     = grp["close_T"].shift(1)
    daily["day_return"]     = (daily["close_T"] - daily["prev_close"]) / daily["prev_close"].replace(0, np.nan)
    daily["turnover_med20"] = grp["turnover"].transform(lambda s: s.rolling(20, min_periods=5).median())

    daily["is_gocap"]    = (daily["close_T"] < min_price).fillna(False)
    daily["is_illiquid"] = (daily["turnover_med20"].isna()) | (daily["turnover_med20"] < min_turnover_idr)
    daily["is_ara"]      = (daily["day_return"] >= ara_threshold).fillna(False)
    daily["exclude"]     = daily["is_gocap"] | daily["is_illiquid"] | daily["is_ara"]

    return daily[["date", "ticker", "close_T", "day_return", "turnover_med20",
                  "is_gocap", "is_illiquid", "is_ara", "exclude"]]


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------
def build_closing_momentum(yf_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Closing session momentum features (available at entry ~15:30):
    - close_ret_last1h: return of last 1h candle (14:xx → 15:xx)
    - close_vs_open_day: intraday return dari open 09:xx ke close 15:xx
    - close_range_pct: (high - low) / open on day T (volatility hari ini)
    """
    df = yf_1h.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["ticker"]   = df["ticker"].str.replace(r"\.JK$", "", regex=True)
    # Convert to Jakarta time for correct hour extraction
    df["dt_wib"]   = df["datetime"].dt.tz_convert("Asia/Jakarta") if df["datetime"].dt.tz is not None else df["datetime"]
    df["date"]     = df["dt_wib"].dt.normalize().dt.tz_localize(None).astype("datetime64[ns]")
    df["hour"]     = df["dt_wib"].dt.hour

    open_price = (
        df[df["hour"] == 9].groupby(["date", "ticker"])["open"].first().reset_index()
        .rename(columns={"open": "day_open"})
    )
    close_14 = (
        df[df["hour"] == 14].groupby(["date", "ticker"])["close"].last().reset_index()
        .rename(columns={"close": "close_14"})
    )
    close_15 = (
        df[df["hour"] == 15].groupby(["date", "ticker"])["close"].last().reset_index()
        .rename(columns={"close": "close_15"})
    )
    high_day = (
        df[df["hour"].between(9, 15)].groupby(["date", "ticker"])["high"].max().reset_index()
        .rename(columns={"high": "day_high"})
    )
    low_day = (
        df[df["hour"].between(9, 15)].groupby(["date", "ticker"])["low"].min().reset_index()
        .rename(columns={"low": "day_low"})
    )

    out = open_price.merge(close_14, on=["date", "ticker"], how="left")
    out = out.merge(close_15, on=["date", "ticker"], how="left")
    out = out.merge(high_day, on=["date", "ticker"], how="left")
    out = out.merge(low_day,  on=["date", "ticker"], how="left")

    out["close_ret_last1h"]  = (out["close_15"] - out["close_14"]) / out["close_14"].replace(0, np.nan)
    out["close_vs_open_day"] = (out["close_15"] - out["day_open"]) / out["day_open"].replace(0, np.nan)
    out["close_range_pct"]   = (out["day_high"] - out["day_low"]) / out["day_open"].replace(0, np.nan)

    feat_cols = ["date", "ticker", "close_ret_last1h", "close_vs_open_day", "close_range_pct"]
    out = out[feat_cols].copy()
    for c in feat_cols[2:]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    return out


def build_overnight_history(yf_1h: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """
    Historical overnight gap patterns per ticker (no lookahead via shift(1)):
    - overnight_ret_ma{N}: rolling mean overnight return
    - overnight_positive_rate{N}: % of nights with positive return
    - gap_*: gap character family (prefix-consistent features)
      using previous close as anchor (e.g. high >= prev_close * 1.02)
    """
    if windows is None:
        windows = [5, 20, 60]

    df = yf_1h.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["ticker"]   = df["ticker"].str.replace(r"\.JK$", "", regex=True)
    # Convert to Jakarta time for correct hour extraction
    df["dt_wib"]   = df["datetime"].dt.tz_convert("Asia/Jakarta") if df["datetime"].dt.tz is not None else df["datetime"]
    df["date"]     = df["dt_wib"].dt.normalize().dt.tz_localize(None).astype("datetime64[ns]")
    df["hour"]     = df["dt_wib"].dt.hour

    close_15 = (
        df[df["hour"] == 15].groupby(["date", "ticker"])["close"].last().reset_index()
        .rename(columns={"close": "close_15"})
    )
    high_day = (
        df[df["hour"].between(9, 15)].groupby(["date", "ticker"])["high"].max().reset_index()
        .rename(columns={"high": "day_high"})
    )
    low_day = (
        df[df["hour"].between(9, 15)].groupby(["date", "ticker"])["low"].min().reset_index()
        .rename(columns={"low": "day_low"})
    )
    open_9 = (
        df[df["hour"] == 9].groupby(["date", "ticker"])["open"].first().reset_index()
        .rename(columns={"open": "open_9"})
    )

    merged = close_15.merge(open_9, on=["date", "ticker"], how="inner")
    merged = merged.merge(high_day, on=["date", "ticker"], how="left")
    merged = merged.merge(low_day, on=["date", "ticker"], how="left")
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    # prev_close per ticker (not global shift — avoids cross-ticker contamination)
    prev_close = merged.groupby("ticker", sort=False)["close_15"].shift(1).replace(0, np.nan)
    merged["overnight_ret"]          = (merged["open_9"]    - prev_close) / prev_close
    merged["day_high_vs_prev_close"] = (merged["day_high"]  - prev_close) / prev_close
    merged["day_low_vs_prev_close"]  = (merged["day_low"]   - prev_close) / prev_close
    merged["day_close_vs_prev_close"]= (merged["close_15"]  - prev_close) / prev_close
    # Align: overnight_ret on date D = return from close D-1 to open D
    # For our label we need: at entry on day T, history of past overnight returns
    # So shift(1) again to avoid lookahead
    grp = merged.groupby("ticker", sort=False)

    feat_frames = []
    for ticker, g in grp:
        g = g.copy()
        shifted = g["overnight_ret"].shift(1)
        shifted_high = g["day_high_vs_prev_close"].shift(1)
        shifted_low = g["day_low_vs_prev_close"].shift(1)
        shifted_close = g["day_close_vs_prev_close"].shift(1)
        for w in windows:
            mp = max(3, w // 4)
            roll = shifted.rolling(w, min_periods=mp)
            g[f"overnight_ret_ma{w}"]           = roll.mean()
            g[f"overnight_positive_rate{w}"]     = (shifted > 0).rolling(w, min_periods=mp).mean()
            # Gap down profile per saham
            g[f"gapdown_freq_{w}d"]              = (shifted < -0.02).rolling(w, min_periods=mp).mean()
            g[f"gapdown_severe_freq_{w}d"]       = (shifted < -0.05).rolling(w, min_periods=mp).mean()
            g[f"overnight_worst_{w}d"]           = roll.min()
            g[f"overnight_p10_{w}d"]             = roll.quantile(0.10)

            # Gap character family (prefix gap_):
            # - gap_up2_freq: how often day_high >= prev_close * 1.02
            # - gap_down2_freq: how often day_low <= prev_close * 0.98
            # - gap_up2_down2_edge: asymmetry of up-vs-down expansion
            # - gap_up2_followthrough_freq: up2 days that still close green vs prev close
            up2 = (shifted_high >= 0.02).astype("float32")
            down2 = (shifted_low <= -0.02).astype("float32")
            up2_follow = ((shifted_high >= 0.02) & (shifted_close > 0)).astype("float32")
            g[f"gap_up2_freq_{w}d"] = up2.rolling(w, min_periods=mp).mean()
            g[f"gap_down2_freq_{w}d"] = down2.rolling(w, min_periods=mp).mean()
            g[f"gap_up2_down2_edge_{w}d"] = g[f"gap_up2_freq_{w}d"] - g[f"gap_down2_freq_{w}d"]
            g[f"gap_up2_followthrough_freq_{w}d"] = up2_follow.rolling(w, min_periods=mp).mean()
        feat_frames.append(g)

    out = pd.concat(feat_frames, ignore_index=True)
    feat_cols = (
        ["date", "ticker"]
        + [f"overnight_ret_ma{w}" for w in windows]
        + [f"overnight_positive_rate{w}" for w in windows]
        + [f"gapdown_freq_{w}d" for w in windows]
        + [f"gapdown_severe_freq_{w}d" for w in windows]
        + [f"gap_up2_freq_{w}d" for w in windows]
        + [f"gap_down2_freq_{w}d" for w in windows]
        + [f"gap_up2_down2_edge_{w}d" for w in windows]
        + [f"gap_up2_followthrough_freq_{w}d" for w in windows]
        + [f"overnight_worst_{w}d" for w in windows]
        + [f"overnight_p10_{w}d" for w in windows]
    )
    out = out[feat_cols].copy()
    for c in feat_cols[2:]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    return out


def build_cvd_features(features: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """
    Cumulative Volume Delta (CVD) per ticker dari L1 broker data.
    CVD = rolling sum of net_volume (buy_vol - sell_vol) across ALL brokers.

    Columns: cvd_{N}d, cvd_{N}d_norm (per window N)
    Semua pakai shift(1) implisit karena L1 features sudah T-1 shifted.
    """
    if windows is None:
        windows = [5, 10, 20]

    required = {"date", "ticker", "flow_net_volume"}
    empty_cols = ["date", "ticker"] + [f"cvd_{w}d" for w in windows] + [f"cvd_{w}d_norm" for w in windows]
    if features.empty or not required.issubset(features.columns):
        return pd.DataFrame(columns=empty_cols)

    has_vol = "flow_buy_volume" in features.columns and "flow_sell_volume" in features.columns
    cols = ["date", "ticker", "flow_net_volume"]
    if has_vol:
        cols += ["flow_buy_volume", "flow_sell_volume"]
    df = features[cols].copy()

    df["date"]            = pd.to_datetime(df["date"], errors="coerce").dt.normalize().dt.tz_localize(None).astype("datetime64[ns]")
    df["flow_net_volume"] = pd.to_numeric(df["flow_net_volume"], errors="coerce").fillna(0)
    if has_vol:
        df["flow_buy_volume"]  = pd.to_numeric(df["flow_buy_volume"],  errors="coerce").fillna(0)
        df["flow_sell_volume"] = pd.to_numeric(df["flow_sell_volume"], errors="coerce").fillna(0)

    agg = {"flow_net_volume": "sum"}
    if has_vol:
        agg["flow_buy_volume"] = "sum"
        agg["flow_sell_volume"] = "sum"
    daily = df.groupby(["date", "ticker"]).agg(agg).reset_index()
    daily["total_volume"] = (daily["flow_buy_volume"] + daily["flow_sell_volume"]) if has_vol \
        else daily["flow_net_volume"].abs() * 2

    out_frames = []
    for ticker, grp in daily.groupby("ticker"):
        grp = grp.sort_values("date").copy()
        res = grp[["date"]].copy()
        res["ticker"] = ticker
        for w in windows:
            cvd       = grp["flow_net_volume"].rolling(w, min_periods=1).sum()
            total_vol = grp["total_volume"].rolling(w, min_periods=1).sum()
            res[f"cvd_{w}d"]      = cvd.values.astype("float32")
            res[f"cvd_{w}d_norm"] = np.where(total_vol > 0, cvd / total_vol, np.nan).astype("float32")
        out_frames.append(res)

    if not out_frames:
        return pd.DataFrame(columns=empty_cols)
    return pd.concat(out_frames, ignore_index=True)


def build_stockbit_features(broksum_raw: pd.DataFrame, broker_code: str = "XL") -> pd.DataFrame:
    """
    Stockbit (XL) broker activity — sama dengan BPJS tapi untuk BSJP
    same-day activity T-1 (shifted) adalah safe untuk entry di 15:30 T.

    Kolom: xl_buy_freq_ma5, xl_sell_freq_ma5, xl_buy_freq_ma20, xl_freq_surge
    """
    required = {"broker", "stock_code", "date", "buy_freq", "sell_freq"}
    if broksum_raw.empty or not required.issubset(broksum_raw.columns):
        return pd.DataFrame(columns=["date", "ticker",
                                     "xl_buy_freq_ma5", "xl_sell_freq_ma5",
                                     "xl_buy_freq_ma20", "xl_freq_surge"])

    df = broksum_raw[broksum_raw["broker"] == broker_code].copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "ticker",
                                     "xl_buy_freq_ma5", "xl_sell_freq_ma5",
                                     "xl_buy_freq_ma20", "xl_freq_surge"])

    df["date"]       = pd.to_datetime(df["date"]).dt.tz_localize(None).astype("datetime64[ns]")
    df["buy_freq"]   = pd.to_numeric(df["buy_freq"],  errors="coerce").fillna(0)
    df["sell_freq"]  = pd.to_numeric(df["sell_freq"], errors="coerce").fillna(0)
    df["total_freq"] = df["buy_freq"] + df["sell_freq"]
    df = df.sort_values(["stock_code", "date"])

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
    for c in ["xl_buy_freq_ma5", "xl_sell_freq_ma5", "xl_buy_freq_ma20", "xl_freq_surge"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Resolve default output path based on exit_hour
    if args.training_output is None:
        if args.exit_hour == 9:
            args.training_output = DEFAULT_TRAIN_OUTPUT
        else:
            args.training_output = DATAMART_DIR / f"training_datamart_bsjp_close{args.exit_hour}.parquet"

    print(f"[Init] yf_1h={args.yf_1h_path}")
    print(f"[Init] features={args.features_path}")
    print(f"[Init] exit_hour={args.exit_hour}")
    print(f"[Init] output={args.training_output}")

    # Modules dir: write each feature group to separate parquet for modular loading
    modules_enabled = bool(args.modules_dir) and args.modules_dir != Path(".")
    if modules_enabled:
        args.modules_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Modules] writing feature modules to {args.modules_dir}")

    # Incremental mode: filter inputs to warmup window, only output rows >= date_from
    date_from_ts: pd.Timestamp | None = None
    if args.date_from:
        date_from_ts = pd.Timestamp(args.date_from)

    yf_1h = pd.read_parquet(args.yf_1h_path)
    features_warmup_cutoff: pd.Timestamp | None = None
    if date_from_ts is not None:
        warmup_cutoff = date_from_ts - pd.Timedelta(days=args.warmup_calendar_days)
        if yf_1h["datetime"].dt.tz is not None:
            warmup_cutoff = warmup_cutoff.tz_localize("UTC")
        yf_1h = yf_1h[yf_1h["datetime"] >= warmup_cutoff].copy()
        print(f"[IncrementalMode] yf_warmup={warmup_cutoff.date()}")
        
        features_warmup_cutoff = date_from_ts - pd.Timedelta(days=args.features_warmup_calendar_days)
    print(f"[Load] yf_1h_rows={len(yf_1h):,}")

    # --- Label ---
    labels = build_label(yf_1h, exit_hour=args.exit_hour)
    tp_rate = labels["label_tp"].mean()
    print(f"[Label] rows={len(labels):,}, date={labels['date'].min().date()} -> {labels['date'].max().date()}, tp_rate={tp_rate:.2%}")

    # --- Universe filter (Tier 1: minimize delisting risk) ---
    if args.enable_universe_filter:
        filt = build_universe_filter(
            yf_1h,
            min_price=args.universe_min_price,
            min_turnover_idr=args.universe_min_turnover,
            ara_threshold=args.universe_ara_threshold,
        )
        before_n = len(labels)
        labels = labels.merge(filt[["date", "ticker", "is_gocap", "is_illiquid", "is_ara", "exclude"]],
                              on=["date", "ticker"], how="left")
        n_gocap    = int(labels["is_gocap"].fillna(False).sum())
        n_illiq    = int(labels["is_illiquid"].fillna(False).sum())
        n_ara      = int(labels["is_ara"].fillna(False).sum())
        n_excluded = int(labels["exclude"].fillna(True).sum())
        labels = labels[~labels["exclude"].fillna(True)].copy()
        labels = labels.drop(columns=["is_gocap", "is_illiquid", "is_ara", "exclude"])
        tp_rate_after = labels["label_tp"].mean()
        print(f"[UniverseFilter] before={before_n:,}, after={len(labels):,}, excluded={before_n-len(labels):,} "
              f"(gocap={n_gocap:,}, illiquid={n_illiq:,}, ara={n_ara:,}), tp_rate={tp_rate_after:.2%}")
    else:
        print("[UniverseFilter] DISABLED")

    # --- Closing momentum features ---
    closing_mom = build_closing_momentum(yf_1h)
    train = labels.merge(closing_mom, on=["date", "ticker"], how="left")
    print(f"[ClosingMomentum] merged close_ret_last1h, close_vs_open_day, close_range_pct")
    if modules_enabled:
        closing_mom.to_parquet(args.modules_dir / "closing_momentum_features.parquet", index=False)
        print(f"[Modules] wrote closing_momentum_features.parquet ({len(closing_mom):,} rows)")

    # --- Overnight history features ---
    overnight_hist = build_overnight_history(yf_1h)
    train = train.merge(overnight_hist, on=["date", "ticker"], how="left")
    print(
        "[OvernightHistory] merged overnight_ret_ma, overnight_positive_rate, "
        "gapdown_freq, gapdown_severe_freq, gap_up2_freq, gap_down2_freq, "
        "gap_up2_down2_edge, gap_up2_followthrough_freq, overnight_worst, "
        "overnight_p10 (windows 5/20/60)"
    )
    if modules_enabled:
        overnight_hist.to_parquet(args.modules_dir / "overnight_history_features.parquet", index=False)
        print(f"[Modules] wrote overnight_history_features.parquet ({len(overnight_hist):,} rows)")

    # --- Stockbit (XL) features ---
    broksum_raw = pd.DataFrame()
    if args.broksum_raw_path.exists():
        broksum_raw = pd.read_parquet(args.broksum_raw_path)
        if features_warmup_cutoff is not None:
            broksum_raw = broksum_raw[pd.to_datetime(broksum_raw["date"]) >= features_warmup_cutoff].copy()
    xl_feats = build_stockbit_features(broksum_raw, broker_code=STOCKBIT_BROKER_CODE)
    if not xl_feats.empty:
        train = train.merge(xl_feats, on=["date", "ticker"], how="left")
        print(f"[Stockbit/XL] merged xl_buy_freq_ma5, xl_sell_freq_ma5, xl_buy_freq_ma20, xl_freq_surge")
        if modules_enabled:
            xl_feats.to_parquet(args.modules_dir / "stockbit_xl_features.parquet", index=False)
            print(f"[Modules] wrote stockbit_xl_features.parquet ({len(xl_feats):,} rows)")

    # --- Broker aggregate features + CVD (L1 broksum, T-1 safe) ---
    features = pd.DataFrame()
    if args.features_path.exists():
        read_filters = None
        if features_warmup_cutoff is not None:
            read_filters = [("date", ">=", features_warmup_cutoff)]
        features = pd.read_parquet(args.features_path, filters=read_filters)
        features["date"] = pd.to_datetime(features["date"]).dt.normalize().dt.tz_localize(None).astype("datetime64[ns]")
        print(f"[Load] features_rows={len(features):,}")

        local_fund_brokers, bandar_brokers = load_master_broker(
            args.master_broker_path,
            bandar_localfund_min=args.bandar_localfund_min,
        )
        feat_agg = build_feature_aggregate(
            features=features,
            focus_broker=args.focus_broker,
            local_fund_brokers=local_fund_brokers,
            bandar_brokers=bandar_brokers,
        )
        # feat_agg grain: date/ticker — merge to train
        train = train.merge(feat_agg, on=["date", "ticker"], how="left")
        print(f"[BrokerAgg] merged {len(feat_agg.columns)-2} broker features")
        if modules_enabled:
            feat_agg.to_parquet(args.modules_dir / "broker_aggregate_features.parquet", index=False)
            print(f"[Modules] wrote broker_aggregate_features.parquet ({len(feat_agg):,} rows)")

    # --- CVD (dari L1 broksum, T-1 safe) ---
    cvd_feats = build_cvd_features(features)
    if not cvd_feats.empty:
        train = train.merge(cvd_feats, on=["date", "ticker"], how="left")
        cvd_cols = [c for c in cvd_feats.columns if c not in ("date", "ticker")]
        print(f"[CVD] merged {', '.join(cvd_cols)}")
        if modules_enabled:
            cvd_feats.to_parquet(args.modules_dir / "cvd_features.parquet", index=False)
            print(f"[Modules] wrote cvd_features.parquet ({len(cvd_feats):,} rows)")

    # --- Global indices (US close tersedia sore hari T — extra relevan untuk BSJP) ---
    global_indices = load_global_indices(args.global_indices_path)
    if not global_indices.empty:
        global_indices["date"] = global_indices["date"].dt.tz_localize(None).astype("datetime64[ns]")
        train = train.merge(global_indices, on="date", how="left")
        gi_cols = [c for c in global_indices.columns if c != "date"]
        print(f"[GlobalIndex] merged {', '.join(gi_cols)}")
        if modules_enabled:
            global_indices.to_parquet(args.modules_dir / "global_indices_features.parquet", index=False)
            print(f"[Modules] wrote global_indices_features.parquet ({len(global_indices):,} rows)")

    # --- VWAP features (optional, from Level_1_Features/vwap_features.parquet) ---
    if args.vwap_features_path.exists():
        vwap = pd.read_parquet(args.vwap_features_path)
        vwap["date"] = pd.to_datetime(vwap["date"]).dt.normalize()
        train = train.merge(vwap, on=["date", "ticker"], how="left")
        vwap_cols = [c for c in vwap.columns if c not in ("date", "ticker")]
        print(f"[VWAP] merged {', '.join(vwap_cols)}")
        if modules_enabled:
            vwap.to_parquet(args.modules_dir / "vwap_features.parquet", index=False)
            print(f"[Modules] wrote vwap_features.parquet ({len(vwap):,} rows)")
    else:
        print(f"[VWAP] skipped (not found: {args.vwap_features_path.name})")

    print(f"[Training] rows={len(train):,}, cols={len(train.columns)}, "
          f"date={train['date'].min().date()} -> {train['date'].max().date()}, "
          f"tp_rate={train['label_tp'].mean():.2%}")

    if args.dry_run:
        print("[DryRun] Skip writing outputs.")
        return

    args.training_output.parent.mkdir(parents=True, exist_ok=True)

    if date_from_ts is not None:
        # Incremental write: keep existing history, replace rows >= date_from
        train = train[train["date"] >= date_from_ts].copy()
        if args.training_output.exists():
            existing = pd.read_parquet(args.training_output)
            existing["date"] = pd.to_datetime(existing["date"])
            existing = existing[existing["date"] < date_from_ts]
            train = pd.concat([existing, train], ignore_index=True)
            print(f"[IncrementalWrite] kept {len(existing):,} existing rows, appended {len(train)-len(existing):,} new rows")

    train.to_parquet(args.training_output, index=False)
    print(f"[Done] training_output={args.training_output}")


if __name__ == "__main__":
    main()
