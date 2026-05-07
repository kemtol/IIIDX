#!/usr/bin/env python3
"""
Generate preclose14 feature module for BSJP.

These features use only 1h bars available after the hour-14 candle closes
(roughly 14:59 WIB). They replace close15/EOD features that are not available
before a realistic near-close order decision.

Output:
  data/Level_1_Features/modules/preclose14_features.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


IDX_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = IDX_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "Level_0_Raw"
MODULES_DIR = DATA_DIR / "Level_1_Features" / "modules"

DEFAULT_YF_1H = RAW_DATA_DIR / "yfinance_1h.parquet"
DEFAULT_OUTPUT = MODULES_DIR / "preclose14_features.parquet"

ROUNDTRIP_COST_FRAC = 0.004
MORNING_HOURS = {9, 10, 11}
PRECLOSE_HOURS = {9, 10, 11, 13, 14}
PRE_14_AVG_HOURS = {9, 10, 11, 13}
ARA_BUFFER_PCT = 0.005


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


def idx_ara_limit_pct(reference_price: float) -> float:
    if reference_price <= 0 or pd.isna(reference_price):
        return np.nan
    if reference_price <= 200:
        return 0.35
    if reference_price <= 5000:
        return 0.25
    return 0.20


def build_ara_state_features(bars: pd.DataFrame, prev_close: pd.DataFrame) -> pd.DataFrame:
    state = bars.merge(prev_close, on=["date", "ticker"], how="left")
    state["pre14_ara_limit_pct"] = state["pre14_prev_close"].apply(idx_ara_limit_pct)
    state["pre14_tick_size"] = state["pre14_prev_close"].apply(idx_tick_size)
    state["pre14_ara_price"] = state["pre14_prev_close"] * (1.0 + state["pre14_ara_limit_pct"])

    valid = state["pre14_ara_price"].notna() & (state["pre14_ara_price"] > 0)
    touched = valid & (state["high"] >= (state["pre14_ara_price"] - state["pre14_tick_size"]))
    release = touched & (state["low"] < (state["pre14_ara_price"] - state["pre14_tick_size"]))
    close_at_ara = valid & (state["close"] >= (state["pre14_ara_price"] - state["pre14_tick_size"]))
    flat_locked = (
        touched
        & (state["open"] >= (state["pre14_ara_price"] - state["pre14_tick_size"]))
        & (state["low"] >= (state["pre14_ara_price"] - state["pre14_tick_size"]))
        & (state["close"] >= (state["pre14_ara_price"] - state["pre14_tick_size"]))
    )
    release_depth = np.where(
        release,
        (state["pre14_ara_price"] - state["low"]) / state["pre14_ara_price"].replace(0, np.nan),
        0.0,
    )

    state = state.assign(
        ara_bar_touched=touched.astype("float32"),
        ara_bar_release=release.astype("float32"),
        ara_bar_close_at_ara=close_at_ara.astype("float32"),
        ara_bar_flat_locked=flat_locked.astype("float32"),
        ara_bar_release_depth=release_depth,
    )
    touch_hour = (
        state[state["ara_bar_touched"] > 0]
        .groupby(["date", "ticker"], sort=False)["hour"]
        .min()
        .rename("pre14_ara_touch_hour")
        .reset_index()
    )
    agg = (
        state.groupby(["date", "ticker"], sort=False)
        .agg(
            pre14_ara_price=("pre14_ara_price", "first"),
            pre14_ara_touched=("ara_bar_touched", "max"),
            pre14_ara_release_wick_count=("ara_bar_release", "sum"),
            pre14_ara_close_at_ara_count=("ara_bar_close_at_ara", "sum"),
            pre14_ara_flat_ohlc_count=("ara_bar_flat_locked", "sum"),
            pre14_ara_release_wick_depth_max=("ara_bar_release_depth", "max"),
            pre14_ara_release_wick_depth_mean=("ara_bar_release_depth", "mean"),
        )
        .reset_index()
    )
    agg = agg.merge(touch_hour, on=["date", "ticker"], how="left")

    touched_count = (
        state[state["ara_bar_touched"] > 0]
        .groupby(["date", "ticker"], sort=False)
        .size()
        .rename("pre14_ara_touched_bar_count")
        .reset_index()
    )
    last_bar = (
        state.sort_values(["date", "ticker", "datetime"])
        .groupby(["date", "ticker"], sort=False)
        .tail(1)[
            [
                "date",
                "ticker",
                "ara_bar_touched",
                "ara_bar_release",
                "ara_bar_close_at_ara",
                "ara_bar_flat_locked",
                "ara_bar_release_depth",
            ]
        ]
        .rename(
            columns={
                "ara_bar_touched": "pre14_ara_last_bar_touched",
                "ara_bar_release": "pre14_ara_last_bar_release",
                "ara_bar_close_at_ara": "pre14_ara_last_bar_close_at_ara",
                "ara_bar_flat_locked": "pre14_ara_last_bar_locked",
                "ara_bar_release_depth": "pre14_ara_last_bar_release_depth",
            }
        )
    )
    agg = agg.merge(touched_count, on=["date", "ticker"], how="left")
    agg = agg.merge(last_bar, on=["date", "ticker"], how="left")
    agg["pre14_ara_touched_bar_count"] = agg["pre14_ara_touched_bar_count"].fillna(0.0)
    agg["pre14_ara_release_wick_ratio"] = (
        agg["pre14_ara_release_wick_count"]
        / agg["pre14_ara_touched_bar_count"].replace(0, np.nan)
    )
    agg["pre14_ara_locked_proxy"] = (
        (agg["pre14_ara_touched"] > 0)
        & (agg["pre14_ara_release_wick_count"] <= 0)
        & (agg["pre14_ara_last_bar_close_at_ara"] > 0)
    ).astype("float32")
    agg["pre14_ara_touched_released"] = (
        (agg["pre14_ara_touched"] > 0) & (agg["pre14_ara_release_wick_count"] > 0)
    ).astype("float32")
    return agg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate BSJP preclose14 feature module.")
    p.add_argument("--yf-1h-path", type=Path, default=DEFAULT_YF_1H)
    p.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_preclose14_features(yf_1h: pd.DataFrame) -> pd.DataFrame:
    df = yf_1h.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "ticker"])
    df["ticker"] = df["ticker"].astype(str).str.replace(r"\.JK$", "", regex=True)
    # Convert to Jakarta time for correct hour extraction
    df["dt_wib"] = df["datetime"].dt.tz_convert("Asia/Jakarta") if df["datetime"].dt.tz is not None else df["datetime"]
    df["date"] = df["dt_wib"].dt.normalize().dt.tz_localize(None).astype("datetime64[ns]")
    df["hour"] = df["dt_wib"].dt.hour

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    bars = df[df["hour"].isin(PRECLOSE_HOURS)].copy()
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    bars["tp"] = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    bars["tp_vol"] = bars["tp"] * bars["volume"].fillna(0.0)

    daily_last = (
        df.dropna(subset=["close"])
        .sort_values(["ticker", "date", "datetime"])
        .groupby(["ticker", "date"], sort=False)["close"]
        .last()
        .reset_index(name="daily_last_close")
        .sort_values(["ticker", "date"])
    )
    daily_last["pre14_prev_close"] = daily_last.groupby("ticker", sort=False)["daily_last_close"].shift(1)
    prev_close = daily_last[["date", "ticker", "pre14_prev_close"]]
    ara_state = build_ara_state_features(bars, prev_close)

    g = bars.groupby(["date", "ticker"], sort=False)
    base = g.agg(
        pre14_open9=("open", lambda s: s.iloc[0]),
        pre14_high=("high", "max"),
        pre14_low=("low", "min"),
        pre14_tp_vol=("tp_vol", "sum"),
        pre14_volume_until14=("volume", "sum"),
    ).reset_index()

    hour_14 = (
        bars[bars["hour"] == 14]
        .sort_values(["date", "ticker", "datetime"])
        .groupby(["date", "ticker"], sort=False)
        .agg(
            pre14_open14=("open", "first"),
            pre14_close14=("close", "last"),
            pre14_volume_14h=("volume", "sum"),
        )
        .reset_index()
    )
    # ORB uses the first available morning bar. Strict hour-09 bars are sparse
    # in Yahoo 1h history, so this keeps the opening-range concept executable
    # while preserving source-hour for audit.
    orb_09 = (
        bars[bars["hour"].isin(MORNING_HOURS)]
        .sort_values(["date", "ticker", "datetime"])
        .groupby(["date", "ticker"], sort=False)
        .agg(
            pre14_orb_open=("open", "first"),
            pre14_orb_high=("high", "max"),
            pre14_orb_low=("low", "min"),
            pre14_orb_close=("close", "last"),
            pre14_orb_source_hour=("hour", "first"),
        )
        .reset_index()
    )
    hour_11 = (
        bars[bars["hour"] == 11]
        .sort_values(["date", "ticker", "datetime"])
        .groupby(["date", "ticker"], sort=False)["close"]
        .last()
        .reset_index()
        .rename(columns={"close": "pre14_close11"})
    )
    am_range = (
        bars[bars["hour"].isin(MORNING_HOURS)]
        .groupby(["date", "ticker"], sort=False)
        .agg(
            pre14_am_high=("high", "max"),
            pre14_am_low=("low", "min"),
        )
        .reset_index()
    )
    pm_range = (
        bars[bars["hour"].isin({13, 14})]
        .groupby(["date", "ticker"], sort=False)
        .agg(
            pre14_pm_high=("high", "max"),
            pre14_pm_low=("low", "min"),
        )
        .reset_index()
    )
    pm_open = (
        bars[bars["hour"] >= 13]
        .sort_values(["date", "ticker", "hour", "datetime"])
        .groupby(["date", "ticker"], sort=False)
        .agg(
            pre14_open_pm=("open", "first"),
            pre14_close_pm_open=("close", "first"),
        )
        .reset_index()
    )

    morning = bars[bars["hour"].isin(MORNING_HOURS)].copy()
    am = morning.groupby(["date", "ticker"], sort=False).agg(
        am_tp_vol=("tp_vol", "sum"),
        am_vol=("volume", "sum"),
    ).reset_index()
    am["pre14_vwap_morning"] = am["am_tp_vol"] / am["am_vol"].replace(0, np.nan)
    am = am[["date", "ticker", "am_tp_vol", "am_vol", "pre14_vwap_morning"]]

    pre_avg = (
        bars[bars["hour"].isin(PRE_14_AVG_HOURS)]
        .groupby(["date", "ticker"], sort=False)["volume"]
        .mean()
        .reset_index()
        .rename(columns={"volume": "pre14_avg_hourly_volume_before14"})
    )

    out = (
        base.merge(hour_14, on=["date", "ticker"], how="inner")
        .merge(orb_09, on=["date", "ticker"], how="left")
        .merge(hour_11, on=["date", "ticker"], how="left")
        .merge(am_range, on=["date", "ticker"], how="left")
        .merge(pm_range, on=["date", "ticker"], how="left")
        .merge(pm_open, on=["date", "ticker"], how="left")
        .merge(am, on=["date", "ticker"], how="left")
        .merge(pre_avg, on=["date", "ticker"], how="left")
        .merge(prev_close, on=["date", "ticker"], how="left")
        .merge(ara_state, on=["date", "ticker"], how="left")
    )

    out["pre14_vwap"] = out["pre14_tp_vol"] / out["pre14_volume_until14"].replace(0, np.nan)
    out["pre14_turnover_until14"] = out["pre14_tp_vol"]
    out["pre14_am_volume"] = out["am_vol"]
    out["pre14_am_turnover"] = out["am_tp_vol"]
    out["pre14_pm_volume"] = (
        out["pre14_volume_until14"] - out["pre14_am_volume"]
    )
    out["pre14_pm_turnover"] = (
        out["pre14_turnover_until14"] - out["pre14_am_turnover"]
    )
    out["pre14_pm_am_volume_ratio"] = (
        out["pre14_pm_volume"] / out["pre14_am_volume"].replace(0, np.nan)
    )
    out["pre14_pm_am_turnover_ratio"] = (
        out["pre14_pm_turnover"] / out["pre14_am_turnover"].replace(0, np.nan)
    )
    out["pre14_pm_am_volume_rate_ratio"] = (
        (out["pre14_pm_volume"] / 2.0) / (out["pre14_am_volume"].replace(0, np.nan) / 3.0)
    )
    out["pre14_pm_am_turnover_rate_ratio"] = (
        (out["pre14_pm_turnover"] / 2.0) / (out["pre14_am_turnover"].replace(0, np.nan) / 3.0)
    )

    out["pre14_ret_14h"] = (
        (out["pre14_close14"] - out["pre14_open14"]) / out["pre14_open14"].replace(0, np.nan)
    )
    out["pre14_ret_11_to_14"] = (
        (out["pre14_close14"] - out["pre14_close11"]) / out["pre14_close11"].replace(0, np.nan)
    )
    out["pre14_close_vs_open9"] = (
        (out["pre14_close14"] - out["pre14_open9"]) / out["pre14_open9"].replace(0, np.nan)
    )
    out["pre14_range_pct"] = (
        (out["pre14_high"] - out["pre14_low"]) / out["pre14_open9"].replace(0, np.nan)
    )
    out["pre14_orb_mid"] = (out["pre14_orb_high"] + out["pre14_orb_low"]) / 2.0
    out["pre14_orb_range"] = out["pre14_orb_high"] - out["pre14_orb_low"]
    orb_range = out["pre14_orb_range"].replace(0, np.nan)
    out["pre14_orb_range_pct"] = (
        out["pre14_orb_range"] / out["pre14_orb_open"].replace(0, np.nan)
    )
    out["pre14_close_vs_orb_mid"] = (
        (out["pre14_close14"] - out["pre14_orb_mid"])
        / out["pre14_orb_mid"].replace(0, np.nan)
    )
    out["pre14_close_vs_orb_high"] = (
        (out["pre14_close14"] - out["pre14_orb_high"])
        / out["pre14_orb_high"].replace(0, np.nan)
    )
    out["pre14_close_vs_orb_low"] = (
        (out["pre14_close14"] - out["pre14_orb_low"])
        / out["pre14_orb_low"].replace(0, np.nan)
    )
    out["pre14_orb_position"] = (
        (out["pre14_close14"] - out["pre14_orb_low"]) / orb_range
    )
    out["pre14_orb_breakout_strength"] = (
        (out["pre14_close14"] - out["pre14_orb_high"]).clip(lower=0.0) / orb_range
    )
    out["pre14_orb_breakdown_strength"] = (
        (out["pre14_orb_low"] - out["pre14_close14"]).clip(lower=0.0) / orb_range
    )
    out["pre14_pm_high_vs_orb_high"] = (
        (out["pre14_pm_high"] - out["pre14_orb_high"])
        / out["pre14_orb_high"].replace(0, np.nan)
    )
    out["pre14_pm_high_breakout_strength"] = (
        (out["pre14_pm_high"] - out["pre14_orb_high"]).clip(lower=0.0) / orb_range
    )
    out["pre14_close_orb_high_hold"] = (
        (out["pre14_close14"] - out["pre14_orb_high"]) / orb_range
    )
    out["pre14_orb_body_pct"] = (
        (out["pre14_orb_close"] - out["pre14_orb_open"])
        / out["pre14_orb_open"].replace(0, np.nan)
    )
    out["pre14_orb_upper_wick_pct"] = (
        (out["pre14_orb_high"] - np.maximum(out["pre14_orb_open"], out["pre14_orb_close"]))
        / out["pre14_orb_open"].replace(0, np.nan)
    )
    out["pre14_orb_lower_wick_pct"] = (
        (np.minimum(out["pre14_orb_open"], out["pre14_orb_close"]) - out["pre14_orb_low"])
        / out["pre14_orb_open"].replace(0, np.nan)
    )
    out["pre14_close_above_orb_high"] = (out["pre14_close14"] > out["pre14_orb_high"]).astype("float32")
    out["pre14_close_below_orb_low"] = (out["pre14_close14"] < out["pre14_orb_low"]).astype("float32")
    out["pre14_pm_high_above_orb_high"] = (out["pre14_pm_high"] > out["pre14_orb_high"]).astype("float32")
    out["pre14_close_to_vwap"] = (
        (out["pre14_close14"] - out["pre14_vwap"]) / out["pre14_vwap"].replace(0, np.nan)
    )
    out["pre14_vwap_trend"] = (
        (out["pre14_vwap"] - out["pre14_vwap_morning"])
        / out["pre14_vwap_morning"].replace(0, np.nan)
    )
    out["pre14_close_above_vwap"] = (out["pre14_close14"] > out["pre14_vwap"]).astype("float32")
    out["pre14_pm_drive"] = (
        (out["pre14_close14"] - out["pre14_close_pm_open"])
        / out["pre14_vwap"].replace(0, np.nan)
    )
    out["pre14_pm_open_to_vwap_am"] = (
        (out["pre14_close_pm_open"] - out["pre14_vwap_morning"])
        / out["pre14_vwap_morning"].replace(0, np.nan)
    )
    out["pre14_pm_high_vs_close11"] = (
        (out["pre14_pm_high"] - out["pre14_close11"])
        / out["pre14_close11"].replace(0, np.nan)
    )
    out["pre14_pm_close_vs_close11"] = (
        (out["pre14_close14"] - out["pre14_close11"])
        / out["pre14_close11"].replace(0, np.nan)
    )
    out["pre14_pm_low_vs_close11"] = (
        (out["pre14_pm_low"] - out["pre14_close11"])
        / out["pre14_close11"].replace(0, np.nan)
    )
    out["pre14_pm_open_vs_close11"] = (
        (out["pre14_open_pm"] - out["pre14_close11"])
        / out["pre14_close11"].replace(0, np.nan)
    )
    out["pre14_pm_high_close_spread"] = (
        (out["pre14_pm_high"] - out["pre14_close14"])
        / out["pre14_close11"].replace(0, np.nan)
    )

    bars = bars.sort_values(["ticker", "date", "hour", "datetime"]).copy()
    bars["cum_tp_vol"] = bars.groupby(["ticker", "date"], sort=False)["tp_vol"].cumsum()
    bars["cum_volume"] = bars.groupby(["ticker", "date"], sort=False)["volume"].cumsum()
    bars["cum_vwap"] = bars["cum_tp_vol"] / bars["cum_volume"].replace(0, np.nan)
    bars["vol_above_vwap"] = np.where(bars["close"] > bars["cum_vwap"], bars["volume"], 0.0)
    vol_above = (
        bars.groupby(["date", "ticker"], sort=False)["vol_above_vwap"]
        .sum()
        .reset_index()
    )
    out = out.merge(vol_above, on=["date", "ticker"], how="left")
    out["pre14_vol_above_vwap_pct"] = (
        out["vol_above_vwap"] / out["pre14_volume_until14"].replace(0, np.nan)
    )

    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    grp = out.groupby("ticker", sort=False)
    out["pre14_daily_volume"] = out["pre14_volume_until14"]
    out["pre14_daily_turnover"] = out["pre14_turnover_until14"]
    for w in [5, 10, 20]:
        out[f"pre14_daily_volume_ma{w}"] = grp["pre14_daily_volume"].transform(
            lambda s, window=w: s.shift(1).rolling(window, min_periods=max(3, window // 2)).mean()
        )
        out[f"pre14_daily_turnover_ma{w}"] = grp["pre14_daily_turnover"].transform(
            lambda s, window=w: s.shift(1).rolling(window, min_periods=max(3, window // 2)).mean()
        )
        out[f"pre14_am_volume_to_dailyvol_{w}d"] = (
            out["pre14_am_volume"] / out[f"pre14_daily_volume_ma{w}"].replace(0, np.nan)
        )
        out[f"pre14_until14_volume_to_dailyvol_{w}d"] = (
            out["pre14_volume_until14"] / out[f"pre14_daily_volume_ma{w}"].replace(0, np.nan)
        )
        out[f"pre14_am_turnover_to_dailyturnover_{w}d"] = (
            out["pre14_am_turnover"] / out[f"pre14_daily_turnover_ma{w}"].replace(0, np.nan)
        )
        out[f"pre14_until14_turnover_to_dailyturnover_{w}d"] = (
            out["pre14_turnover_until14"] / out[f"pre14_daily_turnover_ma{w}"].replace(0, np.nan)
        )
    out["pre14_volume_until14_ma20"] = grp["pre14_volume_until14"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=5).mean()
    )
    out["pre14_turnover_until14_ma20"] = grp["pre14_turnover_until14"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=5).mean()
    )
    out["pre14_volume_14h_ratio_to_avg_hourly"] = (
        out["pre14_volume_14h"]
        / out["pre14_avg_hourly_volume_before14"].replace(0, np.nan)
    )
    out["pre14_volume_until14_ratio_20d"] = (
        out["pre14_volume_until14"]
        / out["pre14_volume_until14_ma20"].replace(0, np.nan)
    )
    out["pre14_turnover_until14_ratio_20d"] = (
        out["pre14_turnover_until14"]
        / out["pre14_turnover_until14_ma20"].replace(0, np.nan)
    )

    out["pre14_tick_size"] = out["pre14_close14"].apply(idx_tick_size)
    out["pre14_tick_pct"] = out["pre14_tick_size"] / out["pre14_close14"].replace(0, np.nan)
    out["pre14_am_open_low_dist"] = (
        (out["pre14_open9"] - out["pre14_am_low"])
        / out["pre14_open9"].replace(0, np.nan)
    )
    out["pre14_pm_open_low_dist"] = (
        (out["pre14_open_pm"] - out["pre14_pm_low"])
        / out["pre14_open_pm"].replace(0, np.nan)
    )
    out["pre14_am_open_is_low"] = (
        out["pre14_open9"] <= (out["pre14_am_low"] + out["pre14_tick_size"])
    ).astype("float32")
    out["pre14_pm_open_is_low"] = (
        out["pre14_open_pm"] <= (out["pre14_pm_low"] + out["pre14_tick_size"])
    ).astype("float32")
    pm_range_denom = (out["pre14_pm_high"] - out["pre14_pm_low"]).replace(0, np.nan)
    out["pre14_pm_close_near_high"] = (
        (out["pre14_close14"] - out["pre14_pm_low"]) / pm_range_denom
    ).clip(0.0, 1.0)
    spread_one_way = out["pre14_close14"].apply(estimate_spread_frac)
    out["pre14_spread_cost_est"] = 2.0 * spread_one_way
    out["pre14_market_cost_est"] = ROUNDTRIP_COST_FRAC + out["pre14_spread_cost_est"]
    out["pre14_return_from_prev_close"] = (
        (out["pre14_close14"] - out["pre14_prev_close"])
        / out["pre14_prev_close"].replace(0, np.nan)
    )
    out["pre14_ara_limit_pct"] = out["pre14_prev_close"].apply(idx_ara_limit_pct)
    out["pre14_ara_distance_pct"] = out["pre14_ara_limit_pct"] - out["pre14_return_from_prev_close"]
    out["pre14_is_ara_like"] = (
        out["pre14_return_from_prev_close"] >= (out["pre14_ara_limit_pct"] - ARA_BUFFER_PCT)
    ).astype("float32")

    feature_cols = [
        "date",
        "ticker",
        "pre14_ret_14h",
        "pre14_ret_11_to_14",
        "pre14_close_vs_open9",
        "pre14_range_pct",
        "pre14_orb_range_pct",
        "pre14_orb_source_hour",
        "pre14_close_vs_orb_mid",
        "pre14_close_vs_orb_high",
        "pre14_close_vs_orb_low",
        "pre14_orb_position",
        "pre14_orb_breakout_strength",
        "pre14_orb_breakdown_strength",
        "pre14_pm_high_vs_orb_high",
        "pre14_pm_high_breakout_strength",
        "pre14_close_orb_high_hold",
        "pre14_orb_body_pct",
        "pre14_orb_upper_wick_pct",
        "pre14_orb_lower_wick_pct",
        "pre14_close_above_orb_high",
        "pre14_close_below_orb_low",
        "pre14_pm_high_above_orb_high",
        "pre14_vwap",
        "pre14_close_to_vwap",
        "pre14_vwap_trend",
        "pre14_close_above_vwap",
        "pre14_pm_drive",
        "pre14_vol_above_vwap_pct",
        "pre14_pm_open_to_vwap_am",
        "pre14_pm_high_vs_close11",
        "pre14_pm_close_vs_close11",
        "pre14_pm_low_vs_close11",
        "pre14_pm_open_vs_close11",
        "pre14_pm_high_close_spread",
        "pre14_volume_until14",
        "pre14_turnover_until14",
        "pre14_am_volume",
        "pre14_pm_volume",
        "pre14_am_turnover",
        "pre14_pm_turnover",
        "pre14_pm_am_volume_ratio",
        "pre14_pm_am_turnover_ratio",
        "pre14_pm_am_volume_rate_ratio",
        "pre14_pm_am_turnover_rate_ratio",
        "pre14_volume_14h",
        "pre14_volume_14h_ratio_to_avg_hourly",
        "pre14_am_volume_to_dailyvol_5d",
        "pre14_am_volume_to_dailyvol_10d",
        "pre14_am_volume_to_dailyvol_20d",
        "pre14_until14_volume_to_dailyvol_5d",
        "pre14_until14_volume_to_dailyvol_10d",
        "pre14_until14_volume_to_dailyvol_20d",
        "pre14_am_turnover_to_dailyturnover_5d",
        "pre14_am_turnover_to_dailyturnover_10d",
        "pre14_am_turnover_to_dailyturnover_20d",
        "pre14_until14_turnover_to_dailyturnover_5d",
        "pre14_until14_turnover_to_dailyturnover_10d",
        "pre14_until14_turnover_to_dailyturnover_20d",
        "pre14_volume_until14_ratio_20d",
        "pre14_turnover_until14_ratio_20d",
        "pre14_tick_size",
        "pre14_tick_pct",
        "pre14_am_open_low_dist",
        "pre14_pm_open_low_dist",
        "pre14_am_open_is_low",
        "pre14_pm_open_is_low",
        "pre14_pm_close_near_high",
        "pre14_spread_cost_est",
        "pre14_market_cost_est",
        "pre14_prev_close",
        "pre14_return_from_prev_close",
        "pre14_ara_limit_pct",
        "pre14_ara_distance_pct",
        "pre14_is_ara_like",
        "pre14_ara_price",
        "pre14_ara_touched",
        "pre14_ara_touch_hour",
        "pre14_ara_touched_bar_count",
        "pre14_ara_release_wick_count",
        "pre14_ara_release_wick_ratio",
        "pre14_ara_release_wick_depth_max",
        "pre14_ara_release_wick_depth_mean",
        "pre14_ara_close_at_ara_count",
        "pre14_ara_flat_ohlc_count",
        "pre14_ara_last_bar_touched",
        "pre14_ara_last_bar_release",
        "pre14_ara_last_bar_close_at_ara",
        "pre14_ara_last_bar_locked",
        "pre14_ara_last_bar_release_depth",
        "pre14_ara_locked_proxy",
        "pre14_ara_touched_released",
    ]
    result = out[feature_cols].copy()
    for c in feature_cols[2:]:
        result[c] = pd.to_numeric(result[c], errors="coerce").astype("float32")
    return result.sort_values(["date", "ticker"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    print(f"[preclose14] yf_1h={args.yf_1h_path}")
    yf_1h = pd.read_parquet(args.yf_1h_path)
    features = build_preclose14_features(yf_1h)
    print(
        "[preclose14] rows={:,}, dates={} -> {}, tickers={}, features={}".format(
            len(features),
            pd.to_datetime(features["date"]).min().date(),
            pd.to_datetime(features["date"]).max().date(),
            features["ticker"].nunique(),
            len(features.columns) - 2,
        )
    )
    nan_summary = (
        features.drop(columns=["date", "ticker"])
        .isna()
        .mean()
        .sort_values(ascending=False)
        .head(8)
    )
    print("[preclose14] top_nan_ratio=")
    print(nan_summary.to_string())
    if args.dry_run:
        print("[preclose14] dry-run, not writing")
        return
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output_path, index=False)
    print(f"[preclose14] wrote {args.output_path}")


if __name__ == "__main__":
    main()
