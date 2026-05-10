from __future__ import annotations

from datetime import date

import pandas as pd

from pipeline.fetch.fallback_ipot_trend_to_yfinance_1h import (
    REQUIRED_YF_COLUMNS,
    aggregate_ipot_hour,
    merge_preserving_yfinance,
    select_missing_fallback_rows,
    target_datetime_utc,
)


def test_target_timestamp_wib_to_utc():
    ts = target_datetime_utc(date(2026, 5, 8), 14)

    assert ts.isoformat() == "2026-05-08T07:00:00+00:00"


def test_aggregate_ipot_hour_to_yfinance_schema():
    trend = pd.DataFrame(
        [
            {"ticker": "BBRI", "time": "14:00", "close": 4000, "vol": 100},
            {"ticker": "BBRI", "time": "14:01", "close": 4020, "vol": 150},
            {"ticker": "BBRI", "time": "14:59", "close": 4010, "vol": 200},
            {"ticker": "BBRI", "time": "15:00", "close": 4050, "vol": 300},
        ]
    )

    out = aggregate_ipot_hour(trend, date(2026, 5, 8), 14, volume_mode="sum")

    assert list(out.columns) == REQUIRED_YF_COLUMNS
    assert len(out) == 1
    row = out.iloc[0]
    assert row["datetime"].isoformat() == "2026-05-08T07:00:00+00:00"
    assert row["ticker"] == "BBRI"
    assert row["open"] == 4000
    assert row["high"] == 4020
    assert row["low"] == 4000
    assert row["close"] == 4010
    assert row["volume"] == 450


def test_select_missing_fallback_rows_only_for_missing_universe_tickers():
    target_dt = target_datetime_utc(date(2026, 5, 8), 14)
    yfinance = pd.DataFrame(
        [
            {
                "datetime": target_dt,
                "ticker": "BBRI",
                "open": 1,
                "high": 1,
                "low": 1,
                "close": 1,
                "volume": 1,
            }
        ],
        columns=REQUIRED_YF_COLUMNS,
    )
    ipot = pd.DataFrame(
        [
            {
                "datetime": target_dt,
                "ticker": "BBRI",
                "open": 2,
                "high": 2,
                "low": 2,
                "close": 2,
                "volume": 2,
            },
            {
                "datetime": target_dt,
                "ticker": "TLKM",
                "open": 3,
                "high": 3,
                "low": 3,
                "close": 3,
                "volume": 3,
            },
        ],
        columns=REQUIRED_YF_COLUMNS,
    )

    rows, present_count, missing_count = select_missing_fallback_rows(
        yfinance, ipot, ["BBRI", "TLKM", "ASII"], target_dt
    )

    assert present_count == 1
    assert missing_count == 2
    assert rows["ticker"].tolist() == ["TLKM"]


def test_merge_preserves_existing_yfinance_on_duplicate_key():
    target_dt = target_datetime_utc(date(2026, 5, 8), 14)
    yfinance = pd.DataFrame(
        [
            {
                "datetime": target_dt,
                "ticker": "BBRI",
                "open": 100,
                "high": 110,
                "low": 90,
                "close": 105,
                "volume": 1000,
            }
        ],
        columns=REQUIRED_YF_COLUMNS,
    )
    fallback = pd.DataFrame(
        [
            {
                "datetime": target_dt,
                "ticker": "BBRI",
                "open": 200,
                "high": 220,
                "low": 180,
                "close": 210,
                "volume": 2000,
            }
        ],
        columns=REQUIRED_YF_COLUMNS,
    )

    merged = merge_preserving_yfinance(yfinance, fallback)

    assert len(merged) == 1
    assert merged.iloc[0]["open"] == 100
    assert merged.iloc[0]["close"] == 105


def test_merge_is_idempotent_for_same_fallback_rows():
    target_dt = target_datetime_utc(date(2026, 5, 8), 14)
    yfinance = pd.DataFrame(columns=REQUIRED_YF_COLUMNS)
    fallback = pd.DataFrame(
        [
            {
                "datetime": target_dt,
                "ticker": "TLKM",
                "open": 1,
                "high": 2,
                "low": 1,
                "close": 2,
                "volume": 10,
            }
        ],
        columns=REQUIRED_YF_COLUMNS,
    )

    once = merge_preserving_yfinance(yfinance, fallback)
    twice = merge_preserving_yfinance(once, fallback)

    pd.testing.assert_frame_equal(once, twice)
