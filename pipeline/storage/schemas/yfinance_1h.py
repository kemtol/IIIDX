"""Schema for yfinance_1h — hourly OHLCV bars from Yahoo Finance for IDX tickers.

Source: yfinance + Yahoo v8 chart API (Python `fetch_yfinance.py`, Go `bsjp download`).
Lives in yfinance_1h.duckdb.

Natural key: (ticker, datetime) — one bar per ticker per UTC hour.

Datetime is stored as TIMESTAMPTZ. Canonical write should use UTC tz-aware values
(real Yahoo unix-second timestamps converted via .In(loc) or pd.to_datetime utc=True).
A historical bug stored WIB-naive values that pandas/pyarrow promoted to UTC without
shift, leading to 7-hour misalignment for rows ≤ 2026-04-27. Going forward: never
strip tz; always write tz-aware. Cleanup of legacy rows is a separate one-shot.
"""
from __future__ import annotations

from ._base import ColumnDef, TableSchema

SCHEMA = TableSchema(
    source="yfinance_1h",
    db_file="yfinance_1h.duckdb",
    table="yfinance_1h",
    parquet_path="data/Level_0_Raw/yfinance_1h.parquet",
    natural_key=("ticker", "datetime"),
    columns=(
        ColumnDef("datetime", "TIMESTAMPTZ", nullable=False),
        ColumnDef("ticker", "VARCHAR", nullable=False),
        ColumnDef("open", "DOUBLE"),
        ColumnDef("high", "DOUBLE"),
        ColumnDef("low", "DOUBLE"),
        ColumnDef("close", "DOUBLE"),
        ColumnDef("volume", "DOUBLE"),
    ),
)
