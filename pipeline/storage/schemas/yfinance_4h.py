"""Schema for yfinance_4h — 4-hourly OHLCV bars from Yahoo Finance for IDX tickers.

Source: yfinance Python library (pipeline/fetch/fetch_yfinance.py).
Lives in yfinance_4h.duckdb.

Natural key: (ticker, datetime) — one bar per ticker per UTC window.

Datetime is stored as TIMESTAMPTZ.
"""
from __future__ import annotations

from ._base import ColumnDef, TableSchema

SCHEMA = TableSchema(
    source="yfinance_4h",
    db_file="yfinance_4h.duckdb",
    table="yfinance_4h",
    parquet_path="data/Level_0_Raw/yfinance_4h.parquet",
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
