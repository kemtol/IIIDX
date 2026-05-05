"""Schema for yfinance_daily — daily OHLCV bars from Yahoo Finance for IDX tickers.

Source: yfinance Python library (pipeline/fetch/fetch_yfinance_daily.py).
Lives in yfinance_daily.duckdb.

Natural key: (ticker, date) — one row per ticker per trading day.

Date is stored as TIMESTAMPTZ. Pandas/to_parquet writes NaT-safe values.
"""
from __future__ import annotations

from ._base import ColumnDef, TableSchema

SCHEMA = TableSchema(
    source="yfinance_daily",
    db_file="yfinance_daily.duckdb",
    table="yfinance_daily",
    parquet_path="data/Level_0_Raw/yfinance_daily.parquet",
    natural_key=("ticker", "date"),
    columns=(
        ColumnDef("date", "TIMESTAMPTZ", nullable=False),
        ColumnDef("ticker", "VARCHAR", nullable=False),
        ColumnDef("open", "DOUBLE"),
        ColumnDef("high", "DOUBLE"),
        ColumnDef("low", "DOUBLE"),
        ColumnDef("close", "DOUBLE"),
        ColumnDef("volume", "DOUBLE"),
    ),
)
