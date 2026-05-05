"""Schema for global_indices — daily close for 5 macro indices from Yahoo Finance.

Source: yfinance Python library (pipeline/fetch/fetch_global_indices_simple.py).
Lives in global_indices.duckdb.

Natural key: (symbol, date) — one row per symbol per trading day.

Symbols: ^IXIC (Nasdaq), ^N225 (Nikkei), ^VIX, IDR=X (USDIDR), ^JKSE (IHSG).
"""
from __future__ import annotations

from ._base import ColumnDef, TableSchema

SCHEMA = TableSchema(
    source="global_indices",
    db_file="global_indices.duckdb",
    table="global_indices",
    parquet_path="data/Level_0_Raw/global_indices.parquet",
    natural_key=("symbol", "date"),
    columns=(
        ColumnDef("date", "TIMESTAMPTZ", nullable=False),
        ColumnDef("symbol", "VARCHAR", nullable=False),
        ColumnDef("close", "DOUBLE"),
    ),
)
