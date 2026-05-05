"""Schema for broksum_bybroker — daily broker-summary data from IPOT.

Source: IPOT WebSocket scraper (pipeline/fetch/fetch_broksum_ipot.py).
Lives in broksum_bybroker.duckdb.

Natural key: (date, broker, stock_code) — one row per broker per stock per trading day.

Date is stored as VARCHAR "YYYY-MM-DD" (not TIMESTAMPTZ).
"""
from __future__ import annotations

from ._base import ColumnDef, TableSchema

SCHEMA = TableSchema(
    source="broksum",
    db_file="broksum_bybroker.duckdb",
    table="broksum",
    parquet_path="data/Level_0_Raw/broksum_bybroker.parquet",
    natural_key=("date", "broker", "stock_code"),
    columns=(
        ColumnDef("broker", "VARCHAR", nullable=False),
        ColumnDef("stock_code", "VARCHAR", nullable=False),
        ColumnDef("date", "VARCHAR", nullable=False),
        ColumnDef("broker_type", "VARCHAR"),
        ColumnDef("breadth", "BIGINT"),
        ColumnDef("buy_val", "DOUBLE"),
        ColumnDef("sell_val", "DOUBLE"),
        ColumnDef("net_val", "DOUBLE"),
        ColumnDef("total_val", "DOUBLE"),
        ColumnDef("buy_vol", "DOUBLE"),
        ColumnDef("sell_vol", "DOUBLE"),
        ColumnDef("net_vol", "DOUBLE"),
        ColumnDef("buy_freq", "BIGINT"),
        ColumnDef("sell_freq", "BIGINT"),
        ColumnDef("avg_buy_price", "DOUBLE"),
        ColumnDef("avg_sell_price", "DOUBLE"),
        ColumnDef("scraped_at", "VARCHAR"),
    ),
)
