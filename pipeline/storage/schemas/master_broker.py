"""Schema for master_broker — IDX broker reference data.

Source: IDX Block Anggota Bursa (currently 92 rows).
Lives in master.duckdb (shared with master_emiten per PRD §4.1).

Columns mirror existing master_broker.parquet exactly. Three columns include
percent signs (`foreignfund_%`, `localfund_%`, `retail_%`) — TableSchema
double-quotes all identifiers so DuckDB parses them cleanly.
"""
from __future__ import annotations

from ._base import ColumnDef, TableSchema

SCHEMA = TableSchema(
    source="master_broker",
    db_file="master.duckdb",
    table="master_broker",
    parquet_path="data/Level_0_Raw/master_broker.parquet",
    natural_key=("broker_code",),
    columns=(
        ColumnDef("broker_code", "VARCHAR", nullable=False),
        ColumnDef("broker_name", "VARCHAR", nullable=False),
        ColumnDef("category", "VARCHAR", nullable=False),
        ColumnDef("foreignfund_%", "INTEGER", nullable=False),
        ColumnDef("localfund_%", "INTEGER", nullable=False),
        ColumnDef("retail_%", "INTEGER", nullable=False),
        ColumnDef("status", "VARCHAR", nullable=False),
        ColumnDef("is_active", "BOOLEAN", nullable=False),
        ColumnDef("first_seen_at", "TIMESTAMPTZ", nullable=False),
        ColumnDef("last_seen_at", "TIMESTAMPTZ", nullable=False),
        ColumnDef("last_status_change_at", "TIMESTAMPTZ", nullable=False),
        ColumnDef("source_url", "VARCHAR", nullable=False),
        ColumnDef("fetched_at", "TIMESTAMPTZ", nullable=False),
    ),
)
