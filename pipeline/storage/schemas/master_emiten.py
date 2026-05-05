"""Schema for master_emiten — IDX listed equities reference data.

Source: D1 API (currently 773 rows, all status=ACTIVE).
Lives in master.duckdb (shared with master_broker per PRD §4.1).

Columns mirror the existing master_emiten.parquet exactly. Source-side
timestamps (created_at/updated_at) are preserved as-is — they are
distinct from our pipeline audit cols (_inserted_at/_updated_at) which
are appended by TableSchema.
"""
from __future__ import annotations

from ._base import ColumnDef, TableSchema

SCHEMA = TableSchema(
    source="master_emiten",
    db_file="master.duckdb",
    table="master_emiten",
    parquet_path="data/Level_0_Raw/master_emiten.parquet",
    natural_key=("ticker",),
    columns=(
        ColumnDef("ticker", "VARCHAR", nullable=False),
        ColumnDef("sector", "VARCHAR"),
        ColumnDef("industry", "VARCHAR"),
        ColumnDef("status", "VARCHAR", nullable=False),
        ColumnDef("created_at", "TIMESTAMP", nullable=False),
        ColumnDef("updated_at", "TIMESTAMP", nullable=False),
        ColumnDef("source", "VARCHAR", nullable=False),
        ColumnDef("fetched_at", "TIMESTAMPTZ", nullable=False),
    ),
)
