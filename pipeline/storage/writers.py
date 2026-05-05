"""Dual-write dispatcher driven by per-source canary flags.

Default (stage 0, no env vars): parquet only — current pipeline unchanged.
Stage 1 (dual-write): parquet AND DuckDB upsert in same call.
Stage 3 (cutover): DuckDB only.

Promotion = set L0_<SOURCE>_DUCKDB_WRITE / L0_<SOURCE>_PARQUET_WRITE env vars.
See PRD 0003 §12.2 for the full state machine.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .config import L0SourceConfig, load_config
from .l0 import open_l0_writer
from .schemas import TableSchema, get_schema


def _write_parquet(df: pd.DataFrame, parquet_path: str) -> None:
    target = Path(parquet_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(target)


def _write_duckdb_upsert(
    df: pd.DataFrame,
    schema: TableSchema,
    con: duckdb.DuckDBPyConnection,
) -> None:
    """Run upsert in a single transaction. Caller owns connection lifecycle."""
    con.execute(schema.create_table_sql())
    con.execute("BEGIN")
    try:
        con.register("df", df)
        con.execute(schema.upsert_sql())
        con.execute("COMMIT")
        con.execute("CHECKPOINT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister("df")


def write_l0(
    source: str,
    df: pd.DataFrame,
    cfg: L0SourceConfig | None = None,
) -> None:
    """Persist df to one or both L0 backends per canary flags.

    Loads canary config from env if not provided. Both flags being False is
    rejected at L0SourceConfig construction (no silent data loss).
    """
    if cfg is None:
        cfg = load_config()[source]
    schema = get_schema(source)

    if cfg.parquet_write:
        _write_parquet(df, schema.parquet_path)

    if cfg.duckdb_write:
        with open_l0_writer(schema.db_file) as con:
            _write_duckdb_upsert(df, schema, con)
