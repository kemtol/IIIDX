"""L0 reader with parquet ↔ duckdb toggle controlled by canary flags.

Downstream code (L1 builders, validators) imports `read_l0(source, **filters)`
and never touches paths directly. Promotion of `L0_<SOURCE>_DUCKDB_READ=true`
flips the source over without further code change.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from .config import L0SourceConfig, load_config
from .l0 import l0_path
from .schemas import get_schema


def _apply_eq_filters_pandas(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    for k, v in filters.items():
        if k not in df.columns:
            raise KeyError(f"filter column {k!r} not in DataFrame")
        df = df[df[k] == v]
    return df


def _read_parquet(parquet_path: str, filters: dict) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    return _apply_eq_filters_pandas(df, filters)


def _q(identifier: str) -> str:
    return f'"{identifier}"'


def _read_duckdb(source: str, filters: dict) -> pd.DataFrame:
    schema = get_schema(source)
    cols = ", ".join(_q(c) for c in schema.all_user_columns)
    sql = f"SELECT {cols} FROM {schema.table}"
    params: list = []
    if filters:
        where_parts = []
        for k, v in filters.items():
            if k not in schema.all_user_columns:
                raise KeyError(f"filter column {k!r} not in schema {schema.table}")
            where_parts.append(f"{_q(k)} = ?")
            params.append(v)
        sql += " WHERE " + " AND ".join(where_parts)
    con = duckdb.connect(str(l0_path(schema.db_file)), read_only=True)
    try:
        return con.execute(sql, params).df()
    finally:
        con.close()


def read_l0(
    source: str,
    *,
    cfg: L0SourceConfig | None = None,
    **filters,
) -> pd.DataFrame:
    """Read L0 source. Returns user columns only (audit cols stripped).

    Filters are simple equality predicates: read_l0('master_emiten', ticker='BBCA').
    """
    if cfg is None:
        cfg = load_config()[source]
    if cfg.duckdb_read:
        return _read_duckdb(source, filters)
    schema = get_schema(source)
    return _read_parquet(schema.parquet_path, filters)
