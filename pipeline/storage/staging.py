"""Per-fetch crash-resilient staging buffer for L0 sources.

Formalizes the user's DIY `_TEMP_<source>.duckdb` pattern: during a long-running
fetch loop, persist each chunk (per-ticker, per-broker, per-batch) into a
DuckDB staging table immediately. A crash mid-loop preserves all previously
staged rows. On restart, the fetch script can read staged state and resume
without re-doing finished work. At the end of a clean run, staging is merged
into the L0 parquet via `write_l0()` and cleared.

Why DuckDB rather than appending to parquet directly:
  - Per-row INSERTs to parquet are not transactional; a crash mid-write can
    corrupt the file. DuckDB BEGIN/COMMIT gives true durability per chunk.
  - Parquet has no efficient upsert; DuckDB ON CONFLICT DO UPDATE handles
    re-runs cleanly when a partial chunk is re-fetched.

Usage (see `pipeline/fetch/fetch_yfinance.py` for the canonical example):

    with open_staging("yfinance_1h") as stage:
        last_map = stage.last_seen_per_ticker(existing_df)
        for ticker in tickers:
            fresh = fetch_one(ticker, last_dt=last_map.get(ticker))
            if fresh is not None:
                stage.append(fresh)        # COMMIT per ticker
        stage.commit_to_l0(base_df=existing_df)  # merge → write_l0
        stage.clear()                            # drop staging table

Crash recovery: if the loop dies before `commit_to_l0()`, the next run sees the
staging file, computes `last_seen_per_ticker` including staged rows, and only
fetches what's missing. After commit_to_l0 succeeds, clear() drops the table
so the next clean run starts empty.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb
import pandas as pd

from .l0 import L0_DIR
from .schemas import TableSchema, get_schema
from .schemas._base import _q
from .writers import write_l0

STAGING_PREFIX = "_STAGING_"
STAGING_TABLE = "staging"


def staging_path(source: str) -> Path:
    """Path to the staging DuckDB for a source. Lives under L0 dir."""
    return L0_DIR / f"{STAGING_PREFIX}{source}.duckdb"


class StagingDB:
    """Open staging table for one L0 source. Use via `open_staging()`."""

    def __init__(self, source: str, schema: TableSchema, con: duckdb.DuckDBPyConnection):
        self.source = source
        self.schema = schema
        self.con = con
        self._ensure_table()

    def _ensure_table(self) -> None:
        col_defs = []
        for c in self.schema.columns:
            null_clause = "" if c.nullable else " NOT NULL"
            col_defs.append(f"{_q(c.name)} {c.duckdb_type}{null_clause}")
        pk = ", ".join(_q(k) for k in self.schema.natural_key)
        body = ",\n  ".join(col_defs + [f"PRIMARY KEY ({pk})"])
        self.con.execute(f"CREATE TABLE IF NOT EXISTS {STAGING_TABLE} (\n  {body}\n)")

    def append(self, df: pd.DataFrame) -> int:
        """Upsert a chunk into staging in a single transaction.

        Returns row count inserted/updated. Caller may call this once per
        ticker, per broker, or per any safe granular batch — each call is
        durable and survives a subsequent crash.
        """
        if df is None or df.empty:
            return 0
        cols_in_schema = list(self.schema.all_user_columns)
        df = df[[c for c in cols_in_schema if c in df.columns]]
        if df.empty:
            return 0
        quoted = ", ".join(_q(c) for c in df.columns)
        select_clause = f"SELECT {quoted} FROM df"
        pk = ", ".join(_q(k) for k in self.schema.natural_key)
        update_pairs = ",\n  ".join(
            f"{_q(c)} = EXCLUDED.{_q(c)}"
            for c in self.schema.value_columns
            if c in df.columns
        )
        sql = (
            f"INSERT INTO {STAGING_TABLE} ({quoted})\n"
            f"{select_clause}\n"
            f"ON CONFLICT ({pk}) DO UPDATE SET\n  {update_pairs}"
            if update_pairs
            else f"INSERT INTO {STAGING_TABLE} ({quoted})\n{select_clause}\nON CONFLICT ({pk}) DO NOTHING"
        )
        self.con.execute("BEGIN")
        try:
            self.con.register("df", df)
            self.con.execute(sql)
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        finally:
            self.con.unregister("df")
        return len(df)

    def fetch_all(self) -> pd.DataFrame:
        """Read all staged rows as a DataFrame (user columns only)."""
        cols = ", ".join(_q(c) for c in self.schema.all_user_columns)
        return self.con.execute(f"SELECT {cols} FROM {STAGING_TABLE}").df()

    def staged_count(self) -> int:
        return self.con.execute(f"SELECT COUNT(*) FROM {STAGING_TABLE}").fetchone()[0]

    def last_seen_per_ticker(
        self,
        base_df: pd.DataFrame | None = None,
        ticker_col: str = "ticker",
        time_col: str = "datetime",
    ) -> dict:
        """Latest seen timestamp per ticker across base + staged rows.

        Used by fetch loops to compute "what's missing" — only fetch dates
        beyond max(base, staged) for each ticker.
        """
        if ticker_col not in self.schema.all_user_columns:
            raise KeyError(f"{ticker_col!r} not in schema {self.schema.table}")
        if time_col not in self.schema.all_user_columns:
            raise KeyError(f"{time_col!r} not in schema {self.schema.table}")
        staged = self.fetch_all()
        merged = (
            pd.concat([base_df, staged], ignore_index=True)
            if base_df is not None and not base_df.empty
            else staged
        )
        if merged.empty:
            return {}
        return merged.groupby(ticker_col)[time_col].max().to_dict()

    def commit_to_l0(self, base_df: pd.DataFrame | None = None) -> int:
        """Merge staged rows with base_df (existing parquet) and write via gateway.

        Dedup uses the schema's natural key with `keep="last"` — staged values
        win over base since they reflect the freshest fetch.

        Returns row count written.
        """
        staged = self.fetch_all()
        if base_df is not None and not base_df.empty:
            base_clean = base_df[
                [c for c in self.schema.all_user_columns if c in base_df.columns]
            ]
            combined = pd.concat([base_clean, staged], ignore_index=True)
        else:
            combined = staged
        if combined.empty:
            return 0
        combined = combined.drop_duplicates(
            subset=list(self.schema.natural_key), keep="last"
        ).sort_values(list(self.schema.natural_key)).reset_index(drop=True)
        write_l0(self.source, combined)
        return len(combined)

    def clear(self) -> None:
        """Truncate the staging table. Call after `commit_to_l0` succeeds.

        Keeps the table definition so subsequent `staged_count()` etc still
        work without re-creating. Use `drop_staging(source)` to also remove
        the file from disk.
        """
        self.con.execute(f"DELETE FROM {STAGING_TABLE}")
        self.con.execute("CHECKPOINT")


@contextmanager
def open_staging(source: str) -> Iterator[StagingDB]:
    """Open (or reopen) the staging DB for a source.

    Connection is closed automatically on context exit. The staging file
    persists across invocations until `clear()` is called.
    """
    schema = get_schema(source)
    path = staging_path(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    try:
        yield StagingDB(source=source, schema=schema, con=con)
    finally:
        con.close()


def staging_exists(source: str) -> bool:
    """Whether a staging file is on disk for this source (resume hint)."""
    return staging_path(source).exists()


def drop_staging(source: str) -> bool:
    """Remove the staging file from disk. Idempotent. Returns True if removed."""
    p = staging_path(source)
    if p.exists():
        p.unlink()
        wal = p.with_suffix(p.suffix + ".wal")
        if wal.exists():
            wal.unlink()
        return True
    return False
