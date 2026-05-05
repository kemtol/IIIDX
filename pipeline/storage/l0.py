"""L0 DuckDB file paths and connection helpers.

Per PRD 0003 §4.1: 6 per-source DB files, not a single file. Multiple
tables can share a file (master_emiten + master_broker → master.duckdb)
when their fetch cadence does not contend.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

L0_DIR = Path("data/Level_0_Raw")

L0_DB_FILES: tuple[str, ...] = (
    "master",            # master_emiten + master_broker
    "global_indices",
    "yfinance_daily",
    "yfinance_1h",
    "yfinance_4h",
    "broksum",           # broksum_bybroker
)


def _stem(db_file: str) -> str:
    return db_file.removesuffix(".duckdb")


def l0_path(db_file: str) -> Path:
    """Resolve absolute path. Accepts either 'master' or 'master.duckdb'."""
    stem = _stem(db_file)
    if stem not in L0_DB_FILES:
        raise ValueError(
            f"Unknown L0 db_file: {db_file!r}; expected one of {L0_DB_FILES}"
        )
    return L0_DIR / f"{stem}.duckdb"


def attach_l0(
    con: duckdb.DuckDBPyConnection,
    db_files: tuple[str, ...] | list[str] | None = None,
    read_only: bool = True,
) -> None:
    """ATTACH L0 source DBs to an existing connection for cross-source queries.

    The attached schema name = file stem (e.g. 'master', 'yfinance_daily').
    Per PRD §4.3 example: SELECT ... FROM yfd.yfinance_daily JOIN m.master_emiten ...
    """
    targets = tuple(db_files) if db_files else L0_DB_FILES
    mode = " (READ_ONLY)" if read_only else ""
    for f in targets:
        stem = _stem(f)
        path = l0_path(stem)
        con.execute(f"ATTACH '{path}' AS {stem}{mode}")


def open_l0_writer(db_file: str) -> duckdb.DuckDBPyConnection:
    """Open a write connection. Acquires DuckDB file lock on connect.

    Use as a context manager in writers:
        with open_l0_writer('master') as con:
            con.execute('BEGIN'); ...; con.execute('COMMIT')
    """
    return duckdb.connect(str(l0_path(db_file)))
