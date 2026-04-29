"""
DuckDB helpers for the BPJS inference store.
Mirrors inferences/bsjp/db.py — same schema, separate DB file.
"""
import duckdb
import pandas as pd
from datetime import date, timedelta

from config import INFERENCE_DB, DB_HISTORY_DAYS


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    INFERENCE_DB.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(INFERENCE_DB), read_only=read_only)


def init_schema(con: duckdb.DuckDBPyConnection, feature_names: list[str]) -> None:
    feature_cols = "\n    ,".join(f'"{f}" DOUBLE' for f in feature_names)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS features_store (
            date        DATE NOT NULL
           ,ticker      VARCHAR NOT NULL
           ,{feature_cols}
           ,inserted_at TIMESTAMP DEFAULT now()
           ,PRIMARY KEY (date, ticker)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS picks_log (
            date             DATE NOT NULL
           ,variant          VARCHAR NOT NULL
           ,rank             INTEGER NOT NULL
           ,ticker           VARCHAR NOT NULL
           ,pred_proba       DOUBLE
           ,entry_price      DOUBLE
           ,exit_price       DOUBLE
           ,intraday_return  DOUBLE
           ,hit_tp           BOOLEAN
           ,logged_at        TIMESTAMP DEFAULT now()
           ,PRIMARY KEY (date, variant, rank)
        )
    """)


def upsert_features(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    raise NotImplementedError("implement in Phase 2")


def get_latest_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute("""
        SELECT * EXCLUDE (inserted_at)
        FROM features_store
        WHERE date = (SELECT MAX(date) FROM features_store)
    """).df()


def get_latest_date(con: duckdb.DuckDBPyConnection) -> date | None:
    row = con.execute("SELECT MAX(date) FROM features_store").fetchone()
    return row[0] if row else None


def prune_old_rows(con: duckdb.DuckDBPyConnection) -> int:
    cutoff = date.today() - timedelta(days=DB_HISTORY_DAYS)
    result = con.execute("DELETE FROM features_store WHERE date < ?", [cutoff])
    return result.rowcount


def log_picks(con: duckdb.DuckDBPyConnection, picks: pd.DataFrame, variant: str) -> None:
    raise NotImplementedError("implement in Phase 2")


def settle_picks(con: duckdb.DuckDBPyConnection, settlement_df: pd.DataFrame, variant: str) -> None:
    raise NotImplementedError("implement in Phase 2")
