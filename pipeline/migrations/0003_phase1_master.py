"""PRD 0003 Phase 1: initial load of master tables from parquet.

Loads `master_emiten` and `master_broker` from their parquet sources into
`master.duckdb`. Idempotent via `ON CONFLICT DO NOTHING` — re-runs do not
churn audit timestamps.

Audit semantics:
  - First run: rows get `_inserted_at = _updated_at = now()`.
  - Subsequent runs: existing rows untouched.

Phase 1 next steps (NOT done by this script):
  - Refactor fetch scripts to call write_l0() instead of df.to_parquet().
  - Promote stage 1: export L0_MASTER_EMITEN_DUCKDB_WRITE=true.
  - Run validator daily for 14 days zero-discrepancy streak.

Usage:
    python pipeline/migrations/0003_phase1_master.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.storage.l0 import open_l0_writer  # noqa: E402
from pipeline.storage.schemas import SCHEMAS  # noqa: E402
from pipeline.storage.schemas._base import (  # noqa: E402
    AUDIT_INSERTED,
    AUDIT_UPDATED,
    _q,
)


SOURCES = ("master_emiten", "master_broker")


def _initial_load_sql(schema, parquet_path: str) -> str:
    """INSERT ... FROM read_parquet ON CONFLICT DO NOTHING.

    `_inserted_at` / `_updated_at` set to now() for fresh rows; conflict path
    is no-op so re-runs preserve original audit timestamps.
    """
    user_cols = ", ".join(_q(c) for c in schema.all_user_columns)
    pk = ", ".join(_q(k) for k in schema.natural_key)
    return (
        f"INSERT INTO {schema.table} ({user_cols}, {AUDIT_INSERTED}, {AUDIT_UPDATED})\n"
        f"SELECT {user_cols}, now(), now() FROM read_parquet('{parquet_path}')\n"
        f"ON CONFLICT ({pk}) DO NOTHING"
    )


def up() -> dict[str, dict]:
    """Run Phase 1 initial load for master sources. Returns per-source counts."""
    by_file: dict[str, list] = {}
    for source in SOURCES:
        schema = SCHEMAS[source]
        by_file.setdefault(schema.db_file, []).append(schema)

    summary: dict[str, dict] = {}
    for db_file, schemas in by_file.items():
        with open_l0_writer(db_file) as con:
            for schema in schemas:
                con.execute(schema.create_table_sql())
                before = con.sql(f"SELECT count(*) FROM {schema.table}").fetchone()[0]
                parquet_abs = str(_REPO_ROOT / schema.parquet_path)
                con.execute("BEGIN")
                try:
                    con.execute(_initial_load_sql(schema, parquet_abs))
                    con.execute("COMMIT")
                except Exception:
                    con.execute("ROLLBACK")
                    raise
                after = con.sql(f"SELECT count(*) FROM {schema.table}").fetchone()[0]
                summary[schema.source] = {
                    "table": schema.table,
                    "db_file": schema.db_file,
                    "rows_before": before,
                    "rows_after": after,
                    "rows_inserted": after - before,
                }
            con.execute("CHECKPOINT")
    return summary


def main() -> int:
    import json

    summary = up()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
