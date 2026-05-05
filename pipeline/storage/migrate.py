"""Migration runner helpers for L0 storage.

For Phase 0 the only operation is `init_all()` — idempotent CREATE TABLE for
every registered schema. Later phases (`pipeline/migrations/0003_phase{1..5}_*.py`)
add data loads, validators, and resume-state migrations and call back into
this module for connection management.
"""
from __future__ import annotations

import argparse
import sys

from .l0 import open_l0_writer
from .schemas import SCHEMAS, TableSchema


def _group_by_db_file() -> dict[str, list[TableSchema]]:
    out: dict[str, list[TableSchema]] = {}
    for schema in SCHEMAS.values():
        out.setdefault(schema.db_file, []).append(schema)
    return out


def init_all() -> dict[str, list[str]]:
    """Apply CREATE TABLE IF NOT EXISTS for every registered schema.

    Returns mapping of db_file → list of tables created/verified.
    """
    summary: dict[str, list[str]] = {}
    for db_file, schemas in _group_by_db_file().items():
        with open_l0_writer(db_file) as con:
            for schema in schemas:
                con.execute(schema.create_table_sql())
            con.execute("CHECKPOINT")
        summary[db_file] = [s.table for s in schemas]
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="L0 storage migration runner")
    parser.add_argument(
        "command",
        choices=["init"],
        help="init: idempotent CREATE TABLE for all registered schemas",
    )
    args = parser.parse_args(argv)
    if args.command == "init":
        summary = init_all()
        for db_file, tables in summary.items():
            print(f"{db_file}: {tables}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
