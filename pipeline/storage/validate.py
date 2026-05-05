"""Daily validator runner for L0 canary.

For each registered schema where `duckdb_write` is enabled (stage 1+), compare
parquet vs DuckDB and persist results to `_LOG/duckdb_canary_<source>_<date>.jsonc`.

Skip stage-0 sources — no DuckDB to validate against.

Per PRD 0003 §5.3: pass criteria for cutover is zero discrepancy 14 calendar
days in a row. This runner emits one result per day; aggregating the streak is
out of scope (downstream consumes the JSONC log).

Usage:
    python -m pipeline.storage.validate
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from .config import load_config
from .l0 import l0_path
from .schemas import SCHEMAS
from .validators._base import BaseValidator, ValidationResult

LOG_DIR = Path("_LOG")


def validate_source(source: str) -> ValidationResult:
    schema = SCHEMAS[source]
    pq_df = pd.read_parquet(schema.parquet_path)
    con = duckdb.connect(str(l0_path(schema.db_file)), read_only=True)
    try:
        ddb_df = con.sql(f"SELECT * FROM {schema.table}").df()
    finally:
        con.close()
    return BaseValidator(schema).run(pq_df, ddb_df)


def main(argv: list[str] | None = None) -> int:
    LOG_DIR.mkdir(exist_ok=True)
    cfg = load_config()
    today = date.today().isoformat()
    summary = {
        "date": today,
        "checked": [],
        "skipped": [],
    }
    rc = 0
    for source in sorted(SCHEMAS):
        if not cfg[source].duckdb_write:
            summary["skipped"].append(
                {"source": source, "reason": "stage 0 (duckdb_write=False)"}
            )
            continue
        result = validate_source(source)
        log_path = LOG_DIR / f"duckdb_canary_{source}_{today}.jsonc"
        log_path.write_text(json.dumps(result.as_dict(), indent=2))
        summary["checked"].append(
            {"source": source, "passed": result.passed, "log": str(log_path)}
        )
        if not result.passed:
            rc = 1
    print(json.dumps(summary, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())
