"""BaseValidator — daily canary parity check between parquet and DuckDB.

Per PRD 0003 §5.3:
  Hard checks (strict equality):
    - count(*)
    - schema (column name set)
    - distinct natural-key cardinality
    - null count per column

  Sample comparison (1000 random rows, sorted by natural key, seed=42):
    - int / string / bool / timestamp: strict equality
    - float / double / decimal:  np.isclose(rtol=1e-9, atol=1e-12, equal_nan=True)

Pass criteria for cutover: zero discrepancy 14 calendar days in a row.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..schemas._base import AUDIT_INSERTED, AUDIT_UPDATED, TableSchema

_FLOAT_TYPES = ("FLOAT", "DOUBLE", "REAL", "DECIMAL", "NUMERIC")
_TS_TYPES = ("TIMESTAMP", "TIMESTAMPTZ", "DATE", "TIME")
_TZ_HINTS = ("TIMESTAMPTZ", "TIME ZONE", "TIMETZ")


@dataclass
class ValidationResult:
    source: str
    passed: bool
    hard_checks: dict[str, bool] = field(default_factory=dict)
    sample_diff_count: int = 0
    sample_size: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "passed": self.passed,
            "hard_checks": self.hard_checks,
            "sample_size": self.sample_size,
            "sample_diff_count": self.sample_diff_count,
            "notes": self.notes,
        }


class BaseValidator:
    SAMPLE_SIZE = 1000
    SAMPLE_SEED = 42
    FLOAT_RTOL = 1e-9
    FLOAT_ATOL = 1e-12

    def __init__(self, schema: TableSchema):
        self.schema = schema

    def run(
        self,
        parquet_df: pd.DataFrame,
        duckdb_df: pd.DataFrame,
    ) -> ValidationResult:
        """Compare two DataFrames representing the same logical table.

        duckdb_df may include audit cols; they are stripped before comparison.
        """
        result = ValidationResult(source=self.schema.source, passed=True)
        ddb = duckdb_df.drop(columns=[AUDIT_INSERTED, AUDIT_UPDATED], errors="ignore")
        pq = parquet_df

        result.hard_checks = self._hard_checks(pq, ddb, result.notes)
        if not all(result.hard_checks.values()):
            result.passed = False
            return result

        n_diff, n_sampled = self._sample_compare(pq, ddb, result.notes)
        result.sample_size = n_sampled
        result.sample_diff_count = n_diff
        if n_diff > 0:
            result.passed = False
        return result

    def _hard_checks(
        self,
        pq: pd.DataFrame,
        ddb: pd.DataFrame,
        notes: list[str],
    ) -> dict[str, bool]:
        checks: dict[str, bool] = {}

        checks["count"] = len(pq) == len(ddb)
        if not checks["count"]:
            notes.append(f"count: parquet={len(pq)} duckdb={len(ddb)}")

        pq_cols = set(pq.columns)
        ddb_cols = set(ddb.columns)
        checks["columns"] = pq_cols == ddb_cols
        if not checks["columns"]:
            notes.append(
                f"columns: missing_in_ddb={sorted(pq_cols - ddb_cols)} "
                f"extra_in_ddb={sorted(ddb_cols - pq_cols)}"
            )
            return checks

        nk = list(self.schema.natural_key)
        checks["natural_key_distinct"] = (
            pq[nk].drop_duplicates().shape[0] == ddb[nk].drop_duplicates().shape[0]
        )
        if not checks["natural_key_distinct"]:
            notes.append(
                f"distinct PK: parquet={pq[nk].drop_duplicates().shape[0]} "
                f"duckdb={ddb[nk].drop_duplicates().shape[0]}"
            )

        nulls_match = True
        for col in self.schema.all_user_columns:
            pq_n, ddb_n = int(pq[col].isna().sum()), int(ddb[col].isna().sum())
            if pq_n != ddb_n:
                nulls_match = False
                notes.append(f"nulls[{col}]: parquet={pq_n} duckdb={ddb_n}")
        checks["nulls"] = nulls_match
        return checks

    def _sample_compare(
        self,
        pq: pd.DataFrame,
        ddb: pd.DataFrame,
        notes: list[str],
    ) -> tuple[int, int]:
        nk = list(self.schema.natural_key)
        pq_s = pq.sort_values(nk).reset_index(drop=True)
        ddb_s = ddb.sort_values(nk).reset_index(drop=True)
        n = min(self.SAMPLE_SIZE, len(pq_s))
        if n == 0:
            return 0, 0
        rng = np.random.default_rng(self.SAMPLE_SEED)
        idx = np.sort(rng.choice(len(pq_s), size=n, replace=False))

        diffs = 0
        for col_def in self.schema.columns:
            col = col_def.name
            pq_col = pq_s.loc[idx, col].reset_index(drop=True)
            ddb_col = ddb_s.loc[idx, col].reset_index(drop=True)
            col_diffs = self._compare_column(pq_col, ddb_col, col_def.duckdb_type)
            if col_diffs > 0:
                diffs += col_diffs
                notes.append(f"sample_diff[{col}]: {col_diffs}/{n}")
        return diffs, n

    def _compare_column(
        self,
        a: pd.Series,
        b: pd.Series,
        duckdb_type: str,
    ) -> int:
        dt = duckdb_type.upper()
        if any(t in dt for t in _FLOAT_TYPES):
            both_nan = a.isna().to_numpy() & b.isna().to_numpy()
            try:
                close = np.isclose(
                    a.fillna(0).to_numpy(dtype=float),
                    b.fillna(0).to_numpy(dtype=float),
                    rtol=self.FLOAT_RTOL,
                    atol=self.FLOAT_ATOL,
                )
            except (TypeError, ValueError):
                close = (a.to_numpy() == b.to_numpy())
            return int((~(close | both_nan)).sum())

        if any(t in dt for t in _TS_TYPES):
            # Parquet often stores timestamps as ISO strings; DuckDB returns
            # them as datetime64. Normalize both sides before strict equality.
            utc = any(t in dt for t in _TZ_HINTS)
            a_n = pd.to_datetime(a, utc=utc, errors="coerce")
            b_n = pd.to_datetime(b, utc=utc, errors="coerce")
            both_nan = a_n.isna().to_numpy() & b_n.isna().to_numpy()
            eq = (a_n.to_numpy() == b_n.to_numpy()) | both_nan
            return int((~eq).sum())

        both_nan = a.isna().to_numpy() & b.isna().to_numpy()
        eq = (a.to_numpy() == b.to_numpy()) | both_nan
        return int((~eq).sum())
