"""Continuity merge gate per PRD 0003 Precondition #4.

Asserts that with default canary flags (parquet_write=True, duckdb_write=False,
duckdb_read=False), the gateway write path produces a parquet that is
content-equivalent to a naive `df.to_parquet(..., index=False)` write.

This must pass before any fetch script can be refactored to call write_l0.
A break here means the gateway has drifted and existing parquet consumers
(L1 builders, validator baselines, archived training datamarts) may see
different data after refactor.

For each registered schema, the test:
  1. Reads the production parquet at schema.parquet_path as the baseline DF.
     If the file is absent (clean CI checkout), the source is skipped.
  2. Writes that DF via two paths into a temp directory:
       a) naive: df.to_parquet(naive_path, index=False)
       b) gateway: write_l0(source, df, cfg=defaults) into a redirected path
  3. Re-reads both parquets and compares with assert_frame_equal.

Run:
    pytest pipeline/storage/tests/test_continuity.py -v
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from pipeline.storage import schemas as schemas_pkg
from pipeline.storage.config import L0SourceConfig
from pipeline.storage.schemas import SCHEMAS
from pipeline.storage.writers import write_l0


def _load_baseline(parquet_path: str) -> pd.DataFrame:
    p = Path(parquet_path)
    if not p.exists():
        pytest.skip(
            f"{p} not present — continuity gate requires production parquet "
            f"as baseline. Run a real fetch first or check out data/."
        )
    return pd.read_parquet(p)


@pytest.mark.parametrize("source", sorted(SCHEMAS.keys()))
def test_default_flags_match_naive_parquet(source, tmp_path, monkeypatch):
    schema = SCHEMAS[source]
    baseline_df = _load_baseline(schema.parquet_path)

    naive_path = tmp_path / f"naive_{source}.parquet"
    baseline_df.to_parquet(naive_path, index=False)

    gateway_path = tmp_path / f"gateway_{source}.parquet"
    redirected = replace(schema, parquet_path=str(gateway_path))
    monkeypatch.setitem(schemas_pkg.SCHEMAS, source, redirected)

    cfg_default = L0SourceConfig(source=source)
    write_l0(source, baseline_df, cfg=cfg_default)

    assert gateway_path.exists(), "gateway must produce parquet at default flags"

    naive_df = pd.read_parquet(naive_path)
    gateway_df = pd.read_parquet(gateway_path)
    pd.testing.assert_frame_equal(naive_df, gateway_df)


@pytest.mark.parametrize("source", sorted(SCHEMAS.keys()))
def test_default_flags_skip_duckdb(source, tmp_path, monkeypatch):
    """At default flags, no DuckDB file must be created."""
    schema = SCHEMAS[source]
    baseline_df = _load_baseline(schema.parquet_path)

    gateway_path = tmp_path / f"gateway_{source}.parquet"
    redirected = replace(schema, parquet_path=str(gateway_path))
    monkeypatch.setitem(schemas_pkg.SCHEMAS, source, redirected)

    import pipeline.storage.l0 as l0
    monkeypatch.setattr(l0, "L0_DIR", tmp_path)

    cfg_default = L0SourceConfig(source=source)
    write_l0(source, baseline_df, cfg=cfg_default)

    duckdb_files = list(tmp_path.glob("*.duckdb"))
    assert duckdb_files == [], (
        f"default flags must not write DuckDB, found: {duckdb_files}"
    )
