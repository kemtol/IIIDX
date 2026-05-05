"""L0 source registry and per-source canary feature flags.

Stage progression per PRD 0003 §12.2:
  Stage 0 (pre):         parquet_write=T, duckdb_write=F, duckdb_read=F  ← default
  Stage 1 (dual-write):  parquet_write=T, duckdb_write=T, duckdb_read=F
  Stage 2 (read switch): parquet_write=T, duckdb_write=T, duckdb_read=T
  Stage 3 (cutover):     parquet_write=F, duckdb_write=T, duckdb_read=T

Promotion = set env var. Rollback = unset env var. No code change.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

L0_SOURCES: tuple[str, ...] = (
    "master_emiten",
    "master_broker",
    "global_indices",
    "yfinance_daily",
    "yfinance_1h",
    "yfinance_4h",
    "broksum_bybroker",
)

_TRUE_TOKENS = frozenset({"true", "1", "yes", "on"})


@dataclass(frozen=True)
class L0SourceConfig:
    source: str
    parquet_write: bool = True
    duckdb_write: bool = False
    duckdb_read: bool = False

    def __post_init__(self) -> None:
        if not (self.parquet_write or self.duckdb_write):
            raise ValueError(
                f"L0 source {self.source!r}: both parquet_write and duckdb_write "
                f"are False — no writer would persist data."
            )


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_TOKENS


def _env_key(source: str, flag: str) -> str:
    return f"L0_{source.upper()}_{flag.upper()}"


def load_config() -> dict[str, L0SourceConfig]:
    """Read per-source canary flags from environment.

    Env var pattern: L0_<SOURCE_UPPER>_<FLAG_UPPER>
        L0_MASTER_EMITEN_PARQUET_WRITE   (default: true)
        L0_MASTER_EMITEN_DUCKDB_WRITE    (default: false)
        L0_MASTER_EMITEN_DUCKDB_READ     (default: false)

    Truthy values: true / 1 / yes / on (case-insensitive). Anything else = False.
    """
    out: dict[str, L0SourceConfig] = {}
    for src in L0_SOURCES:
        out[src] = L0SourceConfig(
            source=src,
            parquet_write=_parse_bool(os.environ.get(_env_key(src, "PARQUET_WRITE")), True),
            duckdb_write=_parse_bool(os.environ.get(_env_key(src, "DUCKDB_WRITE")), False),
            duckdb_read=_parse_bool(os.environ.get(_env_key(src, "DUCKDB_READ")), False),
        )
    return out
