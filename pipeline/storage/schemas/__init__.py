"""Schema registry for L0 sources.

Each source has one schema module (e.g. master_emiten.py) that exports
`SCHEMA: TableSchema`. This module collects them into `SCHEMAS` keyed by
source name (matching `pipeline.storage.config.L0_SOURCES`).
"""
from __future__ import annotations

from . import global_indices as _global_indices
from . import master_broker as _master_broker
from . import master_emiten as _master_emiten
from . import yfinance_1h as _yfinance_1h
from . import yfinance_daily as _yfinance_daily
from ._base import AUDIT_INSERTED, AUDIT_UPDATED, ColumnDef, TableSchema

SCHEMAS: dict[str, TableSchema] = {
    _global_indices.SCHEMA.source: _global_indices.SCHEMA,
    _master_emiten.SCHEMA.source: _master_emiten.SCHEMA,
    _master_broker.SCHEMA.source: _master_broker.SCHEMA,
    _yfinance_1h.SCHEMA.source: _yfinance_1h.SCHEMA,
    _yfinance_daily.SCHEMA.source: _yfinance_daily.SCHEMA,
}


def get_schema(source: str) -> TableSchema:
    try:
        return SCHEMAS[source]
    except KeyError as e:
        raise KeyError(
            f"No schema registered for source {source!r}; known: {sorted(SCHEMAS)}"
        ) from e


__all__ = [
    "AUDIT_INSERTED",
    "AUDIT_UPDATED",
    "ColumnDef",
    "SCHEMAS",
    "TableSchema",
    "get_schema",
]
