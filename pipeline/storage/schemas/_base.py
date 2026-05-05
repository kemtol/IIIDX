"""Schema definition primitives for L0 DuckDB tables.

One concrete schema module per source under pipeline/storage/schemas/.
Audit columns (_inserted_at, _updated_at) are handled internally — schema
authors declare user columns only.

Per PRD 0003 §4.2:
  - _inserted_at set on first insert, NOT overwritten on conflict
  - _updated_at set on every insert AND every conflict-update
  - Both populated by app layer (no DB triggers)

Note: PRD examples write `CURRENT_TIMESTAMP`, but DuckDB 1.5.2 misparses
that token in the `ON CONFLICT DO UPDATE SET` clause as a column ref. We
emit `now()` instead — semantically identical (both return TIMESTAMPTZ).
"""
from __future__ import annotations

from dataclasses import dataclass

AUDIT_INSERTED = "_inserted_at"
AUDIT_UPDATED = "_updated_at"


@dataclass(frozen=True)
class ColumnDef:
    name: str
    duckdb_type: str
    nullable: bool = True


@dataclass(frozen=True)
class TableSchema:
    source: str
    db_file: str
    table: str
    parquet_path: str
    natural_key: tuple[str, ...]
    columns: tuple[ColumnDef, ...]

    def __post_init__(self) -> None:
        col_names = [c.name for c in self.columns]
        if len(set(col_names)) != len(col_names):
            raise ValueError(f"{self.table}: duplicate column names")
        for pk in self.natural_key:
            if pk not in col_names:
                raise ValueError(f"{self.table}: natural key {pk!r} not in columns")
        for reserved in (AUDIT_INSERTED, AUDIT_UPDATED):
            if reserved in col_names:
                raise ValueError(
                    f"{self.table}: {reserved!r} is reserved — audit cols are "
                    f"appended internally, do not declare them in `columns`"
                )

    @property
    def value_columns(self) -> tuple[str, ...]:
        """User columns excluding natural key. These get UPDATE-on-conflict."""
        return tuple(c.name for c in self.columns if c.name not in self.natural_key)

    @property
    def all_user_columns(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def create_table_sql(self) -> str:
        col_lines = []
        for c in self.columns:
            null_clause = "" if c.nullable else " NOT NULL"
            col_lines.append(f'  {_q(c.name)} {c.duckdb_type}{null_clause}')
        col_lines.append(f"  {AUDIT_INSERTED} TIMESTAMP NOT NULL")
        col_lines.append(f"  {AUDIT_UPDATED} TIMESTAMP NOT NULL")
        pk = ", ".join(_q(k) for k in self.natural_key)
        col_lines.append(f"  PRIMARY KEY ({pk})")
        body = ",\n".join(col_lines)
        return f"CREATE TABLE IF NOT EXISTS {self.table} (\n{body}\n)"

    def upsert_sql(self, rel: str = "df") -> str:
        """Generate INSERT ... ON CONFLICT DO UPDATE.

        `rel` is the DuckDB relation/view name to read rows from (e.g. a
        registered DataFrame). The relation must expose the user columns
        listed in `self.columns`; audit timestamps are filled by SQL.

        All identifiers are double-quoted so column names with special chars
        (e.g. `foreignfund_%`) parse cleanly in DuckDB.
        """
        quoted_user = ", ".join(_q(c) for c in self.all_user_columns)
        target_cols = f"{quoted_user}, {AUDIT_INSERTED}, {AUDIT_UPDATED}"
        select_clause = f"SELECT {quoted_user}, now(), now() FROM {rel}"
        pk = ", ".join(_q(k) for k in self.natural_key)
        update_pairs = [f"{_q(c)} = EXCLUDED.{_q(c)}" for c in self.value_columns]
        update_pairs.append(f"{AUDIT_UPDATED} = now()")
        update_clause = ",\n  ".join(update_pairs)
        return (
            f"INSERT INTO {self.table} ({target_cols})\n"
            f"{select_clause}\n"
            f"ON CONFLICT ({pk}) DO UPDATE SET\n"
            f"  {update_clause}"
        )


def _q(identifier: str) -> str:
    """Double-quote a SQL identifier so special characters (e.g. `%`) parse."""
    return f'"{identifier}"'
