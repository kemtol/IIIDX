"""L0 DuckDB backup with weekly/monthly/yearly retention tiers.

Per PRD 0003 §6.1:
  Weekly  — every Sunday 02:00 WIB.            Keep last 4.
  Monthly — first Sunday of month.             Keep last 12.
  Yearly  — first Sunday of January.           Never delete (audit archive).

Each backup runs DuckDB `EXPORT DATABASE '<dir>' (FORMAT PARQUET)`, dumping the
schema + every table into a directory of parquet files. Restore with
`IMPORT DATABASE '<dir>'`.

Cron is daily; the script no-ops on non-Sunday and runs 1–3 tiers on Sundays.
"""
from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import duckdb

from .l0 import l0_path

BACKUP_ROOT = Path("_BAK")

WEEKLY_KEEP = 4
MONTHLY_KEEP = 12

_SUNDAY = 6  # date.weekday() returns 6 for Sunday


def _is_first_sunday_of_month(d: date) -> bool:
    return d.weekday() == _SUNDAY and d.day <= 7


def _is_first_sunday_of_january(d: date) -> bool:
    return d.month == 1 and _is_first_sunday_of_month(d)


def tiers_for_date(d: date) -> list[str]:
    """Which backup tiers should run on this date. Empty list = no-op."""
    if d.weekday() != _SUNDAY:
        return []
    tiers = ["weekly"]
    if _is_first_sunday_of_month(d):
        tiers.append("monthly")
    if _is_first_sunday_of_january(d):
        tiers.append("yearly")
    return tiers


def _tier_key(tier: str, d: date) -> str:
    if tier == "weekly":
        return d.strftime("%Y%m%d")
    if tier == "monthly":
        return d.strftime("%Y%m")
    if tier == "yearly":
        return d.strftime("%Y")
    raise ValueError(f"unknown tier: {tier!r}")


def backup_path(
    db_file: str,
    tier: str,
    key: str,
    root: Path = BACKUP_ROOT,
) -> Path:
    stem = db_file.removesuffix(".duckdb")
    return root / f"duckdb_{stem}_{tier}_{key}"


def export_database(db_file: str, target_dir: Path) -> None:
    """Dump a single L0 DB to a parquet directory via EXPORT DATABASE."""
    src = l0_path(db_file)
    if not src.exists():
        raise FileNotFoundError(f"L0 DB does not exist: {src}")
    target_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(src), read_only=True)
    try:
        con.execute(f"EXPORT DATABASE '{target_dir}' (FORMAT PARQUET)")
    finally:
        con.close()


def apply_retention(
    db_file: str,
    tier: str,
    root: Path = BACKUP_ROOT,
) -> list[Path]:
    """Delete backups beyond retention. Returns deleted directories.

    Yearly tier never deletes — full audit archive.
    """
    if tier == "yearly":
        return []
    keep = {"weekly": WEEKLY_KEEP, "monthly": MONTHLY_KEEP}[tier]
    stem = db_file.removesuffix(".duckdb")
    pattern = f"duckdb_{stem}_{tier}_*"
    matches = sorted(root.glob(pattern), key=lambda p: p.name)
    to_delete = matches[:-keep] if len(matches) > keep else []
    for d in to_delete:
        if d.is_dir():
            shutil.rmtree(d)
    return list(to_delete)


def run_backup(
    db_file: str,
    today: date | None = None,
    root: Path = BACKUP_ROOT,
) -> dict:
    """Run applicable tier backups for the given date. No-op on non-Sunday."""
    if today is None:
        today = date.today()
    tiers = tiers_for_date(today)
    summary: dict = {
        "db_file": db_file,
        "date": today.isoformat(),
        "tiers_run": [],
        "deleted": [],
    }
    for tier in tiers:
        key = _tier_key(tier, today)
        target = backup_path(db_file, tier, key, root=root)
        export_database(db_file, target)
        summary["tiers_run"].append({"tier": tier, "key": key, "path": str(target)})
        for d in apply_retention(db_file, tier, root=root):
            summary["deleted"].append(str(d))
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry: iterate L0_DB_FILES, backup ones that exist on disk.

    No-op on non-Sunday (tiers_for_date returns []). Missing files are skipped
    quietly — Phase 0+ rolls out one DB at a time.
    """
    import json
    from .l0 import L0_DB_FILES, l0_path

    today = date.today()
    overall = {"date": today.isoformat(), "files_processed": [], "files_skipped": []}
    for db_file in L0_DB_FILES:
        if not l0_path(db_file).exists():
            overall["files_skipped"].append({"db_file": db_file, "reason": "file not present"})
            continue
        overall["files_processed"].append(run_backup(db_file, today=today))
    print(json.dumps(overall, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
