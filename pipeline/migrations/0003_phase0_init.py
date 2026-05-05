"""PRD 0003 Phase 0: CREATE TABLE for every registered L0 schema.

No data load — that happens per-source in `0003_phase{1..5}_*.py`. Running this
script is safe and idempotent: tables already created stay untouched.

Usage:
    python pipeline/migrations/0003_phase0_init.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Filename starts with a digit so `python -m` is unavailable; ensure repo root
# is on sys.path when the script runs as `python pipeline/migrations/...`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.storage.migrate import init_all  # noqa: E402


def up() -> dict[str, list[str]]:
    return init_all()


def main() -> int:
    summary = up()
    if not summary:
        print("No schemas registered. Nothing to do.")
        return 0
    for db_file, tables in summary.items():
        print(f"{db_file}: {tables}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
