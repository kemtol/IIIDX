"""
fetch.py — Update the BPJS inference DB with today's features.

NOTE: BPJS is currently PAUSED. This script is scaffolded but not in production.

Usage
-----
    python fetch.py [--date YYYY-MM-DD] [--force] [--skip-l1] [--skip-l2]
"""
import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    L1_SCRIPT, L2_SCRIPT, L2_PARQUET,
    L1_WARMUP_MULTIPLIER, LOG_DIR,
)
from variants.v16b import MODEL_PATH
import db as inference_db


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--date",     default=str(date.today()), help="Target trading date YYYY-MM-DD")
    p.add_argument("--force",    action="store_true",        help="Re-run even if date already in DB")
    p.add_argument("--skip-l1", action="store_true",         help="Skip L1 pipeline")
    p.add_argument("--skip-l2", action="store_true",         help="Skip L2 pipeline")
    return p.parse_args()


def run_l1(target_date: str) -> None:
    from datetime import datetime, timedelta
    dt        = datetime.strptime(target_date, "%Y-%m-%d")
    date_from = (dt - timedelta(days=2)).strftime("%Y-%m-%d")
    cmd = [
        sys.executable, str(L1_SCRIPT),
        "--date-from", date_from,
        "--warmup-bdays-multiplier", str(L1_WARMUP_MULTIPLIER),
        "--skip-refresh-broksum",
        "--skip-refresh-yf-intraday",
        "--skip-refresh-yf-daily",
    ]
    print(f"[fetch] Running L1: {' '.join(cmd[-6:])}")
    subprocess.run(cmd, check=True)


def run_l2() -> None:
    cmd = [sys.executable, str(L2_SCRIPT)]
    print(f"[fetch] Running L2: {L2_SCRIPT.name}")
    subprocess.run(cmd, check=True)


def extract_inference_rows(target_date: str, feature_names: list[str]) -> pd.DataFrame:
    dm = pd.read_parquet(
        L2_PARQUET,
        filters=[("date", "=", pd.Timestamp(target_date))],
        columns=["date", "ticker"] + feature_names,
    )
    print(f"[fetch] Extracted {len(dm)} rows for {target_date}")
    return dm


def main() -> None:
    args        = parse_args()
    target_date = args.date

    model         = lgb.Booster(model_file=str(MODEL_PATH))
    feature_names = model.feature_name()

    con = inference_db.connect()
    inference_db.init_schema(con, feature_names)

    latest = inference_db.get_latest_date(con)
    if str(latest) == target_date and not args.force:
        print(f"[fetch] {target_date} already in DB — skipping (use --force to override)")
        return

    if not args.skip_l1:
        run_l1(target_date)
    if not args.skip_l2:
        run_l2()

    df = extract_inference_rows(target_date, feature_names)
    if df.empty:
        print(f"[fetch] WARNING: no rows for {target_date} in L2 — market holiday?")
        return

    # TODO: Phase 2 — call inference_db.upsert_features(con, df)
    print(f"[fetch] TODO: upsert {len(df)} rows into inference.duckdb")

    pruned = inference_db.prune_old_rows(con)
    print(f"[fetch] Pruned {pruned} old rows from DB")
    con.close()


if __name__ == "__main__":
    main()
