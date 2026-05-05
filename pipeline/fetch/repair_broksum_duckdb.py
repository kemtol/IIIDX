#!/usr/bin/env python3
"""
Broker data repair via iterative fetch + verify.
1. Create TEMP DuckDB from existing parquet (fast metadata check)
2. Detect corrupted/missing dates
3. Run broker fetcher with --repair-days (standard Python)
4. Re-import to DuckDB, verify
5. Repeat if still corrupted

Usage:
  python pipeline/fetch/repair_broksum_duckdb.py [--repair-days 14] [--max-retry 3]
"""

import sys
import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta

REPO_ROOT = Path(__file__).resolve().parents[3]
IDX_DIR = REPO_ROOT / "idx"
L0_DIR = IDX_DIR / "data" / "Level_0_Raw"
FINAL_PARQUET = L0_DIR / "broksum_bybroker.parquet"
LOOKBACK_DAYS = 30
HOLIDAYS = {"2026-04-03", "2026-05-01"}


def load_parquet_to_duckdb() -> duckdb.DuckDBPyConnection:
    """Quick metadata scan of parquet into in-memory DuckDB (no file)."""
    db = duckdb.connect(":memory:")
    db.execute(f"CREATE TABLE broker AS SELECT * FROM read_parquet('{FINAL_PARQUET}')")
    return db


def detect_corruption(db: duckdb.DuckDBPyConnection) -> list[tuple[str, str, int, int]]:
    """Find trading days with missing or incomplete broker data."""
    cutoff = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    
    all_dates = []
    d = datetime.strptime(cutoff, "%Y-%m-%d").date()
    today = date.today()
    while d <= today:
        if d.weekday() < 5 and d.strftime("%Y-%m-%d") not in HOLIDAYS:
            all_dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    
    corrupted = []
    for dt in all_dates:
        if dt == today.strftime("%Y-%m-%d"):
            continue
        cnt = db.execute("SELECT COUNT(*) FROM broker WHERE date = ?", [dt]).fetchone()[0]
        brokers = db.execute("SELECT COUNT(DISTINCT broker) FROM broker WHERE date = ?", [dt]).fetchone()[0]
        if cnt == 0:
            corrupted.append((dt, "MISSING", cnt, brokers))
        elif cnt < 3000 or brokers < 50:
            corrupted.append((dt, "PARTIAL", cnt, brokers))
    
    return corrupted


def run_fetcher(repair_days: int, brokers: str = ""):
    """Run the existing broker fetcher (writes to parquet)."""
    import asyncio
    sys.path.insert(0, str(IDX_DIR / "pipeline" / "fetch"))
    
    sys.argv = [
        "fetch_broksum_ipot.py",
        "--all-brokers",
        "--days", "1",
        "--repair-days", str(repair_days),
        "--update-mode", "append",
        "--max-concurrent", "8",
        "--day-sleep", "0.05",
        "--output", str(FINAL_PARQUET),
    ]
    if brokers:
        sys.argv.extend(["--brokers", brokers])
    
    import fetch_broksum_ipot
    asyncio.run(fetch_broksum_ipot.main())


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-days", type=int, default=14)
    parser.add_argument("--max-retry", type=int, default=3)
    parser.add_argument("--brokers", type=str, default="")
    args = parser.parse_args()
    
    for attempt in range(1, args.max_retry + 1):
        print(f"\n{'='*50}")
        print(f" REPAIR ATTEMPT {attempt}/{args.max_retry}")
        print(f"{'='*50}")
        
        # Phase 1: Detect
        print(f"\n[detect] Loading parquet metadata (last {LOOKBACK_DAYS}d)...")
        db = load_parquet_to_duckdb()
        corrupted = detect_corruption(db)
        db.close()
        
        healthy = max(0, len([
            d for d in range(LOOKBACK_DAYS) 
            if datetime.strptime(str(date.today() - timedelta(days=LOOKBACK_DAYS-d)), "%Y-%m-%d").date() not in set(dt for dt, _, _, _ in corrupted)
        ]))
        
        print(f"[detect] Healthy: {healthy}  |  Corrupted: {len(corrupted)}")
        for dt, reason, cnt, brokers in corrupted:
            print(f"  {dt}: {reason} ({cnt} rows, {brokers} brokers)")
        
        if not corrupted:
            print(f"\n✅ NO CORRUPTION. Parquet is clean.")
            return
        
        # Phase 2: Repair
        print(f"\n[repair] Running broker fetcher...")
        run_fetcher(args.repair_days, args.brokers)
        
        # Phase 3: Re-verify
        print(f"\n[verify] Re-checking after repair...")
        db = load_parquet_to_duckdb()
        still_corrupted = detect_corruption(db)
        new = [(d, r) for d, r, _, _ in still_corrupted 
               if (d, r) not in [(x[0], x[1]) for x in corrupted]]
        fixed = len(corrupted) - len(still_corrupted)
        db.close()
        
        if not still_corrupted:
            print(f"\n✅ ALL DATES REPAIRED. Parquet clean after {attempt} attempt(s).")
            return
        
        print(f"[verify] Fixed: {fixed}  |  Still corrupted: {len(still_corrupted)}")
        if new:
            print("  New issues:")
            for d, r in new:
                print(f"    {d}: {r}")
    
    print(f"\n❌ MAX RETRIES REACHED. {len(still_corrupted)} dates still corrupted.")
    sys.exit(1)


if __name__ == "__main__":
    main()
