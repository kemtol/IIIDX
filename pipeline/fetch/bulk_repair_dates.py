#!/usr/bin/env python3
"""Repair all corrupted broker dates using DuckDB staging."""
import sys, asyncio, duckdb, time
from pathlib import Path

sys.path.insert(0, str(Path("pipeline/fetch")))
import fetch_broksum_ipot as fetcher

REPO = Path(".")
TEMP_DB = REPO / "data/Level_0_Raw/_TEMP_broksum.duckdb"
FINAL = REPO / "data/Level_0_Raw/broksum_bybroker.parquet"

DATES = ["2026-04-14", "2026-04-16", "2026-04-27", "2026-04-29"]
COLS = ['broker','stock_code','date','buy_val','sell_val','net_val','total_val',
        'buy_vol','sell_vol','net_vol','buy_freq','sell_freq']

brokers = fetcher.load_brokers_from_master(
    REPO / "data/Level_0_Raw/master_broker.parquet", include_inactive=False)
print(f"Repairing {len(DATES)} dates for {len(brokers)} brokers")
for d in DATES:
    print(f"  {d}")

db = duckdb.connect(str(TEMP_DB))

async def scrape_date(target: str):
    """Scrape all brokers for one date, INSERT to DuckDB."""
    total = 0
    async with fetcher.IPOTScraper() as scraper:
        for i, broker in enumerate(brokers):
            try:
                r = await scraper.scrape_broker_date(broker, target)
                if r and r.get("status") == "success":
                    stocks = r.get("stocks", [])
                    if stocks:
                        db.execute("DELETE FROM broker_staging WHERE date=? AND broker=?",
                                   [target, broker])
                        for s in stocks:
                            v = [broker, s['stock_code'], target,
                                 float(s.get('buy_val',0)), float(s.get('sell_val',0)),
                                 float(s.get('net_val',0)), float(s.get('total_val',0)),
                                 float(s.get('buy_vol',0)), float(s.get('sell_vol',0)),
                                 float(s.get('net_vol',0)),
                                 int(s.get('buy_freq',0)), int(s.get('sell_freq',0))]
                            db.execute(
                                f"INSERT INTO broker_staging ({','.join(COLS)}) "
                                f"VALUES ({','.join('?'*12)})", v)
                        total += len(stocks)
            except Exception as e:
                pass
            await asyncio.sleep(0.05)
    return total

t0 = time.time()
for date in DATES:
    print(f"\n[repair] Scraping {date}...")
    n = asyncio.run(scrape_date(date))
    brs = db.execute("SELECT COUNT(DISTINCT broker) FROM broker_staging WHERE date=?", [date]).fetchone()[0]
    print(f"  {date}: {n:,} rows, {brs} brokers | {'✅' if n >= 6000 and brs >= 80 else '⚠️'}")

print(f"\n[export] Writing to parquet...")
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq

existing = pd.read_parquet(FINAL)
for date in DATES:
    existing = existing[existing['date'] != date]

new_rows = db.execute(f"SELECT * FROM broker_staging WHERE date IN ({','.join(repr(d) for d in DATES)})").df()
result = pd.concat([existing, new_rows], ignore_index=True)
result = result.sort_values(['date','broker','stock_code'])

table = pa.Table.from_pandas(result)
pq.write_table(table, FINAL, compression='zstd', use_dictionary=True)

elapsed = time.time() - t0
print(f"[export] {len(result):,} total rows | {elapsed:.0f}s")

# Quick verify
for date in DATES:
    rows = len(result[result['date'] == date])
    brs = result[result['date'] == date]['broker'].nunique()
    flag = '✅' if rows >= 6000 and brs >= 80 else '⚠️'
    print(f"  {flag} {date}: {rows:,} rows, {brs} brokers")

db.close()
