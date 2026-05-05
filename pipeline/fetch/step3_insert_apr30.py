#!/usr/bin/env python3
"""Step 3: Scrape Apr 30 broker-by-broker, INSERT immediately to DuckDB."""
import sys, asyncio, duckdb, time
from pathlib import Path

sys.path.insert(0, str(Path("pipeline/fetch")))
import fetch_broksum_ipot as fetcher

REPO = Path(".")
TARGET_DATE = "2026-04-30"
TEMP_DB = REPO / "data/Level_0_Raw/_TEMP_broksum.duckdb"

brokers = fetcher.load_brokers_from_master(
    REPO / "data/Level_0_Raw/master_broker.parquet", include_inactive=False
)
print(f"Target: {TARGET_DATE} | Brokers: {len(brokers)}")

db = duckdb.connect(str(TEMP_DB))
total_inserted = 0
total_brokers_done = 0
t0 = time.time()

async def run():
    global total_inserted, total_brokers_done
    async with fetcher.IPOTScraper() as scraper:
        for broker in brokers:
            try:
                result = await scraper.scrape_broker_date(broker, TARGET_DATE)
                if result and result.get("status") == "success":
                    stocks = result.get("stocks", [])
                    if stocks:
                        # Build values list for INSERT
                        for s in stocks:
                            s['broker'] = broker
                            s['date'] = TARGET_DATE
                        
                        # INSERT matching columns from staging
                        cols = ['broker','stock_code','date','buy_val','sell_val',
                                'net_val','total_val','buy_vol','sell_vol','net_vol',
                                'total_vol','buy_freq','sell_freq']
                        
                        values = []
                        for s in stocks:
                            row = []
                            for c in cols:
                                v = s.get(c, 0)
                                if isinstance(v, str) and c not in ('broker','stock_code','date'):
                                    v = float(v) if '.' in v else int(v)
                                row.append(v)
                            values.append(row)
                        
                        # Delete old + insert new for this broker+date
                        db.execute("DELETE FROM broker_staging WHERE date=? AND broker=?",
                                   [TARGET_DATE, broker])
                        
                        placeholders = ','.join(['(?,?,?,?,?,?,?,?,?,?,?,?,?)'] * len(values))
                        flat = [x for row in values for x in row]
                        db.execute(f"INSERT INTO broker_staging VALUES {placeholders}", flat)
                        
                        total_inserted += len(stocks)
                        total_brokers_done += 1
                        
            except Exception as e:
                pass
            await asyncio.sleep(0.05)
            
            if total_brokers_done > 0 and total_brokers_done % 15 == 0:
                elapsed = time.time() - t0
                print(f"  [{total_brokers_done}/{len(brokers)}] {total_inserted:,} stocks | ETA {elapsed/total_brokers_done*(len(brokers)-total_brokers_done):.0f}s")

asyncio.run(run())

# Verify
total = db.execute(f"SELECT COUNT(*) FROM broker_staging WHERE date='{TARGET_DATE}'").fetchone()[0]
brs = db.execute(f"SELECT COUNT(DISTINCT broker) FROM broker_staging WHERE date='{TARGET_DATE}'").fetchone()[0]
tks = db.execute(f"SELECT COUNT(DISTINCT stock_code) FROM broker_staging WHERE date='{TARGET_DATE}'").fetchone()[0]

print(f"\n[verify] Apr 30 in DuckDB:")
print(f"  Rows: {total:,}")
print(f"  Brokers: {brs}")
print(f"  Tickers: {tks}")
print(f"  Elapsed: {time.time()-t0:.0f}s")
print(f"  Status: {'✅ COMPLETE' if total >= 6000 and brs >= 80 else '⚠️ PARTIAL'}")

db.close()
