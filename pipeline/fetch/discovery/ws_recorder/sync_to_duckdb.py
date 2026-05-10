import duckdb
import sys
import os
from pathlib import Path

def sync(jsonl_path, db_path):
    if not os.path.exists(jsonl_path):
        return

    con = duckdb.connect(db_path)
    
    # Create table if not exists with FULL SCHEMA
    con.execute("""
        CREATE TABLE IF NOT EXISTS trending_raw (
            ticker VARCHAR,
            time TIME,
            close INT,
            vol BIGINT,
            val_haka DOUBLE,
            val_haki DOUBLE,
            accdist DOUBLE,
            accdist_1d DOUBLE,
            bvol BIGINT,
            svol BIGINT,
            raw_payload JSON,
            captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Incremental Load: Only load frames that aren't in the DB yet? 
    # For simplicity in discovery: we load everything into a temp and upsert, 
    # or just use DuckDB's native JSON reader capability.
    
    # We use a temporary table to parse the JSONL
    con.execute("CREATE OR REPLACE TEMP TABLE stage AS SELECT * FROM read_json_auto(?, ignore_errors=true)", [jsonl_path])
    
    # Insert only 'stream' events that are TREND_1D
    con.execute("""
        INSERT INTO trending_raw
        SELECT 
            data.code as ticker,
            data.data.data.time as time,
            data.data.data.cl as close,
            data.data.data.vol as vol,
            data.data.data.val_haka as val_haka,
            data.data.data.val_haki as val_haki,
            data.data.data.accdist_meter as accdist,
            data.data.data.accdist_meter_1d as accdist_1d,
            data.data.data.bvol as bvol,
            data.data.data.svol as svol,
            content as raw_payload,
            CURRENT_TIMESTAMP
        FROM (SELECT *, content FROM stage)
        WHERE event = 'stream' 
          AND data.rtype = 'TREND_1D'
          -- Basic dedup: don't insert if same ticker and time already exists for today
          AND NOT EXISTS (
              SELECT 1 FROM trending_raw t 
              WHERE t.ticker = stage.data.code 
                AND t.time = stage.data.data.data.time
                AND t.captured_at::DATE = CURRENT_DATE
          )
    """)
    
    count = con.execute("SELECT count(*) FROM trending_raw WHERE captured_at::DATE = CURRENT_DATE").fetchone()[0]
    print(f"[Sync] DuckDB now has {count} rows for today.")
    con.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 sync_ws.py <jsonl_path> <db_path>")
    else:
        sync(sys.argv[1], sys.argv[2])
