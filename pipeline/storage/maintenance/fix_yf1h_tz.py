import pandas as pd
from pathlib import Path
import sys

def fix_yfinance_1h_tz(parquet_path: str):
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    cutoff_start = pd.Timestamp("2023-03-06", tz="UTC")
    cutoff_end = pd.Timestamp("2026-04-28", tz="UTC")
    
    print(f"Original row count: {len(df):,}")
    
    if df['datetime'].dt.tz is None:
        print("Converting naive datetime to UTC...")
        df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize('UTC')
    
    mask = (df['datetime'] >= cutoff_start) & (df['datetime'] < cutoff_end)
    corrupted_count = mask.sum()
    print(f"Found {corrupted_count:,} rows in corrupted range.")
    
    if corrupted_count == 0:
        print("No corrupted rows found. Exiting.")
        return

    print("Applying -7h shift to corrupted rows...")
    df.loc[mask, 'datetime'] = df.loc[mask, 'datetime'] - pd.Timedelta(hours=7)
    
    print("Deduplicating and sorting...")
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['ticker', 'datetime'], keep='last')
    df = df.sort_values(['ticker', 'datetime']).reset_index(drop=True)
    after_dedup = len(df)
    
    print(f"Rows removed during dedup: {before_dedup - after_dedup:,}")
    print(f"Final row count: {len(df):,}")
    
    print(f"Writing fixed data to {parquet_path}...")
    df.to_parquet(parquet_path, index=False)
    print("Done.")

if __name__ == "__main__":
    path = "data/Level_0_Raw/yfinance_1h.parquet"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    fix_yfinance_1h_tz(path)
