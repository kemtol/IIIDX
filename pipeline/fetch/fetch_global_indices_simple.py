#!/usr/bin/env python3
"""
Fetch global indices + macro signals via yfinance.

Symbols fetched:
  ^IXIC   — NASDAQ Composite
  ^N225   — NIKKEI 225
  ^VIX    — CBOE Volatility Index (fear gauge)
  IDR=X   — USD/IDR exchange rate (rupiah strength)
  ^JKSE   — IHSG (Jakarta Composite, used for HMM regime)

Output: idx/data/Level_0_Raw/global_indices.parquet
Schema: date | symbol | close
"""

import sys
import subprocess

try:
    import yfinance as yf
except ImportError:
    print("[Setup] Installing yfinance...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "yfinance", "-q"])
    import yfinance as yf

import pandas as pd
from pathlib import Path

# pipeline/fetch/<file>.py → parents[2] = idx/
IDX_DIR = Path(__file__).resolve().parents[2]
OUTPUT_PATH = IDX_DIR / "data" / "Level_0_Raw" / "global_indices.parquet"
_SRC = "global_indices"

SYMBOLS = {
    "^IXIC": "NASDAQ",
    "^N225": "NIKKEI",
    "^VIX":  "VIX",
    "IDR=X": "USDIDR",
    "^JKSE": "IHSG",
}


def _fetch_with_staging() -> pd.DataFrame:
    """Crash-resilient variant using DuckDB staging (same pattern as yf_daily)."""
    from pipeline.storage.staging import open_staging

    print(f"\n=== Fetch global indices -> {OUTPUT_PATH} (staging mode) ===", flush=True)

    # Load existing parquet
    existing = pd.read_parquet(OUTPUT_PATH) if OUTPUT_PATH.exists() else pd.DataFrame()
    if not existing.empty:
        existing['date'] = pd.to_datetime(existing['date'])

    with open_staging(_SRC) as stage:
        last_map = stage.last_seen_per_ticker(
            base_df=existing, ticker_col="symbol", time_col="date",
        )
        # Normalize to tz-naive for comparison
        last_map = {k: v.tz_localize(None) if getattr(v, 'tz', None) else v
                    for k, v in last_map.items()}

        for symbol, name in SYMBOLS.items():
            print(f"[Fetch] {symbol} ({name})...")
            try:
                df = yf.download(symbol, period="5y", interval="1d",
                                 progress=False, auto_adjust=True)
                if df.empty:
                    print(f"  [Warning] No data for {symbol}")
                    continue

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df.reset_index()
                date_col = "Date" if "Date" in df.columns else "Datetime"
                df = df[[date_col, "Close"]].copy()
                df.columns = ["date", "close"]
                df["symbol"] = symbol
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                df = df.dropna(subset=["date", "close"])

                stage.append(df)
                print(f"  -> {len(df)} rows | {df['date'].min().date()} -> {df['date'].max().date()}")
            except Exception as e:
                print(f"  [Error] {symbol}: {e}")

        if stage.staged_count() == 0:
            print("  [Warning] No new data staged")
            return existing

        # Normalize existing date to tz-aware for commit
        if not existing.empty:
            existing['date'] = pd.to_datetime(existing['date'], utc=True).dt.tz_convert('Asia/Jakarta')

        rows = stage.commit_to_l0(base_df=existing)
        stage.clear()

    # Post-commit: normalize parquet to tz-naive (pandas-compatible)
    import duckdb, pyarrow.parquet as pq, pyarrow as pa
    norm_db = duckdb.connect(':memory:')
    norm_db.execute(f"""
        CREATE TABLE t AS
        SELECT date::DATE as date, symbol, close
        FROM read_parquet('{OUTPUT_PATH}')
    """)
    result = norm_db.execute("SELECT * FROM t ORDER BY symbol, date").df()
    result['date'] = pd.to_datetime(result['date'])
    pq.write_table(pa.Table.from_pandas(result), OUTPUT_PATH, compression='zstd')
    norm_db.close()

    print(f"\n[Done] {OUTPUT_PATH}")
    print(f"  rows={len(result):,}")
    print(f"  symbols={sorted(result['symbol'].unique())}")
    print(f"  date_range={result['date'].min().date()} -> {result['date'].max().date()}")
    return result


def fetch_global_indices() -> pd.DataFrame:
    all_data = []

    for symbol, name in SYMBOLS.items():
        print(f"[Fetch] {symbol} ({name})...")
        try:
            df = yf.download(symbol, period="5y", interval="1d", progress=False, auto_adjust=True)
            if df.empty:
                print(f"[Warning] No data for {symbol}")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()

            # yfinance sometimes returns "Date" or "Datetime" as index name
            date_col = "Date" if "Date" in df.columns else "Datetime"
            df = df[[date_col, "Close"]].copy()
            df.columns = ["date", "close"]
            df["symbol"] = symbol
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df = df.dropna(subset=["date", "close"])

            all_data.append(df)
            print(f"  → {len(df):,} rows | {df['date'].min().date()} → {df['date'].max().date()}")

        except Exception as e:
            print(f"[Error] {symbol}: {e}")

    if not all_data:
        raise RuntimeError("Failed to fetch any global indices")

    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)

    print(f"\n[Done] {OUTPUT_PATH}")
    print(f"  rows={len(combined):,}")
    print(f"  symbols={combined['symbol'].unique().tolist()}")
    print(f"  date_range={combined['date'].min().date()} → {combined['date'].max().date()}")

    return combined


if __name__ == "__main__":
    # Staging-mode dispatch
    try:
        from pipeline.storage.schemas import SCHEMAS
        if _SRC in SCHEMAS:
            _fetch_with_staging()
            raise SystemExit(0)
    except ImportError:
        pass
    fetch_global_indices()
