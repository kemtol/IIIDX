#!/usr/bin/env python3
"""
Fetch Yahoo Finance daily OHLCV for IDX tickers and upsert into parquet.

Output schema:
  - date
  - ticker
  - open
  - high
  - low
  - close
  - volume
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

_src = "yfinance_daily"


def parse_args() -> argparse.Namespace:
    service_dir = Path(__file__).resolve().parent
    idx_dir = service_dir.parent
    data_dir = idx_dir / "data" / "Level_0_Raw"
    if not data_dir.exists():
        data_dir = idx_dir / "data"
    if not data_dir.exists():
        data_dir = Path(__file__).resolve().parents[2] / "machinelearning" / "data"

    parser = argparse.ArgumentParser(description="Fetch incremental yfinance daily parquet.")
    parser.add_argument(
        "--master-path",
        type=Path,
        default=data_dir / "master_emiten.parquet",
        help="Path to master emiten parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=data_dir / "yfinance_daily.parquet",
        help="Path to output yfinance daily parquet.",
    )
    parser.add_argument(
        "--all-status",
        action="store_true",
        help="Use all tickers regardless of status. Default uses only ACTIVE when status column exists.",
    )
    parser.add_argument("--limit-tickers", type=int, default=0, help="Optional ticker limit for test runs.")
    parser.add_argument("--pause-seconds", type=float, default=0.05, help="Pause between ticker requests.")
    parser.add_argument("--min-fetch-days", type=int, default=14, help="Minimum period days to fetch per ticker.")
    parser.add_argument("--buffer-days", type=int, default=5, help="Extra overlap days to repair late bars.")
    parser.add_argument("--max-fetch-days", type=int, default=1825, help="Maximum period days to fetch per ticker.")
    parser.add_argument(
        "--extra-tickers",
        type=str,
        default="",
        help="Comma-separated list of extra tickers to fetch (e.g., '^IXIC,^N225,^DJI' for global indices).",
    )
    return parser.parse_args()


def load_tickers(master_path: Path, use_active_only: bool, limit_tickers: int, extra_tickers: list[str]) -> list[str]:
    all_tickers: list[str] = []

    # Load from master emiten if available
    if master_path.exists():
        df = pd.read_parquet(master_path)
        if "ticker" in df.columns:
            source = df
            if use_active_only and "status" in df.columns:
                source = source[source["status"].astype(str).str.upper().eq("ACTIVE")]

            t = source["ticker"].dropna().astype(str).str.upper().str.strip()
            t = t[t != ""]
            t = t.str.replace(r"[^A-Z0-9.]", "", regex=True)
            t = t.map(lambda x: x if x.endswith(".JK") else f"{x}.JK")
            all_tickers = sorted(set(t.tolist()))

    # Add extra tickers (e.g., global indices like ^IXIC, ^N225)
    if extra_tickers:
        extra_clean = [t.strip().upper() for t in extra_tickers if t.strip()]
        all_tickers = sorted(set(all_tickers + extra_clean))

    if not all_tickers:
        raise RuntimeError("No tickers found from master emiten parquet or extra-tickers")

    if limit_tickers > 0:
        all_tickers = all_tickers[:limit_tickers]
    return all_tickers


def load_existing(path: Path) -> pd.DataFrame:
    cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_parquet(path, columns=cols)
    except Exception:
        return pd.DataFrame(columns=cols)

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out = out.dropna(subset=["date", "ticker"])
    return out


def clean_daily_download(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    cols = {str(c).lower(): c for c in raw.columns}
    needed = ["open", "high", "low", "close", "volume"]
    if not all(k in cols for k in needed):
        return None

    out = raw[[cols["open"], cols["high"], cols["low"], cols["close"], cols["volume"]]].copy()
    out.columns = needed
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out = out.dropna(subset=["open", "close"])
    out = out[out["volume"].fillna(0) > 0]
    if out.empty:
        return None

    idx = pd.to_datetime(out.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Asia/Jakarta").tz_localize(None)
    out.index = idx
    out.index.name = "date"

    out["ticker"] = ticker
    out = out.reset_index()[["date", "ticker", "open", "high", "low", "close", "volume"]]
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date", "ticker"])
    return out


def fetch_daily_for_ticker(
    ticker: str,
    last_date: pd.Timestamp | None,
    *,
    min_fetch_days: int,
    buffer_days: int,
    max_fetch_days: int,
) -> tuple[pd.DataFrame | None, str]:
    today = pd.Timestamp.today().normalize()
    if last_date is not None:
        days_ago = (today - last_date).days
        if days_ago <= 0:
            return None, "skipped (up to date)"
        fetch_days = min(max(days_ago + buffer_days, min_fetch_days), max_fetch_days)
    else:
        fetch_days = max_fetch_days

    fallback_days = [max_fetch_days, 1095, 730, 365, 180, 90, 30]
    candidates = []
    for day_count in fallback_days:
        if day_count <= fetch_days and day_count not in candidates:
            candidates.append(day_count)
    if fetch_days not in candidates:
        candidates.insert(0, fetch_days)

    last_error: str | None = None
    for day_count in candidates:
        try:
            raw = yf.download(
                ticker,
                period=f"{day_count}d",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            cleaned = clean_daily_download(raw, ticker)
            if cleaned is not None and not cleaned.empty:
                return cleaned, f"ok (+{len(cleaned)} rows, {day_count}d)"
            last_error = f"no data ({day_count}d)"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
    return None, f"no data after fallback ({last_error or 'unknown'})"


def _update_daily_with_staging(
    data_dir: Path, tickers: list[str], existing: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """Crash-resilient variant using DuckDB staging (same pattern as yf 1h)."""
    from pipeline.storage.staging import open_staging

    out_path = args.output
    print(f"\n=== Fetch daily -> {out_path} (staging mode) ===", flush=True)

    errors = 0
    skips = 0
    staged_count = 0

    with open_staging(_src) as stage:
        last_map = stage.last_seen_per_ticker(base_df=existing, time_col="date")
        # Normalize to tz-naive for comparison with datetime.now()
        last_map = {k: v.tz_localize(None) if getattr(v, 'tz', None) else v
                    for k, v in last_map.items()}

        for idx, ticker in enumerate(tickers, start=1):
            last_date = last_map.get(ticker)
            new_df, status = fetch_daily_for_ticker(
                ticker, last_date,
                min_fetch_days=args.min_fetch_days,
                buffer_days=args.buffer_days,
                max_fetch_days=args.max_fetch_days,
            )
            icon = "✓" if status.startswith("ok") or status.startswith("skipped") else "✗"
            print(f"[{idx:4d}/{len(tickers)}] {icon} {ticker:<12} {status}")
            if status.startswith("skipped"):
                skips += 1
            if status.startswith("no data"):
                errors += 1
            if new_df is not None and not new_df.empty:
                stage.append(new_df)
                staged_count += 1
            if args.pause_seconds > 0:
                time.sleep(args.pause_seconds)

        existing_tz = existing.copy()
        if not existing_tz.empty:
            existing_tz['date'] = pd.to_datetime(existing_tz['date'], utc=True).dt.tz_convert('Asia/Jakarta')

        rows_written = stage.commit_to_l0(base_df=existing_tz)
        stage.clear()

    # Post-commit: normalize parquet to tz-naive (pandas-compatible)
    import pyarrow.parquet as pq, pyarrow as pa
    import duckdb
    norm_db = duckdb.connect(':memory:')
    norm_db.execute(f"CREATE TABLE t AS SELECT date::DATE as date, ticker, open, high, low, close, volume FROM read_parquet('{out_path}')")
    norm_df = norm_db.execute("SELECT * FROM t ORDER BY date, ticker").df()
    norm_df['date'] = pd.to_datetime(norm_df['date'])
    pq.write_table(pa.Table.from_pandas(norm_df), out_path, compression='zstd')
    norm_db.close()

    print(
        f"[done] daily rows={rows_written} staged={staged_count} "
        f"errors={errors} skipped={skips}",
        flush=True,
    )


def main() -> int:
    args = parse_args()

    extra_ticker_list = [t.strip() for t in args.extra_tickers.split(",")] if args.extra_tickers else []
    tickers = load_tickers(
        master_path=args.master_path,
        use_active_only=(not args.all_status),
        limit_tickers=args.limit_tickers,
        extra_tickers=extra_ticker_list,
    )
    if not tickers:
        raise RuntimeError("No tickers found from master emiten parquet")

    existing = load_existing(args.output)
    last_map: dict[str, pd.Timestamp] = {}
    if not existing.empty:
        last_map = existing.groupby("ticker", sort=False)["date"].max().to_dict()

    print(f"Master emiten : {args.master_path}")
    print(f"Output file   : {args.output}")
    print(f"Ticker count  : {len(tickers)}")
    print(f"Existing rows : {len(existing):,}")
    print(f"Start time    : {datetime.now().isoformat()}")

    # Staging-mode dispatch: use DuckDB TEMP if schema registered, else legacy
    try:
        from pipeline.storage.schemas import SCHEMAS
        if _src in SCHEMAS:
            _update_daily_with_staging(args.output.parent, tickers, existing, args)
            return 0
    except ImportError:
        pass

    # Legacy in-memory path

    fresh_frames: list[pd.DataFrame] = []
    errors = 0
    skips = 0

    for idx, ticker in enumerate(tickers, start=1):
        last_date = last_map.get(ticker)
        new_df, status = fetch_daily_for_ticker(
            ticker,
            last_date,
            min_fetch_days=args.min_fetch_days,
            buffer_days=args.buffer_days,
            max_fetch_days=args.max_fetch_days,
        )
        icon = "✓" if status.startswith("ok") or status.startswith("skipped") else "✗"
        print(f"[{idx:4d}/{len(tickers)}] {icon} {ticker:<12} {status}")

        if status.startswith("skipped"):
            skips += 1
        if status.startswith("no data"):
            errors += 1
        if new_df is not None and not new_df.empty:
            fresh_frames.append(new_df)
        if args.pause_seconds > 0:
            time.sleep(args.pause_seconds)

    if fresh_frames:
        fresh = pd.concat(fresh_frames, ignore_index=True)
        combined = pd.concat([existing, fresh], ignore_index=True) if not existing.empty else fresh
        combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
        combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
    else:
        combined = existing.sort_values(["date", "ticker"]).reset_index(drop=True) if not existing.empty else existing

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output, index=False)

    print(
        f"[done] rows={len(combined):,} new_chunks={len(fresh_frames)} "
        f"errors={errors} skipped={skips}"
    )
    print(f"Finished at   : {datetime.now().isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
