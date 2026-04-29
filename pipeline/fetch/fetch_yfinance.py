#!/usr/bin/env python3
"""
Fetch Yahoo Finance intraday data (1h / 4h) for IDX tickers and store as parquet.

Ticker source:
  machinelearning/idx/data/Level_0_Raw/master_emiten.parquet (column: ticker, optional status)

Outputs:
  machinelearning/idx/data/Level_0_Raw/yfinance_1h.parquet
  machinelearning/idx/data/Level_0_Raw/yfinance_4h.parquet

Examples:
  python machinelearning/idx/service/fetch_yfinance.py
  python machinelearning/idx/service/fetch_yfinance.py --intervals 1h
  python machinelearning/idx/service/fetch_yfinance.py --limit-tickers 50
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


LOCAL_TZ = "Asia/Jakarta"


@dataclass(frozen=True)
class IntervalConfig:
    interval: str
    output_name: str
    max_days: int
    min_fetch_days: int
    buffer_days: int


INTERVAL_CONFIG: dict[str, IntervalConfig] = {
    "1h": IntervalConfig("1h", "yfinance_1h.parquet", 730, 30, 5),
    "4h": IntervalConfig("4h", "yfinance_4h.parquet", 730, 30, 7),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch yfinance intraday parquet (1h/4h).")
    parser.add_argument("--intervals", default="1h,4h", help="Comma-separated intervals: 1h,4h")
    parser.add_argument("--only-active", action="store_true", default=True, help="Use only ACTIVE emitens if status exists.")
    parser.add_argument("--all-status", action="store_true", help="Ignore status filter and use all emitens.")
    parser.add_argument("--limit-tickers", type=int, default=0, help="Limit number of tickers for test run.")
    parser.add_argument("--pause-seconds", type=float, default=0.2, help="Pause between ticker requests.")
    parser.add_argument("--master-path", type=Path, default=None, help="Override master_emiten parquet path.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override data directory.")
    return parser.parse_args()


def now_wib() -> datetime:
    return datetime.now().astimezone()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    service_dir = Path(__file__).resolve().parent
    idx_dir = service_dir.parent
    default_data_dir = idx_dir / "data" / "Level_0_Raw"
    if not default_data_dir.exists():
        default_data_dir = idx_dir / "data"
    if not default_data_dir.exists():
        default_data_dir = Path(__file__).resolve().parents[2] / "machinelearning" / "data"

    data_dir = args.data_dir if args.data_dir else default_data_dir
    master_path = args.master_path if args.master_path else (data_dir / "master_emiten.parquet")
    return data_dir, master_path


def parse_intervals(raw: str) -> list[str]:
    items = [x.strip().lower() for x in raw.split(",") if x.strip()]
    bad = [x for x in items if x not in INTERVAL_CONFIG]
    if bad:
        raise ValueError(f"Unsupported intervals: {bad}. Allowed: {list(INTERVAL_CONFIG)}")
    if not items:
        raise ValueError("No interval selected")
    return items


def load_tickers(master_path: Path, *, use_active_only: bool, limit_tickers: int) -> list[str]:
    if not master_path.exists():
        raise FileNotFoundError(f"master emiten parquet not found: {master_path}")

    df = pd.read_parquet(master_path)
    if "ticker" not in df.columns:
        raise ValueError("Column 'ticker' not found in master emiten parquet")

    source = df
    if use_active_only and "status" in df.columns:
        source = df[df["status"].astype(str).str.upper().eq("ACTIVE")]

    t = source["ticker"].dropna().astype(str).str.upper().str.strip()
    t = t[t != ""]
    t = t.str.replace(r"[^A-Z0-9.]", "", regex=True)
    t = t.map(lambda x: x if x.endswith(".JK") else f"{x}.JK")
    tickers = sorted(set(t.tolist()))

    if limit_tickers and limit_tickers > 0:
        tickers = tickers[:limit_tickers]
    return tickers


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])
    try:
        df = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])

    if df.empty:
        return pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])

    needed = ["datetime", "ticker", "open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Existing parquet missing columns {missing}: {path}")

    out = df[needed].copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.dropna(subset=["datetime", "ticker"])
    out["ticker"] = out["ticker"].astype(str).str.upper()
    return out


def clean_intraday(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    cols = {str(c).lower(): c for c in raw.columns}
    required = ["open", "high", "low", "close", "volume"]
    if not all(k in cols for k in required):
        return None

    out = raw[[cols["open"], cols["high"], cols["low"], cols["close"], cols["volume"]]].copy()
    out.columns = required

    idx = pd.to_datetime(out.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(LOCAL_TZ).tz_localize(None)
    else:
        idx = idx.tz_localize("UTC").tz_convert(LOCAL_TZ).tz_localize(None)

    out.index = idx
    out.index.name = "datetime"
    out = out.sort_index()
    out = out.between_time("09:00", "16:00")
    out = out.dropna(subset=["open", "close"])
    out = out[out["volume"].fillna(0) > 0]

    if out.empty:
        return None

    out["ticker"] = ticker
    out = out.reset_index()[["datetime", "ticker", "open", "high", "low", "close", "volume"]]
    return out


def fetch_ticker_intraday(ticker: str, cfg: IntervalConfig, last_dt: pd.Timestamp | None) -> tuple[pd.DataFrame | None, str]:
    now_date = datetime.now().date()
    if last_dt is not None:
        days_ago = (now_date - last_dt.date()).days
        if days_ago <= 0:
            return None, "skipped (up to date)"
        fetch_days = min(max(days_ago + cfg.buffer_days, cfg.min_fetch_days), cfg.max_days)
    else:
        fetch_days = cfg.max_days

    # Intraday endpoint limit is strict and can vary by ticker/listing age.
    # Try from farthest range and fallback to smaller windows automatically.
    fallback_days = [700, 540, 365, 240, 180, 120, 90, 60, 30]
    candidates: list[int] = [fetch_days]
    for d in fallback_days:
        if d < fetch_days:
            candidates.append(d)

    last_err: str | None = None
    for days in candidates:
        try:
            raw = yf.download(
                ticker,
                period=f"{days}d",
                interval=cfg.interval,
                auto_adjust=True,
                progress=False,
            )
            cleaned = clean_intraday(raw, ticker)
            if cleaned is not None and not cleaned.empty:
                return cleaned, f"ok (+{len(cleaned)} rows, {days}d)"
            last_err = f"no data ({days}d)"
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"

    return None, f"no data after fallback ({last_err or 'unknown'})"


def update_interval(data_dir: Path, tickers: list[str], cfg: IntervalConfig, pause_seconds: float) -> None:
    out_path = data_dir / cfg.output_name
    existing = load_existing(out_path)
    last_map: dict[str, pd.Timestamp] = {}
    if not existing.empty:
        last_map = existing.groupby("ticker")["datetime"].max().to_dict()

    new_frames: list[pd.DataFrame] = []
    errors = 0
    skips = 0

    print(f"\n=== Fetch {cfg.interval} -> {out_path} ===", flush=True)
    print(f"tickers: {len(tickers)} | existing_rows: {len(existing)}", flush=True)

    for i, ticker in enumerate(tickers, start=1):
        last_dt = last_map.get(ticker)
        new_df, status = fetch_ticker_intraday(ticker, cfg, last_dt)

        icon = "✓" if status.startswith("ok") or status.startswith("skipped") else "✗"
        print(f"[{i:4d}/{len(tickers)}] {icon} {ticker:<12} {status}", flush=True)

        if status.startswith("error"):
            errors += 1
        if status.startswith("skipped"):
            skips += 1

        if new_df is not None and not new_df.empty:
            new_frames.append(new_df)

        if pause_seconds > 0:
            time.sleep(pause_seconds)

    if new_frames:
        fresh = pd.concat(new_frames, ignore_index=True)
        combined = pd.concat([existing, fresh], ignore_index=True) if not existing.empty else fresh
        combined = combined.drop_duplicates(subset=["datetime", "ticker"], keep="last")
        combined = combined.sort_values(["datetime", "ticker"]).reset_index(drop=True)
    else:
        combined = existing.sort_values(["datetime", "ticker"]).reset_index(drop=True) if not existing.empty else existing

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)

    print(
        f"[done] interval={cfg.interval} rows={len(combined)} "
        f"new_chunks={len(new_frames)} errors={errors} skipped={skips}"
    , flush=True)


def main() -> int:
    args = parse_args()
    intervals = parse_intervals(args.intervals)
    data_dir, master_path = resolve_paths(args)

    tickers = load_tickers(
        master_path=master_path,
        use_active_only=(not args.all_status),
        limit_tickers=args.limit_tickers,
    )
    if not tickers:
        raise RuntimeError("No tickers found from master emiten parquet")

    print(f"Master emiten : {master_path}", flush=True)
    print(f"Data dir      : {data_dir}", flush=True)
    print(f"Intervals     : {intervals}", flush=True)
    print(f"Ticker count  : {len(tickers)}", flush=True)
    print(f"Start time    : {now_wib().isoformat()}", flush=True)

    for iv in intervals:
        cfg = INTERVAL_CONFIG[iv]
        update_interval(data_dir, tickers, cfg, args.pause_seconds)

    print(f"Finished at   : {now_wib().isoformat()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
