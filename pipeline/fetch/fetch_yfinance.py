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
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

# L0 storage gateway lives at repo root; ensure it is importable when this
# script is invoked directly (`python pipeline/fetch/fetch_yfinance.py`).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.storage.schemas import SCHEMAS  # noqa: E402
from pipeline.storage.staging import open_staging, staging_exists  # noqa: E402
from pipeline.storage.writers import write_l0  # noqa: E402

LOCAL_TZ = "Asia/Jakarta"

# Storage source key per pipeline/storage/config.py. yfinance_4h is parquet-only
# until its schema lands; gateway falls through to legacy write for unknown sources.
_SOURCE_BY_INTERVAL = {
    "1h": "yfinance_1h",
    "4h": "yfinance_4h",
}


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
    idx_dir = Path(__file__).resolve().parent.parent.parent
    default_data_dir = idx_dir / "data" / "Level_0_Raw"

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

    # Normalize to UTC tz-aware. Yahoo returns tz-aware (typically exchange-local
    # or UTC depending on symbol); we canonicalize to UTC so concat with existing
    # parquet (UTC tz-aware) does not silently shift values. Historical bug:
    # tz_localize(None) here previously stripped tz, then later promotion to
    # UTC-tz on rewrite shifted values by 7h. Audit 2026-05-05 confirmed
    # 734 trading days (2023-03-06..2026-04-27) corrupted by that path.
    idx = pd.to_datetime(out.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC")
    else:
        idx = idx.tz_localize("UTC")

    out.index = idx
    out.index.name = "datetime"
    out = out.sort_index()
    # between_time on tz-aware index uses index tz; convert to LOCAL_TZ briefly
    # for window selection, then back to UTC for storage.
    local_idx = out.index.tz_convert(LOCAL_TZ)
    mask = (local_idx.time >= pd.Timestamp("09:00").time()) & (
        local_idx.time <= pd.Timestamp("16:00").time()
    )
    out = out.loc[mask]
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
        last_local = pd.Timestamp(last_dt)
        if last_local.tzinfo is None:
            last_local = last_local.tz_localize("UTC")
        else:
            last_local = last_local.tz_convert("UTC")
        last_local = last_local.tz_convert(LOCAL_TZ)

        days_ago = (now_date - last_local.date()).days
        if days_ago <= 0:
            if cfg.interval == "1h" and last_local.hour < 15:
                fetch_days = cfg.min_fetch_days
            else:
                return None, "skipped (up to date)"
        else:
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


def _normalize_combined(combined: pd.DataFrame) -> pd.DataFrame:
    """Drop legacy `__index_level_0__` and force UTC tz-aware datetime."""
    if "__index_level_0__" in combined.columns:
        combined = combined.drop(columns=["__index_level_0__"])
    if "datetime" in combined.columns:
        dt = pd.to_datetime(combined["datetime"], utc=True, errors="coerce")
        combined = combined.assign(datetime=dt).dropna(subset=["datetime"])
    return combined


def _update_interval_with_staging(
    data_dir: Path, tickers: list[str], cfg: IntervalConfig, pause_seconds: float, source: str
) -> None:
    """Crash-resilient variant: persist per-ticker chunks to DuckDB staging.

    A crash mid-loop preserves staged rows; the next run resumes by computing
    last_dt from existing parquet UNION staging. After all tickers fetch
    successfully, staging is merged with existing parquet and written
    atomically via `write_l0()`. Staging is then truncated.
    """
    out_path = data_dir / cfg.output_name
    existing = load_existing(out_path)

    print(f"\n=== Fetch {cfg.interval} -> {out_path} (staging mode) ===", flush=True)
    print(
        f"tickers: {len(tickers)} | existing_rows: {len(existing)} | "
        f"staging_resume: {staging_exists(source)}",
        flush=True,
    )

    errors = 0
    skips = 0
    staged_count = 0

    with open_staging(source) as stage:
        # Combine existing + already-staged to compute resume point per ticker.
        # If a previous run crashed, this picks up where it left off.
        last_map = stage.last_seen_per_ticker(base_df=existing)
        prior_staged = stage.staged_count()
        if prior_staged:
            print(f"  resuming over {prior_staged} previously-staged rows", flush=True)

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
                # Per-ticker COMMIT — durable across crashes.
                stage.append(new_df)
                staged_count += len(new_df)

            if pause_seconds > 0:
                time.sleep(pause_seconds)

        # All tickers iterated — merge staged with existing and write atomically.
        # commit_to_l0 calls write_l0 which uses tmp+rename for the parquet path.
        # Only after this returns do we clear() — so a crash here leaves staging
        # intact for the next run to recover.
        rows_written = stage.commit_to_l0(base_df=_normalize_combined(existing))
        stage.clear()

    print(
        f"[done] interval={cfg.interval} rows={rows_written} "
        f"newly_staged={staged_count} errors={errors} skipped={skips}",
        flush=True,
    )


def _update_interval_inmemory(
    data_dir: Path, tickers: list[str], cfg: IntervalConfig, pause_seconds: float
) -> None:
    """Legacy in-memory path for intervals without a registered L0 schema.

    Still atomic at write (tmp+rename) but offers no crash resilience during
    the fetch loop — a mid-loop crash loses everything not yet persisted.
    """
    out_path = data_dir / cfg.output_name
    existing = load_existing(out_path)
    last_map: dict[str, pd.Timestamp] = {}
    if not existing.empty:
        last_map = existing.groupby("ticker")["datetime"].max().to_dict()

    new_frames: list[pd.DataFrame] = []
    errors = 0
    skips = 0

    print(f"\n=== Fetch {cfg.interval} -> {out_path} (in-memory mode) ===", flush=True)
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

    combined = _normalize_combined(combined)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    combined.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)

    print(
        f"[done] interval={cfg.interval} rows={len(combined)} "
        f"new_chunks={len(new_frames)} errors={errors} skipped={skips}",
        flush=True,
    )


def update_interval(data_dir: Path, tickers: list[str], cfg: IntervalConfig, pause_seconds: float) -> None:
    """Dispatch to staging-mode (crash-resilient) or in-memory mode by source.

    Staging mode requires a registered L0 schema; otherwise we fall back to
    the legacy in-memory path with atomic write.
    """
    out_path = data_dir / cfg.output_name
    source = _SOURCE_BY_INTERVAL.get(cfg.interval)

    if source and source in SCHEMAS:
        # Sanity: gateway always writes to the schema path. If --data-dir was
        # overridden, warn rather than silently writing to a different location.
        schema_path = (_REPO_ROOT / SCHEMAS[source].parquet_path).resolve()
        if out_path.resolve() != schema_path:
            print(
                f"[warn] interval={cfg.interval} out_path={out_path} differs from "
                f"schema path {schema_path}; gateway will write to schema path.",
                flush=True,
            )
        _update_interval_with_staging(data_dir, tickers, cfg, pause_seconds, source)
    else:
        _update_interval_inmemory(data_dir, tickers, cfg, pause_seconds)


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
