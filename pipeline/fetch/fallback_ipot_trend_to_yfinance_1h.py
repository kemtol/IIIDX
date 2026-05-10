#!/usr/bin/env python3
"""Fallback-fill missing yfinance_1h hourly bars from IPOT TREND_1D stream.

Operational use:
  At 15:05 WIB, if Yahoo has not published the 14:00 WIB candle yet, derive
  a compatible 1h OHLCV row from ws_trend_1m.duckdb and append only missing
  (datetime, ticker) keys into yfinance_1h.parquet.

This is intentionally a separate fallback fetcher. It does not change the main
Yahoo fetcher and never overwrites an existing Yahoo row.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.storage.config import L0SourceConfig  # noqa: E402
from pipeline.storage import schemas as schemas_pkg  # noqa: E402
from pipeline.storage.schemas import SCHEMAS  # noqa: E402
from pipeline.storage.writers import write_l0  # noqa: E402

WIB = ZoneInfo("Asia/Jakarta")
SOURCE = "yfinance_1h"
DEFAULT_YF_PATH = REPO_ROOT / SCHEMAS[SOURCE].parquet_path
DEFAULT_TREND_DB = REPO_ROOT / "data/Level_0_Raw/ws_trend_1m.duckdb"
DEFAULT_MASTER_PATH = REPO_ROOT / "data/Level_0_Raw/master_emiten.parquet"
DEFAULT_AUDIT_DIR = REPO_ROOT / "_LOG"
REQUIRED_YF_COLUMNS = ["datetime", "ticker", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class FallbackReport:
    status: str
    target_date: str
    target_hour_wib: int
    target_datetime_utc: str
    universe_count: int
    yfinance_present_count: int
    yfinance_missing_count: int
    ipot_candidate_count: int
    fallback_row_count: int
    inserted_count: int
    mode: str
    yfinance_path: str
    trend_db_path: str
    trend_table: str
    audit_path: str | None = None
    warnings: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill missing yfinance_1h bars from IPOT TREND_1D 1-minute stream."
    )
    parser.add_argument("--date", default=None, help="Target date YYYY-MM-DD; default today WIB.")
    parser.add_argument("--hour", type=int, default=14, help="Target candle hour in WIB; default 14.")
    parser.add_argument("--mode", choices=["dry-run", "upsert"], default="dry-run")
    parser.add_argument("--yf-path", type=Path, default=DEFAULT_YF_PATH)
    parser.add_argument("--trend-db-path", type=Path, default=DEFAULT_TREND_DB)
    parser.add_argument("--trend-table", default="trending_all")
    parser.add_argument("--master-path", type=Path, default=DEFAULT_MASTER_PATH)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument(
        "--universe",
        choices=["master-active", "yfinance", "ipot"],
        default="master-active",
        help="Ticker universe used to decide missing rows.",
    )
    parser.add_argument(
        "--volume-mode",
        choices=["sum", "delta", "auto"],
        default="sum",
        help="How to derive hourly volume from TREND_1D vol.",
    )
    parser.add_argument(
        "--min-minutes",
        type=int,
        default=1,
        help="Minimum minute records per ticker in the target hour.",
    )
    return parser.parse_args()


def parse_target_date(raw: str | None) -> date:
    if raw:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    return datetime.now(WIB).date()


def target_datetime_utc(target_date: date, hour_wib: int) -> pd.Timestamp:
    if hour_wib < 0 or hour_wib > 23:
        raise ValueError("--hour must be between 0 and 23")
    local = datetime.combine(target_date, time(hour_wib, 0), tzinfo=WIB)
    return pd.Timestamp(local).tz_convert("UTC")


def normalize_ticker(value: object) -> str:
    ticker = str(value).upper().strip()
    if ticker.endswith(".JK"):
        ticker = ticker[:-3]
    return re.sub(r"[^A-Z0-9]", "", ticker)


def load_yfinance(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=REQUIRED_YF_COLUMNS)

    df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_YF_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing yfinance_1h columns: {missing}")

    out = df[REQUIRED_YF_COLUMNS].copy()
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
    out = out.dropna(subset=["datetime", "ticker"])
    out["ticker"] = out["ticker"].map(normalize_ticker)
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_master_universe(master_path: Path) -> list[str]:
    if not master_path.exists():
        raise FileNotFoundError(f"master_emiten not found: {master_path}")
    df = pd.read_parquet(master_path)
    if "ticker" not in df.columns:
        raise ValueError(f"{master_path} missing ticker column")
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.upper().eq("ACTIVE")]
    tickers = df["ticker"].dropna().map(normalize_ticker)
    return sorted(t for t in set(tickers) if t)


def _quote_identifier(identifier: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        raise ValueError(f"Unsafe DuckDB identifier: {identifier!r}")
    return f'"{identifier}"'


def _trend_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    desc = con.execute(f"DESCRIBE {_quote_identifier(table)}").fetchall()
    return {str(row[0]) for row in desc}


def load_ipot_trend(
    db_path: Path,
    table: str,
    target_date: date,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if not db_path.exists():
        raise FileNotFoundError(f"IPOT trend DuckDB not found: {db_path}")

    warnings: list[str] = []
    with duckdb.connect(str(db_path), read_only=True) as con:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        if table not in tables:
            raise ValueError(f"Table {table!r} not found in {db_path}; available={sorted(tables)}")

        cols = _trend_columns(con, table)
        required = {"ticker", "time", "close", "vol"}
        missing = required - cols
        if missing:
            raise ValueError(f"{table} missing columns: {sorted(missing)}")

        qtable = _quote_identifier(table)
        if "date" in cols:
            query = f"""
                SELECT ticker, time, close, vol
                FROM {qtable}
                WHERE CAST(date AS DATE) = ?
            """
            df = con.execute(query, [target_date.isoformat()]).df()
        elif "captured_at" in cols:
            query = f"""
                SELECT ticker, time, close, vol
                FROM {qtable}
                WHERE CAST(captured_at AS DATE) = ?
            """
            df = con.execute(query, [target_date.isoformat()]).df()
        else:
            warnings.append(
                "trend table has no date/captured_at column; assuming all rows belong to target date"
            )
            df = con.execute(f"SELECT ticker, time, close, vol FROM {qtable}").df()

    if df.empty:
        return pd.DataFrame(columns=["ticker", "time", "close", "vol"]), tuple(warnings)

    df["ticker"] = df["ticker"].map(normalize_ticker)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce").fillna(0)
    df = df.dropna(subset=["ticker", "time", "close"])
    return df, tuple(warnings)


def _minute_of_day(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.extract(r"(?P<h>\d{1,2}):(?P<m>\d{2})", expand=True)
    return pd.to_numeric(text["h"], errors="coerce") * 60 + pd.to_numeric(text["m"], errors="coerce")


def _derive_volume(values: pd.Series, mode: Literal["sum", "delta", "auto"]) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if vals.empty:
        return 0.0
    if mode == "sum":
        return float(vals.sum())
    if mode == "delta":
        return float(max(vals.iloc[-1] - vals.iloc[0], 0.0))

    diffs = vals.diff().dropna()
    if len(diffs) >= 3 and (diffs >= 0).mean() >= 0.95:
        return float(max(vals.iloc[-1] - vals.iloc[0], 0.0))
    return float(vals.sum())


def aggregate_ipot_hour(
    trend: pd.DataFrame,
    target_date: date,
    hour_wib: int,
    *,
    volume_mode: Literal["sum", "delta", "auto"] = "sum",
    min_minutes: int = 1,
) -> pd.DataFrame:
    if trend.empty:
        return pd.DataFrame(columns=REQUIRED_YF_COLUMNS)

    start_min = hour_wib * 60
    end_min = start_min + 60
    df = trend.copy()
    df["minute_of_day"] = _minute_of_day(df["time"])
    df = df[(df["minute_of_day"] >= start_min) & (df["minute_of_day"] < end_min)]
    df = df.dropna(subset=["minute_of_day", "ticker", "close"])
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_YF_COLUMNS)

    df = df.sort_values(["ticker", "minute_of_day"])
    target_dt = target_datetime_utc(target_date, hour_wib)
    rows: list[dict[str, object]] = []
    for ticker, grp in df.groupby("ticker", sort=True):
        if len(grp) < min_minutes:
            continue
        closes = grp["close"].astype(float)
        rows.append(
            {
                "datetime": target_dt,
                "ticker": ticker,
                "open": float(closes.iloc[0]),
                "high": float(closes.max()),
                "low": float(closes.min()),
                "close": float(closes.iloc[-1]),
                "volume": _derive_volume(grp["vol"], volume_mode),
            }
        )

    return pd.DataFrame(rows, columns=REQUIRED_YF_COLUMNS)


def choose_universe(
    mode: str,
    *,
    master_path: Path,
    yfinance: pd.DataFrame,
    ipot_bars: pd.DataFrame,
) -> list[str]:
    if mode == "master-active":
        return load_master_universe(master_path)
    if mode == "yfinance":
        return sorted(t for t in set(yfinance["ticker"].dropna().map(normalize_ticker)) if t)
    if mode == "ipot":
        return sorted(t for t in set(ipot_bars["ticker"].dropna().map(normalize_ticker)) if t)
    raise ValueError(f"Unsupported universe mode: {mode}")


def select_missing_fallback_rows(
    yfinance: pd.DataFrame,
    ipot_bars: pd.DataFrame,
    universe: list[str],
    target_dt_utc: pd.Timestamp,
) -> tuple[pd.DataFrame, int, int]:
    universe_set = {normalize_ticker(t) for t in universe if normalize_ticker(t)}
    target_existing = yfinance[yfinance["datetime"].eq(target_dt_utc)]
    present = set(target_existing["ticker"].dropna().map(normalize_ticker))
    missing = universe_set - present

    candidates = ipot_bars[ipot_bars["ticker"].isin(missing)].copy()
    candidates = candidates.drop_duplicates(subset=["datetime", "ticker"], keep="last")
    return candidates[REQUIRED_YF_COLUMNS], len(present & universe_set), len(missing)


def merge_preserving_yfinance(yfinance: pd.DataFrame, fallback_rows: pd.DataFrame) -> pd.DataFrame:
    if fallback_rows.empty:
        return yfinance.sort_values(["datetime", "ticker"]).reset_index(drop=True)

    combined = pd.concat([yfinance, fallback_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=["datetime", "ticker"], keep="first")
    combined = combined.sort_values(["datetime", "ticker"]).reset_index(drop=True)
    combined = combined[REQUIRED_YF_COLUMNS]
    combined["datetime"] = pd.to_datetime(combined["datetime"], utc=True, errors="coerce")
    combined["ticker"] = combined["ticker"].map(normalize_ticker)
    return combined


def write_yfinance_snapshot(combined: pd.DataFrame, yf_path: Path) -> None:
    schema = SCHEMAS[SOURCE]
    original = schema
    redirected = replace(schema, parquet_path=str(yf_path.resolve()))
    schemas_pkg.SCHEMAS[SOURCE] = redirected
    try:
        write_l0(SOURCE, combined, cfg=L0SourceConfig(source=SOURCE))
    finally:
        schemas_pkg.SCHEMAS[SOURCE] = original


def write_audit(report: FallbackReport, rows: pd.DataFrame, audit_dir: Path) -> Path:
    audit_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(WIB).strftime("%Y%m%d_%H%M%S")
    path = audit_dir / f"ipot_trend_fallback_{report.target_date}_{report.target_hour_wib:02d}_{stamp}.json"
    payload = {
        **report.__dict__,
        "sample_rows": rows.head(20).assign(datetime=lambda x: x["datetime"].astype(str)).to_dict("records")
        if not rows.empty
        else [],
    }
    payload["warnings"] = list(report.warnings)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path


def run(args: argparse.Namespace) -> FallbackReport:
    target_date = parse_target_date(args.date)
    target_dt = target_datetime_utc(target_date, args.hour)

    yfinance = load_yfinance(args.yf_path)
    trend, trend_warnings = load_ipot_trend(args.trend_db_path, args.trend_table, target_date)
    ipot_bars = aggregate_ipot_hour(
        trend,
        target_date,
        args.hour,
        volume_mode=args.volume_mode,
        min_minutes=args.min_minutes,
    )
    universe = choose_universe(
        args.universe,
        master_path=args.master_path,
        yfinance=yfinance,
        ipot_bars=ipot_bars,
    )
    fallback_rows, present_count, missing_count = select_missing_fallback_rows(
        yfinance,
        ipot_bars,
        universe,
        target_dt,
    )

    warnings = list(trend_warnings)
    if ipot_bars.empty:
        warnings.append("no IPOT bars available for target hour")
    if missing_count and 0 < len(fallback_rows) < missing_count:
        warnings.append(f"partial fallback coverage: can fill {len(fallback_rows)} of {missing_count} missing rows")
    if missing_count and fallback_rows.empty:
        status = "FAIL_IPOT_INSUFFICIENT"
    elif missing_count == 0:
        status = "PASS_NO_FALLBACK_NEEDED"
    elif args.mode == "dry-run":
        status = "PASS_FALLBACK_AVAILABLE" if len(fallback_rows) == missing_count else "PASS_FALLBACK_PARTIAL"
    else:
        combined = merge_preserving_yfinance(yfinance, fallback_rows)
        write_yfinance_snapshot(combined, args.yf_path)
        status = "UPSERTED" if len(fallback_rows) == missing_count else "UPSERTED_PARTIAL"

    report = FallbackReport(
        status=status,
        target_date=target_date.isoformat(),
        target_hour_wib=args.hour,
        target_datetime_utc=target_dt.isoformat(),
        universe_count=len(universe),
        yfinance_present_count=present_count,
        yfinance_missing_count=missing_count,
        ipot_candidate_count=len(ipot_bars),
        fallback_row_count=len(fallback_rows),
        inserted_count=len(fallback_rows) if status.startswith("UPSERTED") else 0,
        mode=args.mode,
        yfinance_path=str(args.yf_path),
        trend_db_path=str(args.trend_db_path),
        trend_table=args.trend_table,
        warnings=tuple(warnings),
    )

    audit_path = write_audit(report, fallback_rows, args.audit_dir)
    return FallbackReport(**{**report.__dict__, "audit_path": str(audit_path)})


def main() -> int:
    args = parse_args()
    report = run(args)
    print(json.dumps(report.__dict__, indent=2, default=str))
    if report.status.startswith("FAIL"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
