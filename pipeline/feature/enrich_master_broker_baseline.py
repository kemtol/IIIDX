#!/usr/bin/env python3
"""
Enrich broker panel with no-lookahead moving-average baselines.

Input canonical schema:
  - date
  - broker
  - ticker
  - total_net_buy

Alias support:
  - ticker <- stock_code
  - total_net_buy <- net_buy | net_val

Backfill behavior:
  - If trading-day history < 240 days, fetch missing historical rows from:
      https://broksum-scrapper.mkemalw.workers.dev/ipot/broker-activity
  - Endpoint is called per (broker, date), then flattened to broker-ticker rows.
  - Data is appended, deduplicated on (date, broker, ticker), sorted by date.

Output:
  - master_broker_enriched.parquet with added MA columns:
      ma_total_netbuy_{20,60,120,240}
      ma_ticker_netbuy_{20,60,120,240}
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


WINDOWS: tuple[int, ...] = (20, 60, 120, 240)
MAX_WINDOW = max(WINDOWS)
REQUIRED_UNIQUE_DAYS = MAX_WINDOW + 1  # 240 prior days + current row (because MA uses shift(1))

ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_DATA_DIR = ROOT_DIR / "data"
LEVEL0_DATA_DIR = BASE_DATA_DIR / "Level_0_Raw"
DATA_DIR = LEVEL0_DATA_DIR if LEVEL0_DATA_DIR.exists() else BASE_DATA_DIR
DEFAULT_INPUT = DATA_DIR / "master_broker.parquet"
DEFAULT_BACKFILLED_OUTPUT = DEFAULT_INPUT
DEFAULT_OUTPUT = DATA_DIR / "master_broker_enriched.parquet"

DEFAULT_BROKSUM_API_BASE = "https://broksum-scrapper.mkemalw.workers.dev"
DEFAULT_BROKER_MASTER_API = "https://api-saham.mkemalw.workers.dev/brokers"

CANONICAL_COLUMNS = ["date", "broker", "ticker", "total_net_buy"]


@dataclass
class BackfillStats:
    requested_trading_days: int = 0
    estimated_calendar_days: int = 0
    brokers_used: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    empty_requests: int = 0
    failed_requests: int = 0
    fetched_rows: int = 0


_thread_local = threading.local()


def get_thread_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def _to_float(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def detect_and_normalize_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    col_map: dict[str, str] = {}

    if "date" in raw_df.columns:
        col_map["date"] = "date"
    if "broker" in raw_df.columns:
        col_map["broker"] = "broker"
    if "ticker" in raw_df.columns:
        col_map["ticker"] = "ticker"
    elif "stock_code" in raw_df.columns:
        col_map["stock_code"] = "ticker"

    if "total_net_buy" in raw_df.columns:
        col_map["total_net_buy"] = "total_net_buy"
    elif "net_buy" in raw_df.columns:
        col_map["net_buy"] = "total_net_buy"
    elif "net_val" in raw_df.columns:
        col_map["net_val"] = "total_net_buy"

    normalized = raw_df.loc[:, list(col_map.keys())].rename(columns=col_map)
    missing = [c for c in CANONICAL_COLUMNS if c not in normalized.columns]
    if missing:
        raise ValueError(
            "Input file is missing required columns after alias mapping. "
            f"Missing={missing}, available={list(raw_df.columns)}"
        )

    normalized = normalized.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
    normalized["broker"] = normalized["broker"].astype(str).str.strip().str.upper()
    normalized["ticker"] = normalized["ticker"].astype(str).str.strip().str.upper()
    normalized["total_net_buy"] = pd.to_numeric(normalized["total_net_buy"], errors="coerce")

    # Keep only rows that still have valid keys/values.
    normalized = normalized[CANONICAL_COLUMNS]
    return normalized


def previous_weekdays(before_day: date, count: int) -> list[date]:
    if count <= 0:
        return []
    out: list[date] = []
    cursor = before_day - timedelta(days=1)
    while len(out) < count:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor -= timedelta(days=1)
    out.sort()
    return out


def estimate_calendar_days(earliest_day: date, backfill_days: Iterable[date]) -> int:
    days = list(backfill_days)
    if not days:
        return 0
    oldest_needed = min(days)
    return (earliest_day - oldest_needed).days


def fetch_brokers_from_api(broker_api_url: str, timeout: float) -> list[str]:
    resp = requests.get(broker_api_url, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    brokers = []
    for item in payload.get("brokers", []):
        code = str(item.get("code", "")).strip().upper()
        if code:
            brokers.append(code)
    if not brokers:
        raise RuntimeError(f"No broker codes returned from {broker_api_url}")
    return sorted(set(brokers))


def fetch_one_broker_day(
    *,
    api_base: str,
    broker: str,
    trading_day: date,
    timeout: float,
    retries: int,
    public_key: str | None,
) -> tuple[list[dict], str | None]:
    endpoint = f"{api_base.rstrip('/')}/ipot/broker-activity"
    params: dict[str, str] = {
        "broker": broker,
        "date": trading_day.isoformat(),
    }
    if public_key:
        params["key"] = public_key

    last_error: str | None = None
    for attempt in range(retries + 1):
        try:
            session = get_thread_session()
            resp = session.get(endpoint, params=params, timeout=timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            payload = resp.json()
            if not payload.get("ok"):
                raise RuntimeError(f"Upstream returned ok=false: {payload}")

            stocks = payload.get("stocks") or []
            rows: list[dict] = []
            for stock in stocks:
                ticker = str(stock.get("stock_code", "")).strip().upper()
                if not ticker:
                    continue
                rows.append(
                    {
                        "date": trading_day.isoformat(),
                        "broker": broker,
                        "ticker": ticker,
                        "total_net_buy": _to_float(stock.get("net_val")),
                        "gross_buy": _to_float(stock.get("buy_val")),
                        "gross_sell": _to_float(stock.get("sell_val")),
                    }
                )
            return rows, None
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt < retries:
                time.sleep(0.4 * (2 ** attempt))
            continue
    return [], last_error


def run_backfill(
    *,
    panel: pd.DataFrame,
    missing_days: list[date],
    api_base: str,
    broker_api_url: str,
    timeout: float,
    retries: int,
    max_workers: int,
    log_every: int,
    public_key: str | None,
    broker_source: str,
    brokers_override: list[str] | None,
) -> tuple[pd.DataFrame, BackfillStats, list[str]]:
    stats = BackfillStats()
    stats.requested_trading_days = len(missing_days)
    if not missing_days:
        return panel, stats, []

    existing_brokers = sorted(panel["broker"].dropna().astype(str).str.upper().unique().tolist())
    if brokers_override:
        brokers = sorted(set([b.strip().upper() for b in brokers_override if b.strip()]))
    elif broker_source == "api":
        try:
            brokers = fetch_brokers_from_api(broker_api_url, timeout=timeout)
            print(f"[Backfill] Broker source=api, loaded {len(brokers)} brokers from {broker_api_url}")
        except Exception as exc:  # noqa: BLE001
            print(f"[Backfill] Broker API failed, fallback to input brokers. error={exc}")
            brokers = existing_brokers
    else:
        brokers = existing_brokers

    if not brokers:
        print("[Backfill] No brokers available for backfill; skip fetch.")
        return panel, stats, []

    stats.brokers_used = len(brokers)
    stats.total_requests = len(brokers) * len(missing_days)
    stats.estimated_calendar_days = estimate_calendar_days(panel["date"].min().date(), missing_days)

    print(
        "[Backfill] Need "
        f"{stats.requested_trading_days} trading days "
        f"(~{stats.estimated_calendar_days} calendar days), "
        f"{stats.brokers_used} brokers, "
        f"{stats.total_requests} requests."
    )

    fetched_rows: list[dict] = []
    errors: list[str] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for day in missing_days:
            for broker in brokers:
                fut = executor.submit(
                    fetch_one_broker_day,
                    api_base=api_base,
                    broker=broker,
                    trading_day=day,
                    timeout=timeout,
                    retries=retries,
                    public_key=public_key,
                )
                future_map[fut] = (broker, day)

        for fut in as_completed(future_map):
            broker, day = future_map[fut]
            completed += 1
            rows, error = fut.result()
            if error is not None:
                stats.failed_requests += 1
                if len(errors) < 20:
                    errors.append(f"{day.isoformat()} {broker}: {error}")
            else:
                stats.successful_requests += 1
                if rows:
                    fetched_rows.extend(rows)
                else:
                    stats.empty_requests += 1

            if completed % max(1, log_every) == 0 or completed == stats.total_requests:
                print(
                    f"[Backfill] Progress {completed}/{stats.total_requests} | "
                    f"ok={stats.successful_requests} empty={stats.empty_requests} fail={stats.failed_requests}"
                )

    if fetched_rows:
        fetched_df = pd.DataFrame.from_records(fetched_rows)
        fetched_df["date"] = pd.to_datetime(fetched_df["date"], errors="coerce").dt.normalize()
        fetched_df["broker"] = fetched_df["broker"].astype(str).str.strip().str.upper()
        fetched_df["ticker"] = fetched_df["ticker"].astype(str).str.strip().str.upper()
        fetched_df["total_net_buy"] = pd.to_numeric(fetched_df["total_net_buy"], errors="coerce")
        fetched_df = fetched_df[CANONICAL_COLUMNS]
        stats.fetched_rows = len(fetched_df)

        merged = pd.concat([panel, fetched_df], ignore_index=True)
        merged = (
            merged.sort_values(["date", "broker", "ticker"])
            .drop_duplicates(subset=["date", "broker", "ticker"], keep="last")
            .reset_index(drop=True)
        )
    else:
        merged = panel.copy()

    if errors:
        print("[Backfill] Error samples (first 20):")
        for item in errors:
            print(f"  - {item}")
    return merged, stats, errors


def compute_moving_averages(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out = out.sort_values(["date", "broker", "ticker"]).reset_index(drop=True)

    broker_daily = (
        out.groupby(["broker", "date"], as_index=False)["total_net_buy"]
        .sum()
        .sort_values(["broker", "date"])
        .reset_index(drop=True)
    )
    for window in WINDOWS:
        col = f"ma_total_netbuy_{window}"
        broker_daily[col] = (
            broker_daily.groupby("broker", sort=False)["total_net_buy"]
            .transform(lambda s, w=window: s.shift(1).rolling(window=w, min_periods=w).mean())
        )
    out = out.merge(
        broker_daily.drop(columns=["total_net_buy"]),
        on=["broker", "date"],
        how="left",
        validate="many_to_one",
    )

    out = out.sort_values(["broker", "ticker", "date"]).reset_index(drop=True)
    for window in WINDOWS:
        col = f"ma_ticker_netbuy_{window}"
        out[col] = (
            out.groupby(["broker", "ticker"], sort=False)["total_net_buy"]
            .transform(lambda s, w=window: s.shift(1).rolling(window=w, min_periods=w).mean())
        )

    return out.sort_values(["date", "broker", "ticker"]).reset_index(drop=True)


def verify_no_lookahead(enriched: pd.DataFrame) -> None:
    print("[Check] Verifying no-lookahead behavior on first valid MA points...")
    daily_base = (
        enriched.groupby(["broker", "date"], as_index=False)["total_net_buy"]
        .sum()
        .sort_values(["broker", "date"])
        .reset_index(drop=True)
    )
    daily_ma_cols = ["broker", "date"] + [f"ma_total_netbuy_{w}" for w in WINDOWS]
    daily_with_ma = enriched[daily_ma_cols].drop_duplicates(subset=["broker", "date"])
    daily = daily_base.merge(daily_with_ma, on=["broker", "date"], how="left")

    for w in WINDOWS:
        col = f"ma_total_netbuy_{w}"
        checked = False
        for _, grp in daily.groupby("broker", sort=False):
            grp = grp.sort_values("date").reset_index(drop=True)
            if len(grp) >= w + 1 and pd.notna(grp.loc[w, col]):
                expected = grp.loc[0 : w - 1, "total_net_buy"].mean()
                actual = float(grp.loc[w, col])
                if not math.isclose(actual, float(expected), rel_tol=1e-9, abs_tol=1e-6):
                    raise AssertionError(
                        f"No-lookahead check failed for {col} (broker level): expected={expected}, actual={actual}"
                    )
                checked = True
                break
        if checked:
            print(f"  - {col}: OK")
        else:
            print(f"  - {col}: skipped (insufficient history)")

    for w in WINDOWS:
        col = f"ma_ticker_netbuy_{w}"
        checked = False
        for _, grp in enriched.groupby(["broker", "ticker"], sort=False):
            grp = grp.sort_values("date").reset_index(drop=True)
            if len(grp) >= w + 1 and pd.notna(grp.loc[w, col]):
                expected = grp.loc[0 : w - 1, "total_net_buy"].mean()
                actual = float(grp.loc[w, col])
                if not math.isclose(actual, float(expected), rel_tol=1e-9, abs_tol=1e-6):
                    raise AssertionError(
                        f"No-lookahead check failed for {col} (ticker level): expected={expected}, actual={actual}"
                    )
                checked = True
                break
        if checked:
            print(f"  - {col}: OK")
        else:
            print(f"  - {col}: skipped (insufficient history)")


def validate_and_report(
    *,
    panel_before_ma: pd.DataFrame,
    enriched: pd.DataFrame,
    errors_in_backfill: list[str],
) -> None:
    unique_dates = enriched["date"].dropna().drop_duplicates().sort_values()
    if unique_dates.empty:
        raise AssertionError("Dataset has no valid dates after processing.")
    earliest = unique_dates.iloc[0].date()
    latest = unique_dates.iloc[-1].date()
    span = len(unique_dates)
    print(f"[Date Range] {earliest} -> {latest} ({span} trading days)")

    if len(panel_before_ma) != len(enriched):
        raise AssertionError(
            f"Row count mismatch: before_ma={len(panel_before_ma)} vs output={len(enriched)}"
        )
    print(f"[Assert] Row count stable: {len(enriched):,} rows")

    if enriched[CANONICAL_COLUMNS].isna().any().any():
        bad = enriched[CANONICAL_COLUMNS].isna().sum().to_dict()
        raise AssertionError(f"NaN detected in original canonical columns: {bad}")
    print("[Assert] No NaN found in original canonical columns.")

    ma_cols = (
        [f"ma_total_netbuy_{w}" for w in WINDOWS]
        + [f"ma_ticker_netbuy_{w}" for w in WINDOWS]
    )
    pct_nan = enriched[ma_cols].isna().mean().mul(100.0)
    print("[NaN Distribution] MA columns")
    for c in ma_cols:
        print(f"  - {c}: {pct_nan[c]:.2f}%")

    # Pick one broker+ticker sample with sufficient history if possible.
    key_cols = ["broker", "ticker"]
    sufficient = enriched[enriched["ma_ticker_netbuy_240"].notna()]
    if not sufficient.empty:
        sample_key = (
            sufficient.groupby(key_cols, as_index=False)
            .size()
            .sort_values("size", ascending=False)
            .iloc[0]
        )
        broker = sample_key["broker"]
        ticker = sample_key["ticker"]
        sample_note = "sufficient history for MA-240"
    else:
        fallback = (
            enriched.groupby(key_cols, as_index=False)
            .size()
            .sort_values("size", ascending=False)
            .iloc[0]
        )
        broker = fallback["broker"]
        ticker = fallback["ticker"]
        sample_note = "best available history (MA-240 still NaN)"

    sample = (
        enriched[(enriched["broker"] == broker) & (enriched["ticker"] == ticker)]
        .sort_values("date")
        .tail(5)
    )
    sample_cols = ["date", "broker", "ticker", "total_net_buy"] + ma_cols
    print(f"[Sample] broker={broker}, ticker={ticker} ({sample_note})")
    print(sample[sample_cols].to_string(index=False))

    if errors_in_backfill:
        print(
            "[Warning] Backfill had failed/time-out requests. "
            "MA-240 and other long-window columns may have elevated NaN rates."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build broker baseline moving averages with optional historical backfill."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input parquet path")
    parser.add_argument(
        "--backfilled-output",
        type=Path,
        default=DEFAULT_BACKFILLED_OUTPUT,
        help="Where to save backfilled base panel before enrichment",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output enriched parquet path")

    parser.add_argument("--skip-backfill", action="store_true", help="Skip history backfill step")
    parser.add_argument("--broksum-api-base", type=str, default=DEFAULT_BROKSUM_API_BASE)
    parser.add_argument("--brokers-api-url", type=str, default=DEFAULT_BROKER_MASTER_API)
    parser.add_argument("--public-key", type=str, default=None, help="Optional IPOT public key param")
    parser.add_argument(
        "--broker-source",
        choices=["api", "input"],
        default="api",
        help="How to determine broker universe for backfill",
    )
    parser.add_argument(
        "--brokers",
        type=str,
        default="",
        help="Optional explicit broker list, comma-separated (overrides broker-source)",
    )
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument(
        "--max-backfill-rounds",
        type=int,
        default=3,
        help="Additional backfill rounds to compensate holiday/empty weekdays",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input parquet not found: {args.input}")

    print(f"[Load] Input: {args.input.resolve()}")
    raw = pd.read_parquet(args.input)
    panel = detect_and_normalize_columns(raw)
    panel = panel.sort_values(["date", "broker", "ticker"]).drop_duplicates(
        subset=["date", "broker", "ticker"], keep="last"
    )
    panel = panel.reset_index(drop=True)

    unique_dates = panel["date"].dropna().drop_duplicates().sort_values()
    if unique_dates.empty:
        raise RuntimeError("No valid date found in input dataset.")
    earliest = unique_dates.iloc[0].date()
    latest = unique_dates.iloc[-1].date()
    trading_span = len(unique_dates)
    gap_days = max(0, REQUIRED_UNIQUE_DAYS - trading_span)

    print(f"[Sufficiency] Current date range: {earliest} -> {latest} ({trading_span} trading days)")
    print(
        f"[Sufficiency] MA-{MAX_WINDOW} needs {MAX_WINDOW} prior trading days + current row "
        f"(minimum unique days={REQUIRED_UNIQUE_DAYS}), gap={gap_days}"
    )

    errors_in_backfill: list[str] = []
    cumulative_stats = BackfillStats()

    brokers_override = [b for b in args.brokers.split(",") if b.strip()] if args.brokers else None

    if gap_days > 0 and not args.skip_backfill:
        for round_idx in range(1, max(1, args.max_backfill_rounds) + 1):
            unique_dates = panel["date"].dropna().drop_duplicates().sort_values()
            current_earliest = unique_dates.iloc[0].date()
            current_span = len(unique_dates)
            current_gap = max(0, REQUIRED_UNIQUE_DAYS - current_span)
            if current_gap == 0:
                break

            missing_days = previous_weekdays(current_earliest, current_gap)
            print(
                f"[Backfill] Round {round_idx}/{args.max_backfill_rounds} "
                f"for gap={current_gap} from earliest={current_earliest}"
            )
            panel_before_round = panel.copy()
            panel, round_stats, round_errors = run_backfill(
                panel=panel,
                missing_days=missing_days,
                api_base=args.broksum_api_base,
                broker_api_url=args.brokers_api_url,
                timeout=args.timeout,
                retries=args.retries,
                max_workers=max(1, args.max_workers),
                log_every=max(1, args.log_every),
                public_key=args.public_key,
                broker_source=args.broker_source,
                brokers_override=brokers_override,
            )

            cumulative_stats.requested_trading_days += round_stats.requested_trading_days
            cumulative_stats.estimated_calendar_days += round_stats.estimated_calendar_days
            cumulative_stats.brokers_used = max(cumulative_stats.brokers_used, round_stats.brokers_used)
            cumulative_stats.total_requests += round_stats.total_requests
            cumulative_stats.successful_requests += round_stats.successful_requests
            cumulative_stats.empty_requests += round_stats.empty_requests
            cumulative_stats.failed_requests += round_stats.failed_requests
            cumulative_stats.fetched_rows += round_stats.fetched_rows
            errors_in_backfill.extend(round_errors)

            if len(panel) == len(panel_before_round):
                print(
                    "[Backfill] No additional rows added in this round; "
                    "stopping early to avoid redundant requests."
                )
                break

        print(
            "[Backfill] Cumulative summary: "
            f"fetched_rows={cumulative_stats.fetched_rows:,}, "
            f"req_total={cumulative_stats.total_requests:,}, "
            f"ok={cumulative_stats.successful_requests:,}, "
            f"empty={cumulative_stats.empty_requests:,}, "
            f"fail={cumulative_stats.failed_requests:,}"
        )
    elif gap_days > 0:
        print(
            "[Warning] Backfill skipped while MA-240 history is insufficient. "
            "MA-240 NaN rates will be elevated."
        )
    else:
        print("[Sufficiency] History already satisfies MA-240 window.")

    # Save backfilled panel first (required pre-step output).
    args.backfilled_output.parent.mkdir(parents=True, exist_ok=True)
    panel = panel.sort_values(["date", "broker", "ticker"]).reset_index(drop=True)
    panel.to_parquet(args.backfilled_output, index=False)
    print(f"[Save] Backfilled base panel: {args.backfilled_output.resolve()} ({len(panel):,} rows)")

    post_unique_dates = panel["date"].dropna().drop_duplicates().sort_values()
    post_span = len(post_unique_dates)
    if post_span < REQUIRED_UNIQUE_DAYS:
        print(
            "[Warning] Backfill completed but history is still below MA-240 requirement "
            f"({post_span}/{REQUIRED_UNIQUE_DAYS} unique trading days). "
            "MA-240 NaN rates will remain elevated."
        )

    # Enrichment
    rows_before = len(panel)
    enriched = compute_moving_averages(panel)

    # Explicit no-lookahead verification
    verify_no_lookahead(enriched)

    # Validations + diagnostics
    validate_and_report(
        panel_before_ma=panel,
        enriched=enriched,
        errors_in_backfill=errors_in_backfill,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(args.output, index=False)
    print(f"[Save] Enriched output: {args.output.resolve()} ({len(enriched):,} rows)")

    # Defensive invariant check mirrored in logs
    if rows_before != len(enriched):
        raise AssertionError(f"Output rows changed unexpectedly: before={rows_before}, after={len(enriched)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
