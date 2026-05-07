#!/usr/bin/env python3
"""Daily BSJP PnL recap from picks_log, with optional starter bootstrap."""

from __future__ import annotations

import argparse
import html
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.run.bsjp_heartbeat import (
    load_notify_env,
    send_discord_webhook,
    send_telegram_message,
)

WIB = timezone(timedelta(hours=7), name="WIB")
DEFAULT_VARIANT = "v19d_close10_preclose14_orb_md100_l21.5"
DEFAULT_COST = 0.004
DEFAULT_DAYS = 7

DEFAULT_DB_PATH = REPO_ROOT / "inferences/bsjp/db/inference.duckdb"
DEFAULT_YF_1H_PATH = REPO_ROOT / "data/Level_0_Raw/yfinance_1h.parquet"
DEFAULT_BSJP_PATH = REPO_ROOT / "inferences/bsjp/golang/bsjp"

ICON_OK = "\u2705"
ICON_BAD = "\u274c"
ICON_PENDING = "\u23f3"
ICON_WARN = "\u26a0\ufe0f"


@dataclass(frozen=True)
class Pick:
    signal_date: date
    variant: str
    rank: int
    ticker: str
    pred_proba: float | None
    entry_price: float | None
    source: str = "picks_log"


@dataclass(frozen=True)
class PickPnl:
    pick: Pick
    exit_date: date | None
    exit_price: float | None
    gross_return: float | None
    net_return: float | None
    status: str
    reason: str


@dataclass(frozen=True)
class DayPnl:
    signal_date: date
    picks: list[PickPnl]
    status: str
    gross_return: float | None
    net_return: float | None
    closed_count: int
    expected_count: int
    reason: str


def pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value * 100:+.2f}%"


def price(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    if abs(value - round(value)) < 1e-9:
        return f"{value:.0f}"
    return f"{value:.2f}"


def normalize_ticker(value: Any) -> str:
    ticker = str(value or "").strip().upper()
    if ticker.endswith(".JK"):
        ticker = ticker[:-3]
    return ticker


def parse_date(value: str | date | datetime | pd.Timestamp) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_picks(db_path: Path, variant: str, as_of: date | None = None) -> list[Pick]:
    if not db_path.exists():
        return []

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        table_count = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'picks_log'"
        ).fetchone()[0]
        if not table_count:
            return []

        params: list[Any] = [variant]
        where = "variant = ?"
        if as_of is not None:
            where += " AND date <= ?"
            params.append(as_of.isoformat())

        rows = con.execute(
            f"""
            SELECT date, variant, rank, ticker, pred_proba, entry_price
            FROM picks_log
            WHERE {where}
            ORDER BY date DESC, rank ASC
            """,
            params,
        ).fetchall()
    finally:
        con.close()

    picks: list[Pick] = []
    for row in rows:
        picks.append(
            Pick(
                signal_date=parse_date(row[0]),
                variant=str(row[1]),
                rank=int(row[2]),
                ticker=normalize_ticker(row[3]),
                pred_proba=float(row[4]) if row[4] is not None else None,
                entry_price=float(row[5]) if row[5] is not None else None,
                source="picks_log",
            )
        )
    return picks


def load_yf_1h(yf_1h_path: Path) -> tuple[list[date], dict[tuple[date, str], float]]:
    if not yf_1h_path.exists():
        return [], {}

    df = pd.read_parquet(
        yf_1h_path,
        columns=["datetime", "ticker", "close"],
    )
    if df.empty:
        return [], {}

    dt = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(WIB)
    frame = pd.DataFrame(
        {
            "trade_date": dt.dt.date,
            "hour": dt.dt.hour,
            "ticker": df["ticker"].map(normalize_ticker),
            "close": pd.to_numeric(df["close"], errors="coerce"),
        }
    )
    trading_dates = sorted(frame["trade_date"].dropna().unique().tolist())
    exits = frame[(frame["hour"] == 10) & frame["close"].notna()].copy()
    exits = exits.sort_values(["trade_date", "ticker"]).drop_duplicates(
        ["trade_date", "ticker"], keep="last"
    )
    exit_close = {
        (parse_date(row.trade_date), str(row.ticker)): float(row.close)
        for row in exits.itertuples(index=False)
    }
    return trading_dates, exit_close


def recent_recap_dates(
    *,
    trading_dates: list[date],
    picks: list[Pick],
    as_of: date,
    days: int,
) -> list[date]:
    pick_dates = {p.signal_date for p in picks if p.signal_date <= as_of}
    market_dates = {d for d in trading_dates if d <= as_of}
    candidates = sorted(market_dates | pick_dates)
    return candidates[-days:]


def next_trading_day(signal_date: date, trading_dates: list[date]) -> date | None:
    for candidate in trading_dates:
        if candidate > signal_date:
            return candidate
    return None


def compute_pick_pnl(
    pick: Pick,
    *,
    trading_dates: list[date],
    exit_close: dict[tuple[date, str], float],
    cost: float = DEFAULT_COST,
) -> PickPnl:
    if pick.entry_price is None or not math.isfinite(pick.entry_price) or pick.entry_price <= 0:
        return PickPnl(pick, None, None, None, None, "missing", "missing entry")

    exit_date = next_trading_day(pick.signal_date, trading_dates)
    if exit_date is None:
        return PickPnl(pick, None, None, None, None, "pending", "pending exit T+1 close10")

    exit_price = exit_close.get((exit_date, normalize_ticker(pick.ticker)))
    if exit_price is None or not math.isfinite(exit_price):
        return PickPnl(pick, exit_date, None, None, None, "missing", "missing exit 10:xx")

    gross_return = (exit_price - pick.entry_price) / pick.entry_price
    net_return = gross_return - cost
    return PickPnl(pick, exit_date, exit_price, gross_return, net_return, "closed", "closed")


def compute_day_pnl(
    signal_date: date,
    picks: list[Pick],
    *,
    trading_dates: list[date],
    exit_close: dict[tuple[date, str], float],
    cost: float = DEFAULT_COST,
    expected_count: int = 3,
) -> DayPnl:
    day_picks = sorted((p for p in picks if p.signal_date == signal_date), key=lambda p: p.rank)
    day_picks = [p for p in day_picks if 1 <= p.rank <= expected_count]
    if not day_picks:
        return DayPnl(signal_date, [], "missing", None, None, 0, expected_count, "missing picks_log")

    pick_pnls = [
        compute_pick_pnl(p, trading_dates=trading_dates, exit_close=exit_close, cost=cost)
        for p in day_picks
    ]
    closed = [p for p in pick_pnls if p.status == "closed"]
    if closed:
        gross = sum(p.gross_return for p in closed if p.gross_return is not None) / len(closed)
        net = sum(p.net_return for p in closed if p.net_return is not None) / len(closed)
        return DayPnl(
            signal_date,
            pick_pnls,
            "closed",
            gross,
            net,
            len(closed),
            expected_count,
            "closed",
        )

    if any(p.status == "pending" for p in pick_pnls):
        return DayPnl(
            signal_date,
            pick_pnls,
            "pending",
            None,
            None,
            0,
            expected_count,
            "pending exit T+1 close10",
        )

    return DayPnl(signal_date, pick_pnls, "missing", None, None, 0, expected_count, "missing exits")


def ensure_bsjp_binary(bsjp_path: Path) -> bool:
    if bsjp_path.exists() and os.access(bsjp_path, os.X_OK):
        return True
    golang_dir = bsjp_path.parent
    if not (golang_dir / "cmd/bsjp").exists():
        return False
    env = os.environ.copy()
    env.setdefault("GOTOOLCHAIN", "local")
    result = subprocess.run(
        ["go", "build", "-o", bsjp_path.name, "./cmd/bsjp/"],
        cwd=golang_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
        return False
    return bsjp_path.exists() and os.access(bsjp_path, os.X_OK)


def parse_go_predict_picks(output: str, *, signal_date: date, variant: str) -> list[Pick]:
    picks: list[Pick] = []
    pattern = re.compile(
        r"^\s*#(?P<rank>[0-9]+)\s+"
        r"(?P<ticker>[A-Za-z0-9.]+)\s+"
        r"proba=(?P<proba>[0-9.]+)\s+"
        r"entry=(?P<entry>[0-9.]+)"
    )
    for line in output.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        rank = int(match.group("rank"))
        if rank > 3:
            continue
        picks.append(
            Pick(
                signal_date=signal_date,
                variant=variant,
                rank=rank,
                ticker=normalize_ticker(match.group("ticker")),
                pred_proba=float(match.group("proba")),
                entry_price=float(match.group("entry")),
                source="bootstrap",
            )
        )
    return sorted(picks, key=lambda p: p.rank)


def bootstrap_predict_picks(
    *,
    bsjp_path: Path,
    db_path: Path,
    signal_date: date,
    variant: str,
    fetch_missing_features: bool = False,
) -> list[Pick]:
    if not ensure_bsjp_binary(bsjp_path):
        return []

    def run_predict() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(bsjp_path),
                "predict",
                "--variant",
                variant,
                "--date",
                signal_date.isoformat(),
                "--db",
                str(db_path),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    result = run_predict()
    if result.returncode != 0 and fetch_missing_features and "no executable features" in result.stderr:
        fetch_result = subprocess.run(
            [
                str(bsjp_path),
                "fetch",
                "--date",
                signal_date.isoformat(),
                "--force",
                "--db",
                str(db_path),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if fetch_result.returncode != 0:
            print(
                f"bootstrap fetch failed for {signal_date.isoformat()}: {fetch_result.stderr.strip()}",
                file=sys.stderr,
            )
            return []
        result = run_predict()
    if result.returncode != 0:
        print(
            f"bootstrap predict failed for {signal_date.isoformat()}: {result.stderr.strip()}",
            file=sys.stderr,
        )
        return []
    return parse_go_predict_picks(result.stdout, signal_date=signal_date, variant=variant)


def build_recap(
    *,
    db_path: Path,
    yf_1h_path: Path,
    variant: str = DEFAULT_VARIANT,
    as_of: date | None = None,
    days: int = DEFAULT_DAYS,
    cost: float = DEFAULT_COST,
    bootstrap_missing: bool = False,
    bootstrap_fetch_missing_features: bool = False,
    bsjp_path: Path = DEFAULT_BSJP_PATH,
) -> list[DayPnl]:
    trading_dates, exit_close = load_yf_1h(yf_1h_path)
    if as_of is None:
        as_of = max(trading_dates) if trading_dates else datetime.now(WIB).date()
    picks = load_picks(db_path, variant, as_of=as_of)
    recap_dates = recent_recap_dates(
        trading_dates=trading_dates,
        picks=picks,
        as_of=as_of,
        days=days,
    )
    if bootstrap_missing:
        existing_dates = {p.signal_date for p in picks}
        bootstrapped: list[Pick] = []
        for signal_date in recap_dates:
            if signal_date in existing_dates:
                continue
            bootstrapped.extend(
                bootstrap_predict_picks(
                    bsjp_path=bsjp_path,
                    db_path=db_path,
                    signal_date=signal_date,
                    variant=variant,
                    fetch_missing_features=bootstrap_fetch_missing_features,
                )
            )
        picks = picks + bootstrapped
    return [
        compute_day_pnl(d, picks, trading_dates=trading_dates, exit_close=exit_close, cost=cost)
        for d in recap_dates
    ]


def _best_day(days: list[DayPnl]) -> DayPnl | None:
    closed = [d for d in days if d.status == "closed" and d.net_return is not None]
    return max(closed, key=lambda d: d.net_return, default=None)


def _worst_day(days: list[DayPnl]) -> DayPnl | None:
    closed = [d for d in days if d.status == "closed" and d.net_return is not None]
    return min(closed, key=lambda d: d.net_return, default=None)


def _latest_closed_day(days: list[DayPnl]) -> DayPnl | None:
    closed = [d for d in days if d.status == "closed"]
    return max(closed, key=lambda d: d.signal_date, default=None)


def _pct_unicode(value: float | None) -> str:
    return pct(value).replace("-", "−")


def _price_compact(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "?"
    if abs(value - round(value)) < 1e-9:
        return f"{value:.0f}"
    return f"{value:g}"


def _box_hline(widths: list[int], left: str, mid: str, right: str, h: str = "─") -> str:
    return left + mid.join(h * w for w in widths) + right


def _render_box_table(
    headers: list[str],
    widths: list[int],
    aligns: list[str],
    rows: list[tuple[str, ...]],
) -> str:
    lines = [_box_hline(widths, "┌", "┬", "┐")]
    lines.append("│" + "│".join(h.center(w) for h, w in zip(headers, widths)) + "│")
    lines.append(_box_hline(widths, "├", "┼", "┤"))
    for i, row in enumerate(rows):
        cells = []
        for val, w, a in zip(row, widths, aligns):
            inner = w - 2
            text = str(val)
            if a == "<":
                cells.append(f" {text:<{inner}} ")
            elif a == ">":
                cells.append(f" {text:>{inner}} ")
            else:
                cells.append(text.center(w))
        lines.append("│" + "│".join(cells) + "│")
        if i < len(rows) - 1:
            lines.append(_box_hline(widths, "├", "┼", "┤"))
    lines.append(_box_hline(widths, "└", "┴", "┘"))
    return "\n".join(lines)


def format_recap_message(
    days: list[DayPnl],
    *,
    as_of: date,
    variant: str = DEFAULT_VARIANT,
    cost: float = DEFAULT_COST,
    bootstrap_missing: bool = False,
    bootstrap_fetch_missing_features: bool = False,
) -> str:
    closed_days = [d for d in days if d.status == "closed"]
    pending_days = [d for d in days if d.status == "pending"]
    missing_days = [d for d in days if d.status == "missing"]
    net_total = sum(d.net_return for d in closed_days if d.net_return is not None)
    gross_total = sum(d.gross_return for d in closed_days if d.gross_return is not None)
    win_days = sum(1 for d in closed_days if d.net_return is not None and d.net_return > 0)
    best = _best_day(days)
    worst = _worst_day(days)
    picks_log_ok = any(p.pick.source == "picks_log" for d in days for p in d.picks)
    bootstrap_days = [
        day for day in days if day.picks and all(p.pick.source == "bootstrap" for p in day.picks)
    ]

    L = [
        "<b>BSJP 7D PnL RECAP</b>",
        f"Date: {as_of.isoformat()} | {variant}",
        (
            "Source: picks_log + STARTER bootstrap for missing days (not saved)"
            if bootstrap_missing
            else "Source: Walk-forward picks_log only"
        ),
        "",
        "<b>SUMMARY</b>",
        f"7D Net PnL: {_pct_unicode(net_total)}",
        f"7D Gross PnL: {_pct_unicode(gross_total)}",
        f"Win days: {win_days}/{len(closed_days)}",
    ]
    L.append(
        f"Best: {best.signal_date.isoformat()} {_pct_unicode(best.net_return)}"
        if best else "Best: n/a"
    )
    L.append(
        f"Worst: {worst.signal_date.isoformat()} {_pct_unicode(worst.net_return)}"
        if worst else "Worst: n/a"
    )

    live_rows: list[tuple[str, ...]] = []
    pending_label = ""
    for day in sorted(pending_days, key=lambda x: x.signal_date, reverse=True):
        if not pending_label:
            pending_label = day.signal_date.isoformat()
        for it in day.picks:
            live_rows.append((
                f"#{it.pick.rank} {it.pick.ticker}",
                _price_compact(it.pick.entry_price),
            ))
    if live_rows:
        live_table = _render_box_table(
            headers=["Pick", "Entry"],
            widths=[9, 9],
            aligns=["<", ">"],
            rows=live_rows,
        )
        L.extend([
            "",
            f"<b>LIVE PICKS — {pending_label} (pending T+1 close10)</b>",
            f"<pre>{live_table}</pre>",
        ])

    hist_rows: list[tuple[str, ...]] = []
    for day in sorted(closed_days, key=lambda x: x.signal_date, reverse=True):
        for it in day.picks:
            if it.status != "closed":
                continue
            hist_rows.append((
                day.signal_date.isoformat(),
                f"#{it.pick.rank} {it.pick.ticker}",
                f"{_price_compact(it.pick.entry_price)} → {_price_compact(it.exit_price)}",
                _pct_unicode(it.net_return),
            ))
    title = "HISTORICAL PICKS [BOOTSTRAP]" if bootstrap_days else "HISTORICAL PICKS"
    L.extend(["", f"<b>{title}</b>"])
    if hist_rows:
        hist_table = _render_box_table(
            headers=["Date", "Pick", "Entry → Exit", "Net"],
            widths=[12, 9, 14, 9],
            aligns=["<", "<", "<", ">"],
            rows=hist_rows,
        )
        L.append(f"<pre>{hist_table}</pre>")
    else:
        L.append("(no closed picks yet)")

    notes = [f"picks_log: {'OK' if picks_log_ok else 'MISSING'}"]
    if pending_days:
        notes.append(f"{len(pending_days)} pending")
    if missing_days:
        notes.append(f"{len(missing_days)} missing days")
    if bootstrap_days:
        fetch_note = " w/ fetch" if bootstrap_fetch_missing_features else ""
        notes.append(f"{len(bootstrap_days)} bootstrap{fetch_note} (not saved to picks_log)")
    L.extend([
        "",
        " | ".join(notes),
        f"Cost: {cost * 100:.2f}% roundtrip",
    ])

    return "\n".join(L)


def send_recap(*, text: str, telegram: bool, discord: bool) -> None:
    load_notify_env()
    if telegram:
        token = os.getenv("BSJP_TELEGRAM_TOKEN", "")
        chat_id = os.getenv("BSJP_TELEGRAM_CHAT_ID", "")
        if token and chat_id:
            send_telegram_message(token, chat_id, text)
    if discord:
        webhook_url = os.getenv("BSJP_DISCORD_WEBHOOK_URL", "")
        if not webhook_url:
            token = os.getenv("BSJP_DISCORD_TOKEN", "")
            if token:
                webhook_url = token if token.startswith("https://") else f"https://discord.com/api/webhooks/{token}"
        if webhook_url:
            send_discord_webhook(webhook_url, text)


def _write_message(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--yf-1h-path", type=Path, default=DEFAULT_YF_1H_PATH)
    parser.add_argument("--bsjp-path", type=Path, default=DEFAULT_BSJP_PATH)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--as-of", help="Recap date in YYYY-MM-DD; default latest yfinance_1h date.")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--cost", type=float, default=DEFAULT_COST)
    parser.add_argument("--message-file", type=Path)
    parser.add_argument(
        "--bootstrap-missing",
        action="store_true",
        help="Starter-only mode: score missing recap dates with Go predict without writing picks_log.",
    )
    parser.add_argument(
        "--bootstrap-fetch-missing-features",
        action="store_true",
        help="Starter-only mode: run Go fetch for missing historical features before bootstrap predict.",
    )
    parser.add_argument("--send", action="store_true", help="Send to Telegram and Discord.")
    parser.add_argument("--telegram", action="store_true", help="Send to Telegram.")
    parser.add_argument("--discord", action="store_true", help="Send to Discord.")
    parser.add_argument("--no-print", action="store_true", help="Do not print the rendered message.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.days < 1:
        parser.error("--days must be >= 1")
    if args.cost < 0:
        parser.error("--cost must be >= 0")

    as_of = parse_date(args.as_of) if args.as_of else None
    days = build_recap(
        db_path=args.db_path,
        yf_1h_path=args.yf_1h_path,
        variant=args.variant,
        as_of=as_of,
        days=args.days,
        cost=args.cost,
        bootstrap_missing=args.bootstrap_missing,
        bootstrap_fetch_missing_features=args.bootstrap_fetch_missing_features,
        bsjp_path=args.bsjp_path,
    )
    rendered_as_of = as_of or (max((d.signal_date for d in days), default=datetime.now(WIB).date()))
    message = format_recap_message(
        days,
        as_of=rendered_as_of,
        variant=args.variant,
        cost=args.cost,
        bootstrap_missing=args.bootstrap_missing,
        bootstrap_fetch_missing_features=args.bootstrap_fetch_missing_features,
    )

    if args.message_file:
        _write_message(args.message_file, message)
    if not args.no_print:
        print(html.unescape(message))

    send_telegram = bool(args.send or args.telegram)
    send_discord = bool(args.send or args.discord)
    if send_telegram or send_discord:
        send_recap(text=message, telegram=send_telegram, discord=send_discord)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
