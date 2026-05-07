from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from pipeline.run import bsjp_pnl_recap as recap


def _pick(
    signal_date: date,
    ticker: str = "AAA",
    entry_price: float | None = 100.0,
    rank: int = 1,
) -> recap.Pick:
    return recap.Pick(
        signal_date=signal_date,
        variant=recap.DEFAULT_VARIANT,
        rank=rank,
        ticker=ticker,
        pred_proba=0.5,
        entry_price=entry_price,
    )


def _write_yf_1h(path: Path, rows: list[tuple[str, str, float]]) -> None:
    frame = pd.DataFrame(rows, columns=["datetime", "ticker", "close"])
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    frame["open"] = frame["close"]
    frame["high"] = frame["close"]
    frame["low"] = frame["close"]
    frame["volume"] = 1.0
    frame[["datetime", "ticker", "open", "high", "low", "close", "volume"]].to_parquet(path)


def _create_picks_db(path: Path, picks: list[tuple[str, int, str, float]]) -> None:
    con = duckdb.connect(str(path))
    con.execute(
        """
        CREATE TABLE picks_log (
            date DATE,
            variant VARCHAR,
            rank INTEGER,
            ticker VARCHAR,
            pred_proba DOUBLE,
            entry_price DOUBLE,
            exit_price DOUBLE,
            overnight_return DOUBLE,
            hit_tp BOOLEAN,
            logged_at TIMESTAMP DEFAULT now(),
            PRIMARY KEY (date, variant, rank)
        )
        """
    )
    for signal_date, rank, ticker, entry in picks:
        con.execute(
            """
            INSERT INTO picks_log (date, variant, rank, ticker, pred_proba, entry_price)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [signal_date, recap.DEFAULT_VARIANT, rank, ticker, 0.5, entry],
        )
    con.close()


def test_compute_pick_pnl_math_positive_negative_zero_and_missing():
    signal_date = date(2026, 5, 4)
    exit_date = date(2026, 5, 5)
    trading_dates = [signal_date, exit_date]
    exits = {
        (exit_date, "POS"): 110.0,
        (exit_date, "NEG"): 90.0,
        (exit_date, "FLAT"): 100.0,
    }

    positive = recap.compute_pick_pnl(
        _pick(signal_date, "POS"), trading_dates=trading_dates, exit_close=exits
    )
    negative = recap.compute_pick_pnl(
        _pick(signal_date, "NEG"), trading_dates=trading_dates, exit_close=exits
    )
    flat = recap.compute_pick_pnl(
        _pick(signal_date, "FLAT"), trading_dates=trading_dates, exit_close=exits
    )
    missing_entry = recap.compute_pick_pnl(
        _pick(signal_date, "POS", entry_price=None), trading_dates=trading_dates, exit_close=exits
    )
    missing_exit = recap.compute_pick_pnl(
        _pick(signal_date, "MISS"), trading_dates=trading_dates, exit_close=exits
    )

    assert positive.status == "closed"
    assert positive.gross_return == pytest.approx(0.10)
    assert positive.net_return == pytest.approx(0.096)
    assert negative.gross_return == pytest.approx(-0.10)
    assert negative.net_return == pytest.approx(-0.104)
    assert flat.gross_return == pytest.approx(0.0)
    assert flat.net_return == pytest.approx(-0.004)
    assert missing_entry.status == "missing"
    assert missing_entry.reason == "missing entry"
    assert missing_exit.status == "missing"
    assert missing_exit.reason == "missing exit 10:xx"


def test_trading_day_mapping_uses_next_valid_yfinance_day():
    friday = date(2026, 5, 1)
    monday = date(2026, 5, 4)
    tuesday = date(2026, 5, 5)

    normal = recap.compute_pick_pnl(
        _pick(friday),
        trading_dates=[friday, monday],
        exit_close={(monday, "AAA"): 103.0},
    )
    holiday = recap.compute_pick_pnl(
        _pick(friday),
        trading_dates=[friday, tuesday],
        exit_close={(tuesday, "AAA"): 104.0},
    )
    pending = recap.compute_pick_pnl(
        _pick(friday),
        trading_dates=[friday],
        exit_close={},
    )

    assert normal.exit_date == monday
    assert normal.status == "closed"
    assert holiday.exit_date == tuesday
    assert holiday.status == "closed"
    assert pending.status == "pending"
    assert pending.reason == "pending exit T+1 close10"


def test_build_recap_from_temp_duckdb_mixed_complete_pending_and_missing(tmp_path: Path):
    db_path = tmp_path / "inference.duckdb"
    yf_path = tmp_path / "yfinance_1h.parquet"
    signal_dates = [
        date(2026, 4, 27),
        date(2026, 4, 28),
        date(2026, 4, 29),
        date(2026, 4, 30),
        date(2026, 5, 1),
        date(2026, 5, 4),
        date(2026, 5, 5),
    ]
    picks = []
    for signal_date in signal_dates:
        if signal_date == date(2026, 4, 29):
            continue
        for rank, ticker in enumerate(["AAA", "BBB", "CCC"], start=1):
            picks.append((signal_date.isoformat(), rank, ticker, 100.0))
    _create_picks_db(db_path, picks)

    rows = []
    closes_by_date = {
        date(2026, 4, 27): 100.0,
        date(2026, 4, 28): 101.0,
        date(2026, 4, 29): 102.0,
        date(2026, 4, 30): 99.0,
        date(2026, 5, 1): 104.0,
        date(2026, 5, 4): 103.0,
        date(2026, 5, 5): 105.0,
    }
    for trading_date, close in closes_by_date.items():
        ts = f"{trading_date.isoformat()} 03:00:00+00:00"
        for ticker in ["AAA.JK", "BBB.JK", "CCC.JK"]:
            rows.append((ts, ticker, close))
    _write_yf_1h(yf_path, rows)

    days = recap.build_recap(
        db_path=db_path,
        yf_1h_path=yf_path,
        as_of=date(2026, 5, 5),
    )
    by_date = {d.signal_date: d for d in days}

    assert len(days) == 7
    assert by_date[date(2026, 4, 29)].status == "missing"
    assert by_date[date(2026, 5, 5)].status == "pending"
    assert by_date[date(2026, 5, 4)].status == "closed"
    assert by_date[date(2026, 5, 4)].closed_count == 3

    msg = recap.format_recap_message(days, as_of=date(2026, 5, 5))

    assert "<b>BSJP 7D PnL RECAP</b>" in msg
    assert f"Date: 2026-05-05 | {recap.DEFAULT_VARIANT}" in msg
    assert "<b>SUMMARY</b>" in msg
    assert "Win days:" in msg
    assert "<b>LIVE PICKS — 2026-05-05 (pending T+1 close10)</b>" in msg
    assert "<b>HISTORICAL PICKS</b>" in msg
    assert "<pre>" in msg and "</pre>" in msg
    # Tables contain expected tickers
    assert "#1 AAA" in msg
    assert "#2 BBB" in msg
    # Box-drawing characters present
    assert "┌" in msg and "└" in msg and "│" in msg
    assert "picks_log: OK" in msg
    assert "1 pending" in msg
    assert "1 missing days" in msg


def test_parse_go_predict_picks_marks_bootstrap_source():
    output = """
  debug ticker=AAA features=306 entry=100 proba=0.1234
Top-3 picks for 2026-05-04 (model: v19d_close10_preclose14_orb_md100_l21.5):
  #1 AAA  proba=0.4000  entry=100  features=306
  #2 BBB.JK  proba=0.3000  entry=200  features=306
  #3 CCC  proba=0.2000  entry=300  features=306
  #4 DDD  proba=0.1000  entry=400  features=306
""".strip()

    picks = recap.parse_go_predict_picks(
        output,
        signal_date=date(2026, 5, 4),
        variant=recap.DEFAULT_VARIANT,
    )

    assert [p.rank for p in picks] == [1, 2, 3]
    assert [p.ticker for p in picks] == ["AAA", "BBB", "CCC"]
    assert all(p.source == "bootstrap" for p in picks)


def test_format_recap_labels_bootstrap_source():
    signal_date = date(2026, 5, 4)
    exit_date = date(2026, 5, 5)
    picks = [
        recap.Pick(signal_date, recap.DEFAULT_VARIANT, 1, "AAA", 0.4, 100.0, "bootstrap"),
        recap.Pick(signal_date, recap.DEFAULT_VARIANT, 2, "BBB", 0.3, 100.0, "bootstrap"),
        recap.Pick(signal_date, recap.DEFAULT_VARIANT, 3, "CCC", 0.2, 100.0, "bootstrap"),
    ]
    day = recap.compute_day_pnl(
        signal_date,
        picks,
        trading_dates=[signal_date, exit_date],
        exit_close={(exit_date, "AAA"): 101.0, (exit_date, "BBB"): 101.0, (exit_date, "CCC"): 101.0},
    )

    msg = recap.format_recap_message([day], as_of=exit_date, bootstrap_missing=True)

    assert "Source: picks_log + STARTER bootstrap for missing days (not saved)" in msg
    assert "<b>HISTORICAL PICKS [BOOTSTRAP]</b>" in msg
    assert "2026-05-04" in msg
    assert "#1 AAA" in msg
    assert "picks_log: MISSING" in msg
    assert "1 bootstrap" in msg and "not saved to picks_log" in msg
