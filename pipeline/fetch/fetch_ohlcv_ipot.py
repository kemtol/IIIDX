"""
fetch_ohlcv_ipot.py — Fetch intraday OHLCV bars from IPOT TradingView feed.

Uses the same WebSocket as fetch_broksum_ipot.py (wss://ipotapp.ipot.id/socketcluster/).
Fetches 1-min bars (browser-confirmed resolution), aggregates to 1h.

Protocol (browser-captured):
    1. resolveSymbol("BBRI")        ← required before intraday getBars
    2. resolveSymbol("IDX.BBRI")    ← second resolve with IDX prefix
    3. getBars(resolution="1", ...)  ← intraday uses "1", not "15"
Public endpoint — no auth needed.

Response format (streaming records):
    {"rid":<cid>,"data":{"status":"OK","msg":"reply as record"}}
    {"event":"record","data":{"cmdid":<cmdid>,"recno":N,"data":"{...OHLCV json...}"}}
    ... more records ...
    (stream ends when no new records for IDLE_SECS)

Output: data/Level_0_Raw/ipot_ohlcv_1h.parquet

Usage
-----
    python fetch_ohlcv_ipot.py                       # today, full universe
    python fetch_ohlcv_ipot.py --date 2026-04-23     # specific date
    python fetch_ohlcv_ipot.py --tickers TLKM BBCA   # specific tickers
    python fetch_ohlcv_ipot.py --dry-run             # 1 ticker, print parsed bars
"""
import argparse
import asyncio
import json
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import websockets

WIB = ZoneInfo("Asia/Jakarta")

WS_URL   = "wss://ipotapp.ipot.id/socketcluster/?appsession="
ORIGIN   = "https://indopremier.com"

# resolution "1" matches what the browser uses for intraday (not "15")
RESOLUTION = "1"

REPO_ROOT   = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "data/Level_0_Raw/ipot_ohlcv_1h.parquet"
MASTER_PATH = REPO_ROOT / "data/Level_0_Raw/master_emiten.parquet"

IDLE_SECS   = 2.0   # stop collecting if no record arrives within this window
MAX_SECS    = 15.0  # hard timeout per ticker


def _symbol_info(ticker: str) -> dict:
    return {
        "name":                  ticker,
        "ticker":                f"IDX.{ticker}",
        "exchange":              "IDX",
        "listed_exchange":       "IDX",
        "type":                  "stock",
        "timezone":              "Asia/Jakarta",
        "sector":                "ORDI_PREOPEN",
        "pricescale":            1,
        "minmov":                1,
        "fractional":            False,
        "has_intraday":          True,
        "supportedResolutions":  ["1", "15", "D"],
        "intraday_multipliers":  ["1"],
        "has_seconds":           False,
        "has_daily":             True,
        "has_weekly_and_monthly": False,
        "has_empty_bars":        True,
        "force_session_rebuild": True,
        "volume_precision":      0,
        "data_status":           "streaming",
        "visible_plots_set":     True,
        "base_name":             [ticker],
        "legs":                  [ticker],
        "full_name":             f"IDX:{ticker}",
        "pro_name":              f"IDX:{ticker}",
        "session":               "0900-1130,1400-1615:6|0900-1200,1330-1615",
    }


class IPOTSession:
    """Single WebSocket connection with a shared message dispatcher.

    Multiple concurrent fetch_bars_1m() calls can share one connection safely
    because only one coroutine calls recv() (the _listen loop), and each request
    gets its own asyncio.Queue keyed by cmdid.
    """

    def __init__(self, ws):
        self.ws      = ws
        self._queues: dict[int, asyncio.Queue] = {}
        self._listen_task: asyncio.Task | None = None
        self._cid_counter = 100  # incrementing to avoid collisions

    def _next_cid(self) -> int:
        self._cid_counter += 1
        return self._cid_counter

    async def start(self):
        self._listen_task = asyncio.create_task(self._listen())

    async def stop(self):
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        await self.ws.close()

    async def _listen(self):
        """Single recv loop — dispatches messages to per-request queues."""
        try:
            async for raw in self.ws:
                try:
                    pkt = json.loads(raw)
                except Exception:
                    continue

                # Status reply: {"rid":<cid>, "data":{...}}
                rid = pkt.get("rid")
                if rid is not None and rid in self._queues:
                    await self._queues[rid].put(("status", pkt.get("data", {})))
                    continue

                # Record stream: {"event":"record","data":{"cmdid":...,"data":"..."}}
                if pkt.get("event") == "record":
                    rec   = pkt.get("data", {})
                    cmdid = rec.get("cmdid")
                    if cmdid in self._queues:
                        try:
                            bar = json.loads(rec["data"])
                            await self._queues[cmdid].put(("bar", bar))
                        except Exception:
                            pass
        except Exception:
            pass  # connection closed

    async def _resolve_symbol(self, symbol_name: str) -> bool:
        """Send resolveSymbol and wait for status reply.

        Browser always sends two resolveSymbol calls (plain name + IDX.-prefixed)
        before getBars for intraday data. The server seems to require this to
        initialize the symbol context for intraday feeds.
        """
        cid   = self._next_cid()
        cmdid = self._next_cid()
        q: asyncio.Queue = asyncio.Queue()
        self._queues[cid] = q

        req = {
            "event": "tvdfeed",
            "data": {
                "cmdid": cmdid,
                "param": {
                    "cmd":        "resolveSymbol",
                    "symbolName": symbol_name,
                }
            },
            "cid": cid,
        }
        await self.ws.send(json.dumps(req))
        try:
            _, payload = await asyncio.wait_for(q.get(), timeout=5.0)
            return payload.get("status") == "OK"
        except asyncio.TimeoutError:
            return False
        finally:
            self._queues.pop(cid, None)

    async def fetch_bars_1m(self, ticker: str, target_date: date) -> list[dict]:
        """Fetch 1-min bars. Safe to call concurrently.

        Mirrors the browser flow: resolveSymbol × 2 → getBars.
        Resolution "1" is what IPOT intraday uses (browser-confirmed).
        """
        await self._resolve_symbol(f"IDX.{ticker}")

        day_start = int(datetime(target_date.year, target_date.month, target_date.day,
                                 9, 0, tzinfo=WIB).timestamp())
        day_end   = int(datetime(target_date.year, target_date.month, target_date.day,
                                 16, 15, tzinfo=WIB).timestamp())

        cid   = self._next_cid()
        cmdid = self._next_cid()
        q: asyncio.Queue = asyncio.Queue()
        self._queues[cid]   = q
        self._queues[cmdid] = q

        req = {
            "event": "tvdfeed",
            "data": {
                "cmdid": cmdid,
                "param": {
                    "cmd":              "getBars",
                    "symbolInfo":       _symbol_info(ticker),
                    "resolution":       RESOLUTION,
                    "from":             day_start,
                    "to":               day_end,
                    "firstDataRequest": True,
                }
            },
            "cid": cid,
        }
        await self.ws.send(json.dumps(req))

        bars     = []
        deadline = time.time() + MAX_SECS

        try:
            while time.time() < deadline:
                remaining = deadline - time.time()
                try:
                    kind, payload = await asyncio.wait_for(q.get(), timeout=min(IDLE_SECS, remaining))
                except asyncio.TimeoutError:
                    if bars:
                        break  # idle after receiving data = stream ended
                    continue

                if kind == "status":
                    if payload.get("status") != "OK":
                        break  # NODATA or error
                elif kind == "bar":
                    bars.append(payload)
        finally:
            self._queues.pop(cid, None)
            self._queues.pop(cmdid, None)

        return bars


async def create_session() -> IPOTSession:
    ws = await websockets.connect(
        WS_URL,
        additional_headers={"Origin": ORIGIN, "User-Agent": "ohlcv-fetcher/1.0"},
        compression=None,
        ping_interval=None,
    )
    await ws.send(json.dumps({"event": "#handshake", "data": {"authToken": None}, "cid": 1}))
    await asyncio.sleep(0.5)  # let handshake settle
    session = IPOTSession(ws)
    await session.start()
    return session


def bars_to_df(bars: list[dict], ticker: str) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    df["datetime"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert(WIB)
    df["date"]     = df["datetime"].dt.date
    df["ticker"]   = ticker
    return df[["ticker", "date", "datetime", "open", "high", "low", "close", "volume"]]


def aggregate_to_1h(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["hour"] = df["datetime"].dt.floor("1h")
    agg = df.groupby(["ticker", "date", "hour"]).agg(
        open   =("open",   "first"),
        high   =("high",   "max"),
        low    =("low",    "min"),
        close  =("close",  "last"),
        volume =("volume", "sum"),
    ).reset_index().rename(columns={"hour": "datetime"})
    return agg


def load_universe() -> list[str]:
    df  = pd.read_parquet(MASTER_PATH)
    col = next(c for c in ["ticker", "stock_code", df.columns[0]] if c in df.columns)
    tickers = df[col].dropna().unique().tolist()
    # master_emiten uses yfinance format (BBRI.JK) — IPOT needs bare code (BBRI)
    tickers = [t.replace(".JK", "") for t in tickers]
    return sorted(tickers)


def upsert_parquet(new_df: pd.DataFrame, target_date: date) -> None:
    if OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        existing["date"] = pd.to_datetime(existing["date"]).dt.date
        existing = existing[existing["date"] != target_date]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.sort_values(["date", "ticker", "datetime"]).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"[OHLCV] Saved → {OUTPUT_PATH.name}  ({len(new_df)} new rows, {len(combined)} total)")


async def run(target_date: date, tickers: list[str],
              dry_run: bool, concurrency: int) -> pd.DataFrame:
    print(f"[OHLCV] date={target_date} | tickers={len(tickers)} | concurrency={concurrency}")

    session  = await create_session()
    sem      = asyncio.Semaphore(concurrency)
    all_rows: list[pd.DataFrame] = []
    errors:   list[str] = []
    done      = 0

    async def fetch_one(ticker: str):
        nonlocal done
        async with sem:
            bars = await session.fetch_bars_1m(ticker, target_date)
            done += 1
            if not bars:
                errors.append(ticker)
                return
            df_1m = bars_to_df(bars, ticker)
            df_1h = aggregate_to_1h(df_1m)
            if dry_run:
                print(f"\n=== {ticker}: {len(bars)} bars (1m) → {len(df_1h)} bars (1h) ===")
                print(df_1h.to_string(index=False))
            else:
                all_rows.append(df_1h)
                if done % 50 == 0:
                    print(f"[OHLCV] {done}/{len(tickers)} done, {len(errors)} errors")

    tasks = [asyncio.create_task(fetch_one(t)) for t in tickers]
    await asyncio.gather(*tasks)
    await session.stop()

    if errors:
        print(f"[OHLCV] No data: {len(errors)} tickers "
              f"({', '.join(errors[:10])}{'...' if len(errors) > 10 else ''})")

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date",        default=str(date.today()))
    p.add_argument("--tickers",     nargs="*")
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--dry-run",     action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    target  = date.fromisoformat(args.date)
    tickers = args.tickers or load_universe()

    if args.dry_run:
        tickers = tickers[:3]

    result = asyncio.run(run(target, tickers, args.dry_run, args.concurrency))

    if not args.dry_run and not result.empty:
        upsert_parquet(result, target)
    elif not args.dry_run:
        print("[OHLCV] No data fetched.")
