#!/usr/bin/env python3
"""
Fetch Broker Summary Data Directly from IPOT (IndoPremier)

Script ini scraping langsung dari IPOT WebSocket untuk mendapatkan
data broker activity per-broker-per-date dengan detail per-stock.

Logic diambil dari broksum-scrapper (JavaScript/Cloudflare Worker) dan
diimplementasikan ulang di Python untuk kebutuhan ML.

Data flow:
  1. Fetch appsession dari indopremier.com
  2. Connect WebSocket ke ipotapp.ipot.id
  3. Query broker activity (buy-sorted dan sell-sorted)
  4. Parse dan merge records
  5. Simpan ke parquet (per broker per date, gross, no aggregation)

Usage:
  python fetch-broksum-ipot.py --broker MG --date 2026-04-09
  python fetch-broksum-ipot.py --broker MG --days 20  # Fetch 20 hari
  python fetch-broksum-ipot.py --brokers "MG,ZP,YP" --days 5
"""

import argparse
import asyncio
import json
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import aiohttp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import websockets

# ── Config ──
IDX_DIR = Path(__file__).parent.parent.parent  # pipeline/fetch/ -> pipeline/ -> idx/
BASE_DATA_DIR = IDX_DIR / "data"
LEVEL0_DATA_DIR = BASE_DATA_DIR / "Level_0_Raw"
DATA_DIR = LEVEL0_DATA_DIR if LEVEL0_DATA_DIR.exists() else BASE_DATA_DIR
DEFAULT_PROGRESS_LOG = IDX_DIR / "_LOG" / "log_broksum.jsonc"
DEFAULT_RESUME_STATE = IDX_DIR / "_LOG" / "broksum_resume_state.json"

IPOT_CONFIG = {
    "appsession_url": "https://indopremier.com/ipc/appsession.js",
    "origin": "https://indopremier.com",
    "ws_base": "wss://ipotapp.ipot.id/socketcluster/",
    "idle_ms": 800,
    "max_ms": 8000,
    "user_agent": "broksum-scrapper/1.0",
}

POPULAR_BROKERS = [
    'ZP', 'YU', 'KZ', 'RX', 'ML', 'CC', 'CS', 'DB', 'MS', 'YP', 'MG', 'LG', 'BK', 'AK', 'CG', 'DX', 'HP',
    'NI', 'EP', 'IF', 'PD', 'AI', 'DR', 'KI', 'TP', 'BZ', 'DH', 'AZ', 'XA', 'IB', 'GI',
    'YO', 'PG', 'AP', 'GL', 'SH', 'CP'
]


def generate_recent_weekdays(days: int, end_date: datetime | None = None) -> list[str]:
    """Generate recent weekday dates in YYYY-MM-DD (newest -> oldest)."""
    dates: list[str] = []
    current = end_date or datetime.now()
    while len(dates) < max(0, days):
        if current.weekday() < 5:
            dates.append(current.strftime("%Y-%m-%d"))
        current -= timedelta(days=1)
    return dates


def generate_weekdays_between(from_date: str, to_date: str) -> list[str]:
    """
    Generate weekday dates between from_date and to_date (inclusive), newest -> oldest.
    """
    start = datetime.strptime(from_date, "%Y-%m-%d")
    end = datetime.strptime(to_date, "%Y-%m-%d")
    if start > end:
        raise ValueError("--from-date must be <= --to-date")

    dates: list[str] = []
    current = end
    while current >= start:
        if current.weekday() < 5:
            dates.append(current.strftime("%Y-%m-%d"))
        current -= timedelta(days=1)
    return dates


def _safe_tag(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip())
    return safe or "all"


def build_standard_json_filename(
    brokers: list[str],
    emiten: str | None,
    date_from: str,
    date_until: str,
) -> str:
    broker_tag = brokers[0].upper() if len(brokers) == 1 else "MULTI"
    emiten_tag = _safe_tag(emiten.upper()) if emiten else "all"
    return f"broksum_{broker_tag}_{emiten_tag}_{date_from}_{date_until}.json"


def save_export_json(
    df: pd.DataFrame,
    json_path: Path,
    *,
    brokers: list[str],
    emiten: str | None,
    date_from: str,
    date_until: str,
):
    """Save fetched result rows as standardized JSON export."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_df = df.copy()
    if "date" in json_df.columns:
        json_df["date"] = json_df["date"].astype(str)
    if {"date", "broker", "stock_code"}.issubset(set(json_df.columns)):
        json_df = json_df.sort_values(["date", "broker", "stock_code"])

    payload = {
        "generated_at": datetime.now().isoformat(),
        "source": "IPOT Direct (WebSocket)",
        "brokers": brokers,
        "emiten_filter": emiten.upper() if emiten else "all",
        "date_from": date_from,
        "date_until": date_until,
        "rows": len(json_df),
        "data": json_df.to_dict(orient="records"),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n[JSON] Exported {len(json_df):,} rows to {json_path}")
    print(f"[JSON] File size: {json_path.stat().st_size / 1024 / 1024:.2f} MB")


def append_progress_log(
    log_path: Path,
    *,
    service: str,
    action: str,
    result: str,
    details: dict[str, Any],
):
    """
    Append one event into a JSONC document so editors can parse without errors.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "timesstamp": datetime.now().isoformat(),  # keep compatibility with user's dummy key
        "service": service,
        "action": action,
        "result": result,
        "details": details,
    }

    def strip_jsonc_comments(text: str) -> str:
        """Remove // and /* */ comments from JSONC-like text."""
        out = []
        i = 0
        in_str = False
        escaped = False
        while i < len(text):
            ch = text[i]
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if in_str:
                out.append(ch)
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                i += 1
                continue
            if ch == '"':
                in_str = True
                out.append(ch)
                i += 1
                continue
            if ch == "/" and nxt == "/":
                i += 2
                while i < len(text) and text[i] not in ("\n", "\r"):
                    i += 1
                continue
            if ch == "/" and nxt == "*":
                i += 2
                while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                i += 2
                continue
            out.append(ch)
            i += 1
        return "".join(out)

    def load_doc() -> dict[str, Any]:
        if not log_path.exists() or log_path.stat().st_size == 0:
            return {"format": "broksum-progress-log-v1", "events": []}

        raw = log_path.read_text(encoding="utf-8", errors="ignore")
        cleaned = strip_jsonc_comments(raw).strip()
        if cleaned:
            try:
                obj = json.loads(cleaned)
                if isinstance(obj, dict) and isinstance(obj.get("events"), list):
                    return obj
            except Exception:
                pass

        # Backward compatibility: migrate JSONL-style historical logs.
        events: list[dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or not line.startswith("{") or not line.endswith("}"):
                continue
            try:
                one = json.loads(line)
            except Exception:
                continue
            if isinstance(one, dict):
                events.append(one)
        return {"format": "broksum-progress-log-v1", "events": events}

    doc = load_doc()
    events = doc.get("events")
    if not isinstance(events, list):
        events = []
        doc["events"] = events
    events.append(payload)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_resume_state(path: Path) -> dict[str, Any]:
    """Load per-broker completed date checkpoints for resume-safe runs."""
    if not path.exists() or path.stat().st_size == 0:
        return {"format": "broksum-resume-state-v1", "updated_at": None, "completed_dates": {}}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"format": "broksum-resume-state-v1", "updated_at": None, "completed_dates": {}}

    if not isinstance(doc, dict):
        return {"format": "broksum-resume-state-v1", "updated_at": None, "completed_dates": {}}

    completed = doc.get("completed_dates")
    if not isinstance(completed, dict):
        completed = {}

    normalized: dict[str, list[str]] = {}
    for broker, dates in completed.items():
        b = str(broker).strip().upper()
        if not b:
            continue
        if isinstance(dates, list):
            normalized[b] = sorted(set(str(d) for d in dates if d))

    return {
        "format": "broksum-resume-state-v1",
        "updated_at": doc.get("updated_at"),
        "completed_dates": normalized,
    }


def save_resume_state(path: Path, state: dict[str, Any]):
    """Persist resume state atomically to avoid partial writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    state["format"] = "broksum-resume-state-v1"
    state["updated_at"] = datetime.now().isoformat()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def get_resume_completed_dates(state: dict[str, Any], broker: str) -> set[str]:
    completed = state.get("completed_dates", {})
    if not isinstance(completed, dict):
        return set()
    raw = completed.get(broker.upper(), [])
    if not isinstance(raw, list):
        return set()
    return set(str(x) for x in raw if x)


def add_resume_completed_dates(state: dict[str, Any], broker: str, dates: list[str]) -> int:
    """Add completed dates for one broker. Returns number of newly added dates."""
    completed = state.setdefault("completed_dates", {})
    if not isinstance(completed, dict):
        completed = {}
        state["completed_dates"] = completed
    b = broker.upper()
    prev = set(completed.get(b, [])) if isinstance(completed.get(b, []), list) else set()
    new_set = prev.union(set(str(d) for d in dates if d))
    completed[b] = sorted(new_set)
    return len(new_set) - len(prev)


def load_brokers_from_master(master_path: Path, include_inactive: bool = False) -> list[str]:
    """Load broker codes from local master_broker parquet."""
    if not master_path.exists():
        print(f"[BrokerList] Master not found at {master_path}, fallback popular list.")
        return POPULAR_BROKERS

    try:
        df = pd.read_parquet(master_path)
    except Exception as e:
        print(f"[BrokerList] Failed reading master broker parquet: {e}, fallback popular list.")
        return POPULAR_BROKERS

    if "broker_code" in df.columns:
        code_col = "broker_code"
    elif "code" in df.columns:
        code_col = "code"
    else:
        print("[BrokerList] No broker code column in master broker parquet, fallback popular list.")
        return POPULAR_BROKERS

    active_mask = pd.Series([True] * len(df), index=df.index)
    if not include_inactive:
        if "is_active" in df.columns:
            active_mask = df["is_active"].fillna(False).astype(bool)
        elif "status" in df.columns:
            active_mask = df["status"].astype(str).str.upper().eq("ACTIVE")

    brokers = (
        df.loc[active_mask, code_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    brokers = [b for b in brokers.tolist() if b]
    brokers = sorted(set(brokers))
    if not brokers:
        print("[BrokerList] Empty broker list from master, fallback popular list.")
        return POPULAR_BROKERS

    return brokers


def merge_append_with_upsert(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Append+upsert by (broker, stock_code, date), keeping the latest scraped_at."""
    if existing_df is None or existing_df.empty:
        return new_df
    if new_df is None or new_df.empty:
        return existing_df

    combined = pd.concat([existing_df, new_df], ignore_index=True)
    if "scraped_at" in combined.columns:
        combined["scraped_at"] = pd.to_datetime(combined["scraped_at"], errors="coerce")
        combined = combined.sort_values(["date", "broker", "stock_code", "scraped_at"])
    else:
        combined = combined.sort_values(["date", "broker", "stock_code"])
    combined = combined.drop_duplicates(subset=["broker", "stock_code", "date"], keep="last")
    return combined.reset_index(drop=True)


def normalize_date_series_to_str(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    out = dt.dt.strftime("%Y-%m-%d")
    return out


def load_cached_subset(
    parquet_path: Path,
    brokers: list[str],
    target_dates: list[str],
) -> pd.DataFrame:
    """Load cached rows for selected brokers + date list from parquet store."""
    if not parquet_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"[Cache] Failed reading parquet store {parquet_path}: {e}")
        return pd.DataFrame()

    if "broker" not in df.columns or "date" not in df.columns:
        print(f"[Cache] Parquet store missing broker/date columns: {parquet_path}")
        return pd.DataFrame()

    df = df.copy()
    df["broker"] = df["broker"].astype(str).str.upper().str.strip()
    df["date"] = normalize_date_series_to_str(df["date"])
    mask = df["broker"].isin(set(brokers)) & df["date"].isin(set(target_dates))
    return df.loc[mask].copy()


def compute_missing_dates_map(
    cached_subset: pd.DataFrame,
    brokers: list[str],
    target_dates: list[str],
) -> dict[str, list[str]]:
    """Return per-broker missing date list (based on any broker-day row presence in cache)."""
    if cached_subset.empty:
        return {b: list(target_dates) for b in brokers}

    coverage = (
        cached_subset.loc[:, ["broker", "date"]]
        .dropna()
        .drop_duplicates()
        .groupby("broker")["date"]
        .apply(set)
        .to_dict()
    )
    out: dict[str, list[str]] = {}
    for broker in brokers:
        covered = coverage.get(broker, set())
        out[broker] = [d for d in target_dates if d not in covered]
    return out


def resolve_missing_dates_for_broker(
    parquet_path: Path,
    broker: str,
    target_dates: list[str],
) -> list[str]:
    """
    Recompute missing dates for one broker directly from parquet store.
    Useful for resume-safe runs and incremental flush workflows.
    """
    cached_subset = load_cached_subset(parquet_path, brokers=[broker], target_dates=target_dates)
    missing_map = compute_missing_dates_map(cached_subset, brokers=[broker], target_dates=target_dates)
    return missing_map.get(broker, list(target_dates))


def flush_broker_rows_to_store(
    parquet_store: Path,
    broker_rows: pd.DataFrame,
    *,
    broker: str,
    stage=None,
) -> dict[str, int]:
    """
    Flush one broker chunk into parquet store with upsert semantics.
    If stage is provided, appends to DuckDB staging (crash-safe).
    """
    if broker_rows is None or broker_rows.empty:
        return {"before_rows": 0, "after_rows": 0, "delta_rows": 0}

    if stage is not None:
        stage.append(broker_rows)
        return {"before_rows": 0, "after_rows": len(broker_rows), "delta_rows": len(broker_rows)}

    if parquet_store.exists():
        existing_df = pd.read_parquet(parquet_store)
        before_rows = len(existing_df)
        merged_df = merge_append_with_upsert(existing_df, broker_rows)
    else:
        before_rows = 0
        merged_df = broker_rows.copy()

    save_to_parquet(merged_df, parquet_store)
    after_rows = len(merged_df)
    delta_rows = after_rows - before_rows
    print(
        f"[Flush] Broker {broker}: input_rows={len(broker_rows):,}, "
        f"store_rows {before_rows:,} -> {after_rows:,} (delta={delta_rows:+,})"
    )
    return {"before_rows": before_rows, "after_rows": after_rows, "delta_rows": delta_rows}


class IPOTScraper:
    """Scraper for IPOT broker activity data via WebSocket."""
    
    def __init__(self):
        self.appsession: Optional[str] = None
        self.ws = None
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                "User-Agent": IPOT_CONFIG["user_agent"],
                "Accept": "*/*",
                "Origin": IPOT_CONFIG["origin"],
                "Referer": IPOT_CONFIG["origin"] + "/",
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        if self.ws and not self.ws.closed:
            await self.ws.close()
            
    def _make_ws_key(self) -> str:
        """Generate RFC 6455 compliant WebSocket key."""
        import base64
        import os
        return base64.b64encode(os.urandom(16)).decode()
    
    async def _get_appsession(self) -> str:
        """Fetch appsession token from indopremier.com."""
        if self.appsession:
            return self.appsession
            
        url = IPOT_CONFIG["appsession_url"]
        
        try:
            async with self.session.get(url) as resp:
                if not resp.ok:
                    raise Exception(f"appsession fetch failed: {resp.status}")
                    
                text = await resp.text()
                
                # Try multiple patterns to extract appsession
                patterns = [
                    r'appsession\s*[:=]\s*["\']([^"\']+)["\']',
                    r'appsession=([a-zA-Z0-9\-_]+)',
                    r'["\']appsession["\']\s*[:,]\s*["\']([^"\']+)["\']',
                ]
                
                token = None
                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        token = match.group(1)
                        break
                        
                if not token:
                    # Fallback: look for token near "appsession"
                    near_match = re.search(r'appsession[^A-Za-z0-9\-_]{0,30}([A-Za-z0-9\-_]{16,128})', text, re.IGNORECASE)
                    if near_match:
                        token = near_match.group(1)
                        
                if not token:
                    raise Exception(f"Could not parse appsession from {url}")
                    
                self.appsession = token
                print(f"[IPOT] Got appsession: {token[:20]}...")
                return token
                
        except Exception as e:
            raise Exception(f"Failed to get appsession: {e}")
    
    async def _connect_websocket(self):
        """Connect to IPOT WebSocket."""
        appsession = await self._get_appsession()
        
        # Build WebSocket URL with appsession
        ws_url = f"{IPOT_CONFIG['ws_base']}?appsession={appsession}"
        
        # Headers for WebSocket upgrade (additional_headers for newer websockets versions)
        headers = {
            "Origin": IPOT_CONFIG["origin"],
            "User-Agent": IPOT_CONFIG["user_agent"],
        }
        
        try:
            ws = await websockets.connect(
                ws_url,
                additional_headers=headers,
                compression=None,  # Disable compression to avoid issues
                ping_interval=None,
            )
            
            # Send handshake
            handshake = {
                "event": "#handshake",
                "data": {"authToken": None},
                "cid": 1
            }
            await ws.send(json.dumps(handshake))
            
            print(f"[IPOT] WebSocket connected")
            return ws
            
        except Exception as e:
            print(f"[IPOT] WS connect failed: {e}, retrying with new session...")
            # Clear session and retry once
            self.appsession = None
            appsession = await self._get_appsession()
            ws_url = f"{IPOT_CONFIG['ws_base']}?appsession={appsession}"
            
            ws = await websockets.connect(
                ws_url,
                extra_headers=headers,
                compression=None,
                ping_interval=None,
            )
            
            handshake = {
                "event": "#handshake",
                "data": {"authToken": None},
                "cid": 1
            }
            await ws.send(json.dumps(handshake))
            
            print(f"[IPOT] WebSocket connected (retry)")
            return ws
    
    async def _query_broker(self, ws, broker: str, date: str, ordercol: int) -> tuple[list[str], Optional[list]]:
        """
        Query broker activity from WebSocket.
        
        Args:
            ws: WebSocket connection
            broker: Broker code (e.g., 'MG')
            date: Date in YYYY-MM-DD format
            ordercol: 1 for buy-sorted, 4 for sell-sorted
            
        Returns:
            (records, sum_data)
        """
        import time
        
        # Format date for IPOT: YYYY-M-DD (non-zero-padded month)
        y, m, d = date.split("-")
        ipot_date = f"{y}-{int(m)}-{d}"
        
        cmdid = f"{int(time.time() * 1000)}_{random.random()}"
        cid = f"{int(time.time() * 1000)}_{random.random()}"
        
        # Build query message
        msg = {
            "event": "cmd",
            "data": {
                "cmdid": cmdid,
                "param": {
                    "cmd": "query",
                    "service": "midata",
                    "param": {
                        "source": "datafeed",
                        "index": "en_qu_top_bs",
                        "args": ["s", "", broker, "", "%", ipot_date, ipot_date],
                        "info": {
                            "orderby": [[ordercol, "DESC", "N"]],
                            "sum": [1, 4, 7, 9, 10, 2, 8, 3, 5, 6]
                        },
                        "pagelen": 100,
                        "slid": ""
                    }
                }
            },
            "cid": cid
        }
        
        await ws.send(json.dumps(msg))
        print(f"  [Query] ordercol={ordercol}, cmdid={cmdid[:20]}...")
        
        # Collect responses
        records = []
        sum_data = None
        
        start_at = time.time()
        last_at = start_at
        got_res = False
        res_at = None
        
        IDLE_MS = IPOT_CONFIG["idle_ms"]
        MAX_MS = IPOT_CONFIG["max_ms"]
        
        try:
            while True:
                now = time.time()
                
                # Timeout checks
                if (now - start_at) * 1000 > MAX_MS:
                    print(f"  [Query] Max timeout reached")
                    break
                if got_res and res_at and (now - res_at) * 1000 > 0.5:
                    break
                if len(records) > 0 and (now - last_at) * 1000 > IDLE_MS:
                    break
                if len(records) == 0 and (now - last_at) * 1000 > 2000:
                    break
                    
                # Try to receive message with timeout
                try:
                    msg_text = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    last_at = time.time()
                    
                    # Handle ping/pong
                    if msg_text == "#1":
                        await ws.send("#2")
                        continue
                        
                    try:
                        data = json.loads(msg_text)
                    except:
                        continue
                    
                    # Check for record event with matching cmdid
                    if data.get("event") == "record" and data.get("data", {}).get("cmdid") == cmdid:
                        rec_data = data.get("data", {}).get("data", {})
                        line = rec_data.get("rec", {}).get("en_qu_top_bs") if rec_data.get("rec") else rec_data.get("en_qu_top_bs")
                        
                        if isinstance(line, str):
                            records.append(line)
                            
                        # Get sum data if available
                        if rec_data.get("sum") and not sum_data:
                            sum_data = rec_data["sum"]
                            
                    # Check for response event
                    if data.get("event") == "res" and data.get("data", {}).get("cmdid") == cmdid:
                        got_res = True
                        res_at = time.time()
                        
                except asyncio.TimeoutError:
                    continue
                    
        except Exception as e:
            print(f"  [Query] Error: {e}")
            
        print(f"  [Query] ordercol={ordercol}: {len(records)} records, sum={sum_data is not None}")
        return records, sum_data
    
    def _parse_record(self, line: str) -> Optional[dict]:
        """Parse a pipe-separated record line."""
        parts = line.split("|")
        if len(parts) < 10:
            return None
            
        return {
            "stock_code": parts[0],
            "buy_val": parts[1] or "0",
            "buy_vol": parts[2] or "0",
            "buy_freq": parts[3] or "0",
            "sell_val": parts[4] or "0",
            "sell_vol": parts[5] or "0",
            "sell_freq": parts[6] or "0",
            "net_val": parts[7] or "0",
            "net_vol": parts[8] or "0",
            "total_val": parts[9] or "0",
            "total_vol": parts[10] if len(parts) > 10 else "0",
        }
    
    def _merge_records(self, buy_records: list[str], sell_records: list[str]) -> list[dict]:
        """Merge buy and sell records, dedup by stock code."""
        stock_map = {}
        
        for line in buy_records:
            parsed = self._parse_record(line)
            if parsed:
                code = parsed["stock_code"]
                if code not in stock_map:
                    stock_map[code] = parsed
                    
        for line in sell_records:
            parsed = self._parse_record(line)
            if parsed:
                code = parsed["stock_code"]
                if code not in stock_map:
                    stock_map[code] = parsed
                    
        # Convert to list and sort by absolute net_val desc
        stocks = list(stock_map.values())
        stocks.sort(key=lambda x: abs(int(x.get("net_val", "0").replace(",", "")) or 0), reverse=True)
        
        return stocks
    
    def _get_broker_type(self, broker: str) -> str:
        """Determine broker type (placeholder - can be enhanced with D1 lookup)."""
        # TODO: Implement proper broker classification
        # For now, return Unknown - can be enriched later
        return "Unknown"
    
    async def scrape_broker_date(self, broker: str, date: str) -> dict[str, Any]:
        """
        Scrape broker activity for a specific date.
        
        Returns data in format (status-aware):
        {
            "ok": True,
            "status": "success",
            "broker": "MG",
            "broker_type": "Asing",
            "date": "2026-04-09",
            "breadth": 25,
            "stocks": [...]
        }
        """
        print(f"[Scrape] {broker} / {date}")
        
        try:
            # Connect WebSocket
            ws = await self._connect_websocket()
            
            try:
                # Query buy-sorted
                buy_records, _ = await self._query_broker(ws, broker, date, ordercol=1)
                
                # Small delay between queries
                await asyncio.sleep(0.08)
                
                # Query sell-sorted
                sell_records, sum_data = await self._query_broker(ws, broker, date, ordercol=4)
                
                # Merge records
                stocks = self._merge_records(buy_records, sell_records)
                
                if not stocks:
                    print(f"  [Scrape] No stocks found for {broker}/{date}")
                    return {
                        "ok": False,
                        "status": "no_data",
                        "broker": broker,
                        "date": date,
                        "scraped_at": datetime.now().isoformat(),
                        "stocks": [],
                    }
                    
                # Build summary from sum_data if available
                summary = None
                if sum_data and len(sum_data) >= 10:
                    summary = {
                        "total_buy_val": str(sum_data[0] or 0),
                        "total_sell_val": str(sum_data[1] or 0),
                        "total_net_val": str(sum_data[2] or 0),
                        "total_trading_val": str(sum_data[3] or 0),
                        "total_vol": str(sum_data[4] or 0),
                        "total_buy_vol": str(sum_data[5] or 0),
                        "total_net_vol": str(sum_data[6] or 0),
                        "total_buy_freq": str(sum_data[7] or 0),
                        "total_sell_vol": str(sum_data[8] or 0),
                        "total_sell_freq": str(sum_data[9] or 0),
                    }
                
                result = {
                    "ok": True,
                    "status": "success",
                    "broker": broker,
                    "broker_type": self._get_broker_type(broker),
                    "date": date,
                    "scraped_at": datetime.now().isoformat(),
                    "breadth": len(stocks),
                    "summary": summary,
                    "stocks": stocks
                }
                
                print(f"  [Scrape] Success: {len(stocks)} stocks")
                return result
                
            finally:
                await ws.close()
                
        except Exception as e:
            print(f"  [Scrape] Error: {e}")
            return {
                "ok": False,
                "status": "error",
                "broker": broker,
                "date": date,
                "error": str(e),
                "error_type": type(e).__name__,
                "scraped_at": datetime.now().isoformat(),
                "stocks": [],
            }
    
    async def scrape_broker_days(self, broker: str, days: int) -> list[dict]:
        """Scrape broker activity for multiple days."""
        dates = generate_recent_weekdays(days)
        return await self.scrape_broker_dates(broker, dates)

    async def scrape_broker_dates(self, broker: str, dates: list[str], day_sleep: float = 0.5) -> list[dict]:
        """Scrape broker activity for a list of dates (newest->oldest or any order)."""
        results = []
        if not dates:
            print(f"[Scrape] {broker}: 0 dates (skip)")
            return results
        print(f"[Scrape] {broker}: {len(dates)} dates from {dates[-1]} to {dates[0]}")

        for idx, date_str in enumerate(dates):
            result = await self.scrape_broker_date(broker, date_str)
            results.append(result)
            if idx < len(dates) - 1 and day_sleep > 0:
                await asyncio.sleep(day_sleep)

        return results


def flatten_to_master(data_list: list[dict]) -> list[dict]:
    """
    Flatten scraped data to master format (one row per broker-stock-date).
    """
    rows = []
    
    for data in data_list:
        broker = data["broker"]
        date = data["date"]
        broker_type = data.get("broker_type", "Unknown")
        breadth = data.get("breadth", 0)
        
        for stock in data.get("stocks", []):
            # Convert string values to numeric
            buy_val = float(stock.get("buy_val", "0").replace(",", "") or 0)
            sell_val = float(stock.get("sell_val", "0").replace(",", "") or 0)
            buy_vol = float(stock.get("buy_vol", "0").replace(",", "") or 0)
            sell_vol = float(stock.get("sell_vol", "0").replace(",", "") or 0)
            buy_freq = int(stock.get("buy_freq", "0").replace(",", "") or 0)
            sell_freq = int(stock.get("sell_freq", "0").replace(",", "") or 0)
            net_val = float(stock.get("net_val", "0").replace(",", "") or 0)
            total_val = float(stock.get("total_val", "0").replace(",", "") or 0)
            net_vol = float(stock.get("net_vol", "0").replace(",", "") or 0)
            
            row = {
                "broker": broker,
                "stock_code": stock.get("stock_code"),
                "date": date,
                "broker_type": broker_type,
                "breadth": breadth,
                
                "buy_val": buy_val,
                "sell_val": sell_val,
                "net_val": net_val,
                "total_val": total_val,
                
                "buy_vol": buy_vol,
                "sell_vol": sell_vol,
                "net_vol": net_vol,
                
                "buy_freq": buy_freq,
                "sell_freq": sell_freq,
                
                "avg_buy_price": buy_val / buy_vol if buy_vol > 0 else 0,
                "avg_sell_price": sell_val / sell_vol if sell_vol > 0 else 0,
                
                "scraped_at": data.get("scraped_at"),
            }
            rows.append(row)
            
    return rows


def save_to_parquet(df: pd.DataFrame, output_path: Path, stage=None):
    """Save DataFrame to parquet. If stage is provided, append to DuckDB staging."""
    if stage is not None:
        stage.append(df)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort for optimal query performance
    df = df.sort_values(["date", "broker", "stock_code"])
    
    table = pa.Table.from_pandas(df)
    pq.write_table(
        table,
        output_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    
    print(f"\nSaved {len(df):,} rows to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def print_summary(df: pd.DataFrame):
    """Print data summary."""
    print("\n" + "=" * 60)
    print("IPOT BROKER DATA SUMMARY (Gross, Per Day)")
    print("=" * 60)
    print(f"Total rows: {len(df):,}")
    print(f"Unique brokers: {df['broker'].nunique()}")
    print(f"Unique stocks: {df['stock_code'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total trading days: {df['date'].nunique()}")
    
    print(f"\nRecords per date:")
    print(df.groupby("date").size().to_string())
    
    print("\nSample data (latest date):")
    latest = df["date"].max()
    sample = df[df["date"] == latest][["broker", "stock_code", "date", "net_val", "total_val"]].head(10)
    print(sample.to_string())


async def main():
    # ── Staging-wrapper: use DuckDB TEMP if schema registered ──
    _stage = None
    try:
        from pipeline.storage.schemas import SCHEMAS
        if "broksum" in SCHEMAS:
            from pipeline.storage.staging import open_staging
            _stage_ctx = open_staging("broksum")
            _stage = _stage_ctx.__enter__()
            print(f"\n=== IPOT Scraper -> broksum_bybroker.parquet (staging mode) ===", flush=True)
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Fetch broker data directly from IPOT")
    parser.add_argument("--broker", type=str, help="Single broker code (e.g., MG)")
    parser.add_argument("--date", type=str, help="Single date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=5, help="Number of days to fetch")
    parser.add_argument("--from-date", type=str, help="Date range start (YYYY-MM-DD, inclusive)")
    parser.add_argument("--to-date", type=str, help="Date range end (YYYY-MM-DD, inclusive)")
    parser.add_argument("--brokers", type=str, help="Comma-separated broker codes")
    parser.add_argument("--emiten", type=str, help="Optional stock code filter (e.g., BBCA)")
    parser.add_argument("--all-brokers", action="store_true", help="Fetch all brokers from master_broker.parquet")
    parser.add_argument(
        "--reverse-brokers",
        action="store_true",
        help="Reverse final broker execution order (useful for split-run on another server).",
    )
    parser.add_argument("--include-inactive", action="store_true", help="Include inactive brokers when --all-brokers")
    parser.add_argument(
        "--master-broker",
        type=str,
        default=str(DATA_DIR / "master_broker.parquet"),
        help="Path to master broker parquet (used by --all-brokers)",
    )
    parser.add_argument(
        "--update-mode",
        choices=["overwrite", "append"],
        default="overwrite",
        help="overwrite: replace output, append: upsert into existing parquet",
    )
    parser.add_argument(
        "--repair-days",
        type=int,
        default=0,
        help="When append mode, re-fetch this many latest existing dates for smart repair",
    )
    parser.add_argument(
        "--day-sleep",
        type=float,
        default=0.5,
        help="Delay between day-requests within same broker",
    )
    parser.add_argument(
        "--export",
        choices=["parquet", "json", "both"],
        default="parquet",
        help="Output format. parquet=default pipeline, json=quick export, both=save both.",
    )
    parser.add_argument(
        "--source-mode",
        choices=["auto", "fetch", "parquet"],
        default="auto",
        help=(
            "auto: read parquet cache first, fetch only missing dates; "
            "fetch: always fetch from IPOT; parquet: read only from parquet cache"
        ),
    )
    parser.add_argument(
        "--parquet-store",
        type=str,
        default=str(DATA_DIR / "broksum_bybroker.parquet"),
        help="Parquet cache store used by source-mode auto/parquet",
    )
    parser.add_argument(
        "--progress-log",
        type=str,
        default=str(DEFAULT_PROGRESS_LOG),
        help="Append-only JSONC progress log path",
    )
    parser.add_argument(
        "--disable-progress-log",
        action="store_true",
        help="Disable progress log append events",
    )
    parser.add_argument("--json-output", type=str, help="Optional explicit JSON output path")
    parser.add_argument("--output", type=str, help="Output parquet path")
    parser.add_argument("--max-concurrent", type=int, default=2, help="Max concurrent brokers")
    parser.add_argument(
        "--flush-per-broker",
        dest="flush_per_broker",
        action="store_true",
        default=True,
        help="Flush/upsert to parquet store right after each broker completes.",
    )
    parser.add_argument(
        "--no-flush-per-broker",
        dest="flush_per_broker",
        action="store_false",
        help="Disable per-broker flush (legacy batch write at end).",
    )
    parser.add_argument(
        "--resume-safe",
        dest="resume_safe",
        action="store_true",
        default=True,
        help="Re-check parquet coverage before each broker fetch (safer resume).",
    )
    parser.add_argument(
        "--no-resume-safe",
        dest="resume_safe",
        action="store_false",
        help="Disable per-broker coverage re-check before fetch.",
    )
    parser.add_argument(
        "--resume-state",
        type=str,
        default=str(DEFAULT_RESUME_STATE),
        help="Path to JSON resume checkpoint state (per-broker completed dates).",
    )
    parser.add_argument(
        "--disable-resume-state",
        action="store_true",
        help="Disable resume checkpoint read/write file.",
    )
    args = parser.parse_args()

    if (args.from_date and not args.to_date) or (args.to_date and not args.from_date):
        parser.error("--from-date and --to-date must be provided together")
    if args.date and (args.from_date or args.to_date):
        parser.error("Use either --date OR --from-date/--to-date")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        #timestamp = datetime.now().strftime("%Y%m%d")
        output_path = DATA_DIR / f"broksum_bybroker.parquet"
    
    # Determine brokers to fetch
    if args.broker:
        brokers = [b.strip().upper() for b in args.broker.split(",") if b.strip()]
    elif args.brokers:
        brokers = [b.strip().upper() for b in args.brokers.split(",")]
    elif args.all_brokers:
        brokers = load_brokers_from_master(Path(args.master_broker), include_inactive=args.include_inactive)
    else:
        brokers = POPULAR_BROKERS[:5]  # Default to first 5 popular

    brokers = sorted(set([b for b in brokers if b]))
    if args.reverse_brokers:
        brokers = list(reversed(brokers))

    # Determine dates to fetch
    if args.date:
        requested_dates = [args.date]
    elif args.from_date and args.to_date:
        requested_dates = generate_weekdays_between(args.from_date, args.to_date)
    else:
        requested_dates = generate_recent_weekdays(args.days)

    target_dates = requested_dates.copy()

    # Smart repair dates (append mode)
    if args.update_mode == "append" and args.repair_days > 0 and output_path.exists():
        try:
            existing_df = pd.read_parquet(output_path, columns=["date"])
            existing_dates = (
                pd.to_datetime(existing_df["date"], errors="coerce")
                .dropna()
                .dt.strftime("%Y-%m-%d")
                .drop_duplicates()
                .sort_values()
                .tolist()
            )
            repair_dates = existing_dates[-args.repair_days:] if existing_dates else []
            if repair_dates:
                target_dates = sorted(set(target_dates + repair_dates), reverse=True)
                print(
                    f"[Repair] Added {len(repair_dates)} existing dates for smart repair "
                    f"(window={args.repair_days})."
                )
        except Exception as e:
            print(f"[Repair] Failed to load existing dates: {e}")
    
    emiten_filter = args.emiten.upper().strip() if args.emiten else None
    date_from_tag = min(requested_dates) if requested_dates else ""
    date_until_tag = max(requested_dates) if requested_dates else ""
    progress_log_path = Path(args.progress_log)
    resume_state_path = Path(args.resume_state)
    resume_state_enabled = not args.disable_resume_state
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    resume_state_doc = load_resume_state(resume_state_path) if resume_state_enabled else {
        "format": "broksum-resume-state-v1",
        "updated_at": None,
        "completed_dates": {},
    }

    def log_event(action: str, result: str, details: dict[str, Any]):
        if args.disable_progress_log:
            return
        try:
            append_progress_log(
                progress_log_path,
                service="fetch_broksum_ipot",
                action=action,
                result=result,
                details={"run_id": run_id, **details},
            )
        except Exception as e:
            print(f"[Log] Failed append {progress_log_path}: {e}")

    parquet_store = Path(args.parquet_store)
    use_local_first = args.source_mode in ("auto", "parquet")
    cached_subset = pd.DataFrame()
    broker_dates_map: dict[str, list[str]] = {b: list(target_dates) for b in brokers}

    if use_local_first:
        cached_subset = load_cached_subset(parquet_store, brokers=brokers, target_dates=target_dates)
        if args.source_mode == "parquet":
            broker_dates_map = {b: [] for b in brokers}
        else:
            broker_dates_map = compute_missing_dates_map(cached_subset, brokers=brokers, target_dates=target_dates)
        missing_total = sum(len(v) for v in broker_dates_map.values())
        cached_days = 0
        if not cached_subset.empty:
            cached_days = len(cached_subset.loc[:, ["broker", "date"]].drop_duplicates())
        print(
            f"[Cache] source-mode={args.source_mode}, store={parquet_store}, "
            f"cached_broker_days={cached_days}, missing_fetch_requests={missing_total}"
        )

    log_event(
        "backfill_run_start",
        "started",
        {
            "brokers": brokers,
            "broker_count": len(brokers),
            "emiten": emiten_filter or "all",
            "source_mode": args.source_mode,
            "export": args.export,
            "date_from": date_from_tag,
            "date_until": date_until_tag,
            "requested_days": len(target_dates),
            "missing_fetch_requests_total": sum(len(v) for v in broker_dates_map.values()),
            "parquet_store": str(parquet_store),
            "output": str(output_path),
            "flush_per_broker": bool(args.flush_per_broker),
            "resume_safe": bool(args.resume_safe),
            "resume_state_enabled": bool(resume_state_enabled),
            "resume_state_path": str(resume_state_path),
        },
    )

    print(f"IPOT Scraper")
    print(f"Brokers: {brokers}")
    if target_dates:
        print(f"Dates: {len(target_dates)} ({target_dates[-1]} -> {target_dates[0]})")
    else:
        print("Dates: 0")
    print(f"Update mode: {args.update_mode}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"Flush per broker: {args.flush_per_broker}")
    print(f"Resume safe check: {args.resume_safe}")
    print(f"Resume state file: {resume_state_path} (enabled={resume_state_enabled})")
    print(f"Output: {output_path}")
    print("=" * 60)

    if not target_dates:
        print("No target dates resolved. Exiting.")
        return
    
    fetched_raw_parts: list[pd.DataFrame] = []
    broker_success = 0
    broker_error = 0
    total_payload_days = 0
    total_rows_fetched = 0
    total_rows_flushed = 0
    total_no_data_dates = 0
    total_error_dates = 0

    # Fetch only when needed.
    should_fetch = args.source_mode == "fetch" or any(len(dates) > 0 for dates in broker_dates_map.values())
    if should_fetch:
        async with IPOTScraper() as scraper:
            for idx, broker in enumerate(brokers, start=1):
                dates_requested_for_broker = broker_dates_map.get(broker, target_dates)
                if args.resume_safe and use_local_first and args.source_mode in ("auto", "parquet"):
                    # 1. Start with what's missing from parquet
                    dates_requested_for_broker = resolve_missing_dates_for_broker(
                        parquet_store,
                        broker,
                        target_dates,
                    )
                    # 2. Subtract what's already in staging (crash-safe resume)
                    if _stage is not None:
                        staged_df = _stage.fetch_all()
                        if not staged_df.empty and "broker" in staged_df.columns and "date" in staged_df.columns:
                            staged_dates = set(staged_df[staged_df["broker"] == broker]["date"].unique())
                            if staged_dates:
                                before = len(dates_requested_for_broker)
                                dates_requested_for_broker = [d for d in dates_requested_for_broker if d not in staged_dates]
                                skipped = before - len(dates_requested_for_broker)
                                if skipped > 0:
                                    print(f"[Staging] {broker}: skip {skipped} already in staging duckdb")
                if resume_state_enabled:
                    completed_dates = get_resume_completed_dates(resume_state_doc, broker)
                    if completed_dates:
                        before = len(dates_requested_for_broker)
                        dates_requested_for_broker = [d for d in dates_requested_for_broker if d not in completed_dates]
                        skipped = before - len(dates_requested_for_broker)
                        if skipped > 0:
                            print(f"[ResumeState] {broker}: skip {skipped} completed dates from checkpoint")

                if not dates_requested_for_broker:
                    print(f"[Batch] [{idx}/{len(brokers)}] Skip {broker}: fully covered by parquet cache")
                    broker_success += 1
                    log_event(
                        "broker_complete",
                        "cache_hit",
                        {
                            "broker": broker,
                            "emiten": emiten_filter or "all",
                            "source_mode": args.source_mode,
                            "dates_requested": 0,
                            "dates_fetched_payloads": 0,
                            "rows_fetched": 0,
                            "rows_flushed": 0,
                            "dates_requested_list": [],
                            "dates_returned_list": [],
                            "sequence": idx,
                            "sequence_total": len(brokers),
                            "flush_per_broker": bool(args.flush_per_broker),
                        },
                    )
                    continue

                try:
                    print(f"[Batch] [{idx}/{len(brokers)}] Start {broker} (dates={len(dates_requested_for_broker)})")
                    broker_attempts = await scraper.scrape_broker_dates(
                        broker,
                        dates_requested_for_broker,
                        day_sleep=max(0.0, args.day_sleep),
                    )
                except Exception as e:
                    print(f"[Batch] Error broker {broker}: {e}")
                    broker_error += 1
                    log_event(
                        "broker_complete",
                        "error",
                        {
                            "broker": broker,
                            "emiten": emiten_filter or "all",
                            "source_mode": args.source_mode,
                            "dates_requested": len(dates_requested_for_broker),
                            "dates_fetched_payloads": 0,
                            "rows_fetched": 0,
                            "rows_flushed": 0,
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "sequence": idx,
                            "sequence_total": len(brokers),
                            "flush_per_broker": bool(args.flush_per_broker),
                        },
                    )
                    continue

                broker_payloads = [x for x in broker_attempts if bool(x.get("ok"))]
                no_data_dates = [x.get("date") for x in broker_attempts if x.get("status") == "no_data" and x.get("date")]
                error_dates = [x.get("date") for x in broker_attempts if x.get("status") == "error" and x.get("date")]
                print(
                    f"[Batch] [{idx}/{len(brokers)}] Done {broker}: "
                    f"success_payloads={len(broker_payloads)}, no_data_dates={len(no_data_dates)}, error_dates={len(error_dates)}"
                )

                flat_rows = flatten_to_master(broker_payloads) if broker_payloads else []
                broker_raw_df = pd.DataFrame(flat_rows) if flat_rows else pd.DataFrame()

                payload_days = len(broker_payloads)
                fetched_rows = len(broker_raw_df)
                total_payload_days += payload_days
                total_rows_fetched += fetched_rows
                total_no_data_dates += len(no_data_dates)
                total_error_dates += len(error_dates)

                flush_meta: dict[str, int] = {"before_rows": 0, "after_rows": 0, "delta_rows": 0}
                if error_dates:
                    status = "partial_error" if (payload_days > 0 or no_data_dates) else "error"
                else:
                    status = "success" if payload_days > 0 else "no_data"

                if use_local_first and args.flush_per_broker and not broker_raw_df.empty:
                    try:
                        flush_meta = flush_broker_rows_to_store(
                            parquet_store,
                            broker_raw_df,
                            broker=broker,
                            stage=_stage,
                        )
                        total_rows_flushed += fetched_rows
                    except Exception as e:
                        print(f"[Flush] Error broker {broker}: {e}")
                        broker_error += 1
                        log_event(
                            "broker_complete",
                            "error",
                            {
                                "broker": broker,
                                "emiten": emiten_filter or "all",
                                "source_mode": args.source_mode,
                                "dates_requested": len(dates_requested_for_broker),
                                "dates_fetched_payloads": payload_days,
                                "rows_fetched": fetched_rows,
                                "rows_flushed": 0,
                                "error": str(e),
                                "error_type": type(e).__name__,
                                "error_stage": "flush_per_broker",
                                "sequence": idx,
                                "sequence_total": len(brokers),
                                "flush_per_broker": bool(args.flush_per_broker),
                            },
                        )
                        continue

                # Keep in-memory fetched parts only when needed by non-local-first branch,
                # or when local-first flush is explicitly disabled.
                if (not use_local_first or not args.flush_per_broker) and not broker_raw_df.empty:
                    fetched_raw_parts.append(broker_raw_df)

                resume_added = 0
                resume_total = len(get_resume_completed_dates(resume_state_doc, broker)) if resume_state_enabled else 0
                if resume_state_enabled:
                    completed_for_resume = [d for d in dates_requested_for_broker if d not in set(error_dates)]
                    if completed_for_resume:
                        resume_added = add_resume_completed_dates(resume_state_doc, broker, completed_for_resume)
                        save_resume_state(resume_state_path, resume_state_doc)
                        resume_total = len(get_resume_completed_dates(resume_state_doc, broker))
                        if resume_added > 0:
                            print(
                                f"[ResumeState] {broker}: +{resume_added} dates checkpointed "
                                f"(total={resume_total})"
                            )

                broker_success += 1
                log_event(
                    "broker_complete",
                    status,
                    {
                        "broker": broker,
                        "emiten": emiten_filter or "all",
                        "source_mode": args.source_mode,
                        "dates_requested": len(dates_requested_for_broker),
                        "dates_fetched_payloads": payload_days,
                        "dates_no_data": len(no_data_dates),
                        "dates_error": len(error_dates),
                        "rows_fetched": fetched_rows,
                        "rows_flushed": fetched_rows if (use_local_first and args.flush_per_broker and fetched_rows > 0) else 0,
                        "parquet_rows_before": flush_meta["before_rows"],
                        "parquet_rows_after": flush_meta["after_rows"],
                        "parquet_rows_delta": flush_meta["delta_rows"],
                        "dates_requested_list": dates_requested_for_broker,
                        "dates_returned_list": [x.get("date") for x in broker_payloads if x.get("date")],
                        "dates_no_data_list": no_data_dates,
                        "dates_error_list": error_dates,
                        "resume_state_added_dates": resume_added,
                        "resume_state_total_dates": resume_total,
                        "sequence": idx,
                        "sequence_total": len(brokers),
                        "flush_per_broker": bool(args.flush_per_broker),
                    },
                )

        log_event(
            "fetch_phase_complete",
            "success" if broker_error == 0 else "partial",
            {
                "brokers_success": broker_success,
                "brokers_error": broker_error,
                "total_payload_days": total_payload_days,
                "total_rows_fetched": total_rows_fetched,
                "total_rows_flushed": total_rows_flushed,
                "total_no_data_dates": total_no_data_dates,
                "total_error_dates": total_error_dates,
            },
        )
    else:
        print("[Fetch] Skip IPOT fetch: all requested dates are covered by parquet cache.")
        for idx, broker in enumerate(brokers, start=1):
            log_event(
                "broker_complete",
                "cache_hit",
                {
                    "broker": broker,
                    "emiten": emiten_filter or "all",
                    "source_mode": args.source_mode,
                    "dates_requested": 0,
                    "dates_fetched_payloads": 0,
                    "rows_fetched": 0,
                    "rows_flushed": 0,
                    "dates_requested_list": [],
                    "dates_returned_list": [],
                    "sequence": idx,
                    "sequence_total": len(brokers),
                    "flush_per_broker": bool(args.flush_per_broker),
                },
            )
        log_event(
            "fetch_phase_complete",
            "success",
            {
                "brokers_success": len(brokers),
                "brokers_error": 0,
                "total_payload_days": 0,
                "total_rows_fetched": 0,
                "total_rows_flushed": 0,
                "total_no_data_dates": 0,
                "total_error_dates": 0,
            },
        )

    fetched_raw_df = (
        pd.concat(fetched_raw_parts, ignore_index=True)
        if fetched_raw_parts
        else pd.DataFrame()
    )
    if fetched_raw_df.empty and args.source_mode == "fetch" and not use_local_first:
        print("\nNo data scraped. Exiting.")
        return

    # In local-first mode, optionally backfill parquet store in one batch when
    # per-broker flush is disabled (legacy behavior).
    if use_local_first and not fetched_raw_df.empty and not args.flush_per_broker:
        try:
            if parquet_store.exists():
                existing_store_df = pd.read_parquet(parquet_store)
                merged_store_df = merge_append_with_upsert(existing_store_df, fetched_raw_df)
            else:
                merged_store_df = fetched_raw_df.copy()
            save_to_parquet(merged_store_df, parquet_store, stage=_stage)
            print(f"[Backfill] Parquet cache updated: {parquet_store}")
        except Exception as e:
            print(f"[Backfill] Failed updating parquet cache: {e}")

    # Build export source dataset.
    if use_local_first:
        export_source_df = load_cached_subset(parquet_store, brokers=brokers, target_dates=target_dates)
        if export_source_df.empty and not fetched_raw_df.empty:
            # Fallback to fetched rows when cache read fails.
            export_source_df = fetched_raw_df.copy()
    else:
        export_source_df = fetched_raw_df.copy()

    if export_source_df.empty and args.export in ("json", "both"):
        print("No data available for requested brokers/dates (cache + fetch). Exiting.")
        return

    # Keep old df variable for parquet/both branch behavior.
    df = fetched_raw_df.copy()
    fetched_df_for_json = export_source_df.copy()
    same_output_as_store = output_path.resolve() == parquet_store.resolve()

    # Optional emiten filter for JSON export payload.
    if emiten_filter and not fetched_df_for_json.empty:
        before_json = len(fetched_df_for_json)
        fetched_df_for_json = fetched_df_for_json[
            fetched_df_for_json["stock_code"].astype(str).str.upper() == emiten_filter
        ].copy()
        print(f"[Filter] Emiten={emiten_filter}: {before_json:,} -> {len(fetched_df_for_json):,} rows")
        if fetched_df_for_json.empty:
            print("No rows after emiten filter. Exiting.")
            return

    # Append/upsert mode
    if args.export in ("parquet", "both"):
        if use_local_first and args.flush_per_broker and same_output_as_store:
            print(
                "[Parquet] Skip final write: output == parquet_store and data already "
                "flushed per broker during fetch phase."
            )
            if not export_source_df.empty:
                print_summary(export_source_df)
            meta = {
                "scraped_at": datetime.now().isoformat(),
                "source": "IPOT Direct (WebSocket)",
                "total_rows": len(export_source_df),
                "unique_brokers": export_source_df["broker"].nunique() if not export_source_df.empty else 0,
                "unique_stocks": export_source_df["stock_code"].nunique() if not export_source_df.empty else 0,
                "date_range": {
                    "min": export_source_df["date"].min() if not export_source_df.empty else None,
                    "max": export_source_df["date"].max() if not export_source_df.empty else None,
                },
                "brokers": brokers,
                "note": "metadata generated from requested subset after per-broker flush mode",
            }
            meta_path = output_path.with_suffix(".json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, default=str)
            print(f"\nMetadata saved to {meta_path}")
        elif df.empty:
            print("No fetched rows to write parquet export output. Exiting parquet branch.")
        else:
            if args.update_mode == "append" and output_path.exists():
                try:
                    existing_df = pd.read_parquet(output_path)
                    before_rows = len(existing_df)
                    df = merge_append_with_upsert(existing_df, df)
                    print(f"[Append] Existing rows: {before_rows:,} -> merged rows: {len(df):,}")
                except Exception as e:
                    print(f"[Append] Failed to merge existing parquet, fallback overwrite. error={e}")

            # Save to parquet
            save_to_parquet(df, output_path, stage=_stage)

            # Print summary
            print_summary(df)

            # Save metadata (parquet pipeline metadata)
            meta = {
                "scraped_at": datetime.now().isoformat(),
                "source": "IPOT Direct (WebSocket)",
                "total_rows": len(df),
                "unique_brokers": df["broker"].nunique(),
                "unique_stocks": df["stock_code"].nunique(),
                "date_range": {"min": df["date"].min(), "max": df["date"].max()},
                "brokers": brokers,
            }
            meta_path = output_path.with_suffix(".json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, default=str)
            print(f"\nMetadata saved to {meta_path}")

    if args.export in ("json", "both"):
        if args.json_output:
            json_out = Path(args.json_output)
        elif args.export == "json" and args.output and Path(args.output).suffix.lower() == ".json":
            json_out = Path(args.output)
        else:
            if args.output:
                out_arg = Path(args.output)
                if out_arg.exists() and out_arg.is_dir():
                    json_dir = out_arg
                elif out_arg.suffix == "":
                    # Treat suffix-less path as directory intention for quick exports.
                    json_dir = out_arg
                else:
                    # output likely parquet filepath; place json beside it.
                    json_dir = out_arg.parent
            else:
                json_dir = DATA_DIR
            json_name = build_standard_json_filename(
                brokers=brokers,
                emiten=emiten_filter,
                date_from=date_from_tag,
                date_until=date_until_tag,
            )
            json_out = json_dir / json_name

        save_export_json(
            fetched_df_for_json,
            json_out,
            brokers=brokers,
            emiten=emiten_filter,
            date_from=date_from_tag,
            date_until=date_until_tag,
        )
        log_event(
            "json_export_complete",
            "success",
            {
                "brokers": brokers,
                "emiten": emiten_filter or "all",
                "rows_exported": len(fetched_df_for_json),
                "json_path": str(json_out),
                "date_from": date_from_tag,
                "date_until": date_until_tag,
            },
        )

    # ── Staging commit: merge all staged → parquet ──
    if _stage is not None:
        try:
            existing = pd.read_parquet(parquet_store) if parquet_store.exists() else pd.DataFrame()
            if not existing.empty:
                existing['date'] = existing['date'].astype(str)
            rows_written = _stage.commit_to_l0(base_df=existing)
            _stage.clear()
            print(f"\n[staging] Committed {rows_written:,} rows, staging cleared", flush=True)
        except Exception as e:
            print(f"\n[staging] Commit error: {e}", flush=True)

    log_event(
        "backfill_run_complete",
        "success",
        {
            "brokers": brokers,
            "emiten": emiten_filter or "all",
            "export": args.export,
            "output": str(output_path),
            "parquet_store": str(parquet_store),
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
