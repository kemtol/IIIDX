#!/usr/bin/env python3
"""
Fetch master emiten list and save to parquet.

Default source is api-saham `/emiten` endpoint (which reads D1 `emiten` table),
so this script is aligned with the existing Cloudflare workers architecture.

Examples:
  python machinelearning/idx/service/fetch_emiten.py
  python machinelearning/idx/service/fetch_emiten.py --source idx
  python machinelearning/idx/service/fetch_emiten.py --status ACTIVE --limit 1000
  python machinelearning/idx/service/fetch_emiten.py --trigger-sync --admin-token "$IDX_SYNC_TOKEN"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

# L0 storage gateway lives at repo root; ensure it is importable when this
# script is invoked directly (`python pipeline/fetch/fetch_emiten.py`).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.storage.schemas import SCHEMAS  # noqa: E402
from pipeline.storage.writers import write_l0  # noqa: E402

_SOURCE = "master_emiten"

DEFAULT_API_BASE = "https://api-saham.mkemalw.workers.dev"
BASE_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LEVEL0_DATA_DIR = BASE_DATA_DIR / "Level_0_Raw"
DEFAULT_OUTPUT = (LEVEL0_DATA_DIR if LEVEL0_DATA_DIR.exists() else BASE_DATA_DIR) / "master_emiten.parquet"
IDX_EMITEN_URL = "https://www.idx.co.id/primary/Helper/GetEmiten?emitenType=*"

IDX_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.idx.co.id/",
}

TICKER_KEYS = ["KodeEmiten", "kodeEmiten", "Kode_Saham", "Kode", "Code", "ticker"]
SECTOR_KEYS = ["Sektor", "Sector", "NamaSektor", "sector", "sektor", "SectorName"]
INDUSTRY_KEYS = ["SubSektor", "SubSector", "Industry", "Industri", "NamaSubSektor", "SubSubSektor", "sub_industry"]
STATUS_KEYS = ["Status", "StatusPerusahaan", "StatusEmiten", "status", "KodeStatus", "statusEmiten"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch master emiten and save as parquet.")
    parser.add_argument("--source", choices=["d1", "idx"], default="d1", help="Data source: D1 API (/emiten) or direct IDX.")
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE, help="Base URL for api-saham worker.")
    parser.add_argument("--status", default="all", help="Status filter for /emiten endpoint (default: all).")
    parser.add_argument("--q", default="", help="Ticker search filter for /emiten endpoint.")
    parser.add_argument("--sector", default="", help="Sector filter for /emiten endpoint.")
    parser.add_argument("--limit", type=int, default=2000, help="Limit for /emiten endpoint (max 2000).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output parquet path.")
    parser.add_argument("--trigger-sync", action="store_true", help="Trigger /admin/emiten/sync before fetching D1 data.")
    parser.add_argument("--admin-token", default="", help="X-Admin-Token for /admin/emiten/sync.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds.")
    return parser.parse_args()


def http_json(url: str, *, method: str = "GET", headers: dict[str, str] | None = None, timeout: int = 30) -> Any:
    base_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
    }
    merged = dict(base_headers)
    if headers:
        merged.update(headers)
    req = Request(url=url, method=method, headers=merged)
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed request to {url}: {exc.reason}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from {url}: {exc}") from exc


def trigger_sync(api_base_url: str, admin_token: str, timeout: int) -> None:
    url = f"{api_base_url.rstrip('/')}/admin/emiten/sync"
    headers: dict[str, str] = {}
    if admin_token:
        headers["x-admin-token"] = admin_token
    payload = http_json(url, method="POST", headers=headers, timeout=timeout)
    print(f"[sync] response: {payload}")


def fetch_from_d1_api(args: argparse.Namespace) -> list[dict[str, Any]]:
    params = {
        "status": args.status,
        "limit": max(1, min(int(args.limit), 2000)),
    }
    if args.q:
        params["q"] = args.q
    if args.sector:
        params["sector"] = args.sector

    url = f"{args.api_base_url.rstrip('/')}/emiten?{urlencode(params)}"
    payload = http_json(url, timeout=args.timeout)
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected /emiten payload: {payload}")
    return data


def pick(obj: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return None


def ensure_ticker_format(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    trimmed = value.strip().upper()
    if not trimmed:
        return None
    if trimmed.endswith(".JK"):
        return trimmed
    alnum = "".join(ch for ch in trimmed if ch.isalnum())
    if not alnum:
        return None
    return f"{alnum}.JK"


def normalize_status(value: Any) -> str:
    text = str(value).strip().upper() if value is not None else ""
    if not text:
        return "ACTIVE"
    mapping = {
        "A": "ACTIVE",
        "N": "ACTIVE",
        "S": "SUSPENDED",
        "SP": "SUSPENDED",
        "SUS": "SUSPENDED",
        "I": "SUSPENDED",
        "D": "DELISTED",
        "DEL": "DELISTED",
    }
    if text in mapping:
        return mapping[text]
    if "SUSP" in text:
        return "SUSPENDED"
    if "DELIST" in text:
        return "DELISTED"
    return text


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def unwrap_idx_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return payload["data"]
        if isinstance(payload.get("Data"), list):
            return payload["Data"]
    return []


def fetch_from_idx(timeout: int) -> list[dict[str, Any]]:
    payload = http_json(IDX_EMITEN_URL, headers=IDX_HEADERS, timeout=timeout)
    rows = unwrap_idx_payload(payload)
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker = ensure_ticker_format(pick(row, TICKER_KEYS))
        if not ticker:
            continue
        out.append(
            {
                "ticker": ticker,
                "sector": normalize_text(pick(row, SECTOR_KEYS)),
                "industry": normalize_text(pick(row, INDUSTRY_KEYS)),
                "status": normalize_status(pick(row, STATUS_KEYS)),
            }
        )
    return out


def normalize_rows(rows: list[dict[str, Any]], source: str) -> pd.DataFrame:
    now_iso = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker = ensure_ticker_format(row.get("ticker"))
        if not ticker:
            continue
        records.append(
            {
                "ticker": ticker,
                "sector": normalize_text(row.get("sector")),
                "industry": normalize_text(row.get("industry")),
                "status": normalize_status(row.get("status")),
                "created_at": normalize_text(row.get("created_at")),
                "updated_at": normalize_text(row.get("updated_at")),
                "source": source,
                "fetched_at": now_iso,
            }
        )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "sector",
                "industry",
                "status",
                "created_at",
                "updated_at",
                "source",
                "fetched_at",
            ]
        )

    df = df.sort_values("ticker").drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)
    return df


def main() -> int:
    args = parse_args()

    if args.source == "d1" and args.trigger_sync:
        trigger_sync(args.api_base_url, args.admin_token, args.timeout)

    if args.source == "d1":
        raw_rows = fetch_from_d1_api(args)
        source_label = "d1_api"
    else:
        raw_rows = fetch_from_idx(args.timeout)
        source_label = "idx_direct"

    df = normalize_rows(raw_rows, source_label)

    # Gateway write: parquet path is canonicalized by the schema. write_l0
    # also handles dual-write to DuckDB once the canary flag is promoted.
    schema_path = (_REPO_ROOT / SCHEMAS[_SOURCE].parquet_path).resolve()
    if args.output.resolve() != schema_path:
        print(
            f"[warn] --output={args.output} differs from schema path "
            f"{schema_path}; gateway will write to schema path."
        )
    write_l0(_SOURCE, df)

    print(f"[ok] source={args.source} rows={len(df)} output={schema_path}")
    if not df.empty:
        print(df.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
