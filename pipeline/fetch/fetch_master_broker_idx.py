#!/usr/bin/env python3
"""
Fetch broker master directly from IDX website and update ACTIVE/INACTIVE status.

Single source of truth:
  - IDX profile page (Nuxt payload): /id/anggota-bursa-dan-partisipan/profil-anggota-bursa

Behavior:
  - Brokers visible on latest IDX page -> status ACTIVE
  - Brokers missing from latest IDX page but present in local master -> status INACTIVE
  - Keep historical rows (so temporary suspension/delisting can be tracked)
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


BASE_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LEVEL0_DATA_DIR = BASE_DATA_DIR / "Level_0_Raw"
if LEVEL0_DATA_DIR.exists():
    DATA_DIR = LEVEL0_DATA_DIR
else:
    DATA_DIR = BASE_DATA_DIR

DEFAULT_OUTPUT = DATA_DIR / "master_broker.parquet"
DEFAULT_METADATA = DATA_DIR / "master_broker_idx_sync.json"
DEFAULT_URLS = [
    "https://block.idx.id/id/anggota-bursa-dan-partisipan/profil-anggota-bursa",
    "https://www.idx.co.id/id/anggota-bursa-dan-partisipan/profil-anggota-bursa",
]

BASE_COLS = ["broker_code", "broker_name", "category", "foreignfund_%", "localfund_%", "retail_%"]
STATUS_COLS = [
    "status",
    "is_active",
    "first_seen_at",
    "last_seen_at",
    "last_status_change_at",
    "source_url",
    "fetched_at",
]
ALL_COLS = BASE_COLS + STATUS_COLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch master broker from IDX profile page and update status.")
    parser.add_argument("--url", action="append", help="IDX page URL (can be passed multiple times).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output master broker parquet path.")
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=DEFAULT_METADATA,
        help="Output metadata json path.",
    )
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds.")
    return parser.parse_args()


def http_get(url: str, timeout: int) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
    }
    req = Request(url=url, headers=headers, method="GET")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {detail[:240]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc.reason}") from exc


def split_js_args(raw: str) -> list[str]:
    """Split top-level JS arguments by comma while respecting quotes/brackets."""
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    in_str: str | None = None
    escaped = False

    for ch in raw:
        if in_str:
            buf.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_str:
                in_str = None
            continue

        if ch in ('"', "'"):
            in_str = ch
            buf.append(ch)
            continue

        if ch in "([{":
            depth += 1
            buf.append(ch)
            continue

        if ch in ")]}":
            depth = max(depth - 1, 0)
            buf.append(ch)
            continue

        if ch == "," and depth == 0:
            token = "".join(buf).strip()
            if token:
                out.append(token)
            buf = []
            continue

        buf.append(ch)

    token = "".join(buf).strip()
    if token:
        out.append(token)
    return out


def parse_js_atom(token: str) -> Any:
    token = token.strip()
    if token == "null":
        return None
    if token == "true":
        return True
    if token == "false":
        return False
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    if re.fullmatch(r"-?\d+\.\d+", token):
        return float(token)
    if (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
        try:
            return ast.literal_eval(token)
        except Exception:
            return token[1:-1]
    return token


def decode_js_escapes(value: str) -> str:
    return (
        value.replace("\\u0026", "&")
        .replace("\\u002F", "/")
        .replace("\\/", "/")
        .replace("\\u0027", "'")
        .replace("\\u0022", '"')
    )


def parse_nuxt_runtime(html: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """
    Parse window.__NUXT__ runtime payload and extract broker rows.

    Handles both literal codes (Code:"YP") and symbol codes (Code:D where D="AO").
    """
    m = re.search(
        r"window\.__NUXT__=\(function\((?P<args>.*?)\)\{(?P<body>.*)\}\((?P<vals>.*)\)\);",
        html,
        re.S,
    )
    if not m:
        raise RuntimeError("Could not parse Nuxt payload from IDX page.")

    arg_names = [x.strip() for x in m.group("args").split(",") if x.strip()]
    arg_values = [parse_js_atom(x) for x in split_js_args(m.group("vals"))]
    symbols = {k: arg_values[i] if i < len(arg_values) else None for i, k in enumerate(arg_names)}
    body = m.group("body")

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    pattern = r'\{Code:([A-Za-z0-9_$"\'-]+),Name:"([^"]+)",License:'
    for code_token, broker_name in re.findall(pattern, body):
        if code_token.startswith('"') or code_token.startswith("'"):
            code_val = parse_js_atom(code_token)
        else:
            code_val = symbols.get(code_token, code_token)

        code = str(code_val or "").strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{2,3}", code):
            continue
        if code in seen:
            continue

        rows.append(
            {
                "broker_code": code,
                "broker_name": decode_js_escapes(broker_name).strip(),
            }
        )
        seen.add(code)

    if not rows:
        raise RuntimeError("No broker rows extracted from Nuxt payload.")

    return rows, symbols


def fetch_idx_brokers(urls: list[str], timeout: int) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for url in urls:
        try:
            html = http_get(url, timeout=timeout)
            rows, _symbols = parse_nuxt_runtime(html)
            df = pd.DataFrame(rows).drop_duplicates(subset=["broker_code"], keep="first")
            df = df.sort_values("broker_code").reset_index(drop=True)
            return df, url
        except Exception as exc:
            errors.append(f"{url} -> {exc}")
    raise RuntimeError("All IDX URLs failed:\n" + "\n".join(errors))


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=ALL_COLS)
    df = pd.read_parquet(path).copy()
    for col in ALL_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    df["broker_code"] = df["broker_code"].astype(str).str.strip().str.upper()
    df["broker_name"] = df["broker_name"].astype(str).str.strip()
    return df[ALL_COLS].drop_duplicates(subset=["broker_code"], keep="last")


def merge_status(existing_df: pd.DataFrame, idx_df: pd.DataFrame, source_url: str) -> tuple[pd.DataFrame, dict[str, int]]:
    now_iso = datetime.now(timezone.utc).isoformat()

    existing_map = {r["broker_code"]: r for _, r in existing_df.iterrows()}
    idx_map = {r["broker_code"]: r for _, r in idx_df.iterrows()}

    all_codes = sorted(set(existing_map.keys()) | set(idx_map.keys()))
    rows: list[dict[str, Any]] = []

    new_active = 0
    deactivated = 0
    reactivated = 0

    for code in all_codes:
        old = existing_map.get(code)
        cur = idx_map.get(code)

        is_active = cur is not None
        new_status = "ACTIVE" if is_active else "INACTIVE"
        old_status = (str(old["status"]).upper() if old is not None and pd.notna(old["status"]) else None)

        if old is None and is_active:
            new_active += 1
        elif old_status == "ACTIVE" and new_status == "INACTIVE":
            deactivated += 1
        elif old_status == "INACTIVE" and new_status == "ACTIVE":
            reactivated += 1

        if old is not None and pd.notna(old["first_seen_at"]):
            first_seen = old["first_seen_at"]
        elif old is not None and pd.notna(old["fetched_at"]):
            first_seen = old["fetched_at"]
        else:
            first_seen = now_iso

        if is_active:
            last_seen = now_iso
        elif old is not None and pd.notna(old["last_seen_at"]):
            last_seen = old["last_seen_at"]
        elif old is not None and pd.notna(old["fetched_at"]):
            last_seen = old["fetched_at"]
        else:
            last_seen = pd.NA

        if old is None or old_status != new_status:
            last_status_change = now_iso
        else:
            last_status_change = old["last_status_change_at"]

        rows.append(
            {
                "broker_code": code,
                "broker_name": (
                    cur["broker_name"]
                    if cur is not None and pd.notna(cur["broker_name"])
                    else (old["broker_name"] if old is not None else pd.NA)
                ),
                "category": old["category"] if old is not None else pd.NA,
                "foreignfund_%": old["foreignfund_%"] if old is not None else pd.NA,
                "localfund_%": old["localfund_%"] if old is not None else pd.NA,
                "retail_%": old["retail_%"] if old is not None else pd.NA,
                "status": new_status,
                "is_active": bool(is_active),
                "first_seen_at": first_seen,
                "last_seen_at": last_seen,
                "last_status_change_at": last_status_change,
                "source_url": source_url,
                "fetched_at": now_iso,
            }
        )

    out = pd.DataFrame(rows)[ALL_COLS].sort_values("broker_code").reset_index(drop=True)
    stats = {
        "total_brokers": int(len(out)),
        "active_brokers": int((out["status"] == "ACTIVE").sum()),
        "inactive_brokers": int((out["status"] == "INACTIVE").sum()),
        "new_active_brokers": int(new_active),
        "deactivated_brokers": int(deactivated),
        "reactivated_brokers": int(reactivated),
    }
    return out, stats


def main() -> int:
    args = parse_args()
    urls = args.url if args.url else list(DEFAULT_URLS)

    idx_df, source_url = fetch_idx_brokers(urls=urls, timeout=args.timeout)
    existing_df = load_existing(args.output)
    final_df, stats = merge_status(existing_df=existing_df, idx_df=idx_df, source_url=source_url)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(args.output, index=False)

    meta = {
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "source_url": source_url,
        "source_visible_brokers": int(len(idx_df)),
        **stats,
        "inactive_codes": final_df.loc[final_df["status"] == "INACTIVE", "broker_code"].tolist(),
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[ok] source={source_url}")
    print(f"[ok] output={args.output}")
    print(f"[ok] metadata={args.metadata_output}")
    print(
        "[ok] total={total_brokers} active={active_brokers} inactive={inactive_brokers} "
        "new={new_active_brokers} deactivated={deactivated_brokers} reactivated={reactivated_brokers}".format(**stats)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
