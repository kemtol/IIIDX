#!/usr/bin/env python3
"""State, Telegram commands, and message formatting for BSJP heartbeat."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

WIB = timezone(timedelta(hours=7), name="WIB")

DEFAULT_INTERVAL_MINUTES = 5
MIN_INTERVAL_MINUTES = 1
MAX_INTERVAL_MINUTES = 60


def default_state() -> dict[str, Any]:
    return {
        "status_format": "verbose",
        "interval_minutes": DEFAULT_INTERVAL_MINUTES,
        "last_update_id": 0,
        "last_sent_epoch": 0,
    }


def load_notify_env(path: Path | None = None) -> None:
    env_path = path or (Path.home() / ".bsjp_notify.env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or not line.startswith("export "):
            continue
        key_value = line[len("export ") :]
        if "=" not in key_value:
            continue
        key, value = key_value.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def load_state(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = default_state()
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            if isinstance(raw, dict):
                state.update(raw)
        except json.JSONDecodeError:
            backup = path.with_suffix(path.suffix + ".corrupt")
            path.replace(backup)
    state["status_format"] = str(state.get("status_format", "verbose")).lower()
    if state["status_format"] not in {"verbose", "compact"}:
        state["status_format"] = "verbose"
    try:
        state["interval_minutes"] = int(state.get("interval_minutes", DEFAULT_INTERVAL_MINUTES))
    except (TypeError, ValueError):
        state["interval_minutes"] = DEFAULT_INTERVAL_MINUTES
    state["interval_minutes"] = max(
        MIN_INTERVAL_MINUTES, min(MAX_INTERVAL_MINUTES, state["interval_minutes"])
    )
    state["last_update_id"] = int(state.get("last_update_id", 0) or 0)
    state["last_sent_epoch"] = int(state.get("last_sent_epoch", 0) or 0)
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _canonical_command(text: str) -> tuple[str, list[str]]:
    parts = text.strip().split()
    if not parts:
        return "", []
    command = parts[0].split("@", 1)[0].lower()
    return command, parts[1:]


def apply_command(state: dict[str, Any], text: str) -> str | None:
    command, args = _canonical_command(text)
    if command == "/compactstatus":
        if args:
            value = args[0].lower()
            if value in {"on", "compact", "true", "1"}:
                state["status_format"] = "compact"
            elif value in {"off", "verbose", "false", "0"}:
                state["status_format"] = "verbose"
            else:
                return "Invalid format. Use /compactstatus, /compactstatus on, or /compactstatus off."
        else:
            state["status_format"] = "compact" if state.get("status_format") != "compact" else "verbose"
        return f"BSJP status format changed: {state['status_format'].upper()}"

    if command == "/intervalstatus":
        if not args:
            return (
                f"BSJP heartbeat interval: {state['interval_minutes']} minutes\n"
                f"Status format: {str(state['status_format']).upper()}"
            )
        try:
            interval = int(args[0])
        except ValueError:
            return "Invalid interval. Use /intervalstatus N with N between 1 and 60."
        if interval < MIN_INTERVAL_MINUTES or interval > MAX_INTERVAL_MINUTES:
            return "Invalid interval. Use /intervalstatus N with N between 1 and 60."
        state["interval_minutes"] = interval
        return f"BSJP heartbeat interval changed: {interval} minutes"

    return None


def should_send(state: dict[str, Any], now_epoch: int | None = None) -> bool:
    if now_epoch is None:
        now_epoch = int(time.time())
    elapsed = now_epoch - int(state.get("last_sent_epoch", 0) or 0)
    return elapsed >= int(state.get("interval_minutes", DEFAULT_INTERVAL_MINUTES)) * 60


def mark_sent(state: dict[str, Any], now_epoch: int | None = None) -> None:
    epoch = int(time.time() if now_epoch is None else now_epoch)
    state["last_sent_epoch"] = epoch - (epoch % 60)


def _normalize_status(raw: str) -> str:
    raw = raw.strip()
    if raw in {"[OK]", "OK"} or raw.startswith("\u2705"):
        return "[OK]"
    if raw in {"[FAIL]", "FAIL"} or raw.startswith("\u274c"):
        return "[FAIL]"
    if raw in {"[WARN]", "WARN"} or raw.startswith("\u26a0"):
        return "[WARN]"
    if raw in {"[WAIT]", "WAIT"}:
        return "[WAIT]"
    if raw in {"[SKIP]", "SKIP"}:
        return "[SKIP]"
    return "[INFO]"


def _parse_context(check_output: str) -> tuple[str, str]:
    match = re.search(r"D-Day:\s*([0-9-]+)\s*\|\s*T-1:\s*([0-9-]+)", check_output)
    if match:
        return match.group(1), match.group(2)
    now = datetime.now(WIB)
    return now.strftime("%Y-%m-%d"), ""


def _parse_check_items(check_output: str) -> list[tuple[str, str, str]]:
    items: list[tuple[str, str, str]] = []
    for raw_line in check_output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("---") or line.startswith("Error:"):
            continue

        if line[0:1] in {"\u2705", "\u274c"} or line.startswith("\u26a0"):
            status = _normalize_status(line[0])
            rest = line[1:].strip()
        else:
            match = re.match(r"^(\[[A-Z]+\])\s+(.*)$", line)
            if not match:
                continue
            status = _normalize_status(match.group(1))
            rest = match.group(2).strip()

        parts = rest.split(None, 1)
        if not parts:
            continue
        name = parts[0]
        detail = parts[1] if len(parts) > 1 else ""
        items.append((status, name, detail))
    return items


def _friendly_name(name: str) -> str:
    mapping = {
        "L0_broksum": "broksum",
        "L0_yf_daily": "yf_daily",
        "L0_global": "global",
        "L0_yf_1h": "yf_1h",
        "DB_State": "DuckDB",
        "DB_Sync": "DuckDB",
        "DB_Price": "entry_price",
        "DB_Brok": "broker_features",
        "Model": "model",
    }
    return mapping.get(name, name)


def _display_status(status: str) -> str:
    mapping = {
        "[OK]": "✅",
        "[FAIL]": "❌",
        "[WARN]": "⚠️",
        "[WAIT]": "⏳",
        "[SKIP]": "⏭️",
        "[INFO]": "ℹ️",
    }
    return mapping.get(status, status)


def _split_groups(items: list[tuple[str, str, str]]) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    context_names = {"L0_broksum", "L0_yf_daily", "L0_global", "Model"}
    intraday_names = {"L0_yf_1h", "DB_State", "DB_Sync", "DB_Price", "DB_Brok"}
    context = [item for item in items if item[1] in context_names]
    intraday = [item for item in items if item[1] in intraday_names]
    return context, intraday


def _group_status(items: list[tuple[str, str, str]]) -> str:
    statuses = {status for status, _, _ in items}
    if "[FAIL]" in statuses:
        return "FAIL"
    if "[WAIT]" in statuses:
        return "WAIT"
    if "[WARN]" in statuses:
        return "WARN"
    return "OK"


def _stage(now: datetime, failed: bool, check_output: str) -> str:
    if "Weekend" in check_output or "MARKET CLOSED" in check_output:
        return "MARKET_CLOSED"
    if failed and ("L0_yf_1h" in check_output or "entry_price" in check_output):
        return "WAITING_PRE14" if now.hour < 15 else "BLOCKED"
    if failed:
        return "BLOCKED"
    if now.hour < 9:
        return "PRE_MARKET"
    if now.hour < 14:
        return "MORNING_WATCH"
    return "READY"


def _next_check(now: datetime, state: dict[str, Any]) -> datetime:
    return now + timedelta(minutes=int(state.get("interval_minutes", DEFAULT_INTERVAL_MINUTES)))


def _cook_line(now: datetime, cook_success: bool | None, cook_seconds: int | None) -> str:
    if cook_success is True:
        suffix = f" | {cook_seconds}s" if cook_seconds is not None else ""
        return f"Data updated: {now.strftime('%H:%M:%S')}{suffix}"
    if cook_success is False:
        return "Data update: failed"
    return "Data update: not needed"


def _bold(text: str) -> str:
    return f"<b>{html.escape(text)}</b>"


def _escape(text: str) -> str:
    return html.escape(text, quote=False)


def format_status_message(
    state: dict[str, Any],
    *,
    check_output: str,
    failed: bool,
    now: datetime | None = None,
    cook_success: bool | None = None,
    cook_seconds: int | None = None,
) -> str:
    if now is None:
        now = datetime.now(WIB)
    if now.tzinfo is None:
        now = now.replace(tzinfo=WIB)

    d_day, t_minus_1 = _parse_context(check_output)
    items = _parse_check_items(check_output)
    context_items, intraday_items = _split_groups(items)
    stage = _stage(now, failed, check_output)
    overall = "READY" if not failed else ("WAITING" if stage == "WAITING_PRE14" else "BLOCKED")
    next_check = _next_check(now, state).strftime("%H:%M")

    if state.get("status_format") == "compact":
        cook = "skipped"
        if cook_success is True:
            cook = f"{cook_seconds}s" if cook_seconds is not None else "OK"
        elif cook_success is False:
            cook = "FAIL"
        return "\n".join(
            [
                _bold(f"BSJP [{now.strftime('%H:%M')}] {overall}"),
                _escape(f"D {d_day} | T-1 {t_minus_1}"),
                (
                    _escape(
                        f"T-1: {_group_status(context_items)} | "
                        f"T-pre14: {_group_status(intraday_items)} | "
                        f"DB: {_group_status([i for i in intraday_items if i[1].startswith('DB_')])} | "
                        f"Cook: {cook}"
                    )
                ),
                _escape(f"Next: {next_check}"),
            ]
        )

    lines = [
        _bold(f"BSJP HEARTBEAT [{now.strftime('%H:%M')} WIB]"),
        _escape(f"Date: {d_day} | T-1: {t_minus_1}"),
        f"Stage: {_bold(stage)}",
        f"Overall: {_bold(overall)}",
        "",
        _bold("T-1 CONTEXT"),
    ]
    for status, name, detail in context_items:
        lines.append(_escape(f"{_display_status(status)} {_friendly_name(name):<12} {detail}".rstrip()))
    if not context_items:
        lines.append(_escape("⚠️ no context checks parsed"))

    lines.extend(["", _bold("T INTRADAY")])
    for status, name, detail in intraday_items:
        lines.append(_escape(f"{_display_status(status)} {_friendly_name(name):<12} {detail}".rstrip()))
    if not intraday_items:
        lines.append(_escape("⚠️ no intraday checks parsed"))

    action = "Ready for recommendation" if not failed else "Needs attention before recommendation"
    if stage == "WAITING_PRE14":
        action = "Waiting for intraday data"
    elif stage == "MARKET_CLOSED":
        action = "Market is closed"
    elif cook_success is False:
        action = "Data update failed"

    lines.extend(
        [
            "",
            _bold("DATA UPDATE"),
            _escape(_cook_line(now, cook_success, cook_seconds)),
            _escape(f"Next status: {next_check}"),
            _escape(f"Recommendation: {action}"),
        ]
    )
    return "\n".join(lines)


def telegram_api(token: str, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = None
    if params:
        data = urllib.parse.urlencode(params).encode()
    with urllib.request.urlopen(url, data=data, timeout=10) as response:
        return json.loads(response.read().decode())


def send_telegram_message(token: str, chat_id: str, text: str) -> None:
    telegram_api(token, "sendMessage", {"chat_id": chat_id, "text": text, "parse_mode": "HTML"})


def discord_text_from_html(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<b>(.*?)</b>", r"**\1**", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def send_discord_webhook(webhook_url: str, text: str) -> None:
    payload = json.dumps({"content": discord_text_from_html(text)[:2000]}).encode()
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "BSJP-Heartbeat/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        response.read()


def process_telegram_commands(state: dict[str, Any], token: str, chat_id: str) -> int:
    offset = int(state.get("last_update_id", 0) or 0) + 1
    response = telegram_api(token, "getUpdates", {"offset": offset, "timeout": 1})
    updates = response.get("result", []) if response.get("ok", False) else []
    processed = 0
    for update in updates:
        update_id = int(update.get("update_id", 0) or 0)
        state["last_update_id"] = max(int(state.get("last_update_id", 0) or 0), update_id)
        message = update.get("message") or update.get("edited_message") or {}
        text = str(message.get("text", "") or "").strip()
        update_chat = message.get("chat") or {}
        if not text or str(update_chat.get("id", "")) != str(chat_id):
            continue
        reply = apply_command(state, text)
        if reply:
            send_telegram_message(token, chat_id, reply)
            processed += 1
    return processed


def _cmd_process_telegram(args: argparse.Namespace) -> int:
    state_path = Path(args.state_path)
    state = load_state(state_path)
    token = os.getenv("BSJP_TELEGRAM_TOKEN", "")
    chat_id = os.getenv("BSJP_TELEGRAM_CHAT_ID", "")
    if token and chat_id:
        try:
            process_telegram_commands(state, token, chat_id)
        except Exception as exc:  # Network failure must not block preflight.
            print(f"telegram command polling failed: {exc}", file=sys.stderr)
    save_state(state_path, state)
    return 0


def _cmd_should_send(args: argparse.Namespace) -> int:
    if args.force:
        return 0
    state = load_state(Path(args.state_path))
    if should_send(state):
        return 0
    next_epoch = int(state.get("last_sent_epoch", 0) or 0) + int(state["interval_minutes"]) * 60
    print(datetime.fromtimestamp(next_epoch, WIB).strftime("%H:%M:%S"))
    return 75


def _cmd_format(args: argparse.Namespace) -> int:
    state = load_state(Path(args.state_path))
    check_output = Path(args.check_output_file).read_text()
    cook_success = None
    if args.cook_success == "1":
        cook_success = True
    elif args.cook_success == "0":
        cook_success = False
    cook_seconds = int(args.cook_seconds) if args.cook_seconds else None
    msg = format_status_message(
        state,
        check_output=check_output,
        failed=bool(int(args.failed)),
        cook_success=cook_success,
        cook_seconds=cook_seconds,
    )
    Path(args.message_file).write_text(msg + "\n")
    return 0


def _cmd_mark_sent(args: argparse.Namespace) -> int:
    state_path = Path(args.state_path)
    state = load_state(state_path)
    mark_sent(state)
    save_state(state_path, state)
    return 0


def _cmd_send_telegram(args: argparse.Namespace) -> int:
    load_notify_env()
    token = os.getenv("BSJP_TELEGRAM_TOKEN", "")
    chat_id = os.getenv("BSJP_TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return 0
    text = Path(args.message_file).read_text()
    send_telegram_message(token, chat_id, text)
    return 0


def _cmd_send_discord(args: argparse.Namespace) -> int:
    load_notify_env()
    webhook_url = os.getenv("BSJP_DISCORD_WEBHOOK_URL", "")
    if not webhook_url:
        token = os.getenv("BSJP_DISCORD_TOKEN", "")
        if token:
            webhook_url = token if token.startswith("https://") else f"https://discord.com/api/webhooks/{token}"
    if not webhook_url:
        return 0
    text = Path(args.message_file).read_text()
    send_discord_webhook(webhook_url, text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("process-telegram")
    p.add_argument("--state-path", required=True)
    p.set_defaults(func=_cmd_process_telegram)

    p = sub.add_parser("should-send")
    p.add_argument("--state-path", required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_cmd_should_send)

    p = sub.add_parser("format")
    p.add_argument("--state-path", required=True)
    p.add_argument("--check-output-file", required=True)
    p.add_argument("--message-file", required=True)
    p.add_argument("--failed", required=True, choices=["0", "1"])
    p.add_argument("--cook-success", default="", choices=["", "0", "1"])
    p.add_argument("--cook-seconds", default="")
    p.set_defaults(func=_cmd_format)

    p = sub.add_parser("mark-sent")
    p.add_argument("--state-path", required=True)
    p.set_defaults(func=_cmd_mark_sent)

    p = sub.add_parser("send-telegram")
    p.add_argument("--message-file", required=True)
    p.set_defaults(func=_cmd_send_telegram)

    p = sub.add_parser("send-discord")
    p.add_argument("--message-file", required=True)
    p.set_defaults(func=_cmd_send_discord)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
