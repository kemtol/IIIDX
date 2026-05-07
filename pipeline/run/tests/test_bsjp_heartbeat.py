from datetime import datetime, timezone
from pathlib import Path

from pipeline.run import bsjp_heartbeat as hb


def test_compactstatus_toggles_format():
    state = hb.default_state()

    response = hb.apply_command(state, "/compactstatus")

    assert state["status_format"] == "compact"
    assert response == "BSJP status format changed: COMPACT"

    response = hb.apply_command(state, "/compactstatus")

    assert state["status_format"] == "verbose"
    assert response == "BSJP status format changed: VERBOSE"


def test_intervalstatus_reads_and_sets_interval():
    state = hb.default_state()

    assert hb.apply_command(state, "/intervalstatus") == (
        "BSJP heartbeat interval: 5 minutes\nStatus format: VERBOSE"
    )

    response = hb.apply_command(state, "/intervalstatus 15")

    assert state["interval_minutes"] == 15
    assert response == "BSJP heartbeat interval changed: 15 minutes"


def test_intervalstatus_rejects_invalid_values():
    state = hb.default_state()

    response = hb.apply_command(state, "/intervalstatus 0")

    assert state["interval_minutes"] == 5
    assert response == "Invalid interval. Use /intervalstatus N with N between 1 and 60."


def test_should_send_respects_last_sent_epoch():
    state = hb.default_state()
    state["interval_minutes"] = 5
    state["last_sent_epoch"] = 1_000

    assert hb.should_send(state, now_epoch=1_299) is False
    assert hb.should_send(state, now_epoch=1_300) is True


def test_format_verbose_status_groups_readiness():
    state = hb.default_state()
    now = datetime(2026, 5, 7, 14, 5, tzinfo=hb.WIB)
    check_output = """
--- D-Day: 2026-05-07 | T-1: 2026-05-06 ---
[OK] L0_broksum   2026-05-06 19.4KB
[OK] L0_yf_daily  2026-05-06 40.6MB
[OK] L0_yf_1h     bars: 09,10,11,13,14
[OK] L0_global    2026-05-06 59.7KB
[OK] DB_State     Ready
[OK] Model        v19d loaded
""".strip()

    msg = hb.format_status_message(
        state,
        check_output=check_output,
        failed=False,
        now=now,
        cook_success=True,
        cook_seconds=8,
    )

    assert "<b>BSJP HEARTBEAT [14:05 WIB]</b>" in msg
    assert "Date: 2026-05-07 | T-1: 2026-05-06" in msg
    assert "Stage: <b>READY</b>" in msg
    assert "Overall: <b>READY</b>" in msg
    assert "<b>T-1 CONTEXT</b>" in msg
    assert "✅ broksum" in msg
    assert "<b>T INTRADAY</b>" in msg
    assert "✅ yf_1h" in msg
    assert "<b>DATA UPDATE</b>" in msg
    assert "Data updated: 14:05:00 | 8s" in msg
    assert "Next status: 14:10" in msg
    assert "Recommendation: Ready for recommendation" in msg


def test_format_compact_status():
    state = hb.default_state()
    state["status_format"] = "compact"
    now = datetime(2026, 5, 7, 14, 5, tzinfo=hb.WIB)

    msg = hb.format_status_message(
        state,
        check_output="--- D-Day: 2026-05-07 | T-1: 2026-05-06 ---\n[OK] DB_State Ready",
        failed=False,
        now=now,
        cook_success=True,
        cook_seconds=8,
    )

    assert msg == (
        "<b>BSJP [14:05] READY</b>\n"
        "D 2026-05-07 | T-1 2026-05-06\n"
        "T-1: OK | T-pre14: OK | DB: OK | Cook: 8s\n"
        "Next: 14:10"
    )


def test_load_state_creates_parent_directory(tmp_path: Path):
    state_path = tmp_path / "_STATE" / "bsjp_heartbeat.json"

    state = hb.load_state(state_path)

    assert state["status_format"] == "verbose"
    assert state_path.parent.exists()


def test_discord_text_from_html_uses_markdown_bold():
    text = hb.discord_text_from_html("<b>BSJP HEARTBEAT</b>\nT-1 &amp; T intraday")

    assert text == "**BSJP HEARTBEAT**\nT-1 & T intraday"


def test_load_notify_env_reads_export_file(tmp_path: Path, monkeypatch):
    env_path = tmp_path / ".bsjp_notify.env"
    env_path.write_text("export BSJP_DISCORD_WEBHOOK_URL='https://example.test/webhook'\n")
    monkeypatch.delenv("BSJP_DISCORD_WEBHOOK_URL", raising=False)

    hb.load_notify_env(env_path)

    assert hb.os.environ["BSJP_DISCORD_WEBHOOK_URL"] == "https://example.test/webhook"
