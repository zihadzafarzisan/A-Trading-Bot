"""Phase 2 Step 2.1 — local trade ledger unit tests.

Verifies the append-only ``TradeLedger`` primitive (directory creation, clean
JSONL lines, ordered ``load_recent_trades``) and that a real ``CarryHarvesterDaemon``
unwind writes one forensic line to the ledger. All hermetic — no network.
"""

import json
import os

import pytest

from crypto_quant.analytics import TradeLedger


# ---------------------------------------------------------------------------
# TradeLedger primitive
# ---------------------------------------------------------------------------


def test_ledger_creates_parent_dir_and_appends_clean_jsonl(tmp_path):
    ledger_path = tmp_path / "nested" / "deeper" / "trade_ledger.jsonl"
    ledger = TradeLedger(str(ledger_path))

    ledger.record_closed_trade(
        {
            "trade_id": "T1",
            "symbol": "SOLUSDT",
            "strategy": "CARRY_ARBITRAGE",
            "exit_reason": "YIELD_COMPRESSION",
            "net_pnl_usd": 1.25,
        }
    )

    assert ledger_path.exists(), "record_closed_trade should create parent dirs + file"
    line = ledger_path.read_text(encoding="utf-8").strip()
    record = json.loads(line)
    # Exactly one JSON object, with the required fields and an auto-added UTC stamp.
    assert record["trade_id"] == "T1"
    assert record["symbol"] == "SOLUSDT"
    assert record["strategy"] == "CARRY_ARBITRAGE"
    assert record["exit_reason"] == "YIELD_COMPRESSION"
    assert record["net_pnl_usd"] == pytest.approx(1.25)
    assert record["recorded_at"]  # timestamp added when absent


def test_record_closed_trade_does_not_mutate_caller_dict(tmp_path):
    ledger = TradeLedger(str(tmp_path / "ledger.jsonl"))
    original = {
        "trade_id": "T2",
        "symbol": "ETHUSDT",
        "strategy": "CARRY_ARBITRAGE",
        "exit_reason": "MANUAL",
        "net_pnl_usd": -0.10,
    }
    snapshot = dict(original)

    ledger.record_closed_trade(original)

    assert original == snapshot, "caller dict must not gain the recorded_at stamp"


def test_record_closed_trade_rejects_missing_required_field(tmp_path):
    ledger = TradeLedger(str(tmp_path / "ledger.jsonl"))
    record = {
        "trade_id": "T3",
        "symbol": "SOLUSDT",
        "strategy": "CARRY_ARBITRAGE",
        # missing exit_reason and net_pnl_usd
    }
    with pytest.raises(ValueError):
        ledger.record_closed_trade(record)
    assert not (tmp_path / "ledger.jsonl").exists(), "invalid records must not be written"


def test_preserves_explicit_recorded_at_stamp(tmp_path):
    ledger = TradeLedger(str(tmp_path / "ledger.jsonl"))
    record = {
        "trade_id": "T4",
        "symbol": "BTCUSDT",
        "strategy": "CARRY_ARBITRAGE",
        "exit_reason": "MANUAL",
        "net_pnl_usd": 0.0,
        "recorded_at": "2026-01-01T00:00:00+00:00",
    }
    ledger.record_closed_trade(record)
    line = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(line)["recorded_at"] == "2026-01-01T00:00:00+00:00"


def test_load_recent_trades_returns_records_in_file_order(tmp_path):
    ledger = TradeLedger(str(tmp_path / "ledger.jsonl"))
    for i in range(5):
        ledger.record_closed_trade(
            {
                "trade_id": f"T{i}",
                "symbol": "SOLUSDT",
                "strategy": "CARRY_ARBITRAGE",
                "exit_reason": "MANUAL",
                "net_pnl_usd": i,
            }
        )

    records = ledger.load_recent_trades()
    assert [r["trade_id"] for r in records] == ["T0", "T1", "T2", "T3", "T4"]
    assert [r["net_pnl_usd"] for r in records] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_load_recent_trades_respects_limit_and_missing_file(tmp_path):
    # Missing file -> empty list.
    assert TradeLedger(str(tmp_path / "nope.jsonl")).load_recent_trades() == []

    ledger = TradeLedger(str(tmp_path / "ledger.jsonl"))
    for i in range(10):
        ledger.record_closed_trade(
            {
                "trade_id": f"T{i}",
                "symbol": "SOLUSDT",
                "strategy": "CARRY_ARBITRAGE",
                "exit_reason": "MANUAL",
                "net_pnl_usd": i,
            }
        )
    records = ledger.load_recent_trades(limit=3)
    assert [r["trade_id"] for r in records] == ["T7", "T8", "T9"]


# ---------------------------------------------------------------------------
# CarryDaemon integration: an unwind writes a ledger line
# ---------------------------------------------------------------------------


def test_daemon_unwind_writes_trade_ledger_line(tmp_path):
    from crypto_quant.db.connection import DatabaseManager
    from test_carry_daemon import FakeCarryBroker, FakeConnector, RecordingNotifier, make_daemon

    ledger_path = tmp_path / "trade_ledger.jsonl"

    db = DatabaseManager(in_memory=True)
    db.create_tables()
    # Healthy entry buffer (90% clears the pre-flight gatekeeper); maintenance
    # then spikes so the live buffer (10%) breaches the health-check minimum.
    futures = FakeConnector(account={"totalMarginBalance": "100", "totalMaintMargin": "10"})
    carry = FakeCarryBroker(futures_connector=futures)
    daemon = make_daemon(db, carry, min_margin_ratio=0.20, ledger_path=str(ledger_path))
    daemon._notifier = RecordingNotifier()

    daemon.tick()  # opens (entry margin ratio 90% >= 25% pre-flight hurdle)
    futures.account["totalMaintMargin"] = "90"
    daemon.tick()  # 10% buffer < 20% min -> emergency unwind -> ledger write

    assert carry.unwound, "expected an unwind to have been executed"
    assert ledger_path.exists(), "ledger file should exist after an unwind"
    lines = [l for l in ledger_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1, f"expected exactly one ledger line, got {len(lines)}"

    record = json.loads(lines[0])
    assert record["trade_id"] == carry.unwound[0]
    assert record["symbol"] == "SOLUSDT"
    assert record["strategy"] == "CARRY_ARBITRAGE"
    # net_pnl is mapped from the close metrics' net_pnl_usdt; a numeric value.
    assert isinstance(record["net_pnl_usd"], (int, float))
    assert record["recorded_at"]

    # The same closed trade should also be readable back via load_recent_trades.
    loaded = daemon.trade_ledger.load_recent_trades(limit=1)
    assert len(loaded) == 1 and loaded[0]["trade_id"] == carry.unwound[0]