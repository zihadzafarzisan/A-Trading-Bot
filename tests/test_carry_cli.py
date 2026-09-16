"""Hermetic CLI tests for cash-and-carry terminal controls."""
import re
from datetime import datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

import crypto_quant.cli.main as main
from crypto_quant.db.connection import DatabaseManager
from crypto_quant.db.models import CarryFundingPaymentRecord, CarryPositionRecord

runner = CliRunner()


def _plain(output: str) -> str:
    """Strip rich ANSI colour/bold escapes so substring assertions match text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", output)


@pytest.fixture
def cli_db(monkeypatch):
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    monkeypatch.setattr(main, "get_db_manager", lambda db_path=None: db)
    return db


def seed_position(db, position_id="TWIN-1", status="OPEN"):
    session = db.get_session()
    try:
        row = CarryPositionRecord(
            position_id=position_id, symbol="SOLUSDT", quantity=0.5,
            spot_fill_price=100.0, futures_fill_price=100.2,
            entry_basis_spread_pct=0.2, status=status,
            opened_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        session.add(row)
        session.commit()
        return row
    finally:
        session.close()


def test_carry_help_lists_commands():
    result = runner.invoke(main.app, ["carry", "--help"])
    assert result.exit_code == 0
    for command in ("status", "payments", "unwind", "start"):
        assert command in _plain(result.output)


def test_status_empty_ledger(cli_db):
    result = runner.invoke(main.app, ["carry", "status"])
    assert result.exit_code == 0
    assert "No carry positions recorded" in _plain(result.output)


def test_status_renders_history_and_live_metrics(cli_db, monkeypatch):
    seed_position(cli_db)
    session = cli_db.get_session()
    try:
        session.add(CarryFundingPaymentRecord(
            position_id="TWIN-1", symbol="SOLUSDT", funding_rate=0.0001,
            funding_payment_usdt=0.25, mark_price=101.0,
            timestamp=datetime.now(timezone.utc),
        ))
        session.commit()
    finally:
        session.close()
    monkeypatch.setattr(main, "_carry_live_metrics", lambda record, funding: {
        "mark": 101.0, "spot": 100.5, "basis_pct": 0.4975,
        "margin_balance": 1000.0, "margin_buffer_pct": 85.0,
        "funding": funding, "net_pnl": 0.65,
    })

    result = runner.invoke(main.app, ["carry", "status"])
    assert result.exit_code == 0, result.output
    assert "Cash-and-Carry Position Ledger" in _plain(result.output)
    assert "TWIN-1" in _plain(result.output)
    assert "Live Carry Health" in _plain(result.output)
    assert "85.00%" in _plain(result.output)
    assert "0.6500" in _plain(result.output)


def test_status_exchange_failure_preserves_ledger(cli_db, monkeypatch):
    seed_position(cli_db)
    monkeypatch.setattr(main, "_carry_live_metrics", lambda *args: (_ for _ in ()).throw(RuntimeError("offline")))
    result = runner.invoke(main.app, ["carry", "status"])
    assert result.exit_code == 0
    assert "TWIN-1" in _plain(result.output)
    assert "Live metrics unavailable" in _plain(result.output)


def test_payments_limit_and_cumulative_total(cli_db):
    seed_position(cli_db)
    session = cli_db.get_session()
    try:
        for index, payment in enumerate((0.1, 0.2, -0.05)):
            session.add(CarryFundingPaymentRecord(
                position_id="TWIN-1", symbol="SOLUSDT", funding_rate=0.0001,
                funding_payment_usdt=payment, mark_price=100.0,
                timestamp=datetime.now(timezone.utc) + timedelta(hours=index),
            ))
        session.commit()
    finally:
        session.close()

    result = runner.invoke(main.app, ["carry", "payments", "--limit", "2"])
    assert result.exit_code == 0, result.output
    assert "Carry Funding Payment Ledger" in _plain(result.output)
    assert "Cumulative confirmed funding income: $+0.250000" in _plain(result.output)


def test_manual_unwind_invokes_broker_and_persists_closed(cli_db, monkeypatch):
    seed_position(cli_db)

    class FakeBroker:
        def unwind_twin_leg_carry(self, position):
            position.status = "CLOSED"
            position.closed_at = datetime.now(timezone.utc).timestamp()
            return position

        class futures:
            class connector:
                @staticmethod
                def get_positions(symbol):
                    return []
        class spot:
            class connector:
                @staticmethod
                def get_balances():
                    return {"SOL": {"total": 0.0}}

    monkeypatch.setattr(main, "_carry_build_broker", lambda dry_run: FakeBroker())
    result = runner.invoke(main.app, ["carry", "unwind", "TWIN-1", "--dry"])
    assert result.exit_code == 0, result.output
    assert "closed successfully" in _plain(result.output)
    session = cli_db.get_session()
    try:
        row = session.get(CarryPositionRecord, "TWIN-1")
        assert row.status == "CLOSED"
        assert row.closed_at is not None
    finally:
        session.close()


def test_manual_unwind_missing_position_never_constructs_broker(cli_db, monkeypatch):
    monkeypatch.setattr(main, "_carry_build_broker", lambda dry_run: pytest.fail("broker must not be built"))
    result = runner.invoke(main.app, ["carry", "unwind", "NOPE"])
    assert result.exit_code == 1
    assert "not found" in _plain(result.output)


def test_carry_start_passes_arguments_to_daemon(cli_db, monkeypatch):
    captured = {}

    class FakeDaemon:
        def __init__(self, config, broker, db, scanner=None):
            captured["config"] = config
            captured["broker"] = broker
            captured["scanner"] = scanner
        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(main, "_carry_build_broker", lambda dry_run: {"dry": dry_run})
    from crypto_quant.execution import carry_daemon
    monkeypatch.setattr(carry_daemon, "CarryHarvesterDaemon", FakeDaemon)

    result = runner.invoke(main.app, [
        "carry", "start", "--max-pairs", "4", "--total-usdt", "2000",
        "--scan-interval", "120", "--min-apr", "0.25", "--dry",
    ])
    assert result.exit_code == 0, result.output
    assert captured["ran"] is True
    assert captured["config"].max_active_pairs == 4
    assert captured["config"].total_allocation_usdt == pytest.approx(2000.0)
    assert captured["config"].scan_interval_seconds == pytest.approx(120.0)
    assert captured["config"].min_apr_threshold == pytest.approx(0.25)
    assert captured["config"].allocation_per_pair_usdt is None  # default split
    assert captured["broker"] == {"dry": True}
    assert captured["scanner"] is not None  # discovery is wired to FundingScanner
