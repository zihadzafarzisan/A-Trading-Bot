"""Offline Cash-and-Carry dashboard persistence and rendering tests."""
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

import crypto_quant.cli.main as main
from crypto_quant.dashboard import DashboardData, DashboardGenerator
from crypto_quant.db.connection import DatabaseManager
from crypto_quant.db.models import CarryFundingPaymentRecord, CarryPositionRecord

runner = CliRunner()


def seed_carry(db):
    session = db.get_session()
    try:
        one = CarryPositionRecord(
            position_id="TWIN-SOL-1", symbol="SOLUSDT", quantity=0.5,
            spot_fill_price=100.0, futures_fill_price=100.2,
            entry_basis_spread_pct=0.2, status="OPEN",
            opened_at=datetime(2026, 1, 1, 0, tzinfo=timezone.utc),
        )
        two = CarryPositionRecord(
            position_id="TWIN-SOL-2", symbol="SOLUSDT", quantity=0.3,
            spot_fill_price=101.0, futures_fill_price=100.8,
            entry_basis_spread_pct=-0.198, status="CLOSED",
            opened_at=datetime(2026, 1, 2, 0, tzinfo=timezone.utc),
            closed_at=datetime(2026, 1, 3, 0, tzinfo=timezone.utc),
        )
        session.add_all([one, two])
        session.add_all([
            CarryFundingPaymentRecord(
                position_id="TWIN-SOL-1", symbol="SOLUSDT", funding_rate=0.0001,
                funding_payment_usdt=0.10, mark_price=100.5,
                timestamp=datetime(2026, 1, 1, 8, tzinfo=timezone.utc),
            ),
            CarryFundingPaymentRecord(
                position_id="TWIN-SOL-1", symbol="SOLUSDT", funding_rate=0.0002,
                funding_payment_usdt=0.20, mark_price=101.0,
                timestamp=datetime(2026, 1, 1, 16, tzinfo=timezone.utc),
            ),
        ])
        session.commit()
    finally:
        session.close()


def test_from_carry_shape_metrics_and_chronological_curves():
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    seed_carry(db)

    data = DashboardGenerator.from_carry(db)
    carry = data.carry_data
    assert data.title == "Cash-and-Carry Performance & Yield Dashboard"
    metrics = carry["metrics"]
    assert metrics["active_carries"] == 1
    assert metrics["total_positions"] == 2
    assert metrics["total_funding_usdt"] == pytest.approx(0.30)
    assert metrics["avg_entry_basis_pct"] == pytest.approx((0.2 + -0.198) / 2)
    assert carry["positions"][0]["opened_at"] == "2026-01-01 00:00"
    assert carry["positions"][0]["closed_at"] == "-"
    assert carry["funding_payments"][0]["timestamp"] == "2026-01-01 16:00"
    assert carry["funding_payments"][0]["funding_rate_pct"] == 0.02
    assert [point["equity"] for point in carry["funding_curve"]] == pytest.approx([0.1, 0.3])
    assert [point["equity"] for point in carry["basis_curve"]] == pytest.approx([0.2, -0.198])


def test_empty_carry_dashboard_has_requested_empty_state(tmp_path):
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    data = DashboardGenerator.from_carry(db)
    out = tmp_path / "carry-empty.html"
    DashboardGenerator().generate(data, out)
    content = out.read_text(encoding="utf-8")
    assert "No Cash-and-Carry positions or funding payments recorded in this database." in content
    assert "cdn." not in content.lower()
    assert "<script src=" not in content
    assert "googleapis" not in content.lower()


def test_carry_dashboard_contains_kpis_svg_ledgers_and_badges(tmp_path):
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    seed_carry(db)
    out = tmp_path / "carry.html"
    DashboardGenerator().generate(DashboardGenerator.from_carry(db), out)
    content = out.read_text(encoding="utf-8")
    for expected in (
        "Confirmed Funding", "Active Positions", "Total Carry Trades", "Average Entry Basis", "Strategy Status",
        "Cumulative Realized Funding (USDT)", "Historical Entry Basis Spread (%)",
        "Active & Historical Carry Positions", "Confirmed 8-Hour Funding Settlements",
        "TWIN-SOL-1", "OPEN", "CLOSED", "<svg",
    ):
        assert expected in content
    assert "cdn." not in content.lower()
    assert "<script src=" not in content
    assert "<link rel=" not in content
    assert "googleapis" not in content.lower()


def test_non_carry_dashboard_does_not_render_carry_section(tmp_path):
    out = tmp_path / "normal.html"
    DashboardGenerator().generate(DashboardData(title="Normal", metrics={"win_rate": 0.5}), out)
    assert "Cash-and-Carry Performance &amp; Yield" not in out.read_text(encoding="utf-8")


def test_dashboard_carry_cli_uses_local_ledger(monkeypatch, tmp_path):
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    seed_carry(db)
    monkeypatch.setattr(main, "get_db_manager", lambda db_path=None: db)
    output = tmp_path / "carry-cli.html"

    result = runner.invoke(main.app, ["dashboard", "carry", "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert "Cash-and-carry dashboard written" in result.output
    assert "Cumulative Realized Funding (USDT)" in output.read_text(encoding="utf-8")
