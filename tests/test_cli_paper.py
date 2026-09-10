"""CLI paper command tests (Phase 12).

Uses an in-memory database so nothing touches the real project DB, and
monkeypatches the CLI's config/db accessors to keep the tests isolated.
"""

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import crypto_quant.cli.main as main
from crypto_quant.config.settings import AppConfig
from crypto_quant.db.connection import DatabaseManager
from crypto_quant.data.repository import MarketDataRepository

runner = CliRunner()


@pytest.fixture
def cli_db(monkeypatch):
    """An in-memory DB that the CLI's get_db_manager() resolves to."""
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    monkeypatch.setattr(main, "get_db_manager", lambda db_path=None: db)
    monkeypatch.setattr(main, "get_config", lambda config_path=None: AppConfig())
    return db


def _seed(db, n=120, drift=0.5):
    """Seed a steadily rising BTCUSDT 1h series (guarantees long/TP trades)."""
    close = 100.0 + np.arange(n) * drift
    df = pd.DataFrame({
        "timestamp": [1609459200000 + i * 3600_000 for i in range(n)],
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.full(n, 1_000_000.0),
    })
    repo = MarketDataRepository(db)
    repo.save(df, "BTCUSDT", "1h", market_type="spot")
    return df


def test_paper_command_help():
    result = runner.invoke(main.app, ["paper", "--help"])
    assert result.exit_code == 0
    assert "start" in result.output
    assert "status" in result.output


def test_paper_start_replay_persists(cli_db):
    from crypto_quant.db.models import ExecutionTrade
    _seed(cli_db)
    result = runner.invoke(main.app, [
        "paper", "start", "--strategy", "trend",
        "--symbol", "BTCUSDT", "--timeframe", "1h", "--market", "spot",
    ])
    assert result.exit_code == 0, result.output
    assert "Session Summary" in result.output
    s = cli_db.get_session()
    try:
        n = (s.query(ExecutionTrade)
             .filter(ExecutionTrade.execution_mode == "paper").count())
    finally:
        s.close()
    assert n >= 1


def test_paper_status_empty(cli_db):
    result = runner.invoke(main.app, ["paper", "status"])
    assert result.exit_code == 0
    assert "No paper trades yet" in result.output


def test_paper_status_lists_trades(cli_db):
    test_paper_start_replay_persists(cli_db)
    result = runner.invoke(main.app, ["paper", "status"])
    assert result.exit_code == 0
    assert "Paper Account Summary" in result.output
    assert "Recent Paper Trades" in result.output


def test_paper_stop_no_session(cli_db):
    result = runner.invoke(main.app, ["paper", "stop"])
    assert result.exit_code == 0
    assert "No active live paper session" in result.output


def test_paper_stop_sets_stop_requested(cli_db, monkeypatch, tmp_path):
    sess = tmp_path / "paper_session.json"
    sess.write_text(json.dumps({"phase": "running", "run_id": "R1"}), encoding="utf-8")
    monkeypatch.setattr(main, "SESSION_FILE", str(sess))
    result = runner.invoke(main.app, ["paper", "stop"])
    assert result.exit_code == 0
    assert "Stop requested" in result.output
    assert json.loads(sess.read_text(encoding="utf-8"))["stop_requested"] is True


def test_paper_report_writes_html_without_persisting(cli_db, tmp_path):
    from crypto_quant.db.models import ExecutionTrade
    _seed(cli_db)
    out = tmp_path / "paper.html"

    s = cli_db.get_session()
    try:
        before = (s.query(ExecutionTrade)
                  .filter(ExecutionTrade.execution_mode == "paper").count())
    finally:
        s.close()

    result = runner.invoke(main.app, [
        "paper", "report", "--strategy", "trend",
        "--symbol", "BTCUSDT", "--timeframe", "1h", "--market", "spot",
        "--output", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "Paper Trading" in out.read_text(encoding="utf-8")

    s = cli_db.get_session()
    try:
        after = (s.query(ExecutionTrade)
                 .filter(ExecutionTrade.execution_mode == "paper").count())
    finally:
        s.close()
    # report renders without accumulating duplicate session rows
    assert after == before


def test_paper_start_requires_data(cli_db):
    result = runner.invoke(main.app, [
        "paper", "start", "--strategy", "trend",
        "--symbol", "BTCUSDT", "--timeframe", "1h", "--market", "spot",
    ])
    assert result.exit_code != 0
    assert "Not enough data" in result.output