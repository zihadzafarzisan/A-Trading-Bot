"""Shared test fixtures."""

import sys
import os
from pathlib import Path

# Ensure the tests directory is importable (for shared helpers like fake_adapter)
sys.path.insert(0, os.path.dirname(__file__))

import pytest
import tempfile

from crypto_quant.config.settings import AppConfig
from crypto_quant.db.connection import DatabaseManager, init_database


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config():
    """Provide a test configuration."""
    return AppConfig()


@pytest.fixture
def test_db():
    """Provide a test database (in-memory for testing)."""
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    yield db


@pytest.fixture
def sample_ohlcv():
    """Provide sample OHLCV data."""
    return [
        {"timestamp": 1609459200000, "open": 100.0, "high": 105.0, "low": 98.0, "close": 103.0, "volume": 1000000},
        {"timestamp": 1609462800000, "open": 103.0, "high": 108.0, "low": 101.0, "close": 106.0, "volume": 1200000},
        {"timestamp": 1609466400000, "open": 106.0, "high": 110.0, "low": 104.0, "close": 109.0, "volume": 1500000},
    ]


@pytest.fixture
def sample_trade():
    """Provide a sample trade record."""
    return {
        "id": "TRADE-TEST001",
        "experiment_id": "EXP-TEST001",
        "strategy_id": "STRAT-TEST001",
        "symbol": "BTCUSDT",
        "market_type": "spot",
        "direction": "long",
        "timeframe": "1h",
        "entry_time": 1609459200000,
        "entry_price": 50000.0,
        "quantity": 0.1,
        "leverage": 1,
    }


@pytest.fixture
def sample_experiment():
    """Provide a sample experiment record."""
    return {
        "id": "EXP-TEST001",
        "strategy_id": "STRAT-TEST001",
        "config": "{}",
        "start_date": "2020-01-01",
        "end_date": "2026-09-08",
        "status": "completed",
    }


@pytest.fixture
def sample_strategy():
    """Provide a sample strategy record."""
    return {
        "id": "STRAT-TEST001",
        "name": "Test Strategy",
        "type": "trend",
        "description": "A test strategy",
        "parameters": "{}",
        "version": 1,
    }
