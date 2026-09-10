"""Worker process lifecycle, modes, lock, and detached behavior tests (STEP 10)."""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

from crypto_quant.execution.worker import LiveTradingWorker, WorkerConfig, WorkerLock
from crypto_quant.db.connection import DatabaseManager


def _bars(n=10):
    close = 100.0 + np.arange(n) * 0.1
    return pd.DataFrame({
        "timestamp": [1609459200000 + i * 3600_000 for i in range(n)],
        "open": close, "high": close + 0.1,
        "low": close - 0.1, "close": close,
        "volume": np.full(n, 1_000_000.0),
    })


@pytest.fixture
def db():
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    return db


def test_worker_lock_prevents_duplicates(tmp_path):
    lock_file = tmp_path / "paper.lock"
    lock1 = WorkerLock(path=lock_file, heartbeat_s=300)

    # First acquisition succeeds
    assert lock1.acquire("RUN-1") is True
    assert lock_file.exists()

    # Second acquisition fails (duplicate prevention)
    lock2 = WorkerLock(path=lock_file, heartbeat_s=300)
    assert lock2.acquire("RUN-2") is False

    lock1.release()


def test_worker_lock_reclaims_stale(tmp_path):
    lock_file = tmp_path / "paper.lock"

    # Assume a crash left a lockfile with an old heartbeat
    lock_file.write_text(json.dumps({
        "pid": 99999, "run_id": "CRASHED", "last_seen": 100.0,
    }), encoding="utf-8")

    # A new worker should see it's stale (last_seen way in the past relative to now)
    lock2 = WorkerLock(path=lock_file, heartbeat_s=30)
    assert lock2.acquire("NEW-RUN") is True
    assert lock2._owned is True
    lock2.release()


def test_worker_replay_mode_completes_and_stops(db, tmp_path):
    from crypto_quant.data.repository import MarketDataRepository
    repo = MarketDataRepository(db)
    repo.save(_bars(10), "BTCUSDT", "1h", market_type="spot")

    lock_file = tmp_path / "replay.lock"
    lock = WorkerLock(path=lock_file)
    cfg = WorkerConfig(
        strategy="trend", symbol="BTCUSDT", timeframe="1h", market_type="spot",
        mode="replay", run_id="R1", persist=True,
    )
    worker = LiveTradingWorker(config=cfg, db=db, lock=lock)
    res = worker.run()

    # Worker ran loop and finished cleanly
    assert res.n_orders >= 0
    # Lock was released cleanly in finally block
    assert not lock_file.exists()


def test_worker_realtime_mode_polls_and_stops_on_signal(db, tmp_path):
    """Test the realtime loop processes bars and stops gracefully via lock signal."""
    # We mock out the ExchangeAdapter so we don't hit the network
    class FakeAdapter:
        def __init__(self):
            self.i = 0
            self.bars = _bars(20)
        def get_ohlcv_as_dataframe(self, symbol, tf, limit):
            self.i += 1
            # Return progressively more bars to simulate live feed ticking
            cutoff = min(len(self.bars), 5 + self.i)
            return self.bars.iloc[:cutoff].copy()
        def close(self): pass

    cfg = WorkerConfig(
        strategy="trend", symbol="BTCUSDT", timeframe="1h", market_type="spot",
        mode="realtime", poll_interval=0, run_id="LIVE-1", persist=False,
    )

    lock_file = tmp_path / "rt.lock"
    lock = WorkerLock(path=lock_file)
    worker = LiveTradingWorker(config=cfg, db=db, lock=lock, adapter=FakeAdapter())

    # We mock _stop_flagged to return True after 3 ticks so the infinite loop ends
    tick_count = 0
    original_stop_flagged = worker._stop_flagged
    def stop_later():
        nonlocal tick_count
        tick_count += 1
        return tick_count > 3
    worker._stop_flagged = stop_later

    worker.run()
    # It processed ticks and stopped gracefully
    assert worker.state.bars_processed > 0
    assert not lock_file.exists()


def test_worker_stop_method_writes_flag(tmp_path, monkeypatch):
    """Test that worker.stop() sets the cross-process flag."""
    cfg = WorkerConfig(mode="realtime", run_id="S1", persist=False)
    worker = LiveTradingWorker(config=cfg)

    # Monkeypatch the stop flag path
    flag_file = tmp_path / "session_flag.json"
    worker._stop_flag_path = lambda: flag_file

    assert not worker._stop_flagged()
    worker.stop()
    assert worker._stop_flagged() is True
    assert json.loads(flag_file.read_text())["stop_requested"] is True