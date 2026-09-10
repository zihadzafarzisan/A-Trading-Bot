"""Persistence and recovery state tests (STEP 10).

Verify account state (cash, equity, positions) and worker state are successfully
saved to DB and recovered on resume.
"""

from crypto_quant.execution.persistence import load_account_snapshot, save_account_snapshot
from crypto_quant.execution.worker import LiveTradingWorker, WorkerConfig
from crypto_quant.db.connection import DatabaseManager


def test_save_account_snapshot_and_load(tmp_path):
    import json
    import pandas as pd
    import numpy as np
    db = DatabaseManager(in_memory=True)
    db.create_tables()

    # Seed enough data so the worker initializes
    from crypto_quant.data.repository import MarketDataRepository
    close = 100.0 + np.arange(10) * 0.1
    bars = pd.DataFrame({
        "timestamp": [1609459200000 + i * 3600_000 for i in range(10)],
        "open": close, "high": close + 0.1,
        "low": close - 0.1, "close": close,
        "volume": np.full(10, 1_000_000.0),
    })
    MarketDataRepository(db).save(bars, "BTCUSDT", "1h", market_type="spot")

    cfg = WorkerConfig(
        strategy="trend", symbol="BTCUSDT", timeframe="1h",
        market_type="spot", mode="replay",
        starting_capital=1000.0,
    )
    worker = LiveTradingWorker(config=cfg, db=db)

    # Set up some fake state
    worker.broker.cash = 950.0
    worker.broker.positions = {
        "BTCUSDT": {
            "symbol": "BTCUSDT", "side": "long", "quantity": 0.5,
            "entry_price": 50000.0, "fees_paid": 5.0, "leverage": 1,
            "stop_loss": 48000.0, "take_profit": None, "funding": 0.0,
        }
    }
    worker.broker.closed_trades = [{}, {}, {}]  # 3 trades

    # Snapshot to DB
    run_id = "TEST-RUN-01"
    save_account_snapshot(
        db, run_id=run_id, config=worker.engine.config,
        broker=worker.broker, risk=worker.risk, strategy="trend",
        worker_state={"bars_processed": 55}, status="running", last_market_ts=1700000000000
    )

    # Load it back
    acct = load_account_snapshot(db, run_id)
    assert acct is not None
    assert acct.run_id == run_id
    assert acct.cash == 950.0
    assert acct.closed_trades == 3        # saved from len(closed_trades)
    assert acct.mode == "replay"
    assert acct.status == "running"
    assert acct.last_market_ts == 1700000000000

    pos_dict = json.loads(acct.positions)
    assert len(pos_dict) == 1
    assert pos_dict[0]["quantity"] == 0.5
    assert pos_dict[0]["stop_loss"] == 48000.0

    state_dict = json.loads(acct.worker_state)
    assert state_dict["bars_processed"] == 55


def test_resume_broker_state_on_start(tmp_path):
    """If a resume run_id is supplied, the worker should restore the account cash/positions."""
    from crypto_quant.data.repository import MarketDataRepository
    import pandas as pd
    import numpy as np

    db = DatabaseManager(in_memory=True)
    db.create_tables()
    repo = MarketDataRepository(db)

    # Seed data
    close = 100.0 + np.arange(10) * 0.1
    bars = pd.DataFrame({
        "timestamp": [1609459200000 + i * 3600_000 for i in range(10)],
        "open": close, "high": close + 0.1,
        "low": close - 0.1, "close": close,
        "volume": np.full(10, 1_000_000.0),
    })
    repo.save(bars, "BTCUSDT", "1h", market_type="spot")

    # Run 1: Create a snapshot
    cfg1 = WorkerConfig(
        strategy="trend", symbol="BTCUSDT", timeframe="1h", market_type="spot",
        mode="replay", starting_capital=1000.0, run_id="R1", persist=True,
    )
    worker1 = LiveTradingWorker(config=cfg1, db=db)
    worker1.broker.cash = 1234.56
    worker1.broker.closed_trades = [{}, {}]
    worker1._snapshot(status="stopped")

    # Run 2: Resume from R1
    cfg2 = WorkerConfig(
        strategy="trend", symbol="BTCUSDT", timeframe="1h", market_type="spot",
        mode="replay", starting_capital=1000.0, run_id="R2", resume="R1", persist=True,
    )
    worker2 = LiveTradingWorker(config=cfg2, db=db)

    # The broker in worker2 should have inherited cash=1234.56 and closed=2 from R1
    assert worker2.broker.cash == 1234.56
    assert len(worker2.broker.closed_trades) == 2