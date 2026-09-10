"""Paper safety regression tests (STEP 10).

CRITICAL: These tests prove that paper mode can STRUCTURALLY NEVER call live
execution. A separate ``LiveBroker`` does not even exist in this codebase; the
``PaperTradingEngine`` rejects any non-``PaperBroker`` at construction time;
and every trade record always carries ``execution_mode='paper'``.
"""

import pytest
import numpy as np
import pandas as pd

from crypto_quant.strategies.base import BaseStrategy, Direction
from crypto_quant.risk.manager import RiskManager
from crypto_quant.execution.broker import CallbackPriceSource, Order
from crypto_quant.execution.paper import PaperBroker
from crypto_quant.execution.engine import (
    PaperTradingEngine, PaperTradingConfig, DataframePriceSource,
)
from crypto_quant.execution.worker import LiveTradingWorker, WorkerConfig
from crypto_quant.db.connection import DatabaseManager
from crypto_quant.db.models import ExecutionTrade


class SimpleLong(BaseStrategy):
    name = "SimpleLong"
    strategy_type = "test"
    def validate_params(self): pass
    def setup(self, df): return df.copy()
    def entry_signal(self, df, i, ctx=None):
        return (Direction.LONG, "always") if i >= 5 else (Direction.NONE, "")
    def compute_stop_loss(self, df, i, direction, entry_price):
        return entry_price * 0.98
    def compute_take_profit(self, df, i, direction, entry_price, stop_loss=None):
        return entry_price * 1.05


def _bars(n=120, start=100.0, drift=0.3):
    close = start + np.arange(n) * drift
    return pd.DataFrame({
        "timestamp": [1609459200000 + i * 3600_000 for i in range(n)],
        "open": close - 0.2, "high": close + 1.0,
        "low": close - 1.0, "close": close,
        "volume": np.full(n, 1_000_000.0),
    })


# =====================================================================
# Regression 1: PaperTradingEngine rejects non-PaperBroker
# =====================================================================
class TestEnginePaperBrokerGuard:
    """The engine must raise TypeError when given any non-PaperBroker."""

    def test_engine_rejects_plain_broker_object(self):
        """Passing a mock Broker (not PaperBroker) raises TypeError."""
        bars = _bars()

        class FakeLiveBroker:
            """A mock that pretends to be a live broker."""
            name = "live"
            def get_balance(self): return 50000.0
            def get_positions(self): return []
            def place_order(self, order):
                raise RuntimeError("LIVE ORDER SENT — THIS MUST NEVER HAPPEN")
            def get_price(self, symbol): return 100.0

        from crypto_quant.risk.manager import RiskManager
        with pytest.raises(TypeError, match="PaperBroker"):
            PaperTradingEngine(
                strategy=SimpleLong(),
                broker=FakeLiveBroker(),   # intentionally wrong broker type
                risk_manager=RiskManager(),
            )

    def test_engine_accepts_paper_broker(self):
        """A genuine PaperBroker must be accepted without error."""
        bars = _bars()
        src = DataframePriceSource(bars)
        broker = PaperBroker(price_source=src, starting_capital=1000.0)
        engine = PaperTradingEngine(
            strategy=SimpleLong(), broker=broker, risk_manager=RiskManager(),
        )
        assert engine.broker.name == "paper"

    def test_live_order_is_never_placed(self):
        """Verify every order goes through PaperBroker.place_order (simulated only)."""
        bars = _bars()
        src = DataframePriceSource(bars)
        broker = PaperBroker(price_source=src, starting_capital=1000.0)
        engine = PaperTradingEngine(
            strategy=SimpleLong(), broker=broker, risk_manager=RiskManager(),
        )
        result = engine.run_bars(bars)
        for order in broker.orders:
            assert order.status in ("filled", "rejected"), \
                f"Unexpected order status: {order.status}"
        # All closed trades carry execution_mode='paper'
        for t in result.trades:
            assert t["execution_mode"] == "paper", \
                f"Trade has wrong execution_mode: {t['execution_mode']!r}"


# =====================================================================
# Regression 2: CLI paper start never wires to a live broker
# =====================================================================
class TestCLIPaperNeverLive:
    """paper start constructs only PaperBroker and routes through PaperTradingEngine."""

    def test_worker_broker_is_paper(self, tmp_path):
        """LiveTradingWorker always constructs a PaperBroker."""
        bars = _bars()
        db = DatabaseManager(in_memory=True)
        db.create_tables()
        repo_factory = db
        # Seed data
        from crypto_quant.data.repository import MarketDataRepository
        repo = MarketDataRepository(db)
        repo.save(bars, "BTCUSDT", "1h", market_type="spot")

        cfg = WorkerConfig(
            strategy="trend", symbol="BTCUSDT", timeframe="1h",
            market_type="spot", mode="replay",
            start="2021-01-01", end="2021-04-01",
            starting_capital=1000.0,
        )
        worker = LiveTradingWorker(config=cfg, db=db)
        assert isinstance(worker.broker, PaperBroker), \
            "Worker must always have a PaperBroker — got %r" % type(worker.broker)
        assert worker.broker.name == "paper"

    def test_execution_trades_persisted_as_paper(self, tmp_path):
        """Every trade stored in the DB must carry execution_mode='paper'."""
        bars = _bars()
        db = DatabaseManager(in_memory=True)
        db.create_tables()
        from crypto_quant.data.repository import MarketDataRepository
        repo = MarketDataRepository(db)
        repo.save(bars, "BTCUSDT", "1h", market_type="spot")

        cfg = WorkerConfig(
            strategy="trend", symbol="BTCUSDT", timeframe="1h",
            market_type="spot", mode="replay", starting_capital=1000.0,
        )
        worker = LiveTradingWorker(config=cfg, db=db)
        worker.run_replay()

        session = db.get_session()
        try:
            trades = session.query(ExecutionTrade).all()
        finally:
            session.close()
        for t in trades:
            assert t.execution_mode == "paper", \
                f"Found non-paper trade: execution_mode={t.execution_mode!r} for trade {t.id}"


# =====================================================================
# Regression 3: No LiveBroker, no live execution path
# =====================================================================
class TestNoLiveBrokerInCodebase:
    """Structural safety: there is deliberately no LiveBroker in the codebase.
    This is by design (spec #56). If someone adds one, this test ensures it
    never reaches the paper trading path.
    """

    def test_no_live_broker_class(self):
        """LiveBroker must not exist in the execution module."""
        import crypto_quant.execution as ex
        assert not hasattr(ex, "LiveBroker"), \
            "LiveBroker must not be exported from the execution module"

    def test_no_live_order_in_paper_broker(self):
        """PaperBroker.place_order must never raise (it simulates), never net-calls."""
        broker = PaperBroker(
            price_source=CallbackPriceSource(lambda _: 100.0),
            starting_capital=1000.0,
        )
        filled = broker.place_order(
            Order(order_id="test", symbol="BTCUSDT", side="buy",
                  quantity=0.1, order_type="market")
        )
        assert filled.status == "filled"
        # No real exchange call was made — the broker only mutated local state.
        assert len(broker.positions) == 1
        assert broker.positions["BTCUSDT"]["side"] == "long"