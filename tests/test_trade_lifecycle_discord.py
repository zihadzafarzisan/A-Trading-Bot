"""Targeted unit tests for trade lifecycle Discord dispatch.

These tests are hermetic: tests/conftest.py blocks real Discord HTTP.
We validate that the carry daemon and directional live broker trigger
`notify_trade_entry` and `notify_trade_exit` with expected fields.
"""

import types
import pytest

from crypto_quant.db.connection import DatabaseManager
from crypto_quant.execution.broker import Order
from crypto_quant.execution.carry_daemon import CarryHarvesterDaemon
from crypto_quant.execution.live_broker import BinanceLiveBroker

# Reuse the carry test helpers to avoid constructing large configs.
from tests.test_carry_daemon import make_daemon, FakeCarryBroker, _opp


class RecordingLifecycleNotifier:
    def __init__(self):
        self.calls = []

    def is_enabled(self):
        return True

    def notify_trade_entry(self, **kw):
        self.calls.append(("entry", kw))

    def notify_trade_exit(self, **kw):
        self.calls.append(("exit", kw))

    # Keep compatibility with carry daemon which still calls legacy notifiers.
    def notify_trade_open(self, **kw):
        pass

    def notify_trade_close(self, **kw):
        pass

    def notify_financial_audit(self, **kw):
        pass

    def notify_funding_settlement(self, **kw):
        pass

    def notify_circuit_breaker(self, **kw):
        pass


def test_carry_daemon_dispatches_trade_entry_and_exit(monkeypatch):
    """Carry daemon must dispatch entry once both legs filled and exit once
    unwind completes with a proper WIN/LOSS + fields."""
    db = DatabaseManager(in_memory=True)
    db.create_tables()

    # Force a valid carry admission.
    carry = FakeCarryBroker()
    daemon = make_daemon(db, carry, allocation=50.0)
    rec = RecordingLifecycleNotifier()
    daemon._notifier = rec

    # Open -> unwind on next tick due to min_margin_ratio gate in
    # the FakeConnector account defaults (configured by make_daemon).
    # Open
    daemon.tick()  # opens

    entries = [c for c in rec.calls if c[0] == "entry"]
    assert len(entries) == 1
    entry_kw = entries[0][1]
    assert entry_kw["symbol"] == "SOLUSDT"
    assert entry_kw["strategy"] == "CARRY_ARBITRAGE"
    assert entry_kw["side"] == "CARRY"
    assert entry_kw["order_id"]
    assert entry_kw["spot_fill_price"] is not None
    assert entry_kw["futures_fill_price"] is not None
    assert entry_kw["qty"] > 0

    # For a deterministic unwind close, use the same gate as the existing
    # daemon unit test: set min_margin_ratio high so health_check triggers
    # an emergency unwind on the next tick.
    daemon.config.min_margin_ratio = 0.99
    daemon.tick()

    exits = [c for c in rec.calls if c[0] == "exit"]
    assert len(exits) == 1
    exit_kw = exits[0][1]
    assert exit_kw["symbol"] == "SOLUSDT"
    assert exit_kw["strategy"] == "CARRY_ARBITRAGE"
    assert exit_kw["side"] == "CARRY"
    assert exit_kw["position_id"]
    assert exit_kw["net_pnl_usdt"] is not None
    assert exit_kw["roi_pct"] is not None
    assert exit_kw["exit_reason"]
    assert exit_kw["gross_pnl_usdt"] is not None
    assert exit_kw["funding_harvested_usdt"] is not None


class DummyNotifier:
    def __init__(self):
        self.calls = []

    def is_enabled(self):
        return True

    def notify_trade_entry(self, **kw):
        self.calls.append(("entry", kw))

    def notify_trade_exit(self, **kw):
        self.calls.append(("exit", kw))

    def notify_trade_open(self, **kw):
        pass

    def notify_trade_close(self, **kw):
        pass


def test_directional_live_broker_dispatches_entry_and_exit(monkeypatch):
    """Directional path: LiveBroker._notify_fill should call entry on buy
    fills and exit on sell fills."""
    notifier = DummyNotifier()

    # Minimal LiveBroker stub: we call _notify_fill directly, avoiding any
    # exchange/network logic.
    broker = types.SimpleNamespace()
    broker._notify_entries = {}
    broker._notifier = notifier

    # Construct filled buy order then filled sell order.
    buy = Order(
        order_id="o1",
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.1,
        order_type="market",
    )
    buy.status = "filled"
    buy.filled_quantity = 0.1
    buy.fill_price = 50000.0
    buy.exchange_order_id = "o1"

    sell = Order(
        order_id="o2",
        symbol="BTCUSDT",
        side="SELL",
        quantity=0.1,
        order_type="market",
    )
    sell.status = "filled"
    sell.filled_quantity = 0.1
    sell.fill_price = 51000.0
    sell.exchange_order_id = "o2"

    # Reuse the real method implementation by binding LiveBroker._notify_fill.
    from crypto_quant.execution.live_broker import BinanceLiveBroker
    # But tests don't import BinanceLiveBroker; LiveBroker class may differ.
    # Instead, directly call LiveBroker._notify_fill if it exists.

    # Import the actual broker class containing _notify_fill.
    # Bind the real implementation from BinanceLiveBroker.
    BinanceLiveBroker._notify_fill(broker, buy)
    BinanceLiveBroker._notify_fill(broker, sell)

    entries = [c for c in notifier.calls if c[0] == "entry"]
    exits = [c for c in notifier.calls if c[0] == "exit"]
    assert len(entries) == 1
    assert entries[0][1]["symbol"] == "BTCUSDT"
    assert entries[0][1]["strategy"] == "STRAT-TREND"

    assert len(exits) == 1
    assert exits[0][1]["symbol"] == "BTCUSDT"
    assert exits[0][1]["exit_reason"] == "MANUAL"
    assert exits[0][1]["net_pnl_usdt"] is not None


# --------------------------------------------------------------------------
# Directional exit reason tagging: the paper engine stamps the exit reason on
# the broker (``_last_exit_reason``) mapping the internal short-code ("tp"/"sl")
# to a canonical label, then the live broker's ``_notify_fill`` forwards that
# label to ``notify_trade_exit``. These tests drive the real implementations of
# both methods on a lightweight scripted broker.
# --------------------------------------------------------------------------

def _make_scripted_broker():
    """A minimal broker object supporting PaperTradingEngine._do_exit plus
    BinanceLiveBroker._notify_fill, with a recording lifecycle notifier."""
    closed = []

    class _PriceSource:
        def set_price(self, price):
            pass

    broker = types.SimpleNamespace(
        price_source=_PriceSource(),
        orders=[],
        closed_trades=closed,
        _notify_entries={},
        _notifier=RecordingLifecycleNotifier(),
        _last_exit_reason=None,
    )

    def place_order(order):
        broker.orders.append(order)
        closed.append({
            "trade_id": f"tid_{len(closed)}",
            "symbol": order.symbol,
            "side": "long",
            "quantity": order.quantity,
            "entry_price": 50000.0,
            "exit_price": float(order.fill_price or 0.0),
            "net_pnl": 100.0,
            "gross_pnl": 100.0,
            "direction": "long",
        })

    broker.place_order = place_order
    return broker


def _drive_directional_exit(short_code, expected_label, fill_price=51000.0):
    """Run a take-profit or stop-loss exit through _do_exit -> _notify_fill,
    returning (scripted_broker, exit_kwargs)."""
    from crypto_quant.execution.engine import PaperTradingEngine

    broker = _make_scripted_broker()
    # Seed the tracked entry so _notify_fill can compute realized PnL.
    broker._notify_entries["BTCUSDT"] = {
        "side": "long", "entry_price": 50000.0, "qty": 0.1,
    }

    pos = {
        "symbol": "BTCUSDT", "side": "long", "quantity": 0.1,
        "entry_price": 50000.0,
        "stop_loss": 49000.0, "take_profit": 51000.0,
    }
    # Stamp the exit reason via the paper engine's real _do_exit.
    engine = types.SimpleNamespace(broker=broker)
    PaperTradingEngine._do_exit(engine, pos, fill_price, short_code, bar_time=1)
    # The internal short-code must be canonicalized for the notify layer.
    assert broker._last_exit_reason == expected_label

    # Close the position through the live broker's real _notify_fill.
    sell = Order(
        order_id="o_exit", symbol="BTCUSDT", side="SELL",
        quantity=0.1, order_type="market",
    )
    sell.status = "filled"
    sell.filled_quantity = 0.1
    sell.fill_price = fill_price
    sell.exchange_order_id = "o_exit"
    BinanceLiveBroker._notify_fill(broker, sell)

    exits = [c for c in broker._notifier.calls if c[0] == "exit"]
    assert len(exits) == 1
    return broker, exits[0][1]


def test_directional_exit_reason_take_profit():
    """A take-profit close routed through _do_exit + _notify_fill must reach
    notify_trade_exit with exit_reason='TAKE_PROFIT'."""
    broker, kwargs = _drive_directional_exit("tp", "TAKE_PROFIT")
    assert kwargs["symbol"] == "BTCUSDT"
    assert kwargs["strategy"] == "STRAT-TREND"
    assert kwargs["exit_reason"] == "TAKE_PROFIT"
    assert kwargs["net_pnl_usdt"] is not None
    # The label must be consumed so a subsequent independent exit is not
    # mis-tagged as a take-profit.
    assert broker._last_exit_reason is None


def test_directional_exit_reason_stop_loss():
    """A stop-loss close routed through _do_exit + _notify_fill must reach
    notify_trade_exit with exit_reason='STOP_LOSS'."""
    broker, kwargs = _drive_directional_exit("sl", "STOP_LOSS", fill_price=49000.0)
    assert kwargs["symbol"] == "BTCUSDT"
    assert kwargs["strategy"] == "STRAT-TREND"
    assert kwargs["exit_reason"] == "STOP_LOSS"
    assert kwargs["net_pnl_usdt"] is not None
    assert broker._last_exit_reason is None
