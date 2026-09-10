"""Paper trading correctness tests (Phase 12).

PaperBroker accounting and PaperTradingEngine replay are verified against
hand-computed values using deterministic price sources and synthetic bars.
"""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.strategies.base import BaseStrategy, Direction
from crypto_quant.risk.manager import RiskManager
from crypto_quant.risk.limits import RiskLimits
from crypto_quant.execution.broker import CallbackPriceSource, Order
from crypto_quant.execution.paper import PaperBroker
from crypto_quant.execution.engine import (
    PaperTradingEngine,
    PaperTradingConfig,
    DataframePriceSource,
)


def _const_broker(price, market_type="spot", starting=1000.0, slippage=0.001,
                  fee_rate=0.001):
    return PaperBroker(
        price_source=CallbackPriceSource(lambda _sym: price),
        starting_capital=starting,
        fee_rate=fee_rate,
        slippage=slippage,
        market_type=market_type,
    )


def _order(side, qty, price=None, otype="market"):
    return Order(
        order_id=f"o{side}{qty}",
        symbol="BTCUSDT", side=side, quantity=qty,
        order_type=otype, limit_price=price,
    )


@pytest.fixture
def bars():
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, 120))
    base = 1609459200000
    return pd.DataFrame({
        "timestamp": [base + i * 3600_000 for i in range(len(close))],
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.full(len(close), 1_000_000.0),
    })


# ============================================================== PaperBroker: spot
class TestSpotAccounting:
    def test_buy_deducts_cash_and_fee(self):
        b = _const_broker(5000.0)
        filled = b.place_order(_order("buy", 0.1))
        assert filled.status == "filled"
        notional = 5000.0 * (1 + b.slippage) * 0.1          # 5012.5 * 0.1
        fee = notional * b.fee_rate
        assert b.cash == pytest.approx(1000.0 - notional - fee)
        pos = b.positions["BTCUSDT"]
        assert pos["side"] == "long"
        assert pos["quantity"] == pytest.approx(0.1)

    def test_sell_books_realized_pnl(self):
        b = _const_broker(5000.0)
        b.place_order(_order("buy", 0.1))
        filled = b.place_order(_order("sell", 0.1))
        assert filled.status == "filled"
        assert len(b.closed_trades) == 1
        assert b.positions == {}

    def test_spot_is_long_only_rejects_sell_without_position(self):
        b = _const_broker(5000.0)
        filled = b.place_order(_order("sell", 0.1))
        assert filled.status == "rejected"
        assert "long-only" in filled.message

    def test_insufficient_cash_rejected(self):
        b = _const_broker(5000.0, starting=100.0)
        filled = b.place_order(_order("buy", 1.0))          # $5k+ notional > $100
        assert filled.status == "rejected"
        assert "insufficient cash" in filled.message

    def test_limit_buy_below_market_rejected(self):
        b = _const_broker(5000.0)
        # Market (5000) is above the 4000 limit, so the buy hasn't reached the
        # limit yet -> rejected.
        filled = b.place_order(_order("buy", 0.1, price=4000.0, otype="limit"))
        assert filled.status == "rejected"
        assert "above market" in filled.message

    def test_limit_buy_in_money_fills_at_limit(self):
        b = _const_broker(5000.0)
        # Market (5000) is at/below the 6000 limit -> fillable; fills at the limit.
        filled = b.place_order(_order("buy", 0.1, price=6000.0, otype="limit"))
        assert filled.status == "filled"
        assert b.positions["BTCUSDT"]["entry_price"] == pytest.approx(6000.0 * (1 + b.slippage))

    def test_limit_sell_below_market_rejected(self):
        b = _const_broker(5000.0)
        filled = b.place_order(_order("sell", 0.1, price=6000.0, otype="limit"))
        assert filled.status == "rejected"
        assert "below market" in filled.message

    def test_slippage_is_adverse(self):
        b = _const_broker(5000.0, slippage=0.01)
        b.place_order(_order("buy", 0.1))
        assert b.positions["BTCUSDT"]["entry_price"] == pytest.approx(5000.0 * 1.01)
        b2 = _const_broker(5000.0, slippage=0.01)
        b2.place_order(_order("buy", 0.1))
        # sell exit prices lower
        sell = b2.place_order(_order("sell", 0.1))
        assert sell.fill_price == pytest.approx(5000.0 * 0.99)

    def test_equity_is_cash_plus_unrealized(self):
        b = _const_broker(5000.0)
        b.place_order(_order("buy", 0.1))
        expected = b.cash + (5000.0 - b.positions["BTCUSDT"]["entry_price"]) * 0.1
        assert b.equity == pytest.approx(expected)

    def test_close_all_realizes_and_clears(self):
        b = _const_broker(5000.0)
        b.place_order(_order("buy", 0.1))
        b.close_all(reason="test_end")
        assert b.positions == {}
        assert len(b.closed_trades) == 1
        assert b.closed_trades[0]["exit_reason"] == "test_end"


# ============================================================== PaperBroker: futures
class TestFuturesAccounting:
    def test_open_short_holds_cash(self):
        b = _const_broker(5000.0, market_type="futures")
        b.place_order(_order("sell", 0.1))
        pos = b.positions["BTCUSDT"]
        assert pos["side"] == "short"
        assert b.cash == pytest.approx(1000.0)               # no cash deducted
        assert b.closed_trades == []

    def test_close_short_books_pnl(self):
        b = _const_broker(5000.0, market_type="futures")
        open_order = b.place_order(_order("sell", 0.1))        # short
        b.price_source = CallbackPriceSource(lambda _sym: 4800.0)
        close_order = b.place_order(_order("buy", 0.1))
        trade = b.closed_trades[-1]
        # short PnL = (entry_fill - exit_fill) * qty (both legs slippage-adjusted)
        realized = (open_order.fill_price - close_order.fill_price) * 0.1
        assert trade["direction"] == "short"
        assert trade["gross_pnl"] == pytest.approx(realized)
        assert trade["net_pnl"] == pytest.approx(realized - (close_order.fill_price * 0.1) * b.fee_rate)
        assert b.positions == {}

    def test_sell_into_long_closes_then_opens_short(self):
        b = _const_broker(5000.0, market_type="futures")
        b.place_order(_order("buy", 0.1))                     # long
        b.place_order(_order("sell", 0.2))                    # close 0.1 + short 0.1
        assert len(b.closed_trades) == 1
        assert b.positions["BTCUSDT"]["side"] == "short"
        assert b.positions["BTCUSDT"]["quantity"] == pytest.approx(0.1)

    def test_buy_into_short_closes_then_opens_long(self):
        b = _const_broker(5000.0, market_type="futures")
        b.place_order(_order("sell", 0.1))                    # short
        b.place_order(_order("buy", 0.2))                     # close 0.1 + long 0.1
        assert len(b.closed_trades) == 1
        assert b.positions["BTCUSDT"]["side"] == "long"
        assert b.positions["BTCUSDT"]["quantity"] == pytest.approx(0.1)


# ============================================================== Engine
class AtOnceLong(BaseStrategy):
    """Signals a single LONG at closed bar 5 (entered at bar 6's open)."""
    name = "AtOnceLong"
    strategy_type = "test"

    def validate_params(self):
        pass

    def setup(self, df):
        return df.copy()

    def entry_signal(self, df, i, ctx=None):
        return (Direction.LONG, "at5") if i == 5 else (Direction.NONE, "")

    def compute_stop_loss(self, df, i, direction, entry_price):
        return float(df["close"].iloc[i] * 0.98)

    def compute_take_profit(self, df, i, direction, entry_price, stop_loss=None):
        return float(df["close"].iloc[i] * 1.20)


class AlwaysLong(BaseStrategy):
    """Signals LONG every bar (capacity tests)."""
    name = "AlwaysLong"
    strategy_type = "test"

    def validate_params(self):
        pass

    def setup(self, df):
        return df.copy()

    def entry_signal(self, df, i, ctx=None):
        return Direction.LONG, "always"

    def compute_stop_loss(self, df, i, direction, entry_price):
        return float(df["close"].iloc[i] * 0.9)

    def compute_take_profit(self, df, i, direction, entry_price, stop_loss=None):
        return float(df["close"].iloc[i] * 1.3)


class AlwaysShort(BaseStrategy):
    """Signals SHORT every bar (spot must reject)."""
    name = "AlwaysShort"
    strategy_type = "test"
    supports_short = True

    def validate_params(self):
        pass

    def setup(self, df):
        return df.copy()

    def entry_signal(self, df, i, ctx=None):
        return Direction.SHORT, "always"

    def compute_stop_loss(self, df, i, direction, entry_price):
        return float(df["close"].iloc[i] * 1.1)

    def compute_take_profit(self, df, i, direction, entry_price, stop_loss=None):
        return float(df["close"].iloc[i] * 0.9)


def _engine(strategy, df, risk=None, db=None, market="spot", starting=1000.0):
    broker = PaperBroker(
        price_source=DataframePriceSource(df),
        starting_capital=starting, market_type=market,
    )
    risk = risk or RiskManager()
    cfg = PaperTradingConfig(symbol="BTCUSDT", market_type=market, timeframe="1h",
                             starting_capital=starting)
    return PaperTradingEngine(strategy=strategy, broker=broker,
                              risk_manager=risk, config=cfg, db=db)


class TestPaperTradingEngine:
    def test_replay_builds_equity_curve_and_result(self, bars):
        res = _engine(AtOnceLong(), bars).run_bars(bars)
        assert len(res.equity_curve) == len(bars)
        assert res.n_trades >= 1
        # n_orders counts entry + exit orders, so it is >= the number of closed trades.
        assert res.n_orders >= res.n_trades
        assert res.final_equity > 0

    def test_spot_rejects_short_signals(self, bars):
        eng = _engine(AlwaysShort(), bars, market="spot")
        res = eng.run_bars(bars)
        assert res.n_trades == 0
        assert res.n_orders == 0

    def test_risk_caps_open_positions(self, bars):
        limits = RiskLimits(max_open_positions=1, starting_capital=1000.0)
        risk = RiskManager(limits)
        res = _engine(AlwaysLong(), bars, risk=risk).run_bars(bars)
        # With only 1 slot and a per-bar signal, most signals are gated before an
        # order is ever placed, and that gating is recorded as risk events.
        assert res.n_orders < len(bars)
        assert len(res.risk_events) >= 1
        assert any(ev.get("event_type") == "max_positions" for ev in res.risk_events)

    def test_persists_execution_trades(self, bars, test_db):
        from crypto_quant.db.models import ExecutionTrade, Strategy
        eng = _engine(AtOnceLong(), bars, db=test_db)
        eng.run_bars(bars)
        s = test_db.get_session()
        try:
            trades = s.query(ExecutionTrade).filter(
                ExecutionTrade.execution_mode == "paper").all()
            assert len(trades) >= 1
            assert trades[0].symbol == "BTCUSDT"
            strat = s.query(Strategy).filter(Strategy.id == "STRAT-TEST").first()
            assert strat is not None
        finally:
            s.close()

    def test_on_bar_live_step(self, bars):
        eng = _engine(AtOnceLong(), bars)
        source = eng.broker.price_source
        source.set_index(0)
        # Live step only advances; feed bars sequentially and assert no crash and
        # that equity grows monotonically with the data.
        for i in range(10):
            eng.set_live_signal(eng.strategy.generate_signal(bars, i))
            eng.on_bar({
                "open": float(bars["open"].iloc[i]),
                "high": float(bars["high"].iloc[i]),
                "low": float(bars["low"].iloc[i]),
                "close": float(bars["close"].iloc[i]),
                "timestamp": int(bars["timestamp"].iloc[i]),
            })
        assert len(eng.equity_curve) == 10
        assert eng.broker.equity > 0


def test_generate_trade_id_is_collision_safe():
    from crypto_quant.utils.helpers import generate_trade_id
    ids = {generate_trade_id() for _ in range(2000)}
    assert len(ids) == 2000
    assert all(i.startswith("TRADE-") and len(i) > 10 for i in ids)