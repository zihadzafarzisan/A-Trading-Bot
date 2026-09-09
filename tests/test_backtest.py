"""Backtesting engine correctness tests.

Uses synthetic data where the correct answer is known exactly. These tests are
deliberately strict: entry/exit prices, fees, position sizing, and PnL are
verified against hand-computed values.
"""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.backtesting import (
    BacktestEngine, BacktestConfig, ExecutionConfig, ExecutionModel,
    Portfolio, Position, EquityPoint,
)
from crypto_quant.backtesting.metrics import MetricsCalculator
from crypto_quant.strategies.base import BaseStrategy, Direction


# ---------------------------------------------------------------------------
# Deterministic test strategies
# ---------------------------------------------------------------------------
class OneShotLong(BaseStrategy):
    """Signals a single LONG entry at bar 0 only."""

    name = "OneShotLong"
    strategy_type = "test"

    def validate_params(self):
        pass

    def setup(self, df):
        return df.copy()

    def entry_signal(self, df, i, ctx=None):
        if i == 0:
            return Direction.LONG, "test"
        return Direction.NONE, ""

    def compute_stop_loss(self, df, i, direction, entry_price):
        return float(df["close"].iloc[i] * 0.98)  # 2% stop from signal close

    def compute_take_profit(self, df, i, direction, entry_price, stop_loss=None):
        # R:R 2 on the 2% stop
        risk = abs(entry_price - self.compute_stop_loss(df, i, direction, entry_price))
        return entry_price + 2 * risk if direction == Direction.LONG else entry_price - 2 * risk


class AlwaysLong(BaseStrategy):
    """Signals LONG every bar (for capacity/market-type tests)."""

    name = "AlwaysLong"
    strategy_type = "test"

    def validate_params(self):
        pass

    def setup(self, df):
        return df.copy()

    def entry_signal(self, df, i, ctx=None):
        return Direction.LONG, "always"

    def compute_stop_loss(self, df, i, direction, entry_price):
        return float(df["close"].iloc[i] * 0.95)


class AlwaysShort(BaseStrategy):
    """Signals SHORT every bar (for futures short tests)."""

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
        return float(df["close"].iloc[i] * 1.05)


def build_path(closes):
    """Build OHLCV from closes (open = prev close)."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    opens = np.empty(n)
    opens[0] = closes[0]
    opens[1:] = closes[:-1]
    return pd.DataFrame({
        "timestamp": [1609459200000 + i * 3_600_000 for i in range(n)],
        "open": opens,
        "high": np.maximum(opens, closes) + 0.5,
        "low": np.minimum(opens, closes) - 0.5,
        "close": closes,
        "volume": 1000.0,
    })


class TestExecutionModel:
    """Exact-value unit tests for the execution model."""

    def test_entry_fill_long_has_slippage(self):
        model = ExecutionModel(ExecutionConfig(slippage=0.0005))
        assert model.entry_fill(100, "long", 100) == pytest.approx(100.05)
        assert model.entry_fill(100, "short", 100) == pytest.approx(99.95)

    def test_entry_fill_zero_slippage(self):
        model = ExecutionModel(ExecutionConfig(slippage=0.0))
        assert model.entry_fill(100, "long", 100) == pytest.approx(100.0)

    def test_fee_cost_spot(self):
        model = ExecutionModel(ExecutionConfig(market_type="spot", commission=0.001))
        assert model.fee_cost(500) == pytest.approx(0.5)

    def test_fee_cost_futures_taker(self):
        model = ExecutionModel(ExecutionConfig(market_type="futures", taker_fee=0.0004, use_taker=True))
        assert model.fee_cost(1000) == pytest.approx(0.4)

    def test_funding_only_futures(self):
        spot = ExecutionModel(ExecutionConfig(market_type="spot"))
        assert spot.funding_cost(1000, 24, bars_per_funding=8) == 0.0
        fut = ExecutionModel(ExecutionConfig(market_type="futures", funding_rate=0.0001))
        # 24h / 8h = 3 payments on notional 1000
        assert fut.funding_cost(1000, 24, bars_per_funding=8) == pytest.approx(0.3)

    def test_liquidation_price_long(self):
        model = ExecutionModel()
        liq = model.liquidation_price(100, "long", leverage=2, maintenance_margin_pct=0.005)
        assert liq == pytest.approx(100 * (1 - (0.5 - 0.005)))

    def test_liquidation_price_short(self):
        model = ExecutionModel()
        liq = model.liquidation_price(100, "short", leverage=5, maintenance_margin_pct=0.005)
        assert liq == pytest.approx(100 * (1 + (0.2 - 0.005)))


class TestPortfolioAccounting:
    """Exact-value accounting tests."""

    def test_spot_long_open_close(self):
        p = Portfolio(1000)
        pos = p.open_position(
            symbol="BTCUSDT", direction="long", quantity=5.0, entry_price=100.0,
            entry_time=0, entry_bar=0, stop_loss=98, take_profit=104,
            leverage=1, market_type="spot", fee=0.5,
        )
        # Cash deducted: notional + fee
        assert p.cash == pytest.approx(1000 - 500 - 0.5)

        p.mark_to_market(102)
        assert p.unrealized == pytest.approx((102 - 100) * 5.0)
        assert p.equity == pytest.approx(1000 - 500 - 0.5 + 10)

        p.close_position(pos, 104, exit_time=100, exit_bar=1, reason="tp",
                         exit_fee=0.52, funding=0.0)
        # Cash: initial - entry(500+0.5) + exit proceeds(104*5 - 0.52)
        assert p.cash == pytest.approx(1000 - 500 - 0.5 + 520 - 0.52)
        assert pos.net_pnl(104) == pytest.approx(20 - 0.5 - 0.52)
        assert pos.return_pct == pytest.approx((20 - 1.02) / 500)

    def test_futures_long_margin(self):
        p = Portfolio(1000)
        pos = p.open_position(
            symbol="BTCUSDT", direction="long", quantity=5.0, entry_price=100.0,
            entry_time=0, entry_bar=0, stop_loss=98, take_profit=104,
            leverage=5, market_type="futures", fee=0.2,
        )
        # Margin used = notional/leverage = 500/5 = 100
        assert pos.margin_used == pytest.approx(100)
        assert p.cash == pytest.approx(1000 - 100 - 0.2)

        p.close_position(pos, 104, exit_time=100, exit_bar=1, reason="tp",
                         exit_fee=0.2, funding=0.1)
        gross = (104 - 100) * 5.0
        assert pos.net_pnl(104) == pytest.approx(gross - 0.2 - 0.2 - 0.1)
        # Cash: initial - margin(100) - fee(0.2) + margin + gross - exit_fee - funding
        assert p.cash == pytest.approx(1000 + gross - 0.4 - 0.1)

    def test_short_pnl(self):
        p = Portfolio(1000)
        pos = p.open_position(
            symbol="BTCUSDT", direction="short", quantity=2.0, entry_price=100.0,
            entry_time=0, entry_bar=0, stop_loss=105, take_profit=90,
            leverage=1, market_type="spot", fee=0.2,
        )
        p.close_position(pos, 90, exit_time=10, exit_bar=1, reason="tp",
                         exit_fee=0.18, funding=0.0)
        assert pos.gross_pnl(90) == pytest.approx(20.0)
        assert pos.net_pnl(90) == pytest.approx(20 - 0.2 - 0.18)

    def test_equity_snapshot(self):
        p = Portfolio(1000)
        p.record_snapshot(0)
        assert p.equity_curve[0].equity == pytest.approx(1000)


class TestBacktestEngine:
    """End-to-end engine correctness on known data."""

    def make_engine(self, **overrides):
        config = BacktestConfig(
            initial_capital=1000.0, risk_per_trade=0.01,
            max_open_positions=3, max_position_pct=0.5,
            market_type="spot", timeframe="1h",
            execution=ExecutionConfig(market_type="spot", slippage=0.0, commission=0.001),
            min_bars=6,
        )
        for k, v in overrides.items():
            setattr(config, k, v)
        return BacktestEngine(config)

    def test_take_profit_trade_exact(self):
        """A known rising path: entry at 100, TP at 104, verify exact PnL."""
        # closes: flat 100 x5 then rising 1/bar
        closes = [100.0] * 5 + [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
        df = build_path(closes)
        engine = self.make_engine()
        result = engine.run(OneShotLong(), df)

        trades = result.trades
        assert len(trades) == 1
        t = trades[0]
        assert t["direction"] == "long"
        assert t["exit_reason"] == "tp"
        # Entry at bar 1 open = close[0] = 100 (slippage 0)
        assert t["entry_price"] == pytest.approx(100.0)
        # Stop = 98, tp = entry + 2*(entry-stop) = 100 + 4 = 104
        assert t["stop_loss"] == pytest.approx(98.0)
        assert t["take_profit"] == pytest.approx(104.0)
        assert t["exit_price"] == pytest.approx(104.0)
        # Position size: risk 10 / 2% = 500 notional -> qty 5
        assert t["quantity"] == pytest.approx(5.0)
        # Gross = (104-100)*5 = 20; fees = 500*0.001 + 104*5*0.001 = 1.02
        assert t["gross_pnl"] == pytest.approx(20.0)
        assert t["fees"] == pytest.approx(1.02)
        assert t["net_pnl"] == pytest.approx(18.98)
        assert t["return_pct"] == pytest.approx(18.98 / 500)

    def test_stop_loss_trade(self):
        """A known falling path triggers the stop; PnL is bounded and negative."""
        closes = [100.0] * 5 + [99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0]
        df = build_path(closes)
        engine = self.make_engine()
        result = engine.run(OneShotLong(), df)

        trades = result.trades
        assert len(trades) == 1
        t = trades[0]
        assert t["exit_reason"] == "sl"
        assert t["stop_loss"] == pytest.approx(98.0)
        # Stop exit fill at stop level (no gap below on the stop bar since
        # open ~99 > 98; fill = min(open, 98) = 98)
        assert t["exit_price"] == pytest.approx(98.0)
        assert t["net_pnl"] < 0
        # Max loss bounded by stop distance + fees
        assert t["net_pnl"] > -15

    def test_spot_cannot_short(self):
        """Spot market must ignore short signals."""
        closes = [100.0] * 20
        df = build_path(closes)
        engine = self.make_engine()
        result = engine.run(AlwaysShort(), df)
        assert len(result.trades) == 0

    def test_futures_allows_short(self):
        """Futures market can short."""
        closes = [100.0] * 20
        df = build_path(closes)
        engine = self.make_engine(
            market_type="futures",
            execution=ExecutionConfig(market_type="futures", slippage=0.0, taker_fee=0.0004),
        )
        result = engine.run(AlwaysShort(), df)
        assert len(result.trades) >= 1
        assert all(t["direction"] == "short" for t in result.trades)

    def test_futures_funding_scaled_by_bars(self):
        """Funding must scale with holding bars, not milliseconds.

        Regression: holding-time was passed in ms, producing absurd funding
        (billions of % per hour). A flat market with 30 bars should yield small,
        bounded funding and sane equity.
        """
        closes = [100.0] * 40
        df = build_path(closes)
        engine = self.make_engine(
            market_type="futures",
            execution=ExecutionConfig(
                market_type="futures", slippage=0.0, taker_fee=0.0004,
                funding_rate=0.0001,
            ),
        )
        result = engine.run(AlwaysShort(), df)
        assert len(result.trades) >= 1
        funding = result.metrics.trades.total_funding
        # 30+ bars at 8h funding intervals => < ~6 payments per trade; even with
        # several trades, total funding must be small relative to capital.
        assert funding < 10.0, f"Funding exploded: ${funding}"
        assert result.metrics.final_equity > 0
        assert abs(result.metrics.net_return) < 1.0

    def test_max_open_positions_enforced(self):
        """Never more than max_open_positions concurrent positions."""
        closes = [100.0] * 50
        df = build_path(closes)
        engine = self.make_engine(max_open_positions=3)
        result = engine.run(AlwaysLong(), df)
        # Concurrency check on the equity curve
        assert max(p["n_positions"] for p in result.equity_curve) <= 3

    def test_entry_uses_next_bar_open(self):
        """No look-ahead: signal at bar 0 must fill at bar 1's open."""
        closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        df = build_path(closes)
        engine = self.make_engine()
        result = engine.run(OneShotLong(), df)
        t = result.trades[0]
        # entry_time should equal timestamp of bar 1
        assert t["entry_time"] == df["timestamp"].iloc[1]

    def test_no_trades_flat_no_stop(self):
        """A strategy that never produces a valid stop produces no trades."""
        class NoStop(BaseStrategy):
            name = "NoStop"
            strategy_type = "test"
            def validate_params(self): pass
            def setup(self, df): return df.copy()
            def entry_signal(self, df, i, ctx=None):
                return Direction.LONG, "x"
            def compute_stop_loss(self, df, i, direction, entry_price):
                return None  # no stop -> cannot size

        closes = [100.0] * 30
        df = build_path(closes)
        engine = self.make_engine()
        result = engine.run(NoStop(), df)
        assert len(result.trades) == 0


class TestMetrics:
    """Test the metrics calculator with known trades."""

    def _make_trade(self, pnl, entry=100, qty=1, direction="long"):
        pos = Position(
            symbol="BTCUSDT", direction=direction, market_type="spot",
            quantity=qty, entry_price=entry, entry_time=0, entry_bar=0,
            stop_loss=90, take_profit=110, leverage=1,
        )
        exit_price = entry + pnl / qty
        pos.exit_price = exit_price
        pos.exit_time = 100
        pos.exit_bar = 1
        pos.exit_reason = "tp" if pnl > 0 else "sl"
        pos.entry_fee = 0.1
        pos.exit_fee = 0.1
        pos.funding = 0.0
        return pos

    def test_metrics_computed(self):
        trades = [self._make_trade(10), self._make_trade(10), self._make_trade(-5)]
        equity = [EquityPoint(t, e, e, 0, 0) for t, e in zip(range(3), [1000, 1015, 1015])]
        calc = MetricsCalculator(timeframe="1h")
        m = calc.compute(trades, equity, 1000)

        assert m.trades.total_trades == 3
        assert m.trades.winning_trades == 2
        assert m.trades.losing_trades == 1
        assert m.trades.win_rate == pytest.approx(2 / 3)
        # Net PnL: two +10 winners each minus 0.2 fees => 19.6 profit
        assert m.trades.total_profit == pytest.approx(19.6)
        # One -5 loser minus 0.2 fees => 5.2 loss
        assert m.trades.total_loss == pytest.approx(5.2)
        assert m.trades.profit_factor == pytest.approx(19.6 / 5.2)
        assert m.net_pnl == pytest.approx(15)
        assert m.final_equity == pytest.approx(1015)

    def test_empty_trades(self):
        calc = MetricsCalculator(timeframe="1h")
        m = calc.compute([], [], 1000)
        assert m.trades.total_trades == 0
        assert m.net_pnl == 0
        assert m.max_drawdown == 0

    def test_max_drawdown(self):
        equity = [EquityPoint(t, e, e, 0, 0) for t, e in zip(range(4), [1000, 1200, 900, 1100])]
        calc = MetricsCalculator()
        m = calc.compute([], equity, 1000)
        # peak 1200, trough 900 -> 25%
        assert m.max_drawdown == pytest.approx(0.25)

    def test_sharpe_positive_for_uptrend(self):
        equity = [EquityPoint(t, 1000 + t, 1000 + t, 0, 0) for t in range(100)]
        calc = MetricsCalculator(timeframe="1h")
        m = calc.compute([], equity, 1000)
        assert m.sharpe_ratio > 0