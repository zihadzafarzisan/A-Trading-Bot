"""Tests for the strategy framework."""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.strategies import (
    BaseStrategy, Signal, Direction, StopType, StopLossSpec, TakeProfitSpec,
    TrendStrategy, MomentumStrategy, MeanReversionStrategy, BreakoutStrategy,
    create_strategy, get_strategy_class, available_strategy_types,
    strategy_param_grid,
)


def make_df(closes, highs=None, lows=None, volumes=None, step_h=3600000):
    """Build OHLCV DataFrame from closes."""
    closes = closes.astype(float)
    n = len(closes)
    highs = highs if highs is not None else closes + 1.0
    lows = lows if lows is not None else closes - 1.0
    vols = volumes if volumes is not None else np.full(n, 1000.0)
    base = 1609459200000
    return pd.DataFrame({
        "timestamp": [base + i * step_h for i in range(n)],
        "open": closes - 0.1,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": vols,
    })


def uptrend_df(n=300):
    """Monotonic uptrend."""
    closes = np.linspace(100, 300, n)
    return make_df(closes)


def downtrend_df(n=300):
    """Monotonic downtrend."""
    closes = np.linspace(300, 100, n)
    return make_df(closes)


class TestSignal:
    """Test Signal dataclass."""

    def test_inactive_default(self):
        s = Signal()
        assert s.direction == Direction.NONE
        assert s.is_active is False

    def test_active_long(self):
        s = Signal(direction=Direction.LONG, stop_loss=99.0, take_profit=110.0)
        assert s.is_active is True
        assert s.direction_str == "long"


class TestBaseStrategy:
    """Test base strategy mechanics."""

    def test_default_timeframes(self):
        assert "1h" in BaseStrategy.default_timeframes()


class TestTrendStrategy:
    """Test TrendStrategy signals."""

    def test_long_in_uptrend(self):
        strat = TrendStrategy({"ema_fast": 10, "ema_slow": 50})
        df = strat.setup(uptrend_df())
        found = False
        for i in range(60, len(df)):
            direction, reason = strat.entry_signal(df, i)
            if direction == Direction.LONG:
                found = True
                break
        assert found, "Expected a LONG signal in a strong uptrend"

    def test_short_in_downtrend(self):
        strat = TrendStrategy({"ema_fast": 10, "ema_slow": 50})
        df = strat.setup(downtrend_df())
        found = any(
            strat.entry_signal(df, i)[0] == Direction.SHORT
            for i in range(60, len(df))
        )
        assert found, "Expected a SHORT signal in a strong downtrend"

    def test_atr_stop_derived_on_demand(self):
        """ATR stops must work even when setup() adds no atr column."""
        strat = TrendStrategy({"ema_fast": 10, "ema_slow": 50})  # default ATR stop
        df = strat.setup(uptrend_df())
        assert "atr_14" not in df.columns
        for i in range(200, len(df)):
            sig = strat.generate_signal(df, i)
            if sig.is_active:
                assert sig.stop_loss is not None, "ATR stop should be derived on demand"
                assert sig.take_profit is not None
                return
        # fall through: ensure at least one active signal exists in this trend
        found = any(
            strat.generate_signal(df, i).is_active for i in range(200, len(df))
        )
        assert found

    def test_adx_filter_blocks_flat(self):
        strat = TrendStrategy({"ema_fast": 10, "ema_slow": 50, "adx_threshold": 30})
        # Flat/oscillating series -> low ADX -> no signals
        rng = np.random.default_rng(0)
        closes = 100 + rng.normal(0, 1, 300)
        df = strat.setup(make_df(closes))
        signals = sum(
            1 for i in range(60, len(df)) if strat.entry_signal(df, i)[0] != Direction.NONE
        )
        assert signals < 10, "ADX filter should block most signals in a flat market"

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            TrendStrategy({"ema_fast": 50, "ema_slow": 10})  # fast >= slow

    def test_lookahead_shifted(self):
        strat = TrendStrategy({"ema_fast": 10, "ema_slow": 50})
        df = strat.setup(uptrend_df())
        # ema_fast at bar i must equal ema(fast) of closes up to i-1
        fast_raw = df["close"].ewm(span=10, adjust=False, min_periods=10).mean().shift(1)
        pd.testing.assert_series_equal(
            df["ema_fast"].iloc[30:], fast_raw.iloc[30:], check_names=False, check_dtype=False
        )

    def test_generate_signal_has_stop(self):
        strat = TrendStrategy(
            {"ema_fast": 10, "ema_slow": 50},
            stop_spec=StopLossSpec(stop_type=StopType.PERCENT, percent=0.03),
            tp_spec=TakeProfitSpec(mode="rr", risk_reward_ratio=2.0),
        )
        df = strat.setup(uptrend_df())
        found = False
        for i in range(200, len(df)):
            sig = strat.generate_signal(df, i)
            if sig.is_active:
                assert sig.stop_loss is not None
                assert sig.take_profit is not None
                if sig.direction == Direction.LONG:
                    assert sig.stop_loss < df["close"].iloc[i]
                    assert sig.take_profit > df["close"].iloc[i]
                found = True
                break
        assert found, "Expected at least one active signal in uptrend tail"


class TestMomentumStrategy:
    """Test MomentumStrategy signals."""

    def test_oversold_recovery_gives_long(self):
        # A series that dips then recovers -> RSI oversold then rises
        closes = np.concatenate([
            np.linspace(100, 60, 50),
            np.linspace(60, 120, 60),
        ])
        strat = MomentumStrategy({"rsi_period": 14, "rsi_oversold": 40, "rsi_overbought": 60})
        df = strat.setup(make_df(closes))
        found = any(
            strat.entry_signal(df, i)[0] == Direction.LONG
            for i in range(30, len(df))
        )
        assert found

    def test_bounds_validation(self):
        with pytest.raises(ValueError):
            MomentumStrategy({"rsi_oversold": 60, "rsi_overbought": 40})  # oversold > overbought


class TestMeanReversionStrategy:
    """Test MeanReversionStrategy signals."""

    def test_oversold_reversion_gives_long(self):
        # Whipsaw around a stable mean: force close below lower band
        closes = np.linspace(100, 100, 120)
        # add a sharp dip at one point then rebound
        closes = closes.copy()
        for k in range(20, 60):
            closes[k] = 70 - 0.2 * (k - 20)
        strat = MeanReversionStrategy({"bb_period": 20, "bb_std": 2.0, "rsi_extreme": 35})
        df = strat.setup(make_df(closes))
        found = any(
            strat.entry_signal(df, i)[0] == Direction.LONG
            for i in range(20, len(df))
        )
        assert found


class TestBreakoutStrategy:
    """Test BreakoutStrategy signals."""

    def test_breakout_above_range_gives_long(self):
        # Consolidation then a sharp upside break with high volume
        closes = np.concatenate([
            np.full(50, 100.0),
            np.linspace(100, 130, 30),  # breakout
        ])
        volumes = np.concatenate([
            np.full(50, 1000.0),
            np.full(30, 3000.0),  # elevated volume
        ])
        df = make_df(closes, volumes=volumes)
        strat = BreakoutStrategy({"dc_period": 20, "volume_ratio": 1.5})
        dfp = strat.setup(df)
        found = any(
            strat.entry_signal(dfp, i)[0] == Direction.LONG
            for i in range(30, len(dfp))
        )
        assert found

    def test_no_signal_without_volume(self):
        closes = np.concatenate([
            np.full(50, 100.0),
            np.linspace(100, 130, 30),
        ])
        volumes = np.full(80, 1000.0)  # no volume spike
        df = make_df(closes, volumes=volumes)
        strat = BreakoutStrategy({"dc_period": 20, "volume_ratio": 1.5})
        dfp = strat.setup(df)
        signals = sum(
            1 for i in range(30, len(dfp)) if strat.entry_signal(dfp, i)[0] != Direction.NONE
        )
        assert signals == 0, "Breakout without volume confirmation should not signal"


class TestRegistry:
    """Test strategy registry."""

    def test_available_types(self):
        types = available_strategy_types()
        assert "trend" in types
        assert "momentum" in types
        assert "mean_reversion" in types
        assert "breakout" in types

    def test_get_strategy_class(self):
        cls = get_strategy_class("trend")
        assert cls == TrendStrategy

    def test_create_strategy(self):
        strat = create_strategy("momentum", {"rsi_period": 14})
        assert isinstance(strat, MomentumStrategy)

    def test_unknown_type(self):
        with pytest.raises(KeyError):
            create_strategy("nonsense")

    def test_param_grid(self):
        grid = strategy_param_grid("trend")
        assert "ema_fast" in grid
        assert "ema_slow" in grid

    def test_metadata(self):
        strat = create_strategy("trend", {"ema_fast": 10, "ema_slow": 50})
        meta = strat.metadata()
        assert meta["name"] == "Trend Following"
        assert meta["type"] == "trend"
        assert meta["params"]["ema_fast"] == 10