"""Unit tests for Volatility Squeeze Breakout Strategy (Strategy Family #5)."""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.strategies import create_strategy, Direction
from crypto_quant.strategies.volatility_squeeze_breakout import VolatilitySqueezeBreakoutStrategy
from crypto_quant.backtesting import BacktestConfig, BacktestEngine, ExecutionConfig


def _generate_synthetic_ohlcv(n_bars: int = 200, trend: float = 0.0) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.default_rng(42)
    start_ts = 1640995200000  # 2022-01-01 00:00:00 UTC
    timestamps = [start_ts + i * 3600000 for i in range(n_bars)]

    close = 100.0 + np.cumsum(rng.normal(trend, 1.0, size=n_bars))
    high = close + rng.uniform(0.5, 2.0, size=n_bars)
    low = close - rng.uniform(0.5, 2.0, size=n_bars)
    open_p = close + rng.normal(0, 0.5, size=n_bars)
    volume = rng.uniform(100.0, 1000.0, size=n_bars)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_strategy_instantiation():
    """Verify default strategy instantiation and registration."""
    strat = create_strategy("volatility_squeeze_breakout")
    assert isinstance(strat, VolatilitySqueezeBreakoutStrategy)
    assert strat.name == "Volatility Squeeze Breakout"
    assert strat.strategy_type == "volatility_squeeze_breakout"
    assert strat.supports_futures is True
    assert strat.supports_short is True


def test_param_validation():
    """Verify parameter validation bounds."""
    with pytest.raises(ValueError):
        create_strategy("volatility_squeeze_breakout", params={"bb_std": -1.0})

    with pytest.raises(ValueError):
        create_strategy("volatility_squeeze_breakout", params={"kc_mult": 0.0})


def test_causal_setup_and_signals():
    """Verify setup produces causal shifted indicator columns and valid signals."""
    df = _generate_synthetic_ohlcv(200)
    strat = create_strategy("volatility_squeeze_breakout")
    prepared = strat.setup(df)

    assert "bb_upper" in prepared.columns
    assert "kc_upper" in prepared.columns
    assert "mom" in prepared.columns
    assert "vol_sma" in prepared.columns
    assert "_long_setup" in prepared.columns
    assert "_short_setup" in prepared.columns

    # Check signal generation
    for i in range(len(prepared)):
        sig = strat.generate_signal(prepared, i)
        if i < 30:
            assert sig.direction == Direction.NONE
        if sig.direction != Direction.NONE:
            assert sig.stop_loss is not None
            assert sig.take_profit is not None


def test_backtest_execution_integration():
    """Verify full BacktestEngine execution with VolatilitySqueezeBreakoutStrategy."""
    df = _generate_synthetic_ohlcv(300, trend=0.2)
    strat = create_strategy("volatility_squeeze_breakout")
    cfg = BacktestConfig(
        initial_capital=1000.0,
        risk_per_trade=0.01,
        max_open_positions=3,
        max_leverage=5.0,
        market_type="futures",
        timeframe="1h",
        execution=ExecutionConfig(market_type="futures"),
    )
    engine = BacktestEngine(cfg)
    res = engine.run(strat, df, symbol="ETHUSDT")

    assert res.metrics.final_equity > 0.0
    assert len(res.equity_curve) >= len(df)