"""Unit and causality tests for Strategy Family #6 (TrendChannelBreakoutStrategy)."""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.strategies import create_strategy, get_strategy_class
from crypto_quant.strategies.base import Direction
from crypto_quant.strategies.trend_channel_breakout import TrendChannelBreakoutStrategy
from crypto_quant.backtesting import BacktestConfig, BacktestEngine, ExecutionConfig


def _generate_synthetic_ohlcv(n_bars: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic trending and oscillating OHLCV data."""
    np.random.seed(seed)
    start_ts = 1609459200000  # 2021-01-01 00:00:00 UTC
    interval_ms = 3600000     # 1h

    timestamps = [start_ts + i * interval_ms for i in range(n_bars)]

    # Generate price with strong trend segments
    returns = np.random.normal(0.001, 0.015, n_bars)
    prices = 100.0 * np.exp(np.cumsum(returns))

    highs = prices * (1.0 + np.abs(np.random.normal(0.005, 0.003, n_bars)))
    lows = prices * (1.0 - np.abs(np.random.normal(0.005, 0.003, n_bars)))
    opens = (highs + lows) / 2.0 + np.random.normal(0, 0.2, n_bars)
    closes = prices
    volumes = np.random.uniform(100.0, 1000.0, n_bars)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def test_registry_integration():
    """Verify strategy is properly registered in factory and registry."""
    cls = get_strategy_class("trend_channel_breakout")
    assert cls is TrendChannelBreakoutStrategy

    strat = create_strategy("trend_channel_breakout")
    assert isinstance(strat, TrendChannelBreakoutStrategy)
    assert strat.strategy_type == "trend_channel_breakout"
    assert strat.name == "Trend Channel Breakout"
    assert strat.param_grid() is not None


def test_param_validation():
    """Verify parameter validation bounds."""
    # Valid
    strat = TrendChannelBreakoutStrategy({"channel_period": 20, "adx_threshold": 20.0})
    assert strat.params["channel_period"] == 20

    # Invalid channel_period
    with pytest.raises(ValueError, match="channel_period must be > 0"):
        TrendChannelBreakoutStrategy({"channel_period": 0}).validate_params()

    # Invalid adx_threshold
    with pytest.raises(ValueError, match="adx_threshold must be >= 0"):
        TrendChannelBreakoutStrategy({"adx_threshold": -5.0}).validate_params()


def test_causality_and_no_lookahead():
    """Verify signals on bar i depend STRICTLY on closed bars up to i-1."""
    df = _generate_synthetic_ohlcv(200)
    strat = create_strategy("trend_channel_breakout")
    prepared = strat.setup(df)

    # Check setup columns exist
    assert "_long_setup" in prepared.columns
    assert "_short_setup" in prepared.columns
    assert "dc_high" in prepared.columns
    assert "ema_trend" in prepared.columns
    assert "adx_val" in prepared.columns

    # Causality perturbation test:
    # Modifying future bars (i+1..end) must NOT change setup or signal at bar i
    target_bar = 80
    signal_orig, _ = strat.entry_signal(prepared, target_bar)

    df_perturbed = df.copy()
    # Mutate all bars after target_bar
    df_perturbed.loc[target_bar + 1 :, ["open", "high", "low", "close", "volume"]] *= 3.0

    prepared_perturbed = strat.setup(df_perturbed)
    signal_perturbed, _ = strat.entry_signal(prepared_perturbed, target_bar)

    assert signal_orig == signal_perturbed
    assert prepared.loc[target_bar, "_long_setup"] == prepared_perturbed.loc[target_bar, "_long_setup"]
    assert prepared.loc[target_bar, "_short_setup"] == prepared_perturbed.loc[target_bar, "_short_setup"]


def test_engine_backtest_execution():
    """Verify BacktestEngine runs cleanly with TrendChannelBreakoutStrategy."""
    df = _generate_synthetic_ohlcv(300)
    strat = create_strategy("trend_channel_breakout")

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
    res = engine.run(strat, df, symbol="TESTUSDT")

    assert res is not None
    assert res.metrics is not None
    assert isinstance(res.trades, list)
    assert len(res.equity_curve) >= len(df)
