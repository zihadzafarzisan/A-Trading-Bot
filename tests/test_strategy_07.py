"""Unit and causality tests for Strategy Family #7 (VolumeDisplacementContinuationStrategy)."""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.strategies import create_strategy, get_strategy_class
from crypto_quant.strategies.volume_displacement_continuation import VolumeDisplacementContinuationStrategy
from crypto_quant.backtesting import BacktestConfig, BacktestEngine, ExecutionConfig


def _generate_synthetic_ohlcv(n_bars: int = 250, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with volume surges and price displacements."""
    np.random.seed(seed)
    start_ts = 1609459200000  # 2021-01-01 00:00:00 UTC
    interval_ms = 3600000     # 1h

    timestamps = [start_ts + i * interval_ms for i in range(n_bars)]

    returns = np.random.normal(0.001, 0.015, n_bars)
    prices = 100.0 * np.exp(np.cumsum(returns))

    highs = prices * (1.0 + np.abs(np.random.normal(0.005, 0.003, n_bars)))
    lows = prices * (1.0 - np.abs(np.random.normal(0.005, 0.003, n_bars)))
    opens = (highs + lows) / 2.0 + np.random.normal(0, 0.2, n_bars)
    closes = prices
    volumes = np.random.uniform(100.0, 1000.0, n_bars)

    # Inject large displacement candles
    for idx in [50, 100, 150]:
        if idx < n_bars:
            opens[idx] = prices[idx - 1]
            closes[idx] = prices[idx - 1] * 1.06
            highs[idx] = closes[idx] * 1.01
            lows[idx] = opens[idx] * 0.99
            volumes[idx] = 5000.0

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def test_registry_integration():
    """Verify Strategy #7 is registered in factory and registry."""
    cls = get_strategy_class("volume_displacement_continuation")
    assert cls is VolumeDisplacementContinuationStrategy

    strat = create_strategy("volume_displacement_continuation")
    assert isinstance(strat, VolumeDisplacementContinuationStrategy)
    assert strat.strategy_type == "volume_displacement_continuation"
    assert strat.name == "Volume Displacement Continuation"
    assert strat.param_grid() is not None


def test_param_validation():
    """Verify parameter validation bounds."""
    # Valid
    strat = VolumeDisplacementContinuationStrategy({"body_mult": 1.5, "volume_mult": 1.5})
    assert strat.params["body_mult"] == 1.5

    # Invalid body_mult
    with pytest.raises(ValueError, match="body_mult must be > 0"):
        VolumeDisplacementContinuationStrategy({"body_mult": -1.0}).validate_params()

    # Invalid volume_mult
    with pytest.raises(ValueError, match="volume_mult must be > 0"):
        VolumeDisplacementContinuationStrategy({"volume_mult": 0.0}).validate_params()


def test_causality_and_no_lookahead():
    """Verify signals on bar i depend STRICTLY on closed bars up to i-1."""
    df = _generate_synthetic_ohlcv(200)
    strat = create_strategy("volume_displacement_continuation")
    prepared = strat.setup(df)

    assert "_long_setup" in prepared.columns
    assert "_short_setup" in prepared.columns
    assert "atr_14" in prepared.columns
    assert "vol_sma" in prepared.columns
    assert "ema_50" in prepared.columns

    # Causality perturbation test:
    # Modifying future bars (i+1..end) must NOT change setup or signal at bar i
    target_bar = 90
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
    """Verify BacktestEngine runs cleanly with VolumeDisplacementContinuationStrategy."""
    df = _generate_synthetic_ohlcv(300)
    strat = create_strategy("volume_displacement_continuation")

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
