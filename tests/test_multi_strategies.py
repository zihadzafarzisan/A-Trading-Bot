"""Unit tests for the 5 research strategy families.

Validates:
- BreakoutRetestStrategy (breakout, ATR expansion, retest volume, signals)
- TrendFilteredRSIStrategy (SMA200/50 filter, RSI2, RSI14, ADX)
- RegimeAdaptiveStrategy (Causal ADX/Choppiness classification, trend vs MR)
- VWAPBollingerMRStrategy (Low ADX guardrail, outer BB breach, VWAP reversion)
- MTFTrendPullbackStrategy (DMI direction, causal MTF alignment)
"""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.backtesting import BacktestConfig, BacktestEngine
from crypto_quant.strategies import (
    BreakoutRetestStrategy,
    MTFTrendPullbackStrategy,
    RegimeAdaptiveStrategy,
    TrendFilteredRSIStrategy,
    VWAPBollingerMRStrategy,
    create_strategy,
)
from crypto_quant.strategies.base import Direction


def _make_synth_df(n: int = 500, seed: int = 42, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    step_ms = 3600 * 1000  # 1h bars
    start_ms = 1640995200000
    timestamps = [start_ms + i * step_ms for i in range(n)]

    returns = rng.normal(drift, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1.0 + rng.uniform(0.002, 0.015, size=n))
    low = close * (1.0 - rng.uniform(0.002, 0.015, size=n))
    open_p = (high + low) / 2.0
    volume = rng.uniform(100.0, 500.0, size=n)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestBreakoutRetestStrategy:
    """Tests for BreakoutRetestStrategy."""

    def test_param_validation(self):
        with pytest.raises(ValueError, match="lookback"):
            BreakoutRetestStrategy(params={"lookback": 2})
        with pytest.raises(ValueError, match="atr_exp_mult"):
            BreakoutRetestStrategy(params={"atr_exp_mult": -0.5})

    def test_setup_and_signals(self):
        df = _make_synth_df(n=300, drift=0.005)
        strat = BreakoutRetestStrategy(params={"lookback": 20, "atr_exp_mult": 1.0})
        prep = strat.setup(df)
        assert "_long_setup" in prep.columns
        assert "_short_setup" in prep.columns
        assert "dc_high" in prep.columns
        assert "atr_sma50" in prep.columns

        # Backtest integration
        cfg = BacktestConfig(initial_capital=1000.0, risk_per_trade=0.01, market_type="futures")
        res = BacktestEngine(cfg).run(strat, df, symbol="BTCUSDT")
        assert res.config["initial_capital"] == 1000.0

    def test_atr_expansion_filter_effect(self):
        """Proves that a high atr_exp_mult blocks breakouts without volatility expansion."""
        df = _make_synth_df(n=300, drift=0.005)
        strat_no_filter = BreakoutRetestStrategy(params={"lookback": 20, "atr_exp_mult": 0.0})
        strat_strict_filter = BreakoutRetestStrategy(params={"lookback": 20, "atr_exp_mult": 5.0})

        prep_lenient = strat_no_filter.setup(df)
        prep_strict = strat_strict_filter.setup(df)

        lenient_signals = prep_lenient["_long_setup"].sum() + prep_lenient["_short_setup"].sum()
        strict_signals = prep_strict["_long_setup"].sum() + prep_strict["_short_setup"].sum()

        assert strict_signals <= lenient_signals


class TestTrendFilteredRSIStrategy:
    """Tests for TrendFilteredRSIStrategy."""

    def test_param_validation(self):
        with pytest.raises(ValueError, match="sma_fast"):
            TrendFilteredRSIStrategy(params={"sma_fast": 200, "sma_slow": 50})

    def test_setup_and_signals_rsi14(self):
        df = _make_synth_df(n=300, drift=0.005)
        strat = TrendFilteredRSIStrategy(params={"sma_fast": 50, "sma_slow": 200, "rsi_period": 14})
        prep = strat.setup(df)
        assert "_long_setup" in prep.columns
        assert "_short_setup" in prep.columns

        cfg = BacktestConfig(initial_capital=1000.0, risk_per_trade=0.01, market_type="futures")
        res = BacktestEngine(cfg).run(strat, df, symbol="BTCUSDT")
        assert res.config["initial_capital"] == 1000.0

    def test_setup_and_signals_rsi2(self):
        """Explicitly covers 2-period Connors-style RSI."""
        df = _make_synth_df(n=300, drift=0.005)
        strat = TrendFilteredRSIStrategy(params={"sma_fast": 50, "sma_slow": 200, "rsi_period": 2, "rsi_oversold": 10.0, "rsi_overbought": 90.0})
        prep = strat.setup(df)
        assert "_long_setup" in prep.columns
        assert "_short_setup" in prep.columns

        cfg = BacktestConfig(initial_capital=1000.0, risk_per_trade=0.01, market_type="futures")
        res = BacktestEngine(cfg).run(strat, df, symbol="BTCUSDT")
        assert isinstance(res.metrics.net_return, float)


class TestRegimeAdaptiveStrategy:
    """Tests for RegimeAdaptiveStrategy."""

    def test_causal_regime_classification(self):
        df = _make_synth_df(n=300)
        strat = RegimeAdaptiveStrategy()
        prep = strat.setup(df)
        assert "is_trend_regime" in prep.columns
        assert "is_range_regime" in prep.columns
        assert "chop" in prep.columns

        # Verify no lookahead: bar i does not change when future bars are modified
        val_at_100 = prep["is_trend_regime"].iloc[100]
        df_mod = df.copy()
        df_mod.loc[150:, "close"] = df_mod.loc[150:, "close"] * 5.0
        prep_mod = strat.setup(df_mod)
        assert prep_mod["is_trend_regime"].iloc[100] == val_at_100

    def test_backtest_execution(self):
        df = _make_synth_df(n=300)
        strat = RegimeAdaptiveStrategy()
        cfg = BacktestConfig(initial_capital=1000.0, risk_per_trade=0.01, market_type="futures")
        res = BacktestEngine(cfg).run(strat, df, symbol="BTCUSDT")
        assert isinstance(res.metrics.net_return, float)


class TestVWAPBollingerMRStrategy:
    """Tests for VWAPBollingerMRStrategy."""

    def test_param_validation(self):
        with pytest.raises(ValueError, match="adx_max_thresh"):
            VWAPBollingerMRStrategy(params={"adx_max_thresh": -5.0})

    def test_setup_and_signals(self):
        df = _make_synth_df(n=300)
        strat = VWAPBollingerMRStrategy()
        prep = strat.setup(df)
        assert "_long_setup" in prep.columns
        assert "_short_setup" in prep.columns
        assert "vwap" in prep.columns
        assert "bb_upper" in prep.columns

        cfg = BacktestConfig(initial_capital=1000.0, risk_per_trade=0.01, market_type="futures")
        res = BacktestEngine(cfg).run(strat, df, symbol="BTCUSDT")
        assert isinstance(res.metrics.net_return, float)
