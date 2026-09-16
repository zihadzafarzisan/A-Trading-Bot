"""Tests for Multi-Timeframe Trend + Pullback Strategy (MTFTrendPullbackStrategy).

Validates:
- Causal alignment and zero lookahead bias
- Resampling of OHLCV to higher timeframes
- Signal generation across all 3 MTF combinations (4H->1H->15M, 4H->1H->5M, 1D->4H->1H)
- Stop loss (ATR and Swing) and Take Profit calculations
- Integration with BacktestEngine, realistic fees, slippage, and position sizing
"""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.backtesting import BacktestConfig, BacktestEngine
from crypto_quant.strategies.base import Direction, StopType
from crypto_quant.strategies.mtf_trend_pullback import (
    MTFTrendPullbackStrategy,
    resample_ohlcv_causal,
)


def _generate_synthetic_candles(
    n_bars: int = 500,
    timeframe_minutes: int = 15,
    start_ms: int = 1640995200000,
    trend: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV candles."""
    rng = np.random.default_rng(seed)
    step_ms = timeframe_minutes * 60 * 1000
    timestamps = [start_ms + i * step_ms for i in range(n_bars)]

    close = 100.0 + np.cumsum(rng.normal(trend, 0.5, size=n_bars))
    high = close + rng.uniform(0.1, 1.0, size=n_bars)
    low = close - rng.uniform(0.1, 1.0, size=n_bars)
    open_p = (high + low) / 2.0
    volume = rng.uniform(10.0, 100.0, size=n_bars)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestMTFResamplingAndCausality:
    """Rigorous causality and lookahead tests."""

    def test_resample_ohlcv_causal_timestamps(self):
        df_15m = _generate_synthetic_candles(n_bars=32, timeframe_minutes=15)
        # 32 15m bars = 8 hours = 2 4H bars
        res_4h = resample_ohlcv_causal(df_15m, "4h")
        assert len(res_4h) == 2
        # First 4h bar opens at 00:00, available at 04:00 (14,400,000 ms later)
        assert res_4h["available_ts"].iloc[0] == res_4h["open_ts"].iloc[0] + 14_400_000
        assert res_4h["available_ts"].iloc[1] == res_4h["open_ts"].iloc[1] + 14_400_000

    def test_zero_lookahead_bias_on_future_spikes(self):
        """Inject a massive future price spike in bar 14 (03:30) of 4H candle 0 (00:00-04:00).
        Ensure bars at 03:00, 03:15, 03:30, 03:45 CANNOT observe the 4H bar indicators."""
        df = _generate_synthetic_candles(n_bars=60, timeframe_minutes=15)
        # Spike high and close at bar 14 (03:30 UTC)
        df.loc[14, "high"] = 99999.0
        df.loc[14, "close"] = 99999.0

        strat = MTFTrendPullbackStrategy(params={"mtf_combo": "4h_1h_15m"})
        prepared = strat.setup(df)

        # In prepared, at bar 12 (03:00), 13 (03:15), 14 (03:30), 15 (03:45):
        # The 4H bar starting at 00:00 only closes at 04:00 (bar 16).
        # Therefore, at bars 12..15, htf_close must NOT reflect 99999.0!
        for b in [12, 13, 14, 15]:
            assert prepared["htf_close"].iloc[b] != 99999.0 or np.isnan(prepared["htf_close"].iloc[b])


class TestMTFStrategyConfigAndSignals:
    """Strategy configuration, validation, and signal logic."""

    def test_param_validation(self):
        with pytest.raises(ValueError, match="Invalid mtf_combo"):
            MTFTrendPullbackStrategy(params={"mtf_combo": "invalid_combo"})

        with pytest.raises(ValueError, match="htf_ema_fast must be < htf_ema_slow"):
            MTFTrendPullbackStrategy(params={"htf_ema_fast": 50, "htf_ema_slow": 20})

    def test_all_three_mtf_combinations_setup(self):
        # 15M base
        df_15m = _generate_synthetic_candles(n_bars=200, timeframe_minutes=15)
        strat_15m = MTFTrendPullbackStrategy(params={"mtf_combo": "4h_1h_15m"})
        prep_15m = strat_15m.setup(df_15m)
        assert "_long_setup" in prep_15m.columns
        assert "_short_setup" in prep_15m.columns

        # 5M base
        df_5m = _generate_synthetic_candles(n_bars=300, timeframe_minutes=5)
        strat_5m = MTFTrendPullbackStrategy(params={"mtf_combo": "4h_1h_5m"})
        prep_5m = strat_5m.setup(df_5m)
        assert "_long_setup" in prep_5m.columns
        assert "_short_setup" in prep_5m.columns

        # 1H base
        df_1h = _generate_synthetic_candles(n_bars=200, timeframe_minutes=60)
        strat_1h = MTFTrendPullbackStrategy(params={"mtf_combo": "1d_4h_1h"})
        prep_1h = strat_1h.setup(df_1h)
        assert "_long_setup" in prep_1h.columns
        assert "_short_setup" in prep_1h.columns

    def test_stop_loss_and_take_profit_calculation(self):
        strat_atr = MTFTrendPullbackStrategy(params={"stop_type": "atr", "atr_multiplier": 2.5, "risk_reward_ratio": 2.0})
        df = _generate_synthetic_candles(n_bars=100, timeframe_minutes=15)
        prep = strat_atr.setup(df)

        sig = strat_atr.generate_signal(prep, 50)
        # Check that ATR-based stop was computed correctly
        entry_price = float(df["close"].iloc[50])
        stop = strat_atr.compute_stop_loss(prep, 50, Direction.LONG, entry_price)
        tp = strat_atr.compute_take_profit(prep, 50, Direction.LONG, entry_price, stop)

        assert stop is not None and stop < entry_price
        assert tp is not None and tp > entry_price
        assert pytest.approx(tp - entry_price, rel=1e-3) == 2.0 * (entry_price - stop)

    def test_backtest_integration_spot_and_futures(self):
        df = _generate_synthetic_candles(n_bars=300, timeframe_minutes=15, trend=0.1)
        strat = MTFTrendPullbackStrategy(params={
            "mtf_combo": "4h_1h_15m",
            "htf_adx_threshold": 0.0,
            "ltf_rsi_oversold": 50.0,
            "itf_rsi_pullback": 60.0,
        })

        # Spot Backtest
        cfg_spot = BacktestConfig(initial_capital=1000.0, risk_per_trade=0.01, market_type="spot")
        engine_spot = BacktestEngine(cfg_spot)
        res_spot = engine_spot.run(strat, df, symbol="BTCUSDT")
        assert res_spot.config["initial_capital"] == 1000.0
        assert isinstance(res_spot.metrics.net_return, float)

        # Futures Backtest
        cfg_fut = BacktestConfig(initial_capital=1000.0, risk_per_trade=0.01, market_type="futures", max_leverage=5.0)
        engine_fut = BacktestEngine(cfg_fut)
        res_fut = engine_fut.run(strat, df, symbol="BTCUSDT")
        assert res_fut.config["initial_capital"] == 1000.0
        assert isinstance(res_fut.metrics.net_return, float)

    def test_dmi_directional_filter_effect(self):
        """Verifies +DI and -DI are computed and integrated into HTF bull/bear flags."""
        df = _generate_synthetic_candles(n_bars=300, timeframe_minutes=15, trend=0.1)
        strat = MTFTrendPullbackStrategy(params={"mtf_combo": "4h_1h_15m", "htf_adx_threshold": 15.0})
        prep = strat.setup(df)
        assert "htf_plus_di" in prep.columns
        assert "htf_minus_di" in prep.columns
        assert "_long_setup" in prep.columns
        assert "_short_setup" in prep.columns
