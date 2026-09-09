"""Tests for technical indicators.

Includes a dedicated anti-look-ahead test verifying that the default (shift=1)
output at bar t equals the raw indicator evaluated at bar t-1 — proving no
current/future information leaks into the signal.
"""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.indicators import (
    sma, ema, ema_crossover, adx,
    rsi, macd, stochastic, roc, momentum, cci,
    atr, bollinger_bands, historical_volatility, donchian_channel,
    volume_ratio, obv, mfi, vwap, INDICATOR_REGISTRY,
)


@pytest.fixture
def price():
    """A deterministic 1h OHLCV DataFrame."""
    rng = np.random.default_rng(42)
    n = 500
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    volume = np.abs(rng.normal(1000, 200, n))
    ts = [1609459200000 + i * 3_600_000 for i in range(n)]
    return pd.DataFrame({
        "timestamp": ts, "open": close - 0.1, "high": high,
        "low": low, "close": close, "volume": volume,
    })


class TestLookAheadProtection:
    """Verify shift=1 output never uses current/future data."""

    def test_sma_shift_is_lagged(self, price):
        raw = sma(price["close"], 20, shift=0)
        shifted = sma(price["close"], 20, shift=1)
        # shifted[t] == raw[t-1]
        pd.testing.assert_series_equal(shifted.iloc[1:], raw.shift(1).iloc[1:], check_names=False)

    def test_ema_shift_is_lagged(self, price):
        raw = ema(price["close"], 20, shift=0)
        shifted = ema(price["close"], 20, shift=1)
        pd.testing.assert_series_equal(
            shifted.iloc[1:], raw.shift(1).iloc[1:], check_names=False, check_dtype=False
        )

    def test_rsi_shift_is_lagged(self, price):
        raw = rsi(price["close"], 14, shift=0)
        shifted = rsi(price["close"], 14, shift=1)
        pd.testing.assert_series_equal(
            shifted.iloc[1:], raw.shift(1).iloc[1:], check_names=False, check_dtype=False
        )

    def test_atr_shift_is_lagged(self, price):
        raw = atr(price["high"], price["low"], price["close"], 14, shift=0)
        shifted = atr(price["high"], price["low"], price["close"], 14, shift=1)
        pd.testing.assert_series_equal(
            shifted.iloc[1:], raw.shift(1).iloc[1:], check_names=False, check_dtype=False
        )

    def test_shifted_bar_uses_previous_close_only(self, price):
        """The shifted indicator at bar i must not depend on buy[i]. Equality
        of raw @ i-1 and shifted @ i is the proof."""
        # Altered-close invariance: changing today's close must not change
        # yesterday's shifted signal.
        original = sma(price["close"], 5, shift=1)
        altered = price["close"].copy()
        altered.iloc[200] = altered.iloc[200] * 10  # huge spike at bar 200
        modified = sma(altered, 5, shift=1)
        # Bars 201+ should differ (10x move enters window at 201), but bar 200
        # itself (shifted) must equal the value from the unaltered series at 200
        # because a 5-period SMA shifted by 1 at bar 200 uses closes [-5..200).
        assert original.iloc[200] == modified.iloc[200]


class TestTrendIndicators:
    """Test trend indicators."""

    def test_sma_known_value(self):
        s = pd.Series([1, 2, 3, 4, 5])
        result = sma(s, 3, shift=0)
        # sma of last 3 = (3+4+5)/3 = 4
        assert result.iloc[-1] == pytest.approx(4.0)
        # first two are NaN (insufficient data)
        assert np.isnan(result.iloc[0])

    def test_sma_constant_series(self, price):
        s = pd.Series([5.0] * 50)
        result = sma(s, 10, shift=0)
        assert (result.dropna() == 5.0).all()

    def test_ema(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = ema(s, 2, shift=0)
        assert not result.isna().all()

    def test_ema_crossover_signals(self):
        # Fast EMA rising above slow => +1 at the cross row
        s = pd.Series([50.0] * 30 + [60.0, 70.0, 80.0])
        result = ema_crossover(s, 5, 10, shift=0)
        # there will be a +1 once fast overtakes slow
        assert 1 in result.unique()

    def test_adx_bounds(self, price):
        result = adx(price["high"], price["low"], price["close"], 14, shift=0)
        valid = result.dropna()
        if len(valid):
            assert valid.between(0, 100).all()

    def test_adx_trending_series(self):
        # Strong uptrend => ADX should be elevated
        n = 200
        idx = np.arange(n)
        high = pd.Series(idx * 1.0 + 1)
        low = pd.Series(idx * 1.0)
        close = pd.Series(idx * 1.0 + 0.5)
        result = adx(high, low, close, 14, shift=0)
        if result.dropna().any():
            assert result.dropna().mean() > 20


class TestMomentumIndicators:
    """Test momentum indicators."""

    def test_rsi_bounds(self, price):
        result = rsi(price["close"], 14, shift=0)
        valid = result.dropna()
        assert valid.between(0, 100).all()

    def test_rsi_oversold_undersold(self):
        # Monotonic decline => RSI should be low (<30)
        s = pd.Series(np.linspace(100, 10, 100))
        result = rsi(s, 14, shift=0)
        assert result.iloc[-1] < 30

    def test_rsi_overbought_uptrend(self):
        s = pd.Series(np.linspace(10, 100, 100))
        result = rsi(s, 14, shift=0)
        assert result.iloc[-1] > 70

    def test_macd_shape(self, price):
        df = macd(price["close"], 12, 26, 9, shift=0)
        assert list(df.columns) == ["macd", "signal", "histogram"]
        assert len(df) == len(price)

    def test_macd_uptrend_positive(self):
        s = pd.Series(np.linspace(100, 200, 200))
        df = macd(s, 12, 26, 9, shift=0)
        assert df["macd"].iloc[-1] > 0

    def test_stochastic_bounds(self, price):
        df = stochastic(price["high"], price["low"], price["close"], 14, 3, shift=0)
        for col in ("stoch_k", "stoch_d"):
            valid = df[col].dropna()
            assert valid.between(0, 100).all()

    def test_roc(self, price):
        result = roc(price["close"], 12, shift=0)
        assert len(result) == len(price)

    def test_momentum_zero_flat(self):
        s = pd.Series([10.0] * 20)
        result = momentum(s, 5, shift=0)
        assert (result.dropna() == 0).all()

    def test_cci(self, price):
        result = cci(price["high"], price["low"], price["close"], 20, shift=0)
        assert len(result) == len(price)


class TestVolatilityIndicators:
    """Test volatility indicators."""

    def test_atr_positive(self, price):
        result = atr(price["high"], price["low"], price["close"], 14, shift=0)
        valid = result.dropna()
        if len(valid):
            assert (valid > 0).all()

    def test_bollinger_shapes(self, price):
        df = bollinger_bands(price["close"], 20, 2.0, shift=0)
        assert list(df.columns) == ["bb_mid", "bb_upper", "bb_lower", "bb_width"]
        # upper >= lower
        valid = df.dropna()
        if len(valid):
            assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_historical_volatility_flat(self):
        s = pd.Series([100.0] * 50)
        result = historical_volatility(s, 20, shift=0)
        # flat series => zero volatility
        assert (result.dropna() == 0).all() or result.dropna().empty

    def test_donchian_bounds(self, price):
        df = donchian_channel(price["high"], price["low"], 20, shift=0)
        assert "dc_high" in df.columns


class TestVolumeIndicators:
    """Test volume indicators."""

    def test_volume_ratio_above_1(self):
        # Increasing volume => recent volume above average
        v = pd.Series(np.linspace(100, 1000, 50))
        result = volume_ratio(v, 10, shift=0)
        assert result.iloc[-1] > 1

    def test_obv_increases_with_up_volume(self):
        close = pd.Series([100.0, 101, 102, 103, 104])
        vol = pd.Series([10.0, 20, 30, 40, 50])
        result = obv(close, vol, shift=0)
        assert result.iloc[-1] > 0

    def test_mfi_bounds(self, price):
        result = mfi(price["high"], price["low"], price["close"], price["volume"], 14, shift=0)
        valid = result.dropna()
        assert valid.between(0, 100).all()

    def test_vwap(self, price):
        result = vwap(price["high"], price["low"], price["close"], price["volume"], 20, shift=0)
        assert result.dropna().notna().any()
        # VWAP should be near typical price
        assert abs(result.iloc[-1] - price["close"].iloc[-1]) < 5


class TestRegistry:
    """Test indicator registry."""

    def test_registry_contains_all(self):
        for name in ["sma", "ema", "adx", "rsi", "macd", "atr", "bollinger_bands", "obv"]:
            assert name in INDICATOR_REGISTRY

    def test_registry_callable(self):
        s = pd.Series(np.linspace(1, 100, 50))
        result = INDICATOR_REGISTRY["sma"](s, 10, shift=0)
        assert len(result) == 50