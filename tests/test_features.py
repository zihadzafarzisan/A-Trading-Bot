"""Tests for the feature engineering engine."""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.features.engine import FeatureEngine


@pytest.fixture
def price():
    """A deterministic 1h OHLCV DataFrame."""
    rng = np.random.default_rng(7)
    n = 400
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 1.0)
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    volume = np.abs(rng.normal(1000, 200, n)) + 1
    ts = [1609459200000 + i * 3_600_000 for i in range(n)]
    return pd.DataFrame({
        "timestamp": ts, "open": close - 0.1, "high": high,
        "low": low, "close": close, "volume": volume,
    })


class TestFeatureEngine:
    """Test the feature engine."""

    def test_compute_returns_features(self, price):
        engine = FeatureEngine()
        features = engine.compute(price)
        # Original OHLCV plus many feature columns
        assert len(features) == len(price)
        assert "rsi_14" in features.columns
        assert "adx_14" in features.columns
        assert "atr_14" in features.columns
        assert "volume_ratio" in features.columns
        assert "hour" in features.columns

    def test_feature_set_superset(self, price):
        engine = FeatureEngine()
        features = engine.compute(price)
        for name in engine.feature_names():
            assert name in features.columns, f"missing feature: {name}"

    def test_no_nan_unbounded_after_warmup(self, price):
        """After the longest indicator warmup, core momentum/trend features
        should be finite."""
        engine = FeatureEngine()
        features = engine.compute(price)
        tail = features.iloc[-20:]
        for col in ["rsi_14", "adx_14", "atr_ratio", "volume_ratio", "macd_macd",
                    "stoch_k", "roc_12", "hist_vol_20", "mfi_14"]:
            assert tail[col].notna().all(), f"{col} has NaN in tail"
            assert np.isfinite(tail[col]).all(), f"{col} has infinite in tail"

    def test_time_features(self, price):
        engine = FeatureEngine()
        features = engine.compute(price)
        # hour 0..23
        assert features["hour"].between(0, 23).all()
        # valid day of week 0..6
        assert features["day_of_week"].between(0, 6).all()
        # month 1..12
        assert features["month"].between(1, 12).all()

    def test_session_bucket(self, price):
        engine = FeatureEngine()
        features = engine.compute(price, time_bucket_minutes=60)
        assert "session_bucket" in features.columns

    def test_preserves_close(self, price):
        engine = FeatureEngine()
        features = engine.compute(price)
        assert features["close"].iloc[5] == price["close"].iloc[5]

    def test_anti_lookahead_on_features(self, price):
        """The EMA20 feature at bar t must use only closes before t.
        We verify by comparing to the raw ema shifted."""
        from crypto_quant.indicators import ema
        engine = FeatureEngine(shift=1)
        features = engine.compute(price)
        raw = ema(price["close"], 20, shift=0)
        # ema_20 feature at t should equal raw ema at t-1
        expected = raw.shift(1)
        assert_features_approx(features, "ema_20", expected, price)

    def test_commodity_features_bounded(self, price):
        """Volume ratio and bb_pct_b should be finite and reasonable in the tail."""
        engine = FeatureEngine()
        features = engine.compute(price)
        tail = features.iloc[-20:]
        assert (tail["volume_ratio"] > 0).all()
        assert np.isfinite(tail["bb_pct_b"]).all()


def assert_features_approx(features, col, expected_series, df):
    """Assert a feature column approximates an expected series for valid rows."""
    valid = expected_series.notna() & features[col].notna()
    a = features.loc[valid, col].to_numpy()
    b = expected_series.loc[valid].to_numpy()
    np.testing.assert_allclose(a, b, atol=1e-9)