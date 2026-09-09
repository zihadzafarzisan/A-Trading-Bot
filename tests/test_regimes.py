"""Tests for market regime detection and statistical helpers."""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.analysis.regimes import RegimeClassifier
from crypto_quant.analysis.statistics import (
    cohens_d, welch_t_pvalue, compare_effect, percentile_range, category_distribution,
)


def make_df(closes, step_h=3600000):
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    return pd.DataFrame({
        "timestamp": [1609459200000 + i * step_h for i in range(n)],
        "open": closes - 0.1,
        "high": closes + 1.0,
        "low": closes - 1.0,
        "close": closes,
        "volume": 1000.0,
    })


class TestRegimeClassifier:
    """Test regime classification."""

    def test_bull_regime_uptrend(self):
        """A steady uptrend should be classified mostly bull after warmup."""
        closes = np.linspace(100, 400, 400)
        df = make_df(closes)
        classified = RegimeClassifier().classify(df)
        # Only look after the SMA200 warmup
        tail = classified.iloc[250:]
        bull_share = (tail["regime"] == "bull").mean()
        assert bull_share > 0.9, f"bull share too low: {bull_share:.2f}"

    def test_bear_regime_downtrend(self):
        """A steady downtrend should be classified mostly bear after warmup."""
        closes = np.linspace(400, 100, 400)
        df = make_df(closes)
        classified = RegimeClassifier().classify(df)
        tail = classified.iloc[250:]
        bear_share = (tail["regime"] == "bear").mean()
        assert bear_share > 0.9, f"bear share too low: {bear_share:.2f}"

    def test_sideways_flat(self):
        """A flat market with noise should be mostly sideways."""
        rng = np.random.default_rng(0)
        closes = 100 + rng.normal(0, 0.5, 400)
        df = make_df(closes)
        classified = RegimeClassifier().classify(df)
        sideways_share = (classified["regime"] == "sideways").mean()
        assert sideways_share > 0.5, f"sideways share too low: {sideways_share:.2f}"

    def test_high_volatility_detected(self):
        """A high-volatility episode should be flagged high_volatility."""
        rng = np.random.default_rng(1)
        # calm then explosive
        calm = 100 + rng.normal(0, 0.3, 200)
        wild = 100 + rng.normal(0, 8.0, 200)
        closes = np.concatenate([calm, wild])
        df = make_df(closes)
        classified = RegimeClassifier().classify(df)
        # After warmup, the second half should be mostly high_volatility
        tail = classified.iloc[150:]
        high_share = (tail["vol_regime"] == "high_volatility").mean()
        assert high_share > 0.5, f"high vol share too low: {high_share:.2f}"

    def test_no_lookahead_regime(self):
        """Regime at bar t must not depend on future closes."""
        classifier = RegimeClassifier()
        # Changing the last close must not change the regime of earlier bars
        closes = np.linspace(100, 400, 400)
        df1 = make_df(closes)
        df2 = make_df(np.concatenate([closes[:-1], [1000.0]]))  # spike last close
        r1 = classifier.classify(df1)
        r2 = classifier.classify(df2)
        # Regime of bar N-2 should be identical in both (future spike invisible)
        assert r1["regime"].iloc[-2] == r2["regime"].iloc[-2]

    def test_regime_distribution(self):
        closes = np.linspace(100, 400, 300)
        df = make_df(closes)
        classifier = RegimeClassifier()
        classified = classifier.classify(df)
        dist = classifier.regime_distribution(classified)
        assert "bull" in dist["regime"]
        assert abs(sum(dist["regime"].values()) - 1.0) < 1e-6

    def test_describe(self):
        classifier = RegimeClassifier(sma_fast=20, sma_slow=100)
        d = classifier.describe()
        assert d["sma_fast"] == 20
        assert d["sma_slow"] == 100


class TestStatistics:
    """Test statistical helpers."""

    def test_cohens_d_same_distribution(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 500)
        b = rng.normal(0, 1, 500)
        d = cohens_d(a, b)
        assert abs(d) < 0.3

    def test_cohens_d_large_difference(self):
        a = np.random.default_rng(1).normal(10, 1, 200)
        b = np.random.default_rng(2).normal(0, 1, 200)
        d = cohens_d(a, b)
        assert d > 2.0

    def test_welch_pvalue_significant(self):
        a = np.random.default_rng(1).normal(5, 1, 200)
        b = np.random.default_rng(2).normal(0, 1, 200)
        p = welch_t_pvalue(a, b)
        assert p is not None
        assert p < 0.001

    def test_welch_pvalue_not_significant(self):
        a = np.random.default_rng(3).normal(0, 1, 200)
        b = np.random.default_rng(4).normal(0, 1, 200)
        p = welch_t_pvalue(a, b)
        assert p > 0.01

    def test_welch_pvalue_small_sample(self):
        assert welch_t_pvalue([1.0, 2.0], [3.0, 4.0]) is None

    def test_percentile_range(self):
        vals = list(range(1, 101))
        lo, hi = percentile_range(vals, 25, 75)
        assert lo == pytest.approx(25.75)
        assert hi == pytest.approx(75.25)

    def test_category_distribution(self):
        d = category_distribution(["a", "a", "b"])
        assert d["a"]["count"] == 2
        assert d["a"]["share"] == pytest.approx(2 / 3)
        assert d["b"]["share"] == pytest.approx(1 / 3)

    def test_compare_effect(self):
        a = np.random.default_rng(1).normal(5, 1, 200)
        b = np.random.default_rng(2).normal(0, 1, 200)
        e = compare_effect(a, b)
        assert e.meaningful is True
        assert e.winners_higher is True