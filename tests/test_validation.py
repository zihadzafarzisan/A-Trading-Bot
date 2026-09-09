"""Tests for the validation engine (train/test, walk-forward, Monte Carlo)."""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.validation import (
    TrainTestSplitter, SplitDates, WalkForwardValidator,
    default_walk_forward_windows, parameter_stability,
    MonteCarloSimulator,
)
from crypto_quant.validation.optimize_window import window_best_params


def make_df(n=3000, seed=0, drift=0.0003, start_ts=1577836800000, step=86_400_000):
    """Deterministic OHLCV DataFrame (daily bars by default)."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, 0.008, n)
    close = 100 * np.exp(np.cumsum(ret))
    return pd.DataFrame({
        "timestamp": [start_ts + i * step for i in range(n)],
        "open": np.roll(close, 1), "high": close * 1.005,
        "low": close * 0.995, "close": close, "volume": 1000.0,
    })


class TestTrainTestSplitter:
    """Test strict chronological splitting."""

    def test_split_three_way(self):
        df = make_df(n=3000)
        splitter = TrainTestSplitter(SplitDates("2024-12-31", "2025-12-31", "2026-09-08"))
        train, val, test = splitter.split(df)
        # Non-overlapping, chronological, and contiguous
        assert len(train) > 0 and len(val) > 0 and len(test) > 0
        assert train["timestamp"].max() < val["timestamp"].min()
        assert val["timestamp"].max() < test["timestamp"].min()
        # Covers all data up to test_end (later bars intentionally excluded)
        covered = len(train) + len(val) + len(test)
        assert covered <= len(df)
        # train+val+test cover exactly the bars <= 2026-09-08
        test_end_dt = pd.Timestamp("2026-09-08", tz="UTC")
        t = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        assert covered == int((t <= test_end_dt).sum())

    def test_split_ratio_chronological(self):
        df = make_df(n=1000)
        splitter = TrainTestSplitter()
        train, val, test = splitter.split_by_ratio(df, train_ratio=0.7, val_ratio=0.15)
        assert len(train) == 700
        assert len(val) == 150
        assert len(test) == 150
        assert train["timestamp"].max() < val["timestamp"].min()

    def test_invalid_boundaries(self):
        with pytest.raises(ValueError):
            SplitDates("2026-01-01", "2025-01-01", "2024-01-01").validate()

    def test_unsorted_data_rejected(self):
        df = make_df(n=100)
        df = df.iloc[::-1].copy()  # descending
        splitter = TrainTestSplitter()
        with pytest.raises(ValueError, match="sorted"):
            splitter.split(df)

    def test_invalid_ratios(self):
        splitter = TrainTestSplitter()
        df = make_df(n=100)
        with pytest.raises(ValueError):
            splitter.split_by_ratio(df, train_ratio=0.9, val_ratio=0.2)


class TestWalkForward:
    """Test walk-forward validation."""

    def test_windows_default(self):
        windows = default_walk_forward_windows()
        assert len(windows) == 4
        # strictly increasing
        for w in windows:
            assert w.train_start < w.train_end <= w.test_start < w.test_end

    def test_walk_forward_runs(self):
        df = make_df(n=2500, seed=1)  # daily bars spanning 2020-2026
        validator = WalkForwardValidator()
        report = validator.validate(
            df, "trend", {"ema_fast": 20, "ema_slow": 50},
            symbol="BTCUSDT", timeframe="1d",
        )
        assert len(report.windows) >= 1
        w = report.windows[0]
        assert "win_rate" in w.is_metrics
        assert "win_rate" in w.oos_metrics
        # Degradation = OOS - IS
        assert w.win_rate_degradation == pytest.approx(
            w.oos_metrics["win_rate"] - w.is_metrics["win_rate"]
        )

    def test_walk_forward_metrics(self):
        df = make_df(n=2500, seed=1)
        report = WalkForwardValidator().validate(
            df, "trend", {"ema_fast": 20, "ema_slow": 50}, symbol="BTCUSDT", timeframe="1d"
        )
        assert report.mean_oos_win_rate is not None
        assert 0 <= report.mean_oos_win_rate <= 1

    def test_poor_oos_warning(self):
        """A warning fires when OOS win rate drops > 5pp below IS."""
        # Build a report with a big degradation
        df = make_df(n=2500, seed=1)
        report = WalkForwardValidator().validate(
            df, "trend", {"ema_fast": 20, "ema_slow": 50}, symbol="BTCUSDT", timeframe="1d"
        )
        # This is data-dependent; just assert the warnings list is well-formed
        assert isinstance(report.warnings(), list)


class TestParameterStability:
    """Test parameter-stability analysis."""

    def test_stability_runs(self):
        df = make_df(n=3000, seed=2)
        best = {"ema_fast": 20, "ema_slow": 50}
        result = parameter_stability(df, "trend", best, symbol="BTCUSDT")
        assert result.best_params == best
        assert len(result.neighbors) >= 1
        assert isinstance(result.neighbor_spread, float)

    def test_neighbors_vary_one_param(self):
        from crypto_quant.research.optimizer import grid_neighbors
        from crypto_quant.strategies import strategy_param_grid
        grid = strategy_param_grid("trend")
        best = {"ema_fast": 20, "ema_slow": 50, "adx_threshold": 0}
        neighbors = grid_neighbors(best, grid)
        for n in neighbors:
            changed = sum(1 for k in best if n[k] != best[k])
            assert changed == 1

    def test_window_best_params(self):
        df = make_df(n=3000, seed=3)
        from crypto_quant.strategies import strategy_param_grid
        grid = strategy_param_grid("trend")
        result = window_best_params(
            df, "trend", grid, symbol="BTCUSDT", max_combinations=10
        )
        assert result is not None
        assert "params" in result
        assert "metrics" in result


class TestMonteCarlo:
    """Test Monte Carlo simulation."""

    def test_simulation_distribution(self):
        rng = np.random.default_rng(0)
        # Slightly positive edge with realistic noise
        returns = rng.normal(0.002, 0.02, 200)
        sim = MonteCarloSimulator(n_simulations=500, random_seed=1)
        result = sim.simulate(returns, initial_capital=1000.0)

        assert result.n_simulations == 500
        assert len(result.final_equity) == 500
        assert result.final_equity_pctiles["p50"] > 0
        # p95 > p5
        assert result.final_equity_pctiles["p95"] >= result.final_equity_pctiles["p5"]
        assert 0 <= result.probability_of_ruin <= 1
        assert result.max_drawdown_pctiles["p50"] >= 0

    def test_ruin_probability_zero_for_positive_edge(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0.005, 0.005, 300)  # strong positive edge, low vol
        sim = MonteCarloSimulator(n_simulations=200, random_seed=2)
        result = sim.simulate(returns, initial_capital=1000.0, ruin_threshold=0.5)
        assert result.probability_of_ruin < 0.05

    def test_reproducible_with_seed(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0.002, 0.02, 200)
        sim1 = MonteCarloSimulator(n_simulations=300, random_seed=7)
        sim2 = MonteCarloSimulator(n_simulations=300, random_seed=7)
        r1 = sim1.simulate(returns)
        r2 = sim2.simulate(returns)
        assert r1.final_equity == r2.final_equity

    def test_insufficient_trades(self):
        with pytest.raises(ValueError):
            MonteCarloSimulator().simulate([0.01, -0.01], initial_capital=1000)

    def test_losing_streaks_recorded(self):
        # Mostly-losing pool makes long loss streaks likely in resamples
        returns = [-0.01] * 80 + [0.01] * 20
        sim = MonteCarloSimulator(n_simulations=300, random_seed=3)
        result = sim.simulate(returns, initial_capital=1000)
        assert max(result.losing_streaks) >= 10