"""Tests for the validation engine (train/test, walk-forward, multi-objective ranking, Monte Carlo)."""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.validation import (
    MonteCarloSimulator,
    SplitDates,
    TrainTestSplitter,
    WalkForwardValidator,
    default_walk_forward_windows,
    parameter_stability,
)
from crypto_quant.validation.optimize_window import (
    MultiObjectiveFilters,
    MultiObjectiveWeights,
    score_metrics_multi_objective,
    window_best_params,
)


def make_df(n=3000, seed=0, drift=0.0003, start_ts=1577836800000, step=86_400_000):
    """Deterministic OHLCV DataFrame (daily bars by default)."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, 0.008, n)
    close = 100 * np.exp(np.cumsum(ret))
    return pd.DataFrame({
        "timestamp": [start_ts + i * step for i in range(n)],
        "open": np.roll(close, 1),
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": 1000.0,
    })


class TestTrainTestSplitter:
    """Test strict chronological splitting."""

    def test_split_three_way(self):
        df = make_df(n=3000)
        splitter = TrainTestSplitter(SplitDates("2024-12-31", "2025-12-31", "2026-09-08"))
        train, val, test = splitter.split(df)
        assert len(train) > 0 and len(val) > 0 and len(test) > 0
        assert train["timestamp"].max() < val["timestamp"].min()
        assert val["timestamp"].max() < test["timestamp"].min()
        covered = len(train) + len(val) + len(test)
        assert covered <= len(df)
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
        df = df.iloc[::-1].copy()
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
        for w in windows:
            assert w.train_start < w.train_end <= w.test_start < w.test_end

    def test_walk_forward_runs(self):
        df = make_df(n=2500, seed=1)
        validator = WalkForwardValidator()
        report = validator.validate(
            df, "trend", {"ema_fast": 20, "ema_slow": 50},
            symbol="BTCUSDT", timeframe="1d",
        )
        assert len(report.windows) >= 1
        w = report.windows[0]
        assert "win_rate" in w.is_metrics
        assert "win_rate" in w.oos_metrics
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
        df = make_df(n=2500, seed=1)
        report = WalkForwardValidator().validate(
            df, "trend", {"ema_fast": 20, "ema_slow": 50}, symbol="BTCUSDT", timeframe="1d"
        )
        assert isinstance(report.warnings(), list)


class TestMultiObjectiveOptimization:
    """Test multi-objective scoring and window optimization."""

    def test_hard_rejection_insufficient_trades(self):
        metrics = {"total_trades": 5, "expectancy": 2.0, "profit_factor": 1.8, "max_drawdown": 0.1}
        filters = MultiObjectiveFilters(min_trades=15)
        score, passed, reasons, sub = score_metrics_multi_objective(metrics, filters=filters)
        assert not passed
        assert any("Insufficient trades" in r for r in reasons)
        assert score < 0

    def test_hard_rejection_negative_expectancy(self):
        metrics = {"total_trades": 50, "expectancy": -0.5, "profit_factor": 0.9, "max_drawdown": 0.2}
        score, passed, reasons, sub = score_metrics_multi_objective(metrics)
        assert not passed
        assert any("Negative/zero expectancy" in r for r in reasons)

    def test_hard_rejection_excessive_drawdown(self):
        metrics = {"total_trades": 50, "expectancy": 1.0, "profit_factor": 1.5, "max_drawdown": 0.65}
        filters = MultiObjectiveFilters(max_drawdown=0.40)
        score, passed, reasons, sub = score_metrics_multi_objective(metrics, filters=filters)
        assert not passed
        assert any("Excessive max drawdown" in r for r in reasons)

    def test_positive_candidate_scoring_and_sub_scores(self):
        metrics = {
            "total_trades": 60,
            "expectancy": 1.5,
            "profit_factor": 1.8,
            "max_drawdown": 0.15,
            "sharpe_ratio": 1.6,
            "sortino_ratio": 2.2,
            "calmar_ratio": 1.8,
            "win_rate": 0.45,
            "liquidations": 0,
        }
        score, passed, reasons, sub = score_metrics_multi_objective(metrics)
        assert passed
        assert len(reasons) == 0
        assert score > 0.3
        assert "expectancy_utility" in sub
        assert "profit_factor_utility" in sub
        assert "drawdown_penalty" in sub
        assert "sharpe_utility" in sub
        assert "sortino_utility" in sub
        assert "calmar_utility" in sub
        assert "sample_confidence" in sub

    def test_window_best_params_multi_objective(self):
        df = make_df(n=3000, seed=3)
        from crypto_quant.strategies import strategy_param_grid
        grid = strategy_param_grid("trend")
        result = window_best_params(
            df, "trend", grid, symbol="BTCUSDT", max_combinations=10, objective="multi_objective"
        )
        assert result is not None
        assert "params" in result
        assert "metrics" in result
        assert "score" in result
        assert "sub_scores" in result

    def test_parameter_stability_analysis(self):
        df = make_df(n=3000, seed=2)
        best = {"ema_fast": 20, "ema_slow": 50}
        result = parameter_stability(df, "trend", best, symbol="BTCUSDT")
        assert result.best_params == best
        assert len(result.neighbors) >= 1
        assert isinstance(result.neighbor_spread, float)


class TestMonteCarlo:
    """Test Monte Carlo simulation with complete risk distributions."""

    def test_simulation_distribution_and_percentiles(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0.002, 0.02, 200)
        sim = MonteCarloSimulator(n_simulations=500, random_seed=1)
        result = sim.simulate(returns, initial_capital=1000.0)

        assert result.n_simulations == 500
        assert len(result.final_equity) == 500
        assert len(result.sharpes) == 500
        assert len(result.sortinos) == 500
        assert len(result.calmars) == 500

        # Percentile dicts
        for pct_dict in [
            result.final_equity_pctiles,
            result.max_drawdown_pctiles,
            result.sharpe_pctiles,
            result.sortino_pctiles,
            result.calmar_pctiles,
            result.losing_streak_pctiles,
        ]:
            for p in ["p5", "p25", "p50", "p75", "p95"]:
                assert p in pct_dict

        # Monotonic percentiles
        assert result.final_equity_pctiles["p95"] >= result.final_equity_pctiles["p5"]
        assert result.max_drawdown_pctiles["p95"] >= result.max_drawdown_pctiles["p5"]
        assert result.losing_streak_pctiles["p95"] >= result.losing_streak_pctiles["p5"]

        # Probability bounds
        assert 0.0 <= result.probability_of_ruin <= 1.0
        assert 0.0 <= result.prob_severe_dd_20 <= 1.0
        assert 0.0 <= result.prob_severe_dd_30 <= 1.0

        # CVaR / Expected Shortfall of MDD
        assert "cvar_90" in result.cvar_max_drawdown
        assert "cvar_95" in result.cvar_max_drawdown
        assert result.cvar_max_drawdown["cvar_95"] >= result.max_drawdown_pctiles["p95"]

    def test_configurable_ruin_threshold(self):
        rng = np.random.default_rng(0)
        returns = [-0.04] * 50 + [0.01] * 50
        sim = MonteCarloSimulator(n_simulations=200, random_seed=42)
        # Low ruin threshold (ruin if equity < 20% of start)
        r_low = sim.simulate(returns, initial_capital=1000.0, ruin_threshold=0.2)
        # High ruin threshold (ruin if equity < 80% of start)
        r_high = sim.simulate(returns, initial_capital=1000.0, ruin_threshold=0.8)
        assert r_high.probability_of_ruin >= r_low.probability_of_ruin

    def test_trade_dict_inputs_supported(self):
        trade_dicts = [{"net_pnl": 15.0}, {"net_pnl": -10.0}, {"net_pnl": 20.0}] * 10
        sim = MonteCarloSimulator(n_simulations=100, random_seed=42)
        result = sim.simulate(trade_dicts, initial_capital=1000.0)
        assert result.trades_per_sim == 30
        assert result.n_simulations == 100

    def test_reproducible_with_seed(self):
        rng = np.random.default_rng(0)
        returns = rng.normal(0.002, 0.02, 200)
        sim1 = MonteCarloSimulator(n_simulations=300, random_seed=7)
        sim2 = MonteCarloSimulator(n_simulations=300, random_seed=7)
        r1 = sim1.simulate(returns)
        r2 = sim2.simulate(returns)
        assert r1.final_equity == r2.final_equity
        assert r1.max_drawdowns == r2.max_drawdowns
        assert r1.sharpes == r2.sharpes

    def test_severe_drawdown_probabilities(self):
        # Severe losing pool
        returns = [-0.05] * 80 + [0.01] * 20
        sim = MonteCarloSimulator(n_simulations=200, random_seed=3)
        result = sim.simulate(returns, initial_capital=1000)
        assert result.prob_severe_dd_20 > 0.5
        assert result.prob_severe_dd_30 > 0.5
        assert result.probability_of_ruin > 0.5

    def test_insufficient_trades_rejected(self):
        with pytest.raises(ValueError, match="Need >= 10"):
            MonteCarloSimulator().simulate([0.01, -0.01], initial_capital=1000)
