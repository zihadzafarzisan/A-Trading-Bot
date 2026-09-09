"""Tests for the research & optimization engine."""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.research import (
    ParameterSearcher, SearchConfig, RankingEngine, QualityFilters,
    DiscoveryEngine, DiscoveryConfig, ExperimentTracker,
)
from crypto_quant.db.connection import DatabaseManager


@pytest.fixture
def db():
    d = DatabaseManager(in_memory=True)
    d.create_tables()
    return d


@pytest.fixture
def trending_df():
    """A mildly trending market good for trend strategies."""
    n = 1500
    rng = np.random.default_rng(0)
    drift = 0.0004
    ret = rng.normal(drift, 0.01, n)
    close = 100 * np.exp(np.cumsum(ret))
    return pd.DataFrame({
        "timestamp": [1609459200000 + i * 3_600_000 for i in range(n)],
        "open": np.roll(close, 1), "high": close * 1.005,
        "low": close * 0.995, "close": close, "volume": 1000.0,
    })


class TestParameterSearcher:
    """Test parameter search strategies."""

    GRID = {"ema_fast": [10, 20, 30], "ema_slow": [50, 100], "adx_threshold": [0, 25]}

    def test_grid_exhaustive(self):
        s = ParameterSearcher(SearchConfig(search_type="grid", max_combinations=1000))
        combos = s.combinations(self.GRID)
        assert len(combos) == 3 * 2 * 2

    def test_grid_respects_cap(self):
        s = ParameterSearcher(SearchConfig(search_type="grid", max_combinations=5))
        combos = s.combinations(self.GRID)
        assert len(combos) == 5

    def test_random_within_grid(self):
        s = ParameterSearcher(SearchConfig(search_type="random", max_combinations=50, random_seed=1))
        combos = s.combinations(self.GRID)
        assert len(combos) <= 50
        for c in combos:
            assert c["ema_fast"] in self.GRID["ema_fast"]
            assert c["ema_slow"] in self.GRID["ema_slow"]

    def test_adaptive(self):
        s = ParameterSearcher(SearchConfig(search_type="adaptive", max_combinations=40, adaptive_initial=10))
        combos = s.combinations(self.GRID)
        assert len(combos) <= 40
        assert len(combos) >= 10

    def test_unknown_search_type(self):
        with pytest.raises(ValueError):
            ParameterSearcher(SearchConfig(search_type="bayesian"))

    def test_empty_grid(self):
        s = ParameterSearcher()
        assert s.combinations({}) == [{}]

    def test_refine_around(self):
        s = ParameterSearcher(SearchConfig(search_type="adaptive"))
        best = {"ema_fast": 20, "ema_slow": 50, "adx_threshold": 0}
        mutations = s.refine_around(best, self.GRID, n=10)
        assert len(mutations) == 10
        for m in mutations:
            assert m["ema_fast"] in self.GRID["ema_fast"]


class TestQualityFilters:
    """Test hard quality filters."""

    def test_passes_good_candidate(self):
        f = QualityFilters(min_trades=50, max_drawdown=0.3, min_win_rate=0.5, min_profit_factor=1.5)
        metrics = {
            "total_trades": 100, "max_drawdown": 0.2, "win_rate": 0.6,
            "profit_factor": 2.0, "liquidations": 0, "expectancy": 1.0,
        }
        assert f.passes(metrics) is True
        assert f.failed_checks(metrics) == []

    def test_rejects_high_win_low_trades(self):
        """The 5-trade 100% win-rate trap must be rejected."""
        f = QualityFilters(min_trades=100)
        metrics = {"total_trades": 5, "max_drawdown": 0.1, "win_rate": 1.0,
                   "profit_factor": 99, "liquidations": 0, "expectancy": 10}
        assert f.passes(metrics) is False
        assert any("trades" in r for r in f.failed_checks(metrics))

    def test_rejects_high_win_catastrophic_dd(self):
        """95% win rate with huge losses must be rejected."""
        f = QualityFilters(min_trades=100, max_drawdown=0.3)
        metrics = {"total_trades": 200, "max_drawdown": 0.8, "win_rate": 0.95,
                   "profit_factor": 0.3, "liquidations": 0, "expectancy": -1}
        assert f.passes(metrics) is False

    def test_rejects_liquidations(self):
        f = QualityFilters(max_liquidations=0)
        metrics = {"total_trades": 150, "max_drawdown": 0.2, "win_rate": 0.6,
                   "profit_factor": 2.0, "liquidations": 3, "expectancy": 1}
        assert f.passes(metrics) is False


class TestRankingEngine:
    """Test strategy ranking."""

    def _candidate(self, **metrics):
        return {"strategy_type": "trend", "params": {}, "metrics": metrics}

    def test_ranks_by_win_rate(self):
        engine = RankingEngine(
            objective="max_win_rate",
            filters=QualityFilters(min_trades=50, max_drawdown=0.5),
        )
        c1 = self._candidate(total_trades=100, win_rate=0.7, profit_factor=1.5,
                             max_drawdown=0.2, liquidations=0, expectancy=1)
        c2 = self._candidate(total_trades=100, win_rate=0.55, profit_factor=2.5,
                             max_drawdown=0.2, liquidations=0, expectancy=1)
        ranked = engine.rank([c2, c1])
        assert ranked[0].params == {}
        assert ranked[0].metrics["win_rate"] == 0.7
        assert ranked[0].rank == 1

    def test_qualified_before_unqualified(self):
        engine = RankingEngine(objective="max_win_rate", filters=QualityFilters(min_trades=50))
        good = self._candidate(total_trades=100, win_rate=0.6, profit_factor=1.2,
                               max_drawdown=0.2, liquidations=0, expectancy=1)
        bad = self._candidate(total_trades=5, win_rate=1.0, profit_factor=9,
                              max_drawdown=0.1, liquidations=0, expectancy=5)
        ranked = engine.rank([bad, good])
        assert ranked[0].passed_filters is True
        assert ranked[1].passed_filters is False

    def test_top_n(self):
        engine = RankingEngine(objective="max_win_rate", filters=QualityFilters(min_trades=10))
        cands = [
            self._candidate(total_trades=20, win_rate=0.6 + i * 0.05, profit_factor=1.2,
                            max_drawdown=0.2, liquidations=0, expectancy=1)
            for i in range(5)
        ]
        ranked = engine.rank(cands)
        top = engine.top_n(ranked, n=2)
        assert len(top) == 2
        assert top[0].metrics["win_rate"] == max(c["metrics"]["win_rate"] for c in cands)

    def test_unknown_objective(self):
        with pytest.raises(ValueError):
            RankingEngine(objective="nonsense")


class TestDiscoveryEngine:
    """Test end-to-end discovery."""

    def test_discovers_and_ranks(self, trending_df, db):
        cfg = DiscoveryConfig(
            strategy_types=["trend"],
            search_type="grid",
            max_combinations=6,
            min_trades=30,
            max_drawdown=0.7,
            min_profit_factor=0.8,
            timeframe="1h",
        )
        engine = DiscoveryEngine(config=cfg)
        result = engine.discover(trending_df, symbol="BTCUSDT", db=db)

        assert result.n_candidates >= 1
        assert result.experiment_id != "not-persisted"
        assert result.experiment_id.startswith("EXP-")
        # Persistence check
        tracker = ExperimentTracker(db)
        exp = tracker.get_experiment(result.experiment_id)
        assert exp is not None
        assert exp["status"] == "completed"

    def test_min_trades_filter_rejects(self, trending_df, db):
        """With a strict min_trades, nothing passes (no fabrication)."""
        cfg = DiscoveryConfig(
            strategy_types=["trend"],
            search_type="grid",
            max_combinations=6,
            min_trades=10_000,   # impossible
            timeframe="1h",
        )
        engine = DiscoveryEngine(config=cfg)
        result = engine.discover(trending_df, symbol="BTCUSDT", db=db)
        assert result.n_passed == 0

    def test_tracker_candidates_persisted(self, trending_df, db):
        cfg = DiscoveryConfig(
            strategy_types=["trend"], search_type="grid", max_combinations=4,
            min_trades=10, max_drawdown=0.8, min_profit_factor=0.5,
        )
        result = DiscoveryEngine(config=cfg).discover(trending_df, symbol="BTCUSDT", db=db)
        tracker = ExperimentTracker(db)
        top = tracker.top_candidates(result.experiment_id, limit=10)
        assert len(top) >= 1
        assert "metrics" in top[0]
        assert "params" in top[0]

    def test_reproducible_config_stored(self, trending_df, db):
        cfg = DiscoveryConfig(
            strategy_types=["trend"], search_type="grid", max_combinations=4,
            min_trades=10, max_drawdown=0.8, min_profit_factor=0.5,
        )
        result = DiscoveryEngine(config=cfg).discover(trending_df, symbol="BTCUSDT", db=db)
        exp = ExperimentTracker(db).get_experiment(result.experiment_id)
        stored = exp["config"]
        assert stored["search"]["search_type"] == "grid"
        assert stored["search"]["max_combinations"] == 4