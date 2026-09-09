"""Strategy discovery engine.

Orchestrates the research pipeline for a symbol/timeframe/market:

    1. Generate parameter candidates for each strategy family
       (grid / random / adaptive search, honoring max_combinations).
    2. Backtest each candidate on the given data.
    3. Apply hard quality filters.
    4. Rank by the configured objective (default max win rate).
    5. Persist the experiment and candidate results for reproducibility.

The engine never silently hides losing strategies — every evaluated candidate
is recorded; the report shows how many passed and why the rest failed.
"""

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..backtesting import BacktestConfig, BacktestEngine, ExecutionConfig
from ..logging_config import get_logger
from ..strategies import create_strategy, strategy_param_grid
from .experiments import CandidateRecord, ExperimentTracker
from .optimizer import ParameterSearcher, SearchConfig
from .ranking import QualityFilters, RankedCandidate, RankingEngine

logger = get_logger("research")


@dataclass
class DiscoveryConfig:
    """Configuration for a discovery run."""

    strategy_types: List[str] = field(default_factory=lambda: ["trend", "momentum", "breakout", "mean_reversion"])
    search_type: str = "grid"            # grid | random | adaptive
    max_combinations: int = 500          # hard cap per strategy type
    objective: str = "max_win_rate"

    # Quality filters
    min_trades: int = 100
    max_drawdown: float = 0.50
    min_win_rate: float = 0.0
    min_profit_factor: float = 1.0
    max_liquidations: int = 0

    # Risk/execution context
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.01
    max_open_positions: int = 3
    max_leverage: float = 5.0
    market_type: str = "spot"
    timeframe: str = "1h"
    slippage: float = 0.0005

    def quality_filters(self) -> QualityFilters:
        return QualityFilters(
            min_trades=self.min_trades,
            max_drawdown=self.max_drawdown,
            min_win_rate=self.min_win_rate,
            min_profit_factor=self.min_profit_factor,
            max_liquidations=self.max_liquidations,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_types": self.strategy_types,
            "search_type": self.search_type,
            "max_combinations": self.max_combinations,
            "objective": self.objective,
            "quality_filters": {
                "min_trades": self.min_trades,
                "max_drawdown": self.max_drawdown,
                "min_win_rate": self.min_win_rate,
                "min_profit_factor": self.min_profit_factor,
                "max_liquidations": self.max_liquidations,
            },
            "risk": {
                "initial_capital": self.initial_capital,
                "risk_per_trade": self.risk_per_trade,
                "max_open_positions": self.max_open_positions,
                "max_leverage": self.max_leverage,
            },
            "market_type": self.market_type,
            "timeframe": self.timeframe,
            "slippage": self.slippage,
        }


@dataclass
class DiscoveryResult:
    """Result of a discovery run."""

    experiment_id: str
    symbol: str
    timeframe: str
    market_type: str
    n_strategy_types: int
    n_candidates: int
    n_passed: int
    n_tested: int
    duration_seconds: float
    ranked: List[RankedCandidate] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def top(self) -> Optional[RankedCandidate]:
        qualified = [r for r in self.ranked if r.passed_filters]
        return qualified[0] if qualified else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "market_type": self.market_type,
            "n_strategy_types": self.n_strategy_types,
            "n_candidates": self.n_candidates,
            "n_passed": self.n_passed,
            "n_tested": self.n_tested,
            "duration_seconds": round(self.duration_seconds, 2),
            "config": self.config,
            "ranked": [r.to_dict() for r in self.ranked],
        }


class DiscoveryEngine:
    """Runs the automated strategy discovery pipeline."""

    def __init__(
        self,
        config: Optional[DiscoveryConfig] = None,
        tracker: Optional[ExperimentTracker] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """Initialize.

        Args:
            config: Discovery configuration.
            tracker: Experiment tracker (created lazily if omitted).
            progress_callback: Optional callback(evaluated, total) for UI progress.
        """
        self.config = config or DiscoveryConfig()
        self.tracker = tracker
        self._progress = progress_callback or (lambda done, total: None)
        self._searcher = ParameterSearcher(SearchConfig(
            search_type=self.config.search_type,
            max_combinations=self.config.max_combinations,
        ))

    # ------------------------------------------------------------ main API
    def discover(
        self,
        df,
        symbol: str = "BTCUSDT",
        start_date: str = "",
        end_date: str = "",
        db=None,
    ) -> DiscoveryResult:
        """Run discovery over a DataFrame.

        Args:
            df: OHLCV DataFrame (columns: timestamp, open, high, low, close, volume).
            symbol: Symbol label used for reporting/tracking.
            start_date/end_date: For experiment metadata.
            db: DatabaseManager for experiment persistence (optional; without it
                the run is not persisted).

        Returns:
            DiscoveryResult with ranked candidates.
        """
        start = time.time()
        strategy_types = [s for s in self.config.strategy_types if s in ("trend", "momentum", "breakout", "mean_reversion")]

        # Build the full candidate list per strategy type
        all_candidates: List[Dict[str, Any]] = []
        for st in strategy_types:
            grid = strategy_param_grid(st)
            combos = self._searcher.combinations(grid)
            for params in combos:
                all_candidates.append({"strategy_type": st, "params": params})

        n_total = len(all_candidates)
        n_tested = 0
        results: List[Dict[str, Any]] = []

        for i, cand in enumerate(all_candidates):
            try:
                metrics = self._evaluate(df, cand["strategy_type"], cand["params"], symbol)
            except Exception as exc:
                logger.debug("Candidate failed: %s %s (%s)", cand["strategy_type"], cand["params"], exc)
                metrics = None
            n_tested += 1
            if metrics is not None:
                results.append({
                    "strategy_type": cand["strategy_type"],
                    "params": cand["params"],
                    "metrics": metrics,
                })
            self._progress(n_tested, n_total)

        # Adaptive refinement: for 'adaptive' search, perturb around the current
        # best and re-evaluate a small budget (bounded).
        if self.config.search_type == "adaptive" and results:
            results.extend(self._adaptive_refine(df, results, symbol, budget=30))

        # Rank
        ranking = RankingEngine(objective=self.config.objective, filters=self.config.quality_filters())
        ranked = ranking.rank(results)
        n_passed = sum(1 for r in ranked if r.passed_filters)

        # Persist
        experiment_id = self._persist(symbol, start_date, end_date, ranked, results, db)

        duration = time.time() - start
        logger.info(
            "Discovery %s: %d candidates, %d passed filters, %d tested (%.1fs)",
            experiment_id, len(ranked), n_passed, n_tested, duration,
        )

        return DiscoveryResult(
            experiment_id=experiment_id,
            symbol=symbol,
            timeframe=self.config.timeframe,
            market_type=self.config.market_type,
            n_strategy_types=len(strategy_types),
            n_candidates=len(ranked),
            n_passed=n_passed,
            n_tested=n_tested,
            duration_seconds=duration,
            ranked=ranked,
            config=self.config.to_dict(),
        )

    # ------------------------------------------------------------ evaluation
    def _evaluate(self, df, strategy_type: str, params: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
        """Backtest one candidate and return its metrics dict (or None)."""
        strategy = create_strategy(strategy_type, params=params)
        cfg = BacktestConfig(
            initial_capital=self.config.initial_capital,
            risk_per_trade=self.config.risk_per_trade,
            max_open_positions=self.config.max_open_positions,
            max_leverage=self.config.max_leverage,
            market_type=self.config.market_type,
            timeframe=self.config.timeframe,
            execution=ExecutionConfig(
                market_type=self.config.market_type,
                slippage=self.config.slippage,
            ),
            min_bars=30,
        )
        result = BacktestEngine(cfg).run(strategy, df, symbol=symbol)
        return result.metrics.to_dict()

    def _adaptive_refine(self, df, results, symbol, budget: int) -> List[Dict[str, Any]]:
        """Perturb around the best observed params and re-evaluate (bounded)."""
        best = max(results, key=lambda r: r["metrics"].get("win_rate", 0.0))
        if best["metrics"].get("total_trades", 0) < 30:
            return []
        grid = strategy_param_grid(best["strategy_type"])
        mutations = self._searcher.refine_around(best["params"], grid, budget)
        refined = []
        for params in mutations:
            try:
                metrics = self._evaluate(df, best["strategy_type"], params, symbol)
                if metrics:
                    refined.append({
                        "strategy_type": best["strategy_type"],
                        "params": params,
                        "metrics": metrics,
                    })
            except Exception:
                continue
        return refined

    # ------------------------------------------------------------ persistence
    def _persist(self, symbol, start_date, end_date, ranked, results, db) -> str:
        if db is None:
            return "not-persisted"
        if self.tracker is None:
            self.tracker = ExperimentTracker(db)
        tracker = self.tracker

        exp_id = tracker.create_experiment(
            config={
                "symbol": symbol,
                "timeframe": self.config.timeframe,
                "market_type": self.config.market_type,
                "search": self.config.to_dict(),
            },
            start_date=start_date or "",
            end_date=end_date or "",
        )

        records = []
        for r in ranked:
            records.append(CandidateRecord(
                strategy_type=r.strategy_type,
                params=r.params,
                symbol=symbol,
                timeframe=self.config.timeframe,
                market_type=self.config.market_type,
                metrics=r.metrics,
                rank=r.rank,
                passed_filters=r.passed_filters,
            ))
        tracker.save_candidates(exp_id, records)

        # Aggregate summary from the best qualified candidate
        top = self._top_qualified(ranked)
        if top is not None:
            tracker.save_result_summary(exp_id, top.metrics)
        tracker.update_status(exp_id, "completed")
        return exp_id

    @staticmethod
    def _top_qualified(ranked: List[RankedCandidate]) -> Optional[RankedCandidate]:
        for r in ranked:
            if r.passed_filters:
                return r
        return None