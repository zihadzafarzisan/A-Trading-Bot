"""Strategy ranking engine.

Applies hard quality filters first (minimum trades, maximum drawdown, minimum
win rate, profit factor, liquidation cap), then ranks survivors by a
configurable objective. The default objective is maximum win rate — but only
among strategies that pass the quality constraints, so a 95%-win strategy with
catastrophic losses is rejected.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..logging_config import get_logger

logger = get_logger("research")

# Supported ranking objectives
OBJECTIVES = (
    "multi_objective",
    "best_overall",
    "max_win_rate",
    "max_profit_factor",
    "max_risk_adjusted_return",
    "min_drawdown",
)


@dataclass
class QualityFilters:
    """Hard quality constraints a candidate must pass to be ranked."""

    min_trades: int = 100
    max_drawdown: float = 0.50          # 50% max drawdown cap
    min_win_rate: float = 0.0
    min_profit_factor: float = 1.0
    max_liquidations: int = 0           # futures: reject liquidation-heavy strats
    min_expectancy: float = 0.0

    def passes(self, metrics: Dict[str, Any]) -> bool:
        """Return True if a candidate's metrics satisfy all constraints."""
        checks = [
            int(metrics.get("total_trades", 0)) >= self.min_trades,
            float(metrics.get("max_drawdown", 1.0)) <= self.max_drawdown,
            float(metrics.get("win_rate", 0.0)) >= self.min_win_rate,
            float(metrics.get("profit_factor", 0.0)) >= self.min_profit_factor,
            int(metrics.get("liquidations", 0)) <= self.max_liquidations,
            float(metrics.get("expectancy", -1.0)) >= self.min_expectancy,
        ]
        return all(checks)

    def failed_checks(self, metrics: Dict[str, Any]) -> List[str]:
        """Return human-readable reasons a candidate failed the filters."""
        reasons = []
        if int(metrics.get("total_trades", 0)) < self.min_trades:
            reasons.append(f"trades={metrics.get('total_trades', 0)} < {self.min_trades}")
        if float(metrics.get("max_drawdown", 1.0)) > self.max_drawdown:
            reasons.append(f"max_drawdown={metrics.get('max_drawdown', 1.0):.1%} > {self.max_drawdown:.0%}")
        if float(metrics.get("win_rate", 0.0)) < self.min_win_rate:
            reasons.append(f"win_rate={metrics.get('win_rate', 0):.1%} < {self.min_win_rate:.1%}")
        if float(metrics.get("profit_factor", 0.0)) < self.min_profit_factor:
            reasons.append(f"profit_factor={metrics.get('profit_factor', 0):.2f} < {self.min_profit_factor:.2f}")
        if int(metrics.get("liquidations", 0)) > self.max_liquidations:
            reasons.append(f"liquidations={metrics.get('liquidations', 0)} > {self.max_liquidations}")
        if float(metrics.get("expectancy", -1.0)) < self.min_expectancy:
            reasons.append(f"expectancy={metrics.get('expectancy', 0):.2f} < {self.min_expectancy:.2f}")
        return reasons


@dataclass
class RankedCandidate:
    """A candidate with its ranking."""

    candidate_id: int
    strategy_type: str
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    score: float
    rank: int
    passed_filters: bool
    filter_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "strategy_type": self.strategy_type,
            "params": self.params,
            "metrics": self.metrics,
            "score": self.score,
            "rank": self.rank,
            "passed_filters": self.passed_filters,
            "filter_reasons": self.filter_reasons,
        }


class RankingEngine:
    """Filters and ranks strategy candidates."""

    def __init__(
        self,
        objective: str = "max_win_rate",
        filters: Optional[QualityFilters] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize ranking.

        Args:
            objective: One of OBJECTIVES.
            filters: Quality filters applied before ranking.
            weights: For 'best_overall', weight each component.
        """
        if objective not in OBJECTIVES:
            raise ValueError(f"Unknown objective '{objective}'. Options: {OBJECTIVES}")
        self.objective = objective
        self.filters = filters or QualityFilters()
        self.weights = weights or {
            "win_rate": 0.25, "profit_factor": 0.25, "sharpe_ratio": 0.25,
            "drawdown_penalty": 0.25,
        }

    # ------------------------------------------------------------ main API
    def rank(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[RankedCandidate]:
        """Filter, score, and rank candidates.

        Args:
            candidates: List of dicts with keys: strategy_type, params,
                metrics (dict from BacktestMetrics.to_dict()).

        Returns:
            Ranked list (best first). Candidates failing filters are appended
            at the end with passed_filters=False.
        """
        scored = []
        for i, c in enumerate(candidates):
            metrics = c.get("metrics", {})
            passed = self.filters.passes(metrics)
            reasons = self.filters.failed_checks(metrics) if not passed else []
            score = self._score(metrics) if passed else 0.0
            scored.append(RankedCandidate(
                candidate_id=i,
                strategy_type=c.get("strategy_type", "unknown"),
                params=c.get("params", {}),
                metrics=metrics,
                score=score,
                rank=0,
                passed_filters=passed,
                filter_reasons=reasons,
            ))

        # Qualified first (best score), then non-qualified (stable order)
        qualified = [c for c in scored if c.passed_filters]
        unqualified = [c for c in scored if not c.passed_filters]
        qualified.sort(key=lambda c: c.score, reverse=True)
        unqualified.sort(key=lambda c: c.candidate_id)

        ranked = qualified + unqualified
        for pos, c in enumerate(ranked):
            c.rank = pos + 1
        return ranked

    def top_n(self, ranked: List[RankedCandidate], n: int = 10) -> List[RankedCandidate]:
        """Return the top-N qualified candidates."""
        return [c for c in ranked if c.passed_filters][:n]

    # ------------------------------------------------------------ scoring
    def _score(self, metrics: Dict[str, Any]) -> float:
        """Score a candidate's metrics for the configured objective."""
        if self.objective == "multi_objective":
            from ..validation.optimize_window import score_metrics_multi_objective
            score, _, _, _ = score_metrics_multi_objective(metrics)
            return score
        if self.objective == "max_win_rate":
            return float(metrics.get("win_rate", 0.0))
        if self.objective == "max_profit_factor":
            pf = float(metrics.get("profit_factor", 0.0))
            # Cap PF at a sane value to avoid infinite scale distortion
            return min(pf, 10.0)
        if self.objective == "min_drawdown":
            return 1.0 - float(metrics.get("max_drawdown", 1.0))
        if self.objective == "max_risk_adjusted_return":
            return self._risk_adjusted_score(metrics)

        # best_overall composite
        return self._composite_score(metrics)

    def _risk_adjusted_score(self, metrics: Dict[str, Any]) -> float:
        sharpe = float(metrics.get("sharpe_ratio", 0.0))
        net_return = float(metrics.get("net_return", 0.0))
        dd = float(metrics.get("max_drawdown", 1.0)) or 1e-9
        # Blend Sharpe with return/maxDD (Calmar-like), both bounded
        calmar = min(net_return / dd, 10.0)
        return 0.5 * min(max(sharpe, -3), 5) / 5 + 0.5 * min(max(calmar, 0), 10) / 10

    def _composite_score(self, metrics: Dict[str, Any]) -> float:
        wr = float(metrics.get("win_rate", 0.0))
        pf = min(float(metrics.get("profit_factor", 0.0)), 10.0) / 10.0
        sharpe = min(max(float(metrics.get("sharpe_ratio", 0.0)), -3), 5) / 5
        dd = 1.0 - float(metrics.get("max_drawdown", 1.0))
        w = self.weights
        score = (w.get("win_rate", 0.25) * wr
                 + w.get("profit_factor", 0.25) * pf
                 + w.get("sharpe_ratio", 0.25) * sharpe
                 + w.get("drawdown_penalty", 0.25) * dd)
        return max(score, 0.0)