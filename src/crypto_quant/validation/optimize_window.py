"""Per-window multi-objective optimization and parameter stability analysis.

Methodology
-----------
Evaluates candidates using a defensive multi-objective scoring function rather than
naive single-metric maximization (e.g. max win rate), preventing high-win-rate/low-sample
overfitting.

Scoring & Ranking Hierarchy:
1. Hard Quality Filters (Gatekeeper):
   - Minimum trade count (rejects N < min_trades).
   - Positive expectancy (E > 0.0).
   - Profit factor floor (PF >= 1.05).
   - Max drawdown ceiling (MDD <= max_drawdown_limit).
2. Standardized Multi-Objective Score:
   - Expectancy utility U_E = tanh(E / scale).
   - Profit factor utility U_PF = clip((PF - 1.0) / 3.0, 0, 1).
   - Risk-adjusted return utility (Sharpe, Sortino, Calmar).
   - Drawdown penalty P_DD = 1 - (MDD / MDD_max)^2.
   - Sample-size confidence factor C_N = 1 - exp(-N / 25.0).
3. Parameter Stability Analysis:
   - Evaluates performance variance among 1-step grid neighbors.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..backtesting import BacktestConfig, BacktestEngine, ExecutionConfig
from ..logging_config import get_logger
from ..research.optimizer import ParameterSearcher, SearchConfig, grid_neighbors
from ..strategies import create_strategy, strategy_param_grid

logger = get_logger("validation")


@dataclass
class MultiObjectiveWeights:
    """Configurable weights for multi-objective optimization."""

    expectancy: float = 0.25
    profit_factor: float = 0.20
    drawdown_penalty: float = 0.20
    sharpe_ratio: float = 0.15
    sortino_ratio: float = 0.10
    calmar_ratio: float = 0.10


@dataclass
class MultiObjectiveFilters:
    """Hard rejection filters for candidate strategies."""

    min_trades: int = 15
    min_profit_factor: float = 1.05
    min_expectancy: float = 0.0
    max_drawdown: float = 0.40
    max_liquidations: int = 0


@dataclass
class CandidateEvaluation:
    """Evaluation output for a single parameter set."""

    params: Dict[str, Any]
    metrics: Dict[str, Any]
    score: float
    passed_filters: bool
    rejection_reasons: List[str] = field(default_factory=list)
    sub_scores: Dict[str, float] = field(default_factory=dict)


def score_metrics_multi_objective(
    metrics: Dict[str, Any],
    weights: Optional[MultiObjectiveWeights] = None,
    filters: Optional[MultiObjectiveFilters] = None,
) -> Tuple[float, bool, List[str], Dict[str, float]]:
    """Compute deterministic multi-objective score with hard gatekeeping.

    Returns:
        (score, passed_filters, rejection_reasons, sub_scores_dict)
    """
    w = weights or MultiObjectiveWeights()
    f = filters or MultiObjectiveFilters()

    t_cnt = int(metrics.get("total_trades", 0))
    exp = float(metrics.get("expectancy", 0.0))
    pf = float(metrics.get("profit_factor", 0.0))
    mdd = float(metrics.get("max_drawdown", 1.0))
    liq = int(metrics.get("liquidations", 0))
    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    sortino = float(metrics.get("sortino_ratio", 0.0))
    calmar = float(metrics.get("calmar_ratio", 0.0))
    win_rate = float(metrics.get("win_rate", 0.0))

    reasons: List[str] = []
    if t_cnt < f.min_trades:
        reasons.append(f"Insufficient trades: {t_cnt} < {f.min_trades}")
    if exp <= f.min_expectancy:
        reasons.append(f"Negative/zero expectancy: {exp:.2f} <= {f.min_expectancy:.2f}")
    if pf < f.min_profit_factor:
        reasons.append(f"Profit factor below threshold: {pf:.2f} < {f.min_profit_factor:.2f}")
    if mdd > f.max_drawdown:
        reasons.append(f"Excessive max drawdown: {mdd:.1%} > {f.max_drawdown:.1%}")
    if liq > f.max_liquidations:
        reasons.append(f"Liquidations detected: {liq} > {f.max_liquidations}")

    passed = len(reasons) == 0

    # Bounded normalized sub-scores in [0, 1]
    # 1. Expectancy utility (scaled relative to trade notional risk, clamped via tanh)
    u_exp = float(np.tanh(max(0.0, exp) / 10.0))

    # 2. Profit factor utility (1.0 -> 0.0, 4.0+ -> 1.0)
    u_pf = float(np.clip((pf - 1.0) / 3.0, 0.0, 1.0)) if pf >= 1.0 else 0.0

    # 3. Risk-adjusted ratio utilities
    u_sharpe = float(np.clip((sharpe + 1.0) / 4.0, 0.0, 1.0))
    u_sortino = float(np.clip((sortino + 1.0) / 6.0, 0.0, 1.0))
    u_calmar = float(np.clip(calmar / 5.0, 0.0, 1.0)) if calmar > 0 else 0.0

    # 4. Drawdown penalty (quadratic penalty)
    dd_ratio = min(1.0, mdd / max(f.max_drawdown, 0.01))
    p_dd = float(max(0.0, 1.0 - (dd_ratio ** 2)))

    # 5. Sample size confidence factor (asymptotes to 1.0 as N grows)
    c_n = float(1.0 - np.exp(-t_cnt / 25.0)) if t_cnt > 0 else 0.0

    # Composite weighted utility
    composite = (
        w.expectancy * u_exp
        + w.profit_factor * u_pf
        + w.drawdown_penalty * p_dd
        + w.sharpe_ratio * u_sharpe
        + w.sortino_ratio * u_sortino
        + w.calmar_ratio * u_calmar
    )

    final_score = c_n * composite if passed else -1.0 + (c_n * 0.1)

    sub_scores = {
        "expectancy_utility": round(u_exp, 4),
        "profit_factor_utility": round(u_pf, 4),
        "drawdown_penalty": round(p_dd, 4),
        "sharpe_utility": round(u_sharpe, 4),
        "sortino_utility": round(u_sortino, 4),
        "calmar_utility": round(u_calmar, 4),
        "sample_confidence": round(c_n, 4),
        "composite_score": round(final_score, 4),
    }

    return final_score, passed, reasons, sub_scores


def window_best_params(
    df: pd.DataFrame,
    strategy_type: str,
    grid: Dict[str, List[Any]],
    symbol: str = "TEST",
    market_type: str = "spot",
    timeframe: str = "1h",
    max_combinations: int = 20,
    search_type: str = "grid",
    objective: str = "multi_objective",
    weights: Optional[MultiObjectiveWeights] = None,
    filters: Optional[MultiObjectiveFilters] = None,
) -> Optional[Dict[str, Any]]:
    """Evaluate a bounded parameter search on a slice using multi-objective ranking.

    Returns:
        Dict with keys: 'params', 'metrics', 'score', 'passed_filters', 'sub_scores'
        or None if no candidates could be evaluated.
    """
    searcher = ParameterSearcher(SearchConfig(
        search_type=search_type, max_combinations=max_combinations,
    ))
    combos = searcher.combinations(grid)
    evaluations: List[CandidateEvaluation] = []

    for params in combos:
        try:
            strategy = create_strategy(strategy_type, params=params)
            cfg = BacktestConfig(
                market_type=market_type,
                timeframe=timeframe,
                execution=ExecutionConfig(market_type=market_type),
                min_bars=30,
            )
            result = BacktestEngine(cfg).run(strategy, df, symbol=symbol)
            m = result.metrics.to_dict()
        except Exception as exc:
            logger.debug("Evaluation failed for %s with %s: %s", strategy_type, params, exc)
            continue

        if objective == "max_win_rate":
            score = float(m.get("win_rate", 0.0))
            passed = score > 0
            reasons = [] if passed else ["0 win rate"]
            sub_scores = {"win_rate": score}
        elif objective == "max_profit_factor":
            score = float(m.get("profit_factor", 0.0))
            passed = score >= 1.0
            reasons = [] if passed else ["PF < 1.0"]
            sub_scores = {"profit_factor": score}
        else:
            # Default: Multi-objective score
            score, passed, reasons, sub_scores = score_metrics_multi_objective(m, weights, filters)

        evaluations.append(CandidateEvaluation(
            params=params,
            metrics=m,
            score=score,
            passed_filters=passed,
            rejection_reasons=reasons,
            sub_scores=sub_scores,
        ))

    if not evaluations:
        return None

    # Priority: Passed filters first sorted by score descending; then non-passed sorted by score descending
    passed_candidates = [e for e in evaluations if e.passed_filters]
    if passed_candidates:
        passed_candidates.sort(key=lambda e: e.score, reverse=True)
        winner = passed_candidates[0]
    else:
        evaluations.sort(key=lambda e: e.score, reverse=True)
        winner = evaluations[0]

    return {
        "params": winner.params,
        "metrics": winner.metrics,
        "score": winner.score,
        "passed_filters": winner.passed_filters,
        "rejection_reasons": winner.rejection_reasons,
        "sub_scores": winner.sub_scores,
    }


@dataclass
class StabilityResult:
    """Parameter-stability analysis around a best parameter set."""

    best_params: Dict[str, Any]
    best_score: float
    neighbors: List[Dict[str, Any]]
    neighbor_spread: float
    isolated: bool

    def to_dict(self) -> dict:
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "neighbors": self.neighbors,
            "neighbor_spread": self.neighbor_spread,
            "isolated_spike": self.isolated,
        }


def parameter_stability(
    df: pd.DataFrame,
    strategy_type: str,
    best_params: Dict[str, Any],
    symbol: str = "TEST",
    market_type: str = "spot",
    timeframe: str = "1h",
) -> StabilityResult:
    """Analyze the neighborhood around best_params to detect overfit isolated spikes."""
    grid = strategy_param_grid(strategy_type)
    neighbors = grid_neighbors(best_params, grid)

    best_m = _backtest_metrics_dict(df, strategy_type, best_params, symbol, market_type, timeframe)
    best_score, _, _, _ = score_metrics_multi_objective(best_m)

    rows = []
    for nparams in neighbors:
        nm = _backtest_metrics_dict(df, strategy_type, nparams, symbol, market_type, timeframe)
        n_score, n_passed, _, _ = score_metrics_multi_objective(nm)
        rows.append({
            "params": nparams,
            "score": n_score,
            "win_rate": nm.get("win_rate", 0.0),
            "profit_factor": nm.get("profit_factor", 0.0),
            "expectancy": nm.get("expectancy", 0.0),
            "passed_filters": n_passed,
            "delta_from_best": n_score - best_score,
        })

    scores = [r["score"] for r in rows]
    spread = float(np.std(scores)) if len(scores) > 1 else 0.0
    gaps = [best_score - r["score"] for r in rows]
    min_gap = min(gaps) if gaps else 0.0
    isolated = len(rows) >= 2 and min_gap > 0.25 and best_score > 0.4

    return StabilityResult(
        best_params=best_params,
        best_score=best_score,
        neighbors=rows,
        neighbor_spread=spread,
        isolated=isolated,
    )


def _backtest_metrics_dict(df, strategy_type, params, symbol, market_type, timeframe) -> Dict[str, Any]:
    try:
        strategy = create_strategy(strategy_type, params=params)
        cfg = BacktestConfig(
            market_type=market_type, timeframe=timeframe,
            execution=ExecutionConfig(market_type=market_type), min_bars=30,
        )
        result = BacktestEngine(cfg).run(strategy, df, symbol=symbol)
        return result.metrics.to_dict()
    except Exception:
        return {}
