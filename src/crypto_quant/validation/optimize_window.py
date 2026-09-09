"""Per-window optimization and parameter stability analysis.

- ``window_best_params``: small grid/random search on a train slice (used by
  walk-forward reoptimization).
- ``parameter_stability``: evaluates the best params and their neighbors on the
  same data to detect isolated spikes that indicate overfitting (spec #49).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..backtesting import BacktestConfig, BacktestEngine, ExecutionConfig
from ..logging_config import get_logger
from ..strategies import create_strategy, strategy_param_grid
from ..research.optimizer import ParameterSearcher, SearchConfig, grid_neighbors

logger = get_logger("validation")


def window_best_params(
    df: pd.DataFrame,
    strategy_type: str,
    grid: Dict[str, List[Any]],
    symbol: str = "TEST",
    market_type: str = "spot",
    timeframe: str = "1h",
    max_combinations: int = 20,
    search_type: str = "grid",
    objective: str = "max_win_rate",
) -> Optional[Dict[str, Any]]:
    """Evaluate a bounded parameter search on a slice and return the best.

    Returns {'params': best_params, 'metrics': best_metrics} or None.
    """
    from ..research.optimizer import ParameterSearcher

    searcher = ParameterSearcher(SearchConfig(
        search_type=search_type, max_combinations=max_combinations,
    ))
    combos = searcher.combinations(grid)
    best = None
    for params in combos:
        try:
            strategy = create_strategy(strategy_type, params=params)
            cfg = BacktestConfig(
                market_type=market_type, timeframe=timeframe,
                execution=ExecutionConfig(market_type=market_type), min_bars=30,
            )
            result = BacktestEngine(cfg).run(strategy, df, symbol=symbol)
            m = result.metrics.to_dict()
        except Exception:
            continue
        score = m.get("win_rate", 0.0) if objective == "max_win_rate" else m.get("profit_factor", 0.0)
        if best is None or score > best["score"]:
            best = {"params": params, "metrics": m, "score": score}
    return best


@dataclass
class StabilityResult:
    """Parameter-stability analysis around a best parameter set."""

    best_params: Dict[str, Any]
    best_win_rate: float
    neighbors: List[Dict[str, Any]]     # each: params, win_rate, net_return, delta
    neighbor_spread: float              # std of neighbor win rates
    isolated: bool                      # best is an isolated spike

    def to_dict(self) -> dict:
        return {
            "best_params": self.best_params,
            "best_win_rate": self.best_win_rate,
            "neighbors": self.neighbors,
            "neighbor_win_rate_spread": self.neighbor_spread,
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
    """Analyze the neighborhood around best_params.

    Compares the best parameter set against its one-step grid neighbors on the
    same data. If the best win rate is far above all neighbors, the strategy is
    likely overfit (an isolated spike) rather than robust.
    """
    grid = strategy_param_grid(strategy_type)
    neighbors = grid_neighbors(best_params, grid)

    best_wr = _backtest_win_rate(df, strategy_type, best_params, symbol, market_type, timeframe)
    rows = []
    for nparams in neighbors:
        wr = _backtest_win_rate(df, strategy_type, nparams, symbol, market_type, timeframe)
        rows.append({
            "params": nparams,
            "win_rate": wr,
            "delta_from_best": wr - best_wr,
        })

    spread = float(np.std([r["win_rate"] for r in rows])) if len(rows) > 1 else 0.0
    # Isolated if the best beats every neighbor by > 5pp AND the neighbors are
    # clustered well below it (spread small relative to the gap).
    gaps = [best_wr - r["win_rate"] for r in rows]
    min_gap = min(gaps) if gaps else 0.0
    isolated = len(rows) >= 2 and min_gap > 0.05 and best_wr > 0.55

    return StabilityResult(
        best_params=best_params,
        best_win_rate=best_wr,
        neighbors=rows,
        neighbor_spread=spread,
        isolated=isolated,
    )


def _backtest_win_rate(df, strategy_type, params, symbol, market_type, timeframe) -> float:
    try:
        strategy = create_strategy(strategy_type, params=params)
        cfg = BacktestConfig(
            market_type=market_type, timeframe=timeframe,
            execution=ExecutionConfig(market_type=market_type), min_bars=30,
        )
        result = BacktestEngine(cfg).run(strategy, df, symbol=symbol)
        return float(result.metrics.trades.win_rate)
    except Exception:
        return 0.0