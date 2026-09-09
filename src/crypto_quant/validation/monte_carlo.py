"""Monte Carlo trade-sequence simulation.

Resamples the observed trade outcomes to estimate the distribution of possible
equity paths under the SAME trading edge and trade frequency. Used to estimate:
- potential drawdowns
- probability of ruin (equity below a floor)
- expected losing streaks
- equity-curve variability / confidence intervals

These are STATISTICAL SCENARIOS, not predictions. Output is always labeled as
simulation, and sampling never fabricates outcomes outside the observed trade
set.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..logging_config import get_logger

logger = get_logger("validation")


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation output."""

    n_simulations: int
    trades_per_sim: int
    initial_capital: float
    final_equity: List[float]
    max_drawdowns: List[float]
    losing_streaks: List[int]
    final_equity_pctiles: Dict[str, float]
    max_drawdown_pctiles: Dict[str, float]
    probability_of_ruin: float
    ruin_threshold: float

    def to_dict(self) -> dict:
        return {
            "n_simulations": self.n_simulations,
            "trades_per_sim": self.trades_per_sim,
            "initial_capital": self.initial_capital,
            "final_equity_percentiles": self.final_equity_pctiles,
            "max_drawdown_percentiles": self.max_drawdown_pctiles,
            "probability_of_ruin": self.probability_of_ruin,
            "ruin_threshold": self.ruin_threshold,
            "note": "Statistical scenario — NOT a prediction of future results.",
        }


class MonteCarloSimulator:
    """Resamples trade outcomes to estimate scenario distributions."""

    def __init__(
        self,
        n_simulations: int = 1000,
        random_seed: Optional[int] = 42,
    ):
        """Initialize simulator."""
        self.n_simulations = n_simulations
        self._rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------ main API
    def simulate(
        self,
        trade_returns: List[float],
        initial_capital: float = 1000.0,
        ruin_threshold: float = 0.5,     # ruin if equity < 50% of start
    ) -> MonteCarloResult:
        """Run the simulation on observed per-trade returns.

        Args:
            trade_returns: Per-trade net returns (fraction of equity per trade,
                e.g. 0.01 = +1%). Use portfolio-return-per-trade, not raw PnL.
            initial_capital: Starting equity.
            ruin_threshold: Fraction of starting capital below which equity is
                considered ruined.

        Returns:
            MonteCarloResult.
        """
        returns = np.asarray([float(r) for r in trade_returns if np.isfinite(r)])
        if len(returns) < 10:
            raise ValueError(f"Need >= 10 trade returns, got {len(returns)}")

        n_trades = len(returns)
        finals = []
        drawdowns = []
        streaks = []

        for _ in range(self.n_simulations):
            seq = self._rng.choice(returns, size=n_trades, replace=True)
            equity = initial_capital * np.cumprod(1.0 + seq)
            finals.append(float(equity[-1]))
            drawdowns.append(float(self._max_drawdown(equity, initial_capital)))
            streaks.append(int(self._longest_loss_streak(seq)))

        finals = np.array(finals)
        drawdowns = np.array(drawdowns)
        ruin_count = int((finals < initial_capital * ruin_threshold).sum())

        return MonteCarloResult(
            n_simulations=self.n_simulations,
            trades_per_sim=n_trades,
            initial_capital=initial_capital,
            final_equity=finals.tolist(),
            max_drawdowns=drawdowns.tolist(),
            losing_streaks=streaks,
            final_equity_pctiles=self._percentiles(finals),
            max_drawdown_pctiles=self._percentiles(drawdowns),
            probability_of_ruin=ruin_count / self.n_simulations,
            ruin_threshold=ruin_threshold,
        )

    # ------------------------------------------------------------ helpers
    def _max_drawdown(self, equity: np.ndarray, initial: float) -> float:
        peak = equity[0]
        max_dd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            if peak > 0:
                dd = (peak - e) / peak
                if dd > max_dd:
                    max_dd = dd
        return max_dd

    def _longest_loss_streak(self, seq: np.ndarray) -> int:
        best = cur = 0
        for r in seq:
            if r < 0:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    @staticmethod
    def _percentiles(values: np.ndarray) -> Dict[str, float]:
        return {
            f"p{p}": round(float(np.percentile(values, p)), 4)
            for p in (5, 25, 50, 75, 95)
        }