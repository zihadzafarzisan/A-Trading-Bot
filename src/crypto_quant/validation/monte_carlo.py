"""Monte Carlo trade-sequence simulation.

Resamples observed trade outcomes to estimate empirical probability distributions of
equity curves, risk metrics, and drawdown scenarios under the same trading edge:
- Maximum Drawdown distribution (5th, 25th, 50th, 75th, 95th percentiles)
- Expected Shortfall / Conditional Value at Risk (CVaR) of Max Drawdown (worst-tail average)
- Sharpe Ratio distribution
- Sortino Ratio distribution
- Calmar Ratio distribution
- Longest Losing Streak distribution
- Probability of Ruin (equity drops below configurable ruin threshold)
- Probability of severe drawdowns (>20%, >30%)
- Equity curve uncertainty / confidence intervals

Annualization Methodology
-------------------------
For trade-level simulations, Sharpe and Sortino ratios are annualized using:
    Annualized Sharpe = (mean_trade_return / std_trade_return) * sqrt(annualized_trades)
    Annualized Sortino = (mean_trade_return / downside_std_trade_return) * sqrt(annualized_trades)
where `annualized_trades` represents the expected trading frequency per year (default: 100.0).

These are statistical scenarios, not predictions. Sampling uses only observed trade
results without fabricating outcomes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

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
    sharpes: List[float] = field(default_factory=list)
    sortinos: List[float] = field(default_factory=list)
    calmars: List[float] = field(default_factory=list)

    final_equity_pctiles: Dict[str, float] = field(default_factory=dict)
    max_drawdown_pctiles: Dict[str, float] = field(default_factory=dict)
    cvar_max_drawdown: Dict[str, float] = field(default_factory=dict)  # Expected Shortfall of MDD
    sharpe_pctiles: Dict[str, float] = field(default_factory=dict)
    sortino_pctiles: Dict[str, float] = field(default_factory=dict)
    calmar_pctiles: Dict[str, float] = field(default_factory=dict)
    losing_streak_pctiles: Dict[str, int] = field(default_factory=dict)

    prob_severe_dd_20: float = 0.0
    prob_severe_dd_30: float = 0.0
    probability_of_ruin: float = 0.0
    ruin_threshold: float = 0.5
    annualized_trades: float = 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_simulations": self.n_simulations,
            "trades_per_sim": self.trades_per_sim,
            "initial_capital": self.initial_capital,
            "final_equity_percentiles": self.final_equity_pctiles,
            "max_drawdown_percentiles": self.max_drawdown_pctiles,
            "cvar_max_drawdown": self.cvar_max_drawdown,
            "sharpe_percentiles": self.sharpe_pctiles,
            "sortino_percentiles": self.sortino_pctiles,
            "calmar_percentiles": self.calmar_pctiles,
            "losing_streak_percentiles": self.losing_streak_pctiles,
            "probability_of_severe_drawdown_20pct": self.prob_severe_dd_20,
            "probability_of_severe_drawdown_30pct": self.prob_severe_dd_30,
            "probability_of_ruin": self.probability_of_ruin,
            "ruin_threshold": self.ruin_threshold,
            "annualized_trades_assumption": self.annualized_trades,
            "note": "Statistical scenario distribution — NOT a prediction of future results.",
        }


class MonteCarloSimulator:
    """Resamples trade outcomes to estimate scenario distributions."""

    def __init__(
        self,
        n_simulations: int = 1000,
        random_seed: Optional[int] = 42,
    ):
        """Initialize simulator.

        Args:
            n_simulations: Number of synthetic equity curve paths.
            random_seed: Random seed for deterministic reproducibility.
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------ main API
    def simulate(
        self,
        trade_returns: Union[List[float], List[Dict[str, Any]], np.ndarray],
        initial_capital: float = 1000.0,
        ruin_threshold: float = 0.5,
        annualized_trades: float = 100.0,
    ) -> MonteCarloResult:
        """Run the simulation on observed trade returns.

        Args:
            trade_returns: List of fractional returns (e.g. 0.01 = +1%), PnL values,
                or trade dicts containing 'net_pnl' or 'return'.
            initial_capital: Starting equity.
            ruin_threshold: Configurable fraction of starting capital (default 0.5 = 50%)
                below which account is considered ruined.
            annualized_trades: Expected annual trade frequency for Sharpe/Sortino annualization.

        Returns:
            MonteCarloResult with complete distribution percentiles and CVaR metrics.
        """
        parsed_returns: List[float] = []
        if isinstance(trade_returns, np.ndarray):
            parsed_returns = trade_returns.tolist()
        elif isinstance(trade_returns, list):
            for r in trade_returns:
                if isinstance(r, dict):
                    if "net_pnl" in r:
                        parsed_returns.append(float(r["net_pnl"]) / initial_capital)
                    elif "pnl" in r:
                        parsed_returns.append(float(r["pnl"]) / initial_capital)
                    elif "return" in r:
                        parsed_returns.append(float(r["return"]))
                else:
                    parsed_returns.append(float(r))

        returns = np.asarray([float(r) for r in parsed_returns if np.isfinite(r)])
        if len(returns) < 10:
            raise ValueError(f"Need >= 10 trade returns, got {len(returns)}")

        n_trades = len(returns)
        finals: List[float] = []
        drawdowns: List[float] = []
        streaks: List[int] = []
        sharpes: List[float] = []
        sortinos: List[float] = []
        calmars: List[float] = []

        severe_dd_20_count = 0
        severe_dd_30_count = 0
        ruin_count = 0

        # Annualization factor: sqrt(annual trades)
        ann_factor = np.sqrt(max(1.0, annualized_trades))

        for _ in range(self.n_simulations):
            seq = self._rng.choice(returns, size=n_trades, replace=True)
            equity = initial_capital * np.cumprod(1.0 + seq)
            final_eq = float(equity[-1])
            finals.append(final_eq)

            mdd = float(self._max_drawdown(equity, initial_capital))
            drawdowns.append(mdd)

            if mdd >= 0.20:
                severe_dd_20_count += 1
            if mdd >= 0.30:
                severe_dd_30_count += 1

            if np.min(equity) <= initial_capital * ruin_threshold:
                ruin_count += 1

            streaks.append(int(self._longest_loss_streak(seq)))

            # Sharpe & Sortino calculation
            mean_r = float(np.mean(seq))
            std_r = float(np.std(seq))
            neg_returns = seq[seq < 0]
            downside_std = float(np.std(neg_returns)) if len(neg_returns) > 0 else 1e-6

            sh = (mean_r / (std_r + 1e-9)) * ann_factor if std_r > 0 else 0.0
            so = (mean_r / (downside_std + 1e-9)) * ann_factor if downside_std > 0 else 0.0
            tot_ret = (final_eq - initial_capital) / initial_capital
            c = tot_ret / (mdd + 1e-6)

            sharpes.append(float(sh))
            sortinos.append(float(so))
            calmars.append(float(c))

        finals_arr = np.array(finals)
        drawdowns_arr = np.array(drawdowns)
        sharpes_arr = np.array(sharpes)
        sortinos_arr = np.array(sortinos)
        calmars_arr = np.array(calmars)
        streaks_arr = np.array(streaks)

        # Calculate Expected Shortfall / CVaR of Max Drawdown
        # CVaR_alpha(MDD) = mean of MDD values >= percentile_alpha(MDD)
        p90_dd = float(np.percentile(drawdowns_arr, 90))
        p95_dd = float(np.percentile(drawdowns_arr, 95))
        tail_90 = drawdowns_arr[drawdowns_arr >= p90_dd]
        tail_95 = drawdowns_arr[drawdowns_arr >= p95_dd]
        cvar_90 = float(np.mean(tail_90)) if len(tail_90) > 0 else p90_dd
        cvar_95 = float(np.mean(tail_95)) if len(tail_95) > 0 else p95_dd

        cvar_dict = {
            "cvar_90": round(cvar_90, 4),
            "cvar_95": round(cvar_95, 4),
        }

        return MonteCarloResult(
            n_simulations=self.n_simulations,
            trades_per_sim=n_trades,
            initial_capital=initial_capital,
            final_equity=finals,
            max_drawdowns=drawdowns,
            losing_streaks=streaks,
            sharpes=sharpes,
            sortinos=sortinos,
            calmars=calmars,
            final_equity_pctiles=self._percentiles(finals_arr),
            max_drawdown_pctiles=self._percentiles(drawdowns_arr),
            cvar_max_drawdown=cvar_dict,
            sharpe_pctiles=self._percentiles(sharpes_arr),
            sortino_pctiles=self._percentiles(sortinos_arr),
            calmar_pctiles=self._percentiles(calmars_arr),
            losing_streak_pctiles=self._integer_percentiles(streaks_arr),
            prob_severe_dd_20=round(severe_dd_20_count / self.n_simulations, 4),
            prob_severe_dd_30=round(severe_dd_30_count / self.n_simulations, 4),
            probability_of_ruin=round(ruin_count / self.n_simulations, 4),
            ruin_threshold=ruin_threshold,
            annualized_trades=annualized_trades,
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

    @staticmethod
    def _integer_percentiles(values: np.ndarray) -> Dict[str, int]:
        return {
            f"p{p}": int(np.percentile(values, p))
            for p in (5, 25, 50, 75, 95)
        }
