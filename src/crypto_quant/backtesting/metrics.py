"""Performance metrics for backtests.

Computes the full battery of performance and risk metrics from closed trades
and the equity curve. All calculations are defensive (empty inputs handled)
and documented.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .portfolio import EquityPoint, Position


@dataclass
class TradeStats:
    """Aggregate trade statistics."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    win_loss_ratio: float = 0.0
    avg_holding_period: float = 0.0
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_funding: float = 0.0
    liquidations: int = 0
    total_profit: float = 0.0   # sum of net PnL of winning trades
    total_loss: float = 0.0     # abs sum of net PnL of losing trades


@dataclass
class BacktestMetrics:
    """Complete backtest metric set."""

    # Performance
    net_return: float = 0.0
    net_pnl: float = 0.0
    final_equity: float = 0.0

    # Risk
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    recovery_factor: float = 0.0
    volatility: float = 0.0

    # Trades
    trades: TradeStats = field(default_factory=TradeStats)

    # Breakdowns
    long_stats: Optional["TradeStats"] = None
    short_stats: Optional["TradeStats"] = None

    # Raw inputs kept for reference
    equity_series: List[float] = field(default_factory=list)
    n_bars: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        """Flat dict for reporting/DB storage."""
        t = self.trades
        d = {
            "net_return": self.net_return,
            "net_pnl": self.net_pnl,
            "final_equity": self.final_equity,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "recovery_factor": self.recovery_factor,
            "volatility": self.volatility,
            "total_trades": t.total_trades,
            "winning_trades": t.winning_trades,
            "losing_trades": t.losing_trades,
            "win_rate": t.win_rate,
            "profit_factor": t.profit_factor,
            "expectancy": t.expectancy,
            "avg_win": t.avg_win,
            "avg_loss": t.avg_loss,
            "largest_win": t.largest_win,
            "largest_loss": t.largest_loss,
            "win_loss_ratio": t.win_loss_ratio,
            "avg_holding_period": t.avg_holding_period,
            "longest_win_streak": t.longest_win_streak,
            "longest_loss_streak": t.longest_loss_streak,
            "total_fees": t.total_fees,
            "total_slippage": t.total_slippage,
            "total_funding": t.total_funding,
            "liquidations": t.liquidations,
            "total_profit": t.total_profit,
            "total_loss": t.total_loss,
        }
        if self.long_stats:
            d["long"] = _stats_dict(self.long_stats)
        if self.short_stats:
            d["short"] = _stats_dict(self.short_stats)
        return d


def _stats_dict(s: TradeStats) -> Dict[str, float]:
    return {
        "trades": s.total_trades, "wins": s.winning_trades, "losses": s.losing_trades,
        "win_rate": s.win_rate, "profit_factor": s.profit_factor, "expectancy": s.expectancy,
        "avg_win": s.avg_win, "avg_loss": s.avg_loss,
    }


class MetricsCalculator:
    """Computes BacktestMetrics from trades and the equity curve."""

    # Annualization by timeframe (bars per year)
    BARS_PER_YEAR = {
        "1m": 525_600, "5m": 105_120, "15m": 35_040, "30m": 17_520,
        "1h": 8760, "4h": 2190, "1d": 365,
    }

    def __init__(self, timeframe: str = "1h", risk_free_rate: float = 0.0):
        """Initialize with timeframe used for annualization."""
        self.timeframe = timeframe
        self.annualization = self.BARS_PER_YEAR.get(timeframe, 8760)
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------ main API
    def compute(
        self,
        trades: List[Position],
        equity_curve: List[EquityPoint],
        initial_capital: float,
    ) -> BacktestMetrics:
        metrics = BacktestMetrics()

        # ---- Trade stats
        metrics.trades = self._trade_stats(trades)
        metrics.long_stats = self._trade_stats(
            [t for t in trades if t.direction == "long"], skip_missing=True
        )
        metrics.short_stats = self._trade_stats(
            [t for t in trades if t.direction == "short"], skip_missing=True
        )

        # ---- Equity / return
        equity = [p.equity for p in equity_curve] if equity_curve else []
        metrics.equity_series = equity
        metrics.n_bars = len(equity)
        final = equity[-1] if equity else initial_capital
        metrics.final_equity = final
        metrics.net_pnl = final - initial_capital
        metrics.net_return = (final / initial_capital - 1) if initial_capital else 0.0

        # ---- Risk metrics
        dd = self._max_drawdown(equity)
        metrics.max_drawdown = dd

        returns = self._bar_returns(equity)
        metrics.volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        metrics.sharpe_ratio = self._sharpe(returns)
        metrics.sortino_ratio = self._sortino(returns)

        if dd > 0:
            metrics.calmar_ratio = metrics.net_return / dd
            metrics.recovery_factor = abs(metrics.net_return) / dd if metrics.net_return != 0 else 0.0

        # ---- Warnings
        t = metrics.trades
        if t.total_trades < 30:
            metrics.warnings.append("LOW SAMPLE SIZE")
        if dd > 0.30:
            metrics.warnings.append("HIGH DRAWDOWN")
        if t.liquidations > 0:
            metrics.warnings.append("LIQUIDATIONS OCCURRED")

        return metrics

    # ------------------------------------------------------- trade statistics
    def _trade_stats(self, trades: List[Position], skip_missing: bool = False) -> TradeStats:
        stats = TradeStats()
        closed = [t for t in trades if t.exit_price is not None]
        if not closed:
            return stats
        if skip_missing and len(closed) < 10:
            return stats  # avoid noisy per-direction stats

        pnls = np.array([t.net_pnl(t.exit_price) for t in closed])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        stats.total_trades = len(closed)
        stats.winning_trades = len(wins)
        stats.losing_trades = len(losses)
        stats.win_rate = len(wins) / len(closed) if closed else 0.0

        gross_profit = wins.sum() if len(wins) else 0.0
        gross_loss = -losses.sum() if len(losses) else 0.0
        stats.total_profit = gross_profit
        stats.total_loss = gross_loss
        stats.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0.0
        )

        stats.expectancy = float(pnls.mean()) if len(pnls) else 0.0
        stats.avg_win = float(wins.mean()) if len(wins) else 0.0
        stats.avg_loss = float(losses.mean()) if len(losses) else 0.0
        stats.largest_win = float(wins.max()) if len(wins) else 0.0
        stats.largest_loss = float(losses.min()) if len(losses) else 0.0
        if stats.avg_loss != 0:
            stats.win_loss_ratio = abs(stats.avg_win / stats.avg_loss)

        holds = np.array([
            (t.exit_time - t.entry_time) for t in closed if t.exit_time and t.entry_time
        ], dtype=float)
        if len(holds):
            stats.avg_holding_period = float(holds.mean() / 3_600_000.0)  # hours

        # Streaks
        win_flags = pnls > 0
        best_w, best_l, cur_w, cur_l = 0, 0, 0, 0
        for w in win_flags:
            if w:
                cur_w += 1; cur_l = 0
                best_w = max(best_w, cur_w)
            else:
                cur_l += 1; cur_w = 0
                best_l = max(best_l, cur_l)
        stats.longest_win_streak = best_w
        stats.longest_loss_streak = best_l

        # Costs
        stats.total_fees = sum(t.entry_fee + t.exit_fee for t in closed)
        stats.total_slippage = sum(t.slippage for t in closed)
        stats.total_funding = sum(t.funding for t in closed)
        stats.liquidations = sum(1 for t in closed if t.liquidated)

        return stats

    # ------------------------------------------------------------ risk helpers
    def _max_drawdown(self, equity: List[float]) -> float:
        if not equity:
            return 0.0
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

    def _bar_returns(self, equity: List[float]) -> np.ndarray:
        if len(equity) < 2:
            return np.array([])
        eq = np.asarray(equity, dtype=float)
        prev = eq[:-1]
        valid = (prev > 0) & (eq[1:] > 0)
        if not valid.any():
            return np.array([])
        # Continuously compounded log returns eliminate arithmetic compounding distortion
        return np.log(eq[1:][valid] / prev[valid])

    def _sharpe(self, returns: np.ndarray) -> float:
        if len(returns) < 2:
            return 0.0
        std = returns.std(ddof=1)
        if std == 0 or np.isnan(std):
            return 0.0
        excess = returns.mean() - self.risk_free_rate / self.annualization
        return float(excess / std * math.sqrt(self.annualization))

    def _sortino(self, returns: np.ndarray) -> float:
        if len(returns) < 2:
            return 0.0
        downside = returns[returns < 0]
        if len(downside) == 0:
            return float("inf") if returns.mean() > 0 else 0.0
        dd = np.sqrt((downside ** 2).mean())
        if dd == 0:
            return 0.0
        excess = returns.mean() - self.risk_free_rate / self.annualization
        return float(excess / dd * math.sqrt(self.annualization))