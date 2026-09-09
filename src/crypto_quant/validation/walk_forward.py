"""Walk-forward validation.

Evaluates a strategy across rolling train/test windows:
    Train 2020-2022 -> Test 2023
    Train 2021-2023 -> Test 2024
    ...

For each window the strategy is backtested in-sample (train) and out-of-sample
(test). Reports:
- in-sample performance
- out-of-sample performance
- performance degradation (OOS vs IS)
- parameter stability (when re-optimizing per window)

Walk-forward answers: does this strategy degrade gracefully out of sample, or
was its backtest return an artifact of overfitting?
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ..backtesting import BacktestConfig, BacktestEngine, ExecutionConfig
from ..logging_config import get_logger
from ..strategies import create_strategy, strategy_param_grid
from ..strategies.base import BaseStrategy
from .optimize_window import window_best_params  # lightweight helper

logger = get_logger("validation")


@dataclass
class WalkForwardWindow:
    """A single train/test window (dates inclusive at end)."""

    train_start: str
    train_end: str
    test_start: str
    test_end: str

    def to_dict(self) -> dict:
        return {
            "train_start": self.train_start, "train_end": self.train_end,
            "test_start": self.test_start, "test_end": self.test_end,
        }


@dataclass
class WindowResult:
    """Results for one walk-forward window."""

    window: WalkForwardWindow
    is_metrics: Dict[str, Any]
    oos_metrics: Dict[str, Any]
    params: Dict[str, Any]
    win_rate_degradation: float
    return_degradation: float

    def to_dict(self) -> dict:
        return {
            "window": self.window.to_dict(),
            "in_sample": {k: self.is_metrics.get(k) for k in
                          ("win_rate", "net_return", "profit_factor", "max_drawdown", "total_trades")},
            "out_of_sample": {k: self.oos_metrics.get(k) for k in
                              ("win_rate", "net_return", "profit_factor", "max_drawdown", "total_trades")},
            "params": self.params,
            "win_rate_degradation": self.win_rate_degradation,
            "return_degradation": self.return_degradation,
        }


@dataclass
class WalkForwardReport:
    """Aggregate walk-forward result."""

    strategy_type: str
    params: Dict[str, Any]
    windows: List[WindowResult] = field(default_factory=list)

    @property
    def mean_oos_win_rate(self) -> Optional[float]:
        vals = [w.oos_metrics.get("win_rate", 0.0) for w in self.windows]
        return sum(vals) / len(vals) if vals else None

    @property
    def mean_oos_return(self) -> Optional[float]:
        vals = [w.oos_metrics.get("net_return", 0.0) for w in self.windows]
        return sum(vals) / len(vals) if vals else None

    @property
    def mean_degradation(self) -> Optional[float]:
        vals = [w.win_rate_degradation for w in self.windows]
        return sum(vals) / len(vals) if vals else None

    def warnings(self) -> List[str]:
        """Flag concerning patterns."""
        out = []
        if self.mean_degradation is not None and self.mean_degradation < -0.05:
            out.append("POOR OUT-OF-SAMPLE PERFORMANCE (OOS below IS by > 5pp)")
        if len(self.windows) and all(w.oos_metrics.get("total_trades", 0) < 30 for w in self.windows):
            out.append("LOW OOS SAMPLE SIZE")
        return out

    def to_dict(self) -> dict:
        return {
            "strategy_type": self.strategy_type,
            "params": self.params,
            "windows": [w.to_dict() for w in self.windows],
            "mean_oos_win_rate": self.mean_oos_win_rate,
            "mean_oos_return": self.mean_oos_return,
            "mean_win_rate_degradation": self.mean_degradation,
            "warnings": self.warnings(),
        }


def default_walk_forward_windows() -> List[WalkForwardWindow]:
    """The spec's default annual windows (2020-2026)."""
    return [
        WalkForwardWindow("2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        WalkForwardWindow("2021-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        WalkForwardWindow("2022-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
        WalkForwardWindow("2023-01-01", "2025-12-31", "2026-01-01", "2026-09-08"),
    ]


class WalkForwardValidator:
    """Runs walk-forward analysis for a strategy."""

    def __init__(
        self,
        windows: Optional[List[WalkForwardWindow]] = None,
        reoptimize: bool = False,
        search_type: str = "grid",
        max_combinations_per_window: int = 20,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """Initialize validator.

        Args:
            windows: Walk-forward windows (default: annual 2020-2026).
            reoptimize: If True, re-optimize params on each train window;
                otherwise use the fixed params given to validate().
            search_type: Search type for reoptimization.
            max_combinations_per_window: Cap for per-window reoptimization.
            progress_callback: Callback(window_done, total_windows).
        """
        self.windows = windows or default_walk_forward_windows()
        self.reoptimize = reoptimize
        self.search_type = search_type
        self.max_combinations = max_combinations_per_window
        self._progress = progress_callback or (lambda d, t: None)

    # ------------------------------------------------------------ main API
    def validate(
        self,
        df: pd.DataFrame,
        strategy_type: str,
        params: Dict[str, Any],
        symbol: str = "TEST",
        market_type: str = "spot",
        timeframe: str = "1h",
    ) -> WalkForwardReport:
        """Run walk-forward validation.

        Args:
            df: OHLCV DataFrame (must span all windows).
            strategy_type: Registered strategy type.
            params: Fixed params (used when not reoptimizing).
            symbol, market_type, timeframe: Backtest context.

        Returns:
            WalkForwardReport.
        """
        report = WalkForwardReport(strategy_type=strategy_type, params=dict(params))

        for i, w in enumerate(self.windows):
            train = self._slice(df, w.train_start, w.train_end)
            test = self._slice(df, w.test_start, w.test_end)

            if len(train) < 30 or len(test) < 30:
                logger.warning("Window %s has too little data; skipping", w.to_dict())
                self._progress(i + 1, len(self.windows))
                continue

            window_params = params
            if self.reoptimize:
                grid = strategy_param_grid(strategy_type)
                best = window_best_params(
                    train, strategy_type, grid, symbol=symbol,
                    market_type=market_type, timeframe=timeframe,
                    max_combinations=self.max_combinations,
                    search_type=self.search_type,
                )
                window_params = best["params"] if best else params

            is_metrics = self._backtest_metrics(train, strategy_type, window_params, symbol, market_type, timeframe)
            oos_metrics = self._backtest_metrics(test, strategy_type, window_params, symbol, market_type, timeframe)

            wr_degrad = oos_metrics.get("win_rate", 0.0) - is_metrics.get("win_rate", 0.0)
            ret_degrad = oos_metrics.get("net_return", 0.0) - is_metrics.get("net_return", 0.0)

            report.windows.append(WindowResult(
                window=w,
                is_metrics=is_metrics,
                oos_metrics=oos_metrics,
                params=window_params,
                win_rate_degradation=wr_degrad,
                return_degradation=ret_degrad,
            ))
            self._progress(i + 1, len(self.windows))

        return report

    # ------------------------------------------------------------ helpers
    def _slice(self, df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        t = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        start_dt = pd.Timestamp(start, tz="UTC")
        end_dt = pd.Timestamp(end, tz="UTC")
        return df[(t >= start_dt) & (t <= end_dt)].copy()

    def _backtest_metrics(self, df, strategy_type, params, symbol, market_type, timeframe) -> Dict[str, Any]:
        strategy = create_strategy(strategy_type, params=params)
        cfg = BacktestConfig(
            market_type=market_type, timeframe=timeframe,
            execution=ExecutionConfig(market_type=market_type),
            min_bars=30,
        )
        result = BacktestEngine(cfg).run(strategy, df, symbol=symbol)
        return result.metrics.to_dict()