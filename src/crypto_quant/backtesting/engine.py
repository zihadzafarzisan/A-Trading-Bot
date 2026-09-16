"""Event-driven backtesting engine.

Simulates a strategy over OHLCV data one bar at a time, with:
- signal at bar i -> entry at bar i+1 open (execution delay >= 1 bar)
- intra-bar stop/take exits using the current bar's high/low
- conservative same-bar rule: stop is assumed hit before take profit
- fees, slippage, funding, leverage, and isolated-margin liquidation checks
- risk-based position sizing (risk per trade over stop distance)

Guarantees
----------
- Never uses future data: every signal derives from shifted indicators, and
  entries fill on the next bar's open.
- Reports execution assumptions in every result.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from ..logging_config import get_logger
from ..risk import PositionSizer, RiskLimits
from ..strategies.base import BaseStrategy, Direction, Signal
from .execution import ExecutionConfig, ExecutionModel
from .metrics import BacktestMetrics, MetricsCalculator
from .portfolio import Portfolio, Position

logger = get_logger("backtest")


@dataclass
class BacktestResult:
    """Full result of a backtest run."""

    metrics: BacktestMetrics
    trades: List[Dict[str, object]]
    equity_curve: List[Dict[str, object]]
    config: Dict[str, object]
    strategy_metadata: Dict[str, object]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "metrics": self.metrics.to_dict(),
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "config": self.config,
            "strategy": self.strategy_metadata,
            "warnings": self.warnings,
        }


@dataclass
class BacktestConfig:
    """Top-level backtest configuration."""

    initial_capital: float = 1000.0
    risk_per_trade: float = 0.01        # 1% of equity exposed to stop risk
    max_open_positions: int = 3
    max_position_pct: float = 0.5       # cap position notional at 50% of equity
    max_leverage: float = 5.0
    market_type: str = "spot"           # 'spot' | 'futures'
    timeframe: str = "1h"
    # execution assumptions
    execution: ExecutionConfig = field(default_factory=lambda: ExecutionConfig())
    # optional time-based exit
    max_holding_bars: Optional[int] = None
    # stop/take behavior
    honor_take_profit: bool = True
    # futures
    funding_bars: int = 8               # funding interval in bars (8h)
    # guardrail
    min_bars: int = 30                  # refuse backtests shorter than this

    def __post_init__(self):
        if self.execution.market_type != self.market_type:
            self.execution.market_type = self.market_type


class BacktestEngine:
    """Event-driven bar-by-bar backtest engine."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.execution = ExecutionModel(self.config.execution)
        self.metrics_calculator = MetricsCalculator(timeframe=self.config.timeframe)
        # Shared risk-based position sizing (same as paper/live trading)
        self.sizer = PositionSizer(RiskLimits(
            risk_per_trade=self.config.risk_per_trade,
            max_position_pct=self.config.max_position_pct,
            max_leverage=self.config.max_leverage,
        ))

    # ------------------------------------------------------------ main API
    def run(self, strategy: BaseStrategy, df: pd.DataFrame, symbol: str = "TEST") -> BacktestResult:
        """Run a strategy over a prepared OHLCV DataFrame."""
        if df is None or len(df) < self.config.min_bars:
            raise ValueError(
                f"Not enough data to backtest (need >= {self.config.min_bars} bars, got "
                f"{0 if df is None else len(df)})"
            )

        prepared = strategy.setup(df)
        portfolio = Portfolio(self.config.initial_capital)
        delay = max(1, self.config.execution.execution_delay_bars)
        pending: List[Signal] = []

        # Data arrays for speed
        opens = prepared["open"].to_numpy()
        highs = prepared["high"].to_numpy()
        lows = prepared["low"].to_numpy()
        closes = prepared["close"].to_numpy()
        times = prepared["timestamp"].to_numpy()

        n = len(prepared)

        for i in range(n):
            # 0) Per-bar dynamic stop management (Strategy #8 hook).
            #    Called AFTER close of bar i; effective one bar delayed.
            for pos in list(portfolio.open_positions):
                try:
                    strategy.update_stop(prepared, i, pos)
                except Exception as exc:
                    logger.error("Stop management failed at bar %d: %s. Aborting backtest.", i, exc)
                    raise RuntimeError(f"Dynamic risk failure at bar {i}") from exc

            # 1) Fill pending entries at this bar's open
            if pending:
                for sig in list(pending):
                    if portfolio.can_open(self.config.max_open_positions):
                        self._try_enter(portfolio, sig, opens[i], times[i], i, symbol, strategy, prepared)
                    pending.remove(sig)

            # 2) Manage open positions with this bar's high/low
            self._manage_exits(portfolio, highs[i], lows[i], opens[i], closes[i],
                               times[i], i)

            # 3) Mark to market + equity snapshot
            portfolio.mark_to_market(closes[i])
            portfolio.record_snapshot(int(times[i]))

            # 4) Generate signal for a future bar (delay bars ahead)
            if i + delay < n:
                try:
                    sig = strategy.generate_signal(prepared, i)
                except Exception as exc:
                    logger.debug("Signal error at bar %d: %s", i, exc)
                    sig = Signal(direction=Direction.NONE)
                if sig.is_active:
                    pending = [sig]

        # Close any still-open positions at the last close (end of test)
        last_price = closes[-1]
        last_time = int(times[-1])
        last_bar = n - 1
        for pos in list(portfolio.open_positions):
            fee = self.execution.fee_cost(pos.notional)
            holding_bars = max(0, last_bar - pos.entry_bar)
            funding = self.execution.funding_cost(
                pos.notional, holding_bars, self.config.funding_bars
            )
            portfolio.close_position(
                pos, last_price, last_time, last_bar, "end_of_test",
                exit_fee=fee, funding=funding,
            )
        portfolio.mark_to_market(last_price)
        portfolio.record_snapshot(last_time)

        # Metrics
        metrics = self.metrics_calculator.compute(
            portfolio.closed_trades, portfolio.equity_curve, self.config.initial_capital
        )

        trades = [t.to_dict() for t in portfolio.closed_trades]
        equity_curve = [{
            "time": p.time, "equity": round(p.equity, 4),
            "cash": round(p.cash, 4), "unrealized": round(p.unrealized, 4),
            "n_positions": p.n_positions,
        } for p in portfolio.equity_curve]

        return BacktestResult(
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            config=self._config_dict(),
            strategy_metadata=strategy.metadata(),
            warnings=list(metrics.warnings),
        )

    # ------------------------------------------------------------ entry
    def _try_enter(
        self,
        portfolio: Portfolio,
        sig: Signal,
        bar_open: float,
        bar_time: int,
        bar_index: int,
        symbol: str,
        strategy: BaseStrategy,
        df: pd.DataFrame,
    ) -> None:
        """Size and open a position from a signal."""
        direction = sig.direction
        if direction == Direction.NONE:
            return

        # Market-type restrictions
        if self.config.market_type == "spot" and direction == Direction.SHORT:
            logger.debug("Skipping short signal in spot market")
            return

        raw_fill = float(bar_open)
        entry_price = self.execution.entry_fill(raw_fill, direction.value, raw_fill)
        entry_slippage = abs(entry_price - raw_fill)

        # Retrieve the original signal bar index (default to bar_index-1 if missing)
        signal_bar_index = sig.meta.get("signal_bar", max(0, bar_index - 1))

        # Re-anchor stop/take profit via the strategy given the actual fill price
        stop_price, take_profit, extra_state = strategy.prepare_fill(
            df, signal_bar_index, direction, entry_price, sig
        )

        if stop_price is None:
            return  # cannot risk-size without a stop

        leverage = self._resolve_leverage(direction)
        try:
            sizing = self.sizer.size_position(
                equity=portfolio.equity,
                entry_price=entry_price,
                stop_price=stop_price,
                direction=direction.value,
                market_type=self.config.market_type,
                leverage=leverage,
            )
        except ValueError:
            return

        notional = sizing.notional
        quantity = sizing.quantity
        fee = self.execution.fee_cost(notional)

        # Insufficient cash guard
        cost = notional if self.config.market_type == "spot" else notional / leverage
        if portfolio.cash < cost + fee:
            logger.debug("Skipping entry: insufficient cash %.2f < %.2f", portfolio.cash, cost + fee)
            return

        try:
            pos = portfolio.open_position(
                symbol=symbol,
                direction=direction.value,
                quantity=quantity,
                entry_price=entry_price,
                entry_time=bar_time,
                entry_bar=bar_index,
                stop_loss=stop_price,
                take_profit=take_profit,
                leverage=sizing.leverage,
                market_type=self.config.market_type,
                fee=fee,
            )
            pos.slippage = entry_slippage
            pos.initial_stop = float(stop_price)
            pos.active_stop = pos.initial_stop
            pos.extra_state = extra_state
        except ValueError:
            logger.debug("Entry rejected by portfolio accounting")

    def _resolve_leverage(self, direction: Direction) -> int:
        """Leverage to use (futures only; spot is always 1x)."""
        if self.config.market_type == "futures":
            return max(1, min(int(self.config.max_leverage), 5))
        return 1

    # ------------------------------------------------------------ exits
    def _manage_exits(
        self,
        portfolio: Portfolio,
        high: float,
        low: float,
        open_price: float,
        close: float,
        bar_time: int,
        bar_index: int,
    ) -> None:
        """Check stop/take exits for open positions using current bar range."""
        for pos in list(portfolio.open_positions):
            direction = pos.direction

            # Liquidation check (futures)
            if pos.market_type == "futures":
                liq = self.execution.liquidation_price(
                    pos.entry_price, direction, pos.leverage
                )
                if liq and ((direction == "long" and low <= liq) or (direction == "short" and high >= liq)):
                    fill = liq if (direction == "long" and open_price <= liq) else (
                        liq if (direction == "short" and open_price >= liq) else
                        (low if direction == "long" else high)
                    )
                    pos.liquidated = True
                    self._close(portfolio, pos, fill, bar_time, bar_index, "liquidation")
                    continue

            # Stop loss (uses dynamic active_stop if set)
            hit_sl = (direction == "long" and low <= pos.effective_stop) or \
                     (direction == "short" and high >= pos.effective_stop)
            # Take profit (None take_profit -> no TP level; uncapped right tail, Strategy #9 V1.1)
            hit_tp = pos.take_profit is not None and (
                (direction == "long" and high >= pos.take_profit) or
                (direction == "short" and low <= pos.take_profit)
            )

            # Use active_stop for exit fill (dynamic position management)
            stop_exit = pos.effective_stop if pos.active_stop is not None else pos.stop_loss
            if hit_sl and hit_tp:
                # Conservative: stop assumed first
                self._close(portfolio, pos, stop_exit, bar_time, bar_index, "sl")
            elif hit_sl:
                # Gap handling: fill at the worse of open vs stop
                fill = min(open_price, stop_exit) if direction == "long" else max(open_price, stop_exit)
                self._close(portfolio, pos, fill, bar_time, bar_index, "sl")
            elif hit_tp and self.config.honor_take_profit:
                fill = max(open_price, pos.take_profit) if direction == "long" else min(open_price, pos.take_profit)
                self._close(portfolio, pos, fill, bar_time, bar_index, "tp")

            # Time-based exit
            elif self.config.max_holding_bars is not None:
                if bar_index - pos.entry_bar >= self.config.max_holding_bars:
                    self._close(portfolio, pos, close, bar_time, bar_index, "time")

    def _close(
        self, portfolio: Portfolio, pos: Position, fill: float,
        bar_time: int, bar_index: int, reason: str,
    ) -> None:
        """Close a position with fees and funding."""
        exit_notional = fill * pos.quantity
        fee = self.execution.fee_cost(exit_notional)
        holding_bars = max(0, bar_index - pos.entry_bar)
        funding = self.execution.funding_cost(
            pos.notional, holding_bars, self.config.funding_bars
        )
        portfolio.close_position(
            pos, fill, bar_time, bar_index, reason,
            exit_fee=fee, funding=funding,
        )

    # ------------------------------------------------------------ misc
    def _config_dict(self) -> Dict[str, object]:
        c = self.config
        return {
            "initial_capital": c.initial_capital,
            "risk_per_trade": c.risk_per_trade,
            "max_open_positions": c.max_open_positions,
            "max_position_pct": c.max_position_pct,
            "max_leverage": c.max_leverage,
            "market_type": c.market_type,
            "timeframe": c.timeframe,
            "max_holding_bars": c.max_holding_bars,
            "execution": self.config.execution.to_dict(),
        }