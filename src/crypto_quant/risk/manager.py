"""Risk manager.

The central gatekeeper between signal generation and execution. Every proposed
trade passes through check_entry() which enforces (spec #31, #32, #36):

    signal -> risk validation -> position limits -> balance -> leverage
           -> order validation -> exchange

If ANY safety check fails the trade is REJECTED and a RiskEvent is logged.
Also tracks daily/weekly realized loss and peak-to-trough drawdown, and
supports an emergency kill switch that stops new trades (optionally closes
positions).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..logging_config import get_logger
from .limits import (
    EVENT_DAILY_LOSS, EVENT_EMERGENCY_STOP, EVENT_EXPOSURE, EVENT_INVALID_ORDER,
    EVENT_LEVERAGE, EVENT_MAX_DRAWDOWN, EVENT_MAX_POSITIONS, EVENT_WEEKLY_LOSS,
    RiskEvent, RiskLimits,
)
from .position_sizing import PositionSizer, SizingResult

logger = get_logger("risk")


@dataclass
class PortfolioState:
    """Snapshot of portfolio state used by risk checks."""

    equity: float = 1000.0
    cash: float = 1000.0
    open_positions: int = 0
    current_exposure: float = 0.0          # total notional
    realized_pnl_day: float = 0.0          # today's realized pnl (starts at 0)
    realized_pnl_week: float = 0.0
    peak_equity: float = 1000.0


@dataclass
class EntryCheckResult:
    """Outcome of the pre-trade validation chain."""

    allowed: bool
    reasons: List[str] = field(default_factory=list)

    @property
    def is_allowed(self) -> bool:
        return self.allowed

    def to_dict(self) -> dict:
        return {"allowed": self.allowed, "reasons": self.reasons}


class RiskManager:
    """Validates trades against all risk limits and logs risk events."""

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.sizer = PositionSizer(self.limits)
        self.state = PortfolioState(equity=self.limits.starting_capital,
                                    cash=self.limits.starting_capital,
                                    peak_equity=self.limits.starting_capital)
        self.events: List[RiskEvent] = []
        self._shutdown: bool = False

    # ------------------------------------------------------------ main entry gate
    def check_entry(
        self,
        equity: Optional[float] = None,
        open_positions: Optional[int] = None,
        current_exposure: Optional[float] = None,
        proposed_notional: Optional[float] = None,
        leverage: Optional[int] = None,
        symbol: str = "",
    ) -> EntryCheckResult:
        """Run the full pre-trade validation chain.

        Args:
            equity: Current equity (defaults to tracked state).
            open_positions: Number of currently open positions.
            current_exposure: Current total notional exposure.
            proposed_notional: Notional of the proposed trade (if known).
            leverage: Requested leverage (futures).
            symbol: Asset being traded (for exposure-per-asset context).

        Returns:
            EntryCheckResult with reasons for any rejection.
        """
        reasons: List[str] = []
        equity = equity if equity is not None else self.state.equity
        open_positions = open_positions if open_positions is not None else self.state.open_positions

        # 1. Emergency stop (kill switch)
        if self._shutdown or self.limits.emergency_stop:
            reasons.append("EMERGENCY STOP ACTIVE — no new trades")
            self._log(EVENT_EMERGENCY_STOP, "critical",
                      "Trade blocked by emergency stop", {"symbol": symbol})

        # 2. Max open positions
        if open_positions >= self.limits.max_open_positions:
            reasons.append(f"max_open_positions reached ({open_positions})")
            self._log(EVENT_MAX_POSITIONS, "warning",
                      f"Rejected: max positions ({self.limits.max_open_positions}) reached")

        # 3. Daily loss limit
        if self.state.realized_pnl_day <= -self.limits.daily_loss_limit:
            reasons.append(f"daily loss limit hit ({self.state.realized_pnl_day:.2f})")
            self._log(EVENT_DAILY_LOSS, "critical",
                      f"Rejected: daily loss limit reached ({self.state.realized_pnl_day:.2f})")

        # 4. Weekly loss limit
        if self.state.realized_pnl_week <= -self.limits.weekly_loss_limit:
            reasons.append(f"weekly loss limit hit ({self.state.realized_pnl_week:.2f})")
            self._log(EVENT_WEEKLY_LOSS, "critical",
                      f"Rejected: weekly loss limit reached ({self.state.realized_pnl_week:.2f})")

        # 5. Drawdown threshold
        dd = self._current_drawdown(equity)
        if dd > self.limits.max_drawdown_pct:
            reasons.append(f"drawdown {dd:.1%} exceeds limit {self.limits.max_drawdown_pct:.1%}")
            self._log(EVENT_MAX_DRAWDOWN, "critical",
                      f"Rejected: drawdown {dd:.1%} > {self.limits.max_drawdown_pct:.1%}")

        # 6. Exposure caps
        exposure = current_exposure if current_exposure is not None else self.state.current_exposure
        if proposed_notional is not None:
            total = exposure + proposed_notional
            if total > equity * self.limits.max_portfolio_exposure:
                reasons.append(f"portfolio exposure would exceed limit")
                self._log(EVENT_EXPOSURE, "warning",
                          f"Rejected: exposure {total:.2f} > {self.limits.max_portfolio_exposure:.0%} equity")
        elif exposure > equity * self.limits.max_portfolio_exposure:
            reasons.append("portfolio exposure already exceeds limit")
            self._log(EVENT_EXPOSURE, "warning", "Rejected: exposure already over limit")

        # 7. Leverage
        if leverage is not None and leverage > self.limits.max_leverage:
            reasons.append(f"leverage {leverage}x > max {self.limits.max_leverage}x")
            self._log(EVENT_LEVERAGE, "warning",
                      f"Rejected: leverage {leverage}x > {self.limits.max_leverage}x")

        allowed = len(reasons) == 0
        return EntryCheckResult(allowed=allowed, reasons=reasons)

    # ------------------------------------------------------------ sizing
    def size_position(
        self, equity: float, entry_price: float, stop_price: float,
        direction: str, market_type: str = "spot", leverage: Optional[int] = None,
    ) -> SizingResult:
        """Size a position (thin wrapper with risk logging on invalid stops)."""
        try:
            return self.sizer.size_position(equity, entry_price, stop_price,
                                            direction, market_type, leverage)
        except ValueError as exc:
            self._log(EVENT_INVALID_ORDER, "warning", f"Invalid stop/entry: {exc}",
                      {"entry": entry_price, "stop": stop_price, "direction": direction})
            raise

    # ------------------------------------------------------------ lifecycle
    def record_trade_result(self, realized_pnl: float) -> None:
        """Record a realized PnL (updates daily/weekly loss tracking)."""
        self.state.realized_pnl_day += realized_pnl
        self.state.realized_pnl_week += realized_pnl

    def update_equity(self, equity: float) -> None:
        """Update tracked equity and peak equity."""
        self.state.equity = equity
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity

    def _current_drawdown(self, equity: float) -> float:
        peak = max(self.state.peak_equity, equity)
        if peak <= 0:
            return 0.0
        return (peak - equity) / peak

    def emergency_stop(self, close_positions: bool = False) -> None:
        """Engage the kill switch.

        Args:
            close_positions: If True, marks that open positions should be
                closed (executed by the trading layer).
        """
        self._shutdown = True
        self.limits.emergency_stop = True
        self._log(EVENT_EMERGENCY_STOP, "critical",
                  f"EMERGENCY STOP ENGAGED (close_positions={close_positions})")
        logger.critical("EMERGENCY STOP ENGAGED — new trades disabled%s",
                        "; close positions requested" if close_positions else "")

    def release_emergency_stop(self) -> None:
        """Manually release the kill switch (explicit action only)."""
        self._shutdown = False
        self.limits.emergency_stop = False
        logger.warning("Emergency stop manually released")

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown or self.limits.emergency_stop

    # ------------------------------------------------------------ events
    def _log(self, event_type: str, severity: str, message: str,
             context: Optional[Dict[str, Any]] = None) -> None:
        event = RiskEvent(event_type, severity, message, context)
        self.events.append(event)
        log_method = logger.critical if severity == "critical" else logger.warning
        log_method("[%s] %s", event_type, message)

    def recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent risk events (newest first)."""
        return [e.to_dict() for e in self.events[-limit:][::-1]]

    def reset_daily(self) -> None:
        """Reset the daily loss counter (called at day rollover)."""
        self.state.realized_pnl_day = 0.0

    def reset_weekly(self) -> None:
        """Reset the weekly loss counter (called at week rollover)."""
        self.state.realized_pnl_week = 0.0