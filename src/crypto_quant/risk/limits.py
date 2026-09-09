"""Risk limits and risk events.

Central, configurable limits enforced by the RiskManager. Defaults follow the
specification: $1000 starting capital, 1% risk per trade, 3 max open positions,
5x max leverage. Every limit is configurable and every violation is recorded as
a RiskEvent (never silently swallowed).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class RiskLimits:
    """Portfolio risk configuration."""

    # Position sizing
    starting_capital: float = 1000.0
    risk_per_trade: float = 0.01          # 1% of equity exposed to stop risk
    max_open_positions: int = 3
    max_position_pct: float = 0.5         # cap single-position notional at 50% of equity
    max_leverage: float = 5.0

    # Exposure
    max_portfolio_exposure: float = 1.0   # total notional <= 100% of equity
    max_exposure_per_asset: float = 0.5   # per-asset notional <= 50% of equity
    max_correlated_exposure: float = 0.8  # reserved for correlation-aware caps

    # Loss limits (absolute dollars by default)
    daily_loss_limit: float = 100.0       # stop new trades after -$100/day
    weekly_loss_limit: float = 300.0
    max_drawdown_pct: float = 0.25        # 25% peak-to-trough equity drop

    # Kill switch
    emergency_stop: bool = False          # when True: no new trades, optionally close

    def to_dict(self) -> Dict[str, Any]:
        return {
            "starting_capital": self.starting_capital,
            "risk_per_trade": self.risk_per_trade,
            "max_open_positions": self.max_open_positions,
            "max_position_pct": self.max_position_pct,
            "max_leverage": self.max_leverage,
            "max_portfolio_exposure": self.max_portfolio_exposure,
            "max_exposure_per_asset": self.max_exposure_per_asset,
            "daily_loss_limit": self.daily_loss_limit,
            "weekly_loss_limit": self.weekly_loss_limit,
            "max_drawdown_pct": self.max_drawdown_pct,
            "emergency_stop": self.emergency_stop,
        }

    @classmethod
    def from_config(cls, risk_config) -> "RiskLimits":
        """Build RiskLimits from the application RiskConfig model."""
        return cls(
            starting_capital=float(risk_config.starting_capital),
            risk_per_trade=float(risk_config.risk_per_trade),
            max_open_positions=int(risk_config.max_open_positions),
            max_position_pct=float(risk_config.max_position_pct),
            max_leverage=int(risk_config.max_leverage),
            daily_loss_limit=float(risk_config.daily_loss_limit),
            weekly_loss_limit=float(risk_config.weekly_loss_limit),
            max_drawdown_pct=float(risk_config.max_drawdown_pct),
        )


class RiskEvent:
    """A single recorded risk event (violation or warning)."""

    def __init__(
        self,
        event_type: str,
        severity: str,          # 'warning' | 'critical'
        message: str,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ):
        self.event_type = event_type
        self.severity = severity
        self.message = message
        self.context = context or {}
        self.timestamp = timestamp or datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "severity": self.severity,
            "message": self.message,
            "context": self.context,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<RiskEvent {self.severity}:{self.event_type}: {self.message}>"


# Standard event types
EVENT_EMERGENCY_STOP = "emergency_stop"
EVENT_MAX_POSITIONS = "max_positions"
EVENT_DAILY_LOSS = "daily_loss"
EVENT_WEEKLY_LOSS = "weekly_loss"
EVENT_MAX_DRAWDOWN = "max_drawdown"
EVENT_EXPOSURE = "exposure"
EVENT_LEVERAGE = "leverage"
EVENT_INVALID_ORDER = "invalid_order"