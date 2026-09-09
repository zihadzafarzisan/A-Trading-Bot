"""Position sizing.

Implements the specification's risk-based sizing:
    risk_amount = equity * risk_per_trade        (e.g. $1,000 * 1% = $10)
    position_notional = risk_amount / stop_distance_pct   (e.g. $10 / 2% = $500)

subject to caps: maximum position % of equity, maximum leverage (futures), and
portfolio exposure. Also provides trailing-stop helpers (spec #14).
"""

from dataclasses import dataclass
from typing import Optional

from ..logging_config import get_logger
from .limits import RiskLimits

logger = get_logger("risk")


@dataclass
class SizingResult:
    """Result of a position-size calculation."""

    notional: float
    quantity: float
    risk_amount: float
    stop_distance_pct: float
    leverage: int
    capped_by: str          # which cap bound the size ('risk', 'position_pct', 'leverage', 'exposure')

    def to_dict(self) -> dict:
        return {
            "notional": round(self.notional, 6),
            "quantity": round(self.quantity, 8),
            "risk_amount": round(self.risk_amount, 2),
            "stop_distance_pct": round(self.stop_distance_pct, 5),
            "leverage": self.leverage,
            "capped_by": self.capped_by,
        }


class PositionSizer:
    """Computes risk-based position sizes with configurable caps."""

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()

    # ------------------------------------------------------------ main API
    def size_position(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        direction: str,              # 'long' | 'short'
        market_type: str = "spot",   # 'spot' | 'futures'
        leverage: Optional[int] = None,
    ) -> SizingResult:
        """Compute position notional and quantity for a proposed entry.

        Args:
            equity: Current account equity.
            entry_price: Planned entry price.
            stop_price: Stop-loss level.
            direction: Trade direction.
            market_type: 'spot' or 'futures'.
            leverage: Requested leverage (futures only; capped by limits).

        Returns:
            SizingResult. Raises ValueError if the stop is invalid.
        """
        if equity <= 0:
            raise ValueError("Equity must be positive")
        if entry_price <= 0:
            raise ValueError("Entry price must be positive")

        self._validate_stop(entry_price, stop_price, direction)

        # Risk-based core size
        risk_amount = equity * self.limits.risk_per_trade
        stop_dist = abs(entry_price - stop_price) / entry_price
        if stop_dist <= 0:
            raise ValueError("Stop loss must differ from entry price")

        notional = risk_amount / stop_dist
        capped_by = "risk"

        # Cap 1: max single-position notional as % of equity
        pos_cap = equity * self.limits.max_position_pct
        if notional > pos_cap:
            notional = pos_cap
            capped_by = "position_pct"

        # Cap 2: leverage (futures)
        lev = 1
        if market_type == "futures":
            lev = self._resolve_leverage(leverage)
            lev_cap = equity * lev
            if notional > lev_cap:
                notional = lev_cap
                capped_by = "leverage"

        # Cap 3: portfolio exposure (assumes this is the only/additional position)
        exposure_cap = equity * self.limits.max_portfolio_exposure
        if notional > exposure_cap:
            notional = exposure_cap
            capped_by = "exposure"

        quantity = notional / entry_price
        return SizingResult(
            notional=notional, quantity=quantity, risk_amount=risk_amount,
            stop_distance_pct=stop_dist, leverage=lev, capped_by=capped_by,
        )

    # ------------------------------------------------------------ helpers
    def _resolve_leverage(self, requested: Optional[int]) -> int:
        if requested is None:
            return max(1, int(self.limits.max_leverage))
        return max(1, min(int(requested), int(self.limits.max_leverage)))

    @staticmethod
    def _validate_stop(entry: float, stop: float, direction: str) -> None:
        if direction == "long" and stop >= entry:
            raise ValueError(f"Long stop ({stop}) must be below entry ({entry})")
        if direction == "short" and stop <= entry:
            raise ValueError(f"Short stop ({stop}) must be above entry ({entry})")


class TrailingStop:
    """Trailing stop-loss calculator (spec #14: trailing exit)."""

    def __init__(self, activation_pct: float = 0.0, trail_pct: float = 0.02):
        """Initialize trailing stop.

        Args:
            activation_pct: Profit (fraction of entry) before the trail starts.
            trail_pct: Distance the stop trails below the peak (fraction).
        """
        self.activation_pct = activation_pct
        self.trail_pct = trail_pct

    def update(self, entry: float, direction: str, current_price: float, current_stop: Optional[float]) -> float:
        """Return the new trailing stop given the current price.

        Long: stop rises with price once price exceeds activation level.
        Short: stop falls with price once price drops below activation level.
        """
        if direction == "long":
            activation = entry * (1 + self.activation_pct)
            if current_price < activation:
                return current_stop if current_stop is not None else entry * (1 - self.trail_pct)
            new_stop = current_price * (1 - self.trail_pct)
            return max(new_stop, current_stop if current_stop is not None else 0.0)
        else:
            activation = entry * (1 - self.activation_pct)
            if current_price > activation:
                return current_stop if current_stop is not None else entry * (1 + self.trail_pct)
            new_stop = current_price * (1 + self.trail_pct)
            return min(new_stop, current_stop if current_stop is not None else float("inf"))