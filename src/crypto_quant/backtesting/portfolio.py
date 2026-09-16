"""Portfolio accounting for backtesting.

Tracks cash, positions, realized/unrealized PnL, and costs. Supports spot
(no leverage, long-only implied by engine) and isolated-margin futures.

Accounting rules
----------------
- Spot entry: cash -= entry_notional + entry_fee.
- Futures entry: cash -= margin (notional/leverage) + entry_fee.
- Unrealized PnL = (mark - entry) * qty for long, (entry - mark) * qty for short.
- On close: cash += released margin + realized_pnl - exit_fee - funding.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..logging_config import get_logger

logger = get_logger("backtest")


@dataclass
class Position:
    """A single open (or just-closed) position."""

    symbol: str
    direction: str                 # 'long' | 'short'
    market_type: str               # 'spot' | 'futures'
    quantity: float
    entry_price: float
    entry_time: int                # unix ms
    entry_bar: int
    stop_loss: Optional[float]
    take_profit: Optional[float]
    leverage: int = 1
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    funding: float = 0.0
    slippage: float = 0.0
    # fill state
    exit_price: Optional[float] = None
    exit_time: Optional[int] = None
    exit_bar: Optional[int] = None
    exit_reason: Optional[str] = None
    # liquidation
    liquidated: bool = False
    # Dynamic stop management (Strategy #8)
    initial_stop: Optional[float] = None
    active_stop: Optional[float] = None
    risk_r: Optional[float] = None
    breakeven_active: bool = False
    extra_state: Dict[str, Any] = field(default_factory=dict)

    @property
    def effective_stop(self) -> Optional[float]:
        """Stop used for exit checks (dynamic active_stop if set, else initial)."""
        return self.active_stop if self.active_stop is not None else self.stop_loss

    @property
    def notional(self) -> float:
        return self.entry_price * self.quantity

    @property
    def margin_used(self) -> float:
        if self.market_type == "futures":
            return self.notional / self.leverage
        return self.notional

    def unrealized_pnl(self, price: float) -> float:
        if self.direction == "long":
            return (price - self.entry_price) * self.quantity
        return (self.entry_price - price) * self.quantity

    def gross_pnl(self, exit_price: float) -> float:
        if self.direction == "long":
            return (exit_price - self.entry_price) * self.quantity
        return (self.entry_price - exit_price) * self.quantity

    def net_pnl(self, exit_price: float) -> float:
        return self.gross_pnl(exit_price) - self.entry_fee - self.exit_fee - self.funding

    @property
    def return_pct(self) -> float:
        """Return on margin used."""
        if self.exit_price is None or self.margin_used == 0:
            return 0.0
        return self.net_pnl(self.exit_price) / self.margin_used

    def to_dict(self) -> Dict[str, object]:
        """Serializable trade record (schema-compatible with Trade model)."""
        return {
            "symbol": self.symbol,
            "market_type": self.market_type,
            "direction": self.direction,
            "quantity": self.quantity,
            "leverage": self.leverage,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "fees": self.entry_fee + self.exit_fee,
            "slippage": self.slippage,
            "funding_cost": self.funding,
            "gross_pnl": self.gross_pnl(self.exit_price) if self.exit_price else None,
            "net_pnl": self.net_pnl(self.exit_price) if self.exit_price else None,
            "return_pct": self.return_pct,
            "exit_reason": self.exit_reason,
            "liquidated": self.liquidated,
        }


@dataclass
class EquityPoint:
    """Equity snapshot at one bar's close."""

    time: int
    equity: float
    cash: float
    unrealized: float
    n_positions: int


class Portfolio:
    """Tracks capital, positions, and equity through a backtest."""

    def __init__(self, initial_capital: float = 1000.0):
        self.initial_capital = float(initial_capital)
        self.cash = self.initial_capital
        self.open_positions: List[Position] = []
        self.closed_trades: List[Position] = []
        self.equity_curve: List[EquityPoint] = []
        self.n_liquidations = 0

    # ------------------------------------------------------------ equity
    @property
    def unrealized(self) -> float:
        """Sum of unrealized PnL across open positions at last mark."""
        return sum(p.unrealized_pnl(p._last_mark) for p in self.open_positions if hasattr(p, "_last_mark"))

    @property
    def equity(self) -> float:
        return self.cash + self.unrealized

    def mark_to_market(self, price: float) -> None:
        """Record the latest price used to value open positions."""
        for p in self.open_positions:
            p._last_mark = price

    def record_snapshot(self, time: int) -> None:
        self.equity_curve.append(EquityPoint(
            time=time, equity=self.equity, cash=self.cash,
            unrealized=self.unrealized, n_positions=len(self.open_positions),
        ))

    # ------------------------------------------------------------ open/close
    def can_open(self, max_positions: int) -> bool:
        return len(self.open_positions) < max_positions

    def open_position(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        entry_price: float,
        entry_time: int,
        entry_bar: int,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        leverage: int,
        market_type: str,
        fee: float,
    ) -> Position:
        """Open a position, deducting margin/capital and fees from cash."""
        pos = Position(
            symbol=symbol, direction=direction, market_type=market_type,
            quantity=quantity, entry_price=entry_price, entry_time=entry_time,
            entry_bar=entry_bar, stop_loss=stop_loss, take_profit=take_profit,
            leverage=leverage, entry_fee=fee,
        )
        # Cost of the position
        if market_type == "futures":
            cost = pos.margin_used
        else:
            cost = pos.notional
        if self.cash < cost + fee:
            raise ValueError(
                f"Insufficient cash: need {cost + fee:.2f}, have {self.cash:.2f}"
            )
        self.cash -= cost + fee
        pos._last_mark = entry_price
        self.open_positions.append(pos)
        return pos

    def close_position(
        self,
        pos: Position,
        exit_price: float,
        exit_time: int,
        exit_bar: int,
        reason: str,
        exit_fee: float,
        funding: float = 0.0,
    ) -> Position:
        """Close a position, releasing margin and booking PnL."""
        pos.exit_price = exit_price
        pos.exit_time = exit_time
        pos.exit_bar = exit_bar
        pos.exit_reason = reason
        pos.exit_fee = exit_fee
        pos.funding = funding

        gross = pos.gross_pnl(exit_price)
        # Release margin + realized pnl
        self.cash += pos.margin_used + gross - exit_fee - funding

        self.open_positions.remove(pos)
        self.closed_trades.append(pos)
        if pos.liquidated:
            self.n_liquidations += 1
        return pos

    # ------------------------------------------------------------ summary
    @property
    def n_open(self) -> int:
        return len(self.open_positions)

    @property
    def n_closed(self) -> int:
        return len(self.closed_trades)

    def summary(self) -> Dict[str, float]:
        return {
            "initial_capital": self.initial_capital,
            "final_cash": self.cash,
            "final_equity": self.equity,
            "net_pnl": self.equity - self.initial_capital,
            "return_pct": (self.equity / self.initial_capital - 1) * 100.0
            if self.initial_capital else 0.0,
            "n_open": self.n_open,
            "n_closed": self.n_closed,
            "n_liquidations": self.n_liquidations,
        }