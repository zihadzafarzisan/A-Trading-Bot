"""Paper broker.

Simulates order execution against live (or replayed) prices. This broker is
STRUCTURALLY INCAPABLE of sending a real order — it only mutates in-memory
state. Paper and live execution are separate classes (spec #56).

Accounting model
----------------
- Cash = available cash (futures margin is tracked by the caller/risk layer).
- Spot buy:  cash -= (price * qty) + fee.
- Spot sell: cash += (price * qty) - fee   (long-only; cannot go short).
- Futures:   opening a position does not change cash; closing adds
              realized PnL - exit fee. Realized PnL = (exit - entry)*qty
              for longs, (entry - exit)*qty for shorts.
- Equity = cash + unrealized PnL.
"""

from typing import Any, Dict, List

from ..logging_config import get_logger
from ..utils.helpers import generate_trade_id
from .broker import Broker, Order, PriceSource

logger = get_logger("trading")


class PaperBroker(Broker):
    """Simulated broker: fills orders at market price + slippage, tracks state."""

    name = "paper"

    def __init__(
        self,
        price_source: PriceSource,
        starting_capital: float = 1000.0,
        fee_rate: float = 0.001,
        slippage: float = 0.0005,
        market_type: str = "spot",
    ):
        self.price_source = price_source
        self.cash = float(starting_capital)
        self.initial_capital = float(starting_capital)
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.market_type = market_type
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.closed_trades: List[Dict[str, Any]] = []
        self.orders: List[Order] = []

    # ------------------------------------------------------------ Broker API
    def get_balance(self) -> float:
        return self.cash

    def get_positions(self) -> List[Dict[str, Any]]:
        return [dict(p) for p in self.positions.values()]

    def get_price(self, symbol: str) -> float:
        return self.price_source.get_price(symbol)

    @property
    def equity(self) -> float:
        unrealized = 0.0
        for p in self.positions.values():
            unrealized += self._unrealized(p, self.get_price(p["symbol"]))
        return self.cash + unrealized

    def place_order(self, order: Order) -> Order:
        """Simulate a fill. NEVER places a real order."""
        price = self.get_price(order.symbol)

        # Limit-order gate
        if order.order_type == "limit" and order.limit_price is not None:
            if order.side == "buy" and price > order.limit_price:
                order.status = "rejected"
                order.message = "limit buy above market"
                self.orders.append(order)
                return order
            if order.side == "sell" and price < order.limit_price:
                order.status = "rejected"
                order.message = "limit sell below market"
                self.orders.append(order)
                return order
            fill_price = order.limit_price
        else:
            fill_price = price

        # Adverse slippage
        if order.side == "buy":
            fill_price *= (1 + self.slippage)
        else:
            fill_price *= (1 - self.slippage)

        fee = fill_price * order.quantity * self.fee_rate

        if order.side == "buy":
            self._buy(order, fill_price, fee)
        else:
            self._sell(order, fill_price, fee)

        self.orders.append(order)
        logger.debug("Paper %s: %s %s %.6f @ %.4f (fee %.4f)",
                     order.status, order.side, order.symbol, order.quantity, fill_price, fee)
        return order

    # ------------------------------------------------------------ buys
    def _buy(self, order: Order, price: float, fee: float) -> None:
        pos = self.positions.get(order.symbol)

        # Futures: buying into a short closes it first
        if pos is not None and pos["side"] == "short":
            close_qty = min(order.quantity, pos["quantity"])
            realized = (pos["entry_price"] - price) * close_qty
            self.cash += realized - (price * close_qty) * self.fee_rate
            self._record_close(pos, price, close_qty, realized)
            pos["quantity"] -= close_qty
            if pos["quantity"] <= 1e-12:
                del self.positions[order.symbol]
            order.quantity -= close_qty
            if order.quantity <= 1e-12:
                order.status = "filled"
                order.fill_price = price
                order.fee = fee
                return

        notional = price * order.quantity
        if self.cash < notional + fee:
            order.status = "rejected"
            order.message = f"insufficient cash ({self.cash:.2f} < {notional + fee:.2f})"
            return

        self.cash -= notional + fee
        self._open_or_average(order.symbol, "long", order.quantity, price, fee)
        order.status = "filled"
        order.fill_price = price
        order.fee = fee

    def _sell(self, order: Order, price: float, fee: float) -> None:
        pos = self.positions.get(order.symbol)

        # Spot is long-only
        if self.market_type == "spot":
            if pos is None:
                order.status = "rejected"
                order.message = "no position to sell (spot is long-only)"
                return
            close_qty = min(order.quantity, pos["quantity"])
            proceeds = price * close_qty - (price * close_qty) * self.fee_rate
            realized = (price - pos["entry_price"]) * close_qty
            self.cash += proceeds
            self._record_close(pos, price, close_qty, realized)
            pos["quantity"] -= close_qty
            if pos["quantity"] <= 1e-12:
                del self.positions[order.symbol]
            order.status = "filled"
            order.fill_price = price
            order.fee = fee
            return

        # Futures: sell closes long first, then opens short
        if pos is not None and pos["side"] == "long":
            close_qty = min(order.quantity, pos["quantity"])
            realized = (price - pos["entry_price"]) * close_qty
            self.cash += realized - (price * close_qty) * self.fee_rate
            self._record_close(pos, price, close_qty, realized)
            pos["quantity"] -= close_qty
            if pos["quantity"] <= 1e-12:
                del self.positions[order.symbol]
            order.quantity -= close_qty
            if order.quantity <= 1e-12:
                order.status = "filled"
                order.fill_price = price
                order.fee = fee
                return

        # Open/increase short (no cash deduction in this model)
        self._open_or_average(order.symbol, "short", order.quantity, price, fee)
        order.status = "filled"
        order.fill_price = price
        order.fee = fee

    # ------------------------------------------------------------ position book
    def _open_or_average(self, symbol: str, side: str, qty: float, price: float, fee: float) -> None:
        pos = self.positions.get(symbol)
        if pos is None or pos["side"] != side:
            self.positions[symbol] = {
                "symbol": symbol, "side": side, "quantity": qty,
                "entry_price": price, "entry_time": None, "fees_paid": fee,
            }
            return
        total_qty = pos["quantity"] + qty
        pos["entry_price"] = (pos["entry_price"] * pos["quantity"] + price * qty) / total_qty
        pos["quantity"] = total_qty
        pos["fees_paid"] += fee

    def _record_close(self, pos: Dict[str, Any], price: float, qty: float, realized: float) -> None:
        self.closed_trades.append({
            "trade_id": generate_trade_id(),
            "symbol": pos["symbol"],
            "direction": pos["side"],
            "execution_mode": "paper",
            "market_type": self.market_type,
            "entry_price": pos["entry_price"],
            "exit_price": price,
            "quantity": qty,
            "gross_pnl": realized,
            "net_pnl": realized - (price * qty) * self.fee_rate,
            "fees": (price * qty) * self.fee_rate,
            "entry_time": pos.get("entry_time"),
            "exit_time": None,  # set by the engine when it knows the bar time
            "status": "closed",
            "exit_reason": "paper_fill",
        })

    @staticmethod
    def _unrealized(pos: Dict[str, Any], price: float) -> float:
        if pos["side"] == "long":
            return (price - pos["entry_price"]) * pos["quantity"]
        return (pos["entry_price"] - price) * pos["quantity"]

    def mark_positions(self, entry_time: int) -> None:
        """Attach entry timestamp to newly opened positions."""
        for p in self.positions.values():
            if p.get("entry_time") is None:
                p["entry_time"] = entry_time

    def close_all(self, reason: str = "session_end") -> None:
        """Close every open position at the current price."""
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            price = self.get_price(symbol)
            qty = pos["quantity"]
            if pos["side"] == "long":
                realized = (price - pos["entry_price"]) * qty
            else:
                realized = (pos["entry_price"] - price) * qty
            self.cash += realized - (price * qty) * self.fee_rate
            self._record_close(pos, price, qty, realized)
            self.closed_trades[-1]["exit_reason"] = reason
            del self.positions[symbol]

    def reset(self) -> None:
        self.cash = self.initial_capital
        self.positions.clear()
        self.closed_trades.clear()
        self.orders.clear()