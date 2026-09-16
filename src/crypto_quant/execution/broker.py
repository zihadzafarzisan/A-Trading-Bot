"""Broker abstraction.

Separates execution from the rest of the system. PaperBroker and LiveBroker are
DISTINCT implementations — paper trading can never accidentally send a live
order (spec #56). Also defines the PriceSource used to feed market prices.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


class PriceSource(ABC):
    """Supplies current market prices for a symbol."""

    @abstractmethod
    def get_price(self, symbol: str) -> float:
        """Return the latest trade price for a symbol."""

    def get_klines(self, symbol: str, timeframe: str, limit: int = 2) -> pd.DataFrame:
        """Return recent klines (optional; used for indicator warmup)."""
        raise NotImplementedError


class CallbackPriceSource(PriceSource):
    """Price source backed by an arbitrary callable (tests, replay)."""

    def __init__(self, fn):
        self._fn = fn

    def get_price(self, symbol: str) -> float:
        return float(self._fn(symbol))


class AdapterPriceSource(PriceSource):
    """Price source backed by an exchange adapter (live market data).

    Uses the public kline endpoint — no credentials required. Falls back to the
    last close when the latest trade is unavailable.
    """

    def __init__(self, adapter, timeframe: str = "1m"):
        self.adapter = adapter
        self.timeframe = timeframe
        self._cache: Dict[str, float] = {}

    def get_price(self, symbol: str) -> float:
        try:
            df = self.adapter.get_ohlcv_as_dataframe(symbol, self.timeframe, limit=2)
            if df is not None and not df.empty:
                price = float(df["close"].iloc[-1])
                self._cache[symbol] = price
                return price
        except Exception:
            pass
        if symbol in self._cache:
            return self._cache[symbol]
        raise RuntimeError(f"No price available for {symbol}")

    def get_klines(self, symbol: str, timeframe: str, limit: int = 2) -> pd.DataFrame:
        return self.adapter.get_ohlcv_as_dataframe(symbol, timeframe, limit=limit)


@dataclass
class Order:
    """A requested order (paper fills simulate execution)."""

    order_id: str
    symbol: str
    side: str                 # 'buy' | 'sell'
    quantity: float
    order_type: str           # 'market' | 'limit'
    limit_price: Optional[float] = None
    reduce_only: bool = False  # Futures only: never opens a new position, only reduces
    position_side: Optional[str] = None  # Futures: BOTH, LONG, or SHORT
    suppress_notify: bool = False  # Internal/daemon orders: skip broker-level DM alerts
    status: str = "new"       # new -> filled | rejected | new | partially_filled
    exchange_order_id: Optional[str] = None  # Binance numeric orderId, when known
    filled_quantity: Optional[float] = None  # Exchange-confirmed executed base quantity
    fill_price: Optional[float] = None
    fee: float = 0.0
    message: str = ""


class Broker(ABC):
    """Execution interface (implemented by PaperBroker and LiveBroker)."""

    name: str = "base"

    @abstractmethod
    def get_balance(self) -> float:
        """Return available cash balance."""

    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """Return open positions as dicts."""

    @abstractmethod
    def place_order(self, order: Order) -> Order:
        """Place an order and return the result (with fill)."""

    @abstractmethod
    def get_price(self, symbol: str) -> float:
        """Return the current price for a symbol."""

    def cancel_order(self, order_id: str) -> bool:  # pragma: no cover - optional
        return False