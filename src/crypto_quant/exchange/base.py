"""Abstract exchange adapter interface.

Defines the contract every exchange adapter must satisfy so the data engine
and execution layers remain exchange-agnostic. Additional exchanges (Coinbase,
Kraken, Bybit...) can be added by implementing this interface without touching
the rest of the system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ExchangeAdapter(ABC):
    """Base interface for exchange connectivity.

    All methods return plain data structures (dicts/lists) so callers do not
    depend on the underlying HTTP client.
    """

    exchange_name: str = "base"

    # ---------------------------------------------------------------- info
    @abstractmethod
    def get_server_time(self) -> int:
        """Return exchange server time as a UNIX timestamp in milliseconds."""

    @abstractmethod
    def get_exchange_info(self) -> Dict[str, Any]:
        """Return exchange info (symbols, limits, etc.)."""

    @abstractmethod
    def get_symbols(self) -> List[str]:
        """Return list of tradable symbols (e.g. ['BTCUSDT', ...])."""

    # ------------------------------------------------------------ klines
    @abstractmethod
    def get_klines(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> List[List[Any]]:
        """Fetch OHLCV klines.

        Args:
            symbol: Trading pair (e.g. 'BTCUSDT').
            timeframe: Interval (e.g. '1h', '4h', '1d').
            start_time: Inclusive start UNIX ms. None => oldest available.
            end_time: Exclusive end UNIX ms. None => latest available.
            limit: Max candles per request (1..1000).

        Returns:
            List of kline rows as returned by the exchange.
        """

    # ------------------------------------------------------------ futures
    @abstractmethod
    def get_funding_rate_history(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Return historical funding rates (empty if not supported)."""

    # ------------------------------------------------------------ trading
    @abstractmethod
    def get_ohlcv_as_dataframe(self, symbol: str, timeframe: str, **kwargs) -> Any:
        """Convenience: return klines as a normalized DataFrame."""

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the adapter (sessions, sockets)."""
