"""Binance market data adapter.

Implements the ExchangeAdapter interface against Binance's public REST API
using `requests`. Uses only public market-data endpoints so no credentials are
required for research/backtesting data. Execution endpoints are intentionally
left out of this adapter; live trading is handled separately in the execution
layer behind explicit safety controls.
"""

import logging
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from ..config.settings import get_config
from ..logging_config import get_logger
from .base import ExchangeAdapter

# Binance public market data base URLs
SPOT_BASE_URL = "https://api.binance.com"
FUTURES_BASE_URL = "https://fapi.binance.com"

# Timeframe -> Binance interval string
TIMEFRAME_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

# Binance kline column order (raw array indices)
KLINE_COLUMNS = [
    "timestamp",      # 0 open time
    "open",           # 1
    "high",           # 2
    "low",            # 3
    "close",          # 4
    "volume",         # 5
    "close_time",     # 6
    "quote_volume",   # 7
    "trades",         # 8
    "taker_buy_base", # 9
    "taker_buy_quote",# 10
    "ignore",         # 11
]


class BinanceAdapter(ExchangeAdapter):
    """REST adapter for Binance public market data."""

    exchange_name = "binance"

    def __init__(
        self,
        market_type: str = "spot",
        timeout: float = 30.0,
        max_retries: int = 5,
        backoff_factor: float = 2.0,
        proxy: Optional[str] = None,
    ):
        """Initialize the adapter.

        Args:
            market_type: 'spot' or 'futures'.
            timeout: HTTP timeout in seconds.
            max_retries: Retry count for transient failures.
            backoff_factor: Exponential backoff multiplier (seconds).
            proxy: Optional proxy URL.
        """
        if market_type not in ("spot", "futures"):
            raise ValueError(f"Unsupported market_type: {market_type}")

        self.market_type = market_type
        self.base_url = SPOT_BASE_URL if market_type == "spot" else FUTURES_BASE_URL
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._session = requests.Session()

        proxies = {"http": proxy, "https": proxy} if proxy else None
        if proxies:
            self._session.proxies.update(proxies)

        self._logger = get_logger("data")
        self._logger.debug("BinanceAdapter initialized (market=%s)", market_type)

    # ------------------------------------------------------------ helpers
    def _request(self, path: str, params: Optional[Dict] = None) -> Any:
        """Perform a GET request with retries and backoff."""
        url = f"{self.base_url}{path}"
        params = params or {}

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                    headers={"Accept": "application/json"},
                )
                if resp.status_code == 200:
                    return resp.json()

                if resp.status_code == 429:
                    # Rate limited - read Retry-After header if present
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else self.backoff_factor ** attempt
                    self._logger.warning(
                        "Rate limited (429) on %s, waiting %.1fs", path, wait
                    )
                    time.sleep(wait)
                    continue

                if resp.status_code == 418:
                    # IP banned - longer wait
                    wait = self.backoff_factor ** (attempt + 2)
                    self._logger.warning("IP banned (418), waiting %.1fs", wait)
                    time.sleep(wait)
                    continue

                if resp.status_code in (400, 404, 418):
                    self._logger.error(
                        "Request failed: %s %s -> %s",
                        resp.status_code, url, resp.text[:300],
                    )
                    resp.raise_for_status()

                # Other 5xx errors -> retry
                wait = self.backoff_factor ** attempt
                self._logger.warning(
                    "HTTP %s on %s, retrying in %.1fs", resp.status_code, path, wait
                )
                time.sleep(wait)

            except requests.exceptions.Timeout:
                wait = self.backoff_factor ** attempt
                self._logger.warning("Timeout on %s, retrying in %.1fs", path, wait)
                time.sleep(wait)
            except requests.exceptions.ConnectionError:
                wait = self.backoff_factor ** attempt
                self._logger.warning("Connection error on %s, retrying in %.1fs", path, wait)
                time.sleep(wait)

        raise RuntimeError(f"Exhausted retries for {path}")

    # ---------------------------------------------------------------- info
    def get_server_time(self) -> int:
        """Return Binance server time in milliseconds."""
        data = self._request("/api/v3/time")
        return int(data["serverTime"])

    def get_exchange_info(self) -> Dict[str, Any]:
        """Return exchange info dictionary."""
        if self.market_type == "spot":
            return self._request("/api/v3/exchangeInfo")
        return self._request("/fapi/v1/exchangeInfo")

    def get_symbols(self) -> List[str]:
        """Return list of active trading symbols."""
        info = self.get_exchange_info()
        symbols = []
        for s in info.get("symbols", []):
            if self.market_type == "spot":
                # Spot: only SPOT type, trading enabled, ends in USDT
                if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT":
                    symbols.append(s["symbol"])
            else:
                # Futures
                if s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT":
                    symbols.append(s["symbol"])
        return symbols

    # ------------------------------------------------------------ klines
    def get_klines(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> List[List[Any]]:
        """Fetch OHLCV klines from Binance."""
        interval = TIMEFRAME_MAP.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(max(limit, 1), 1000),
        }
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)

        if self.market_type == "spot":
            return self._request("/api/v3/klines", params)
        return self._request("/fapi/v1/klines", params)

    def get_funding_rate_history(
        self,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Return historical funding rates (futures only)."""
        if self.market_type != "futures":
            return []

        params: Dict[str, Any] = {
            "symbol": symbol,
            "limit": min(max(limit, 1), 1000),
        }
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)

        data = self._request("/fapi/v1/fundingRate", params)
        # Normalize: keep symbol + timestampMs + fundingRate
        normalized = []
        for row in data:
            normalized.append({
                "symbol": row.get("symbol"),
                "funding_time": row.get("fundingTime"),
                "funding_rate": float(row.get("fundingRate", 0.0)),
            })
        return normalized

    def get_ohlcv_as_dataframe(self, symbol: str, timeframe: str, **kwargs) -> pd.DataFrame:
        """Return klines as a normalized OHLCV DataFrame."""
        rows = self.get_klines(symbol, timeframe, **kwargs)
        if not rows:
            return pd.DataFrame(columns=KLINE_COLUMNS)

        df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
        # Keep only OHLCV + timestamp
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df.dropna(subset=["timestamp"], inplace=True)
        return df

    # ------------------------------------------------------------ futures
    def get_available_timeframes(self) -> List[str]:
        """Return supported timeframes for this market."""
        return list(TIMEFRAME_MAP.keys())

    def close(self) -> None:
        """Close the underlying HTTP session."""
        try:
            self._session.close()
        except Exception:  # pragma: no cover - trivial cleanup
            pass


class BinanceAdapterFactory:
    """Factory to create Binance adapters from configuration."""

    @staticmethod
    def from_config(market_type: str = "spot") -> BinanceAdapter:
        """Create a BinanceAdapter using config values."""
        config = get_config()
        dl = config.data.rate_limit
        rt = config.data.retry
        return BinanceAdapter(
            market_type=market_type,
            timeout=30.0,
            max_retries=rt.max_attempts,
            backoff_factor=rt.backoff_factor,
        )
