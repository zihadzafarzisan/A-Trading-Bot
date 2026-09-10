"""Binance Live REST Connector.

Handles authenticated API interactions for Binance Spot and Binance USDⓈ-M Futures:
- HMAC-SHA256 signature generation with timestamp and recvWindow
- Public and private endpoints for Spot and Futures (Live & Testnet)
- Time synchronization offset tracking against Binance server time
- Comprehensive order lifecycle operations (submit, cancel, query, open orders)
- Account balances, futures positions, margin, and leverage management
- Rate limit tracking, retry with exponential backoff, and error translation
- Robust masking of API credentials in all logs and string representations
"""

import hashlib
import hmac
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests

from ..logging_config import get_logger

logger = get_logger("trading")

# Endpoint URLs
SPOT_LIVE_URL = "https://api.binance.com"
SPOT_TESTNET_URL = "https://testnet.binance.vision"
FUTURES_LIVE_URL = "https://fapi.binance.com"
FUTURES_TESTNET_URL = "https://testnet.binancefuture.com"


class BinanceAPIError(Exception):
    """Exception raised for Binance API error responses."""

    def __init__(self, code: int, message: str, status_code: Optional[int] = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(f"Binance API Error [{code}]: {message} (HTTP {status_code})")


class BinanceLiveConnector:
    """Production-grade REST connector for Binance Spot and Futures."""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        market_type: str = "spot",
        testnet: bool = True,
        recv_window: int = 5000,
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
    ):
        """Initialize the Binance live connector.

        Args:
            api_key: Binance API key (never logged).
            api_secret: Binance API secret (never logged).
            market_type: 'spot' or 'futures'.
            testnet: True for Binance Testnet, False for Live real money.
            recv_window: Binance timestamp validity window in ms (default 5000).
            timeout: HTTP request timeout in seconds.
            max_retries: Retry attempts for transient 5xx or network errors.
            backoff_factor: Multiplier for exponential backoff.
        """
        if market_type not in ("spot", "futures"):
            raise ValueError(f"Invalid market_type: {market_type}. Must be 'spot' or 'futures'")

        self._api_key = api_key or ""
        self._api_secret = api_secret or ""
        self.market_type = market_type
        self.testnet = bool(testnet)
        self.recv_window = recv_window
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        # Select base URL
        if self.market_type == "spot":
            self.base_url = SPOT_TESTNET_URL if self.testnet else SPOT_LIVE_URL
        else:
            self.base_url = FUTURES_TESTNET_URL if self.testnet else FUTURES_LIVE_URL

        self._session = requests.Session()
        self._time_offset_ms: int = 0  # server_time - local_time
        self._last_time_sync: float = 0.0

        logger.info(
            "BinanceLiveConnector initialized [market=%s, testnet=%s, key_present=%s]",
            self.market_type,
            self.testnet,
            bool(self._api_key),
        )

    def __repr__(self) -> str:
        masked_key = (
            f"{self._api_key[:4]}...{self._api_key[-4:]}" if len(self._api_key) >= 8 else "***"
        )
        return (
            f"BinanceLiveConnector(market={self.market_type}, testnet={self.testnet}, "
            f"key={masked_key})"
        )

    # -----------------------------------------------------------------------
    # Authentication & Time Sync
    # -----------------------------------------------------------------------
    def sync_time(self) -> int:
        """Synchronize local clock with Binance server time."""
        endpoint = "/api/v3/time" if self.market_type == "spot" else "/fapi/v1/time"
        url = f"{self.base_url}{endpoint}"
        try:
            start_local = int(time.time() * 1000)
            resp = self._session.get(url, timeout=self.timeout)
            end_local = int(time.time() * 1000)
            if resp.status_code == 200:
                data = resp.json()
                server_time = int(data["serverTime"])
                latency = (end_local - start_local) // 2
                self._time_offset_ms = server_time - (start_local + latency)
                self._last_time_sync = time.time()
                logger.debug(
                    "Clock synced with Binance. Offset: %d ms, latency: %d ms",
                    self._time_offset_ms,
                    latency,
                )
                return self._time_offset_ms
            else:
                logger.warning("Time sync failed with status %s: %s", resp.status_code, resp.text)
        except Exception as exc:
            logger.warning("Time sync exception: %s", exc)
        return self._time_offset_ms

    def get_synced_timestamp(self) -> int:
        """Get local epoch ms adjusted by synced Binance server offset."""
        if time.time() - self._last_time_sync > 300:  # re-sync every 5 min
            self.sync_time()
        return int(time.time() * 1000) + self._time_offset_ms

    def _sign(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Sign request params with HMAC-SHA256."""
        if not self._api_key or not self._api_secret:
            raise ValueError("Binance API key and secret are required for signed endpoints")

        params = dict(params)
        params["timestamp"] = self.get_synced_timestamp()
        params["recvWindow"] = self.recv_window

        query_string = urlencode(params)
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature

        headers = {
            "X-MBX-APIKEY": self._api_key,
            "Accept": "application/json",
        }
        return params, headers

    # -----------------------------------------------------------------------
    # Core HTTP Request Handler
    # -----------------------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Any:
        """Execute REST request with retries, signature, and error handling."""
        params = params or {}
        url = f"{self.base_url}{path}"
        headers = {"Accept": "application/json"}

        if signed:
            params, headers = self._sign(params)

        for attempt in range(1, self.max_retries + 1):
            try:
                if method.upper() == "GET":
                    resp = self._session.get(
                        url, params=params, headers=headers, timeout=self.timeout
                    )
                elif method.upper() == "POST":
                    resp = self._session.post(
                        url, data=params, headers=headers, timeout=self.timeout
                    )
                elif method.upper() == "DELETE":
                    resp = self._session.delete(
                        url, params=params, headers=headers, timeout=self.timeout
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                # Success
                if resp.status_code == 200:
                    return resp.json()

                # Handle Rate Limiting
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else (self.backoff_factor**attempt)
                    logger.warning(
                        "Rate limited (429) on %s %s, waiting %.1fs", method, path, wait
                    )
                    time.sleep(wait)
                    continue

                if resp.status_code == 418:
                    wait = self.backoff_factor ** (attempt + 2)
                    logger.critical("IP banned (418) on %s %s, waiting %.1fs", method, path, wait)
                    time.sleep(wait)
                    continue

                # Parse JSON Error if available
                err_code = resp.status_code
                err_msg = resp.text
                try:
                    data = resp.json()
                    if isinstance(data, dict):
                        err_code = data.get("code", resp.status_code)
                        err_msg = data.get("msg", resp.text)
                except Exception:
                    pass

                # Timestamp offset error (-1021: Timestamp for this request was 1000ms ahead/behind)
                if err_code == -1021 and signed and attempt < self.max_retries:
                    logger.warning("Timestamp out of sync (-1021), re-syncing clock and retrying")
                    self.sync_time()
                    params, headers = self._sign(
                        {k: v for k, v in params.items() if k not in ("signature", "timestamp")}
                    )
                    continue

                # 4xx client errors (do NOT blindly retry business rejections)
                if 400 <= resp.status_code < 500:
                    logger.error(
                        "Binance request failed (HTTP %d, code %d): %s",
                        resp.status_code,
                        err_code,
                        err_msg,
                    )
                    raise BinanceAPIError(
                        code=err_code, message=err_msg, status_code=resp.status_code
                    )

                # 5xx server errors -> retry with backoff
                wait = self.backoff_factor**attempt
                logger.warning(
                    "HTTP %d on %s %s: %s, retrying in %.1fs",
                    resp.status_code,
                    method,
                    path,
                    err_msg[:200],
                    wait,
                )
                time.sleep(wait)

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                if attempt == self.max_retries:
                    raise RuntimeError(
                        f"Network error on {method} {path} after {self.max_retries} attempts: {exc}"
                    )
                wait = self.backoff_factor**attempt
                logger.warning(
                    "Network error on %s %s (%s), retrying in %.1fs", method, path, exc, wait
                )
                time.sleep(wait)

        raise RuntimeError(f"Exhausted retries for {method} {path}")

    # -----------------------------------------------------------------------
    # Public Market Data
    # -----------------------------------------------------------------------
    def ping(self) -> bool:
        """Check API connectivity."""
        path = "/api/v3/ping" if self.market_type == "spot" else "/fapi/v1/ping"
        try:
            self._request("GET", path)
            return True
        except Exception:
            return False

    def get_server_time(self) -> int:
        """Get Binance server time in epoch ms."""
        path = "/api/v3/time" if self.market_type == "spot" else "/fapi/v1/time"
        res = self._request("GET", path)
        return int(res["serverTime"])

    def get_exchange_info(self) -> Dict[str, Any]:
        """Fetch full exchange metadata, symbol rules, and filters."""
        path = "/api/v3/exchangeInfo" if self.market_type == "spot" else "/fapi/v1/exchangeInfo"
        return self._request("GET", path)

    def get_ticker_price(self, symbol: str) -> float:
        """Get latest ticker trade price."""
        path = "/api/v3/ticker/price" if self.market_type == "spot" else "/fapi/v1/ticker/price"
        res = self._request("GET", path, params={"symbol": symbol.upper()})
        return float(res["price"])

    def get_mark_price(self, symbol: str) -> float:
        """Get mark price (Futures only; falls back to ticker for Spot)."""
        if self.market_type == "futures":
            res = self._request(
                "GET", "/fapi/v1/premiumIndex", params={"symbol": symbol.upper()}
            )
            return float(res["markPrice"])
        return self.get_ticker_price(symbol)

    def get_funding_rate(self, symbol: str) -> float:
        """Get latest funding rate (Futures only)."""
        if self.market_type == "futures":
            res = self._request(
                "GET", "/fapi/v1/premiumIndex", params={"symbol": symbol.upper()}
            )
            return float(res.get("lastFundingRate", 0.0))
        return 0.0

    # -----------------------------------------------------------------------
    # Account & Position Management (Signed)
    # -----------------------------------------------------------------------
    def get_account_info(self) -> Dict[str, Any]:
        """Fetch account balance and permissions."""
        path = "/api/v3/account" if self.market_type == "spot" else "/fapi/v2/account"
        return self._request("GET", path, signed=True)

    def get_balances(self) -> Dict[str, Dict[str, float]]:
        """Return balances mapped by asset.

        Returns:
            Dict[asset, {"free": float, "locked": float, "total": float}]
        """
        acct = self.get_account_info()
        balances: Dict[str, Dict[str, float]] = {}

        if self.market_type == "spot":
            for b in acct.get("balances", []):
                asset = b["asset"]
                free = float(b.get("free", 0.0))
                locked = float(b.get("locked", 0.0))
                if free > 0 or locked > 0:
                    balances[asset] = {"free": free, "locked": locked, "total": free + locked}
        else:  # Futures
            for b in acct.get("assets", []):
                asset = b["asset"]
                wallet = float(b.get("walletBalance", 0.0))
                avail = float(b.get("availableBalance", 0.0))
                unrealized = float(b.get("unrealizedProfit", 0.0))
                balances[asset] = {
                    "free": avail,
                    "locked": max(0.0, wallet - avail),
                    "total": wallet + unrealized,
                }
        return balances

    def get_usdt_balance(self) -> float:
        """Get free USDT balance."""
        balances = self.get_balances()
        return balances.get("USDT", {}).get("free", 0.0)

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open positions (Futures only; returns empty list for Spot)."""
        if self.market_type != "futures":
            return []
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        res = self._request("GET", "/fapi/v2/positionRisk", params=params, signed=True)
        positions = []
        for p in res:
            qty = float(p.get("positionAmt", 0.0))
            if abs(qty) > 0 or symbol is not None:
                positions.append(
                    {
                        "symbol": p["symbol"],
                        "position_amt": qty,
                        "entry_price": float(p.get("entryPrice", 0.0)),
                        "mark_price": float(p.get("markPrice", 0.0)),
                        "unrealized_pnl": float(p.get("unRealizedProfit", 0.0)),
                        "liquidation_price": float(p.get("liquidationPrice", 0.0)),
                        "leverage": int(p.get("leverage", 1)),
                        "margin_type": p.get("marginType", "isolated"),
                        "isolated_margin": float(p.get("isolatedMargin", 0.0)),
                        "side": "long" if qty > 0 else ("short" if qty < 0 else "both"),
                    }
                )
        return positions

    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Set futures leverage (1-125x)."""
        if self.market_type != "futures":
            return {"leverage": 1}
        return self._request(
            "POST",
            "/fapi/v1/leverage",
            params={"symbol": symbol.upper(), "leverage": int(leverage)},
            signed=True,
        )

    def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> Dict[str, Any]:
        """Set futures margin type (ISOLATED or CROSSED)."""
        if self.market_type != "futures":
            return {}
        try:
            return self._request(
                "POST",
                "/fapi/v1/marginType",
                params={"symbol": symbol.upper(), "marginType": margin_type.upper()},
                signed=True,
            )
        except BinanceAPIError as exc:
            # Code -4046: "No need to change margin type" is non-fatal
            if exc.code == -4046:
                return {"msg": "Margin type already set"}
            raise

    # -----------------------------------------------------------------------
    # Trading & Order Management (Signed)
    # -----------------------------------------------------------------------
    def create_order(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        order_type: str,  # "MARKET", "LIMIT", "STOP_MARKET", "TAKE_PROFIT_MARKET"
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
    ) -> Dict[str, Any]:
        """Submit an order to Binance."""
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity,
        }
        if client_order_id:
            if self.market_type == "spot":
                params["newClientOrderId"] = client_order_id
            else:
                params["newClientOrderId"] = client_order_id

        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("Price is required for LIMIT orders")
            params["price"] = price
            params["timeInForce"] = time_in_force

        if stop_price is not None:
            params["stopPrice"] = stop_price

        if self.market_type == "futures" and reduce_only:
            params["reduceOnly"] = "true"

        path = "/api/v3/order" if self.market_type == "spot" else "/fapi/v1/order"
        logger.info(
            "Submitting %s %s order [%s, qty=%s, price=%s, client_id=%s, reduce_only=%s]",
            self.market_type,
            side.upper(),
            symbol.upper(),
            quantity,
            price,
            client_order_id,
            reduce_only,
        )
        return self._request("POST", path, params=params, signed=True)

    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cancel an active order."""
        params: Dict[str, Any] = {"symbol": symbol.upper()}
        if order_id is not None:
            params["orderId"] = int(order_id)
        if client_order_id is not None:
            params["origClientOrderId"] = client_order_id

        path = "/api/v3/order" if self.market_type == "spot" else "/fapi/v1/order"
        return self._request("DELETE", path, params=params, signed=True)

    def cancel_all_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Cancel all open orders for a symbol."""
        path = "/api/v3/openOrders" if self.market_type == "spot" else "/fapi/v1/allOpenOrders"
        return self._request("DELETE", path, params={"symbol": symbol.upper()}, signed=True)

    def query_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query order status."""
        params: Dict[str, Any] = {"symbol": symbol.upper()}
        if order_id is not None:
            params["orderId"] = int(order_id)
        if client_order_id is not None:
            params["origClientOrderId"] = client_order_id

        path = "/api/v3/order" if self.market_type == "spot" else "/fapi/v1/order"
        return self._request("GET", path, params=params, signed=True)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all currently open orders."""
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        path = "/api/v3/openOrders" if self.market_type == "spot" else "/fapi/v1/openOrders"
        return self._request("GET", path, params=params, signed=True)

    def close(self) -> None:
        """Close HTTP session."""
        try:
            self._session.close()
        except Exception:
            pass
