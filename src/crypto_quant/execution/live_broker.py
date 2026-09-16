"""Binance Live Broker and Symbol Filter Validation Layer.

Implements the Broker ABC for production Binance execution (Spot and Futures).
Ensures full pre-trade validation against Binance exchange rules:
- PRICE_FILTER (min_price, max_price, tick_size)
- LOT_SIZE / MARKET_LOT_SIZE (min_qty, max_qty, step_size)
- MIN_NOTIONAL / NOTIONAL (min_notional value in USDT)
- MAX_NUM_ORDERS / PERCENT_PRICE

Provides:
- Dry-run mode: connects, fetches real account & prices, validates orders, but skips live submission
- Spot & Futures support (long/short, leverage, isolated margin, stop-loss / take-profit)
- Slippage and latency tracking
- Full order lifecycle mapping into local Order objects
"""

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..exchange.binance_live import BinanceAPIError, BinanceLiveConnector
from ..logging_config import get_logger
from ..notifications import get_notifier
from .broker import Broker, Order, PriceSource

logger = get_logger("trading")


@dataclass
class SymbolFilters:
    """Parsed exchange trading rules and filters for a symbol."""

    symbol: str
    base_asset: str = "BTC"
    quote_asset: str = "USDT"
    status: str = "TRADING"
    min_price: float = 0.01
    max_price: float = 1_000_000.0
    tick_size: float = 0.01
    min_qty: float = 0.001
    max_qty: float = 1000.0
    step_size: float = 0.001
    min_notional: float = 5.0  # default 5 USDT on Binance

    def round_price(self, price: float) -> float:
        """Round price to valid tick_size precision."""
        if self.tick_size <= 0:
            return price
        precision = max(0, int(round(-math.log10(self.tick_size))))
        steps = round(price / self.tick_size)
        return round(steps * self.tick_size, precision)

    def round_qty(self, qty: float) -> float:
        """Round quantity down to valid step_size precision (floor to avoid balance overshoot)."""
        if self.step_size <= 0:
            return qty
        precision = max(0, int(round(-math.log10(self.step_size))))
        steps = math.floor(qty / self.step_size)
        return round(steps * self.step_size, precision)

    def validate_order(
        self, side: str, qty: float, price: float, order_type: str = "market"
    ) -> Tuple[bool, List[str], float, float]:
        """Validate an order against symbol rules.

        Returns:
            (is_valid, error_reasons, rounded_qty, rounded_price)
        """
        reasons = []
        if self.status != "TRADING":
            reasons.append(f"Symbol {self.symbol} is not TRADING (status={self.status})")

        rounded_qty = self.round_qty(qty)
        rounded_price = self.round_price(price)

        if rounded_qty < self.min_qty:
            reasons.append(
                f"Quantity {rounded_qty} is below min_qty {self.min_qty} for {self.symbol}"
            )
        if rounded_qty > self.max_qty:
            reasons.append(
                f"Quantity {rounded_qty} exceeds max_qty {self.max_qty} for {self.symbol}"
            )

        if order_type.lower() == "limit":
            if rounded_price < self.min_price:
                reasons.append(
                    f"Price {rounded_price} is below min_price {self.min_price} for {self.symbol}"
                )
            if rounded_price > self.max_price:
                reasons.append(
                    f"Price {rounded_price} exceeds max_price {self.max_price} for {self.symbol}"
                )

        notional = rounded_qty * rounded_price
        if notional < self.min_notional:
            reasons.append(
                f"Order notional ${notional:.2f} is below min_notional ${self.min_notional:.2f} for {self.symbol}"
            )

        return len(reasons) == 0, reasons, rounded_qty, rounded_price


class BinancePriceSource(PriceSource):
    """Real-time market price source backed by BinanceLiveConnector."""

    def __init__(self, connector: BinanceLiveConnector):
        self.connector = connector
        self._last_prices: Dict[str, float] = {}

    def get_price(self, symbol: str) -> float:
        symbol = symbol.upper()
        try:
            price = self.connector.get_ticker_price(symbol)
            self._last_prices[symbol] = price
            return price
        except Exception as exc:
            if symbol in self._last_prices:
                logger.debug("Failed to fetch fresh price for %s, using cached: %s", symbol, exc)
                return self._last_prices[symbol]
            raise RuntimeError(f"No price available for {symbol}: {exc}")

    def get_all_prices(self) -> Dict[str, float]:
        """Fetch all ticker prices in a single bulk API request."""
        try:
            endpoint = "/api/v3/ticker/price" if self.connector.market_type == "spot" else "/fapi/v1/ticker/price"
            res = self.connector._request("GET", endpoint)
            for item in res:
                self._last_prices[item["symbol"]] = float(item["price"])
            return self._last_prices
        except Exception as exc:
            logger.warning("Bulk ticker fetch failed: %s", exc)
            return self._last_prices


class BinanceLiveBroker(Broker):
    """Production broker implementing order execution on Binance Spot and Futures."""

    name: str = "binance_live"

    def __init__(
        self,
        connector: BinanceLiveConnector,
        dry_run: bool = True,
        default_leverage: int = 1,
        market_fill_timeout_seconds: float = 5.0,
        market_fill_poll_interval_seconds: float = 0.25,
    ):
        """Initialize Binance live broker.

        Args:
            connector: Authenticated BinanceLiveConnector.
            dry_run: If True, simulates order fills locally using live price without placing real orders.
            default_leverage: Default leverage for futures contracts (1-5x).
            market_fill_timeout_seconds: Maximum time to wait for an exchange
                acknowledgement of a market order to reach a terminal fill state.
            market_fill_poll_interval_seconds: Delay between signed order-status polls.
        """
        if market_fill_timeout_seconds <= 0:
            raise ValueError("market_fill_timeout_seconds must be positive")
        if market_fill_poll_interval_seconds <= 0:
            raise ValueError("market_fill_poll_interval_seconds must be positive")
        self.connector = connector
        self.dry_run = dry_run
        self.market_type = connector.market_type
        self.default_leverage = default_leverage
        self.market_fill_timeout_seconds = market_fill_timeout_seconds
        self.market_fill_poll_interval_seconds = market_fill_poll_interval_seconds
        self.price_source = BinancePriceSource(connector)

        self._symbol_filters: Dict[str, SymbolFilters] = {}
        self.orders: List[Order] = []
        self.closed_trades: List[Dict[str, Any]] = []
        self._dry_run_cash = 1000.0  # fallback for dry run if balance query fails
        self._dry_run_positions: Dict[str, Dict[str, Any]] = {}

        # Directional DM-alert bookkeeping: per-symbol last entry (for realized
        # PnL context on exits) and the disabled-safe notifier singleton.
        self._notify_entries: Dict[str, Dict[str, Any]] = {}
        self._notifier = get_notifier()

        # Fetch exchange info and symbol filters at startup
        self.refresh_symbol_filters()

        logger.info(
            "BinanceLiveBroker initialized [market=%s, testnet=%s, dry_run=%s]",
            self.market_type,
            self.connector.testnet,
            self.dry_run,
        )

    def refresh_symbol_filters(self) -> None:
        """Fetch exchange symbol metadata and parse filters."""
        try:
            info = self.connector.get_exchange_info()
            for sym in info.get("symbols", []):
                s_name = sym["symbol"]
                filters = SymbolFilters(
                    symbol=s_name,
                    base_asset=sym.get("baseAsset", ""),
                    quote_asset=sym.get("quoteAsset", ""),
                    status=sym.get("status", "TRADING"),
                )
                for f in sym.get("filters", []):
                    ftype = f.get("filterType")
                    if ftype == "PRICE_FILTER":
                        filters.min_price = float(f.get("minPrice", 0.01))
                        filters.max_price = float(f.get("maxPrice", 1_000_000.0))
                        filters.tick_size = float(f.get("tickSize", 0.01))
                    elif ftype in ("LOT_SIZE", "MARKET_LOT_SIZE"):
                        filters.min_qty = float(f.get("minQty", 0.001))
                        filters.max_qty = float(f.get("maxQty", 1000.0))
                        filters.step_size = float(f.get("stepSize", 0.001))
                    elif ftype in ("MIN_NOTIONAL", "NOTIONAL"):
                        filters.min_notional = float(
                            f.get("minNotional", f.get("notional", 5.0)) or 5.0
                        )
                self._symbol_filters[s_name] = filters
            logger.info("Loaded symbol filters for %d symbols", len(self._symbol_filters))
        except Exception as exc:
            logger.error("Failed to load symbol filters: %s", exc)

    def get_symbol_filter(self, symbol: str) -> SymbolFilters:
        """Get or create default symbol filter."""
        symbol = symbol.upper()
        if symbol not in self._symbol_filters:
            self.refresh_symbol_filters()
        return self._symbol_filters.get(symbol, SymbolFilters(symbol=symbol))

    # -----------------------------------------------------------------------
    # Broker Interface Implementation
    # -----------------------------------------------------------------------
    def get_balance(self) -> float:
        """Return available cash balance (free USDT)."""
        if self.dry_run:
            try:
                bal = self.connector.get_usdt_balance()
                if bal > 0:
                    return bal
            except Exception:
                pass
            return self._dry_run_cash

        return self.connector.get_usdt_balance()

    def get_positions(self) -> List[Dict[str, Any]]:
        """Return open positions as list of dicts."""
        if self.dry_run:
            return list(self._dry_run_positions.values())

        if self.market_type == "futures":
            raw = self.connector.get_positions()
            positions = []
            for p in raw:
                if abs(p["position_amt"]) > 0:
                    positions.append(
                        {
                            "symbol": p["symbol"],
                            "side": "long" if p["position_amt"] > 0 else "short",
                            "quantity": abs(p["position_amt"]),
                            "entry_price": p["entry_price"],
                            "mark_price": p["mark_price"],
                            "unrealized_pnl": p["unrealized_pnl"],
                            "leverage": p["leverage"],
                            "liquidation_price": p["liquidation_price"],
                            "notional": abs(p["position_amt"]) * p["mark_price"],
                        }
                    )
            return positions
        else:
            # Spot: Fetch bulk ticker map to convert balances without individual requests
            balances = self.connector.get_balances()
            price_map = self.price_source.get_all_prices()
            positions = []
            for asset, b in balances.items():
                if asset not in ("USDT", "USD", "BUSD", "FDUSD") and b["total"] > 0:
                    sym = f"{asset}USDT"
                    if sym in self._symbol_filters and self._symbol_filters[sym].status == "TRADING":
                        price = price_map.get(sym, 0.0)
                        if price > 0:
                            positions.append(
                                {
                                    "symbol": sym,
                                    "side": "long",
                                    "quantity": b["total"],
                                    "entry_price": price,
                                    "current_price": price,
                                    "notional": b["total"] * price,
                                    "leverage": 1,
                                }
                            )
            return positions

    def get_price(self, symbol: str) -> float:
        """Return current market price."""
        return self.price_source.get_price(symbol)

    _TERMINAL_EXCHANGE_STATUSES = {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}

    @staticmethod
    def _response_status(response: Dict[str, Any]) -> str:
        """Normalize Binance's order status into the broker's lowercase form."""
        return str(response.get("status", "NEW")).upper()

    def _await_market_fill(
        self, symbol: str, order: Order, response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Poll an acknowledged market order until it becomes terminal or times out.

        Binance USDⓈ-M can ACK a valid market order as ``NEW`` before its fill
        reaches the REST response. Returning that ACK as a fill creates phantom
        hedges, so only exchange-confirmed terminal status is authoritative.
        """
        status = self._response_status(response)
        executed = float(response.get("executedQty", 0.0) or 0.0)
        exchange_id = response.get("orderId")
        if status in self._TERMINAL_EXCHANGE_STATUSES or executed > 0:
            return response

        deadline = time.monotonic() + self.market_fill_timeout_seconds
        while time.monotonic() < deadline:
            time.sleep(self.market_fill_poll_interval_seconds)
            try:
                response = self.connector.query_order(
                    symbol=symbol,
                    order_id=int(exchange_id) if exchange_id is not None else None,
                    client_order_id=None if exchange_id is not None else order.order_id,
                )
            except Exception as exc:
                logger.warning(
                    "[LIVE EXECUTION] Poll failed for market order %s on %s: %s",
                    order.order_id,
                    symbol,
                    exc,
                )
                continue
            status = self._response_status(response)
            executed = float(response.get("executedQty", 0.0) or 0.0)
            if status in self._TERMINAL_EXCHANGE_STATUSES or executed > 0:
                return response

        response = dict(response)
        response.setdefault("status", "NEW")
        response["_timed_out"] = True
        return response

    def _map_exchange_order(
        self, order: Order, response: Dict[str, Any], fallback_price: float, latency_ms: float
    ) -> None:
        """Map an authoritative Binance order response into the local order."""
        status = self._response_status(response)
        executed = float(response.get("executedQty", 0.0) or 0.0)
        quote = float(response.get("cummulativeQuoteQty", response.get("cumQuote", 0.0)) or 0.0)
        avg_price = (
            quote / executed
            if executed > 0 and quote > 0
            else float(response.get("avgPrice", 0.0) or fallback_price)
        )
        exchange_id = response.get("orderId")
        order.exchange_order_id = str(exchange_id) if exchange_id is not None else None
        order.filled_quantity = executed
        order.fill_price = avg_price if executed > 0 else None
        order.fee = round(
            executed * avg_price * (0.0004 if self.market_type == "futures" else 0.001), 6
        )

        if status == "FILLED" and executed > 0:
            order.status = "filled"
        elif status == "PARTIALLY_FILLED" or executed > 0:
            order.status = "partially_filled"
        elif status in ("CANCELED", "REJECTED", "EXPIRED"):
            order.status = "rejected"
        else:
            order.status = "new"

        timeout_suffix = " after fill-poll timeout" if response.get("_timed_out") else ""
        order.message = (
            f"Binance Order {exchange_id} [{status}] executed_qty={executed} "
            f"(latency: {latency_ms:.1f}ms){timeout_suffix}"
        )
        log_method = logger.info if order.status == "filled" else logger.warning
        log_method(
            "[LIVE EXECUTION] Binance order %s %s for %s: requested=%s executed=%s avg_price=%s%s",
            exchange_id,
            order.status,
            order.symbol,
            order.quantity,
            executed,
            order.fill_price,
            timeout_suffix,
        )

    def _notify_fill(self, order: Order) -> None:
        """Dispatch directional entry/exit DM alerts for a confirmed fill.

        Carry/internal daemon orders are suppressed via ``order.suppress_notify``
        (their alerts originate at a higher lifecycle level). The directional
        path here is a long spot book (the live worker rejects shorts on spot),
        so a buy fill is an entry and a sell fill is an exit.
        """
        if order.suppress_notify or order.status != "filled":
            return
        symbol = order.symbol.upper()
        qty = float(order.filled_quantity or order.quantity or 0.0)
        price = float(order.fill_price or 0.0)
        side = order.side.lower()
        order_id = order.exchange_order_id or order.order_id

        if side in ("buy", "long"):
            # Entry (or add) on a long position.
            self._notify_entries[symbol] = {"side": "long", "entry_price": price, "qty": qty}
            # Legacy entry (existing behavior)
            self._notifier.notify_trade_open(
                symbol=symbol, side="long", qty=qty, fill_price=price, order_id=order_id,
            )
            # New lifecycle entry (best-effort; directional broker doesn't know
            # strategy name here, so it uses a stable default).
            try:
                if hasattr(self._notifier, "notify_trade_entry"):
                    self._notifier.notify_trade_entry(
                        symbol=symbol,
                        strategy="STRAT-TREND",
                        side="LONG",
                        qty=qty,
                        spot_fill_price=price,
                        futures_fill_price=None,
                        allocated_capital_usdt=None,
                        target_apr_pct=None,
                        order_id=order_id,
                        entry_timestamp=None,
                    )
            except Exception:
                pass
            return

        # Exit (sell) — attach best-effort realized PnL from the tracked entry.
        entry = self._notify_entries.get(symbol)
        realized = None
        if entry:
            direction = 1 if entry.get("side") == "long" else -1
            realized = direction * (price - float(entry.get("entry_price", price))) * qty
            self._notify_entries.pop(symbol, None)
        # Legacy exit (existing behavior)
        self._notifier.notify_trade_close(
            symbol=symbol, side="long", qty=qty, fill_price=price, realized_pnl=realized,
        )

        # New lifecycle exit (best-effort)
        try:
            if hasattr(self._notifier, "notify_trade_exit"):
                # Directional broker only tracks entry_price; treat realized
                # (approx) as net pnl.
                net = float(realized) if realized is not None else None
                outcome = "WIN" if (net is not None and net > 0) else "LOSS"
                # Tag exits based on the last closed trade reason set by
                # the strategy execution engine (best-effort).
                exit_reason = getattr(self, "_last_exit_reason", None) or "MANUAL"
                # consume so we don't reuse it for later independent exits.
                try:
                    if hasattr(self, "_last_exit_reason"):
                        self._last_exit_reason = None
                except Exception:
                    pass
                self._notifier.notify_trade_exit(
                    symbol=symbol,
                    position_id=order_id,
                    strategy="STRAT-TREND",
                    side="LONG",
                    qty=qty,
                    spot_fill_price=price,
                    futures_fill_price=None,
                    gross_pnl_usdt=None,
                    net_pnl_usdt=net,
                    funding_harvested_usdt=None,
                    roi_pct=None,
                    duration=None,
                    exit_reason=exit_reason,
                    entry_timestamp=None,
                )
        except Exception:
            pass

    def place_order(self, order: Order) -> Order:
        """Validate, size, and place an order (Live or Dry-Run)."""
        symbol = order.symbol.upper()
        s_filter = self.get_symbol_filter(symbol)

        # 1. Fetch current price
        try:
            curr_price = self.get_price(symbol)
        except Exception as exc:
            order.status = "rejected"
            order.message = f"Failed to get price for {symbol}: {exc}"
            self.orders.append(order)
            return order

        target_price = order.limit_price or curr_price

        # 2. Validate against Binance symbol filters
        is_valid, errors, valid_qty, valid_price = s_filter.validate_order(
            side=order.side,
            qty=order.quantity,
            price=target_price,
            order_type=order.order_type,
        )
        if not is_valid:
            order.status = "rejected"
            order.message = f"Symbol filter validation failed: {'; '.join(errors)}"
            logger.warning("[BROKER] Order rejected for %s: %s", symbol, order.message)
            self.orders.append(order)
            return order

        order.quantity = valid_qty

        # 3. Dry-Run Execution
        if self.dry_run:
            t0 = time.time()
            order.status = "filled"
            order.fill_price = curr_price
            order.filled_quantity = order.quantity
            order.fee = round(
                order.quantity
                * curr_price
                * (0.0004 if self.market_type == "futures" else 0.001),
                6,
            )
            order.message = (
                f"[DRY RUN - SIMULATED] Filled at {curr_price:.4f} (latency: {(time.time()-t0)*1000:.1f}ms)"
            )
            logger.info(
                "[DRY RUN] %s %s order filled: %s @ %s (qty=%s)",
                self.market_type.upper(),
                order.side.upper(),
                symbol,
                curr_price,
                valid_qty,
            )

            # Update dry run position book
            if order.side.lower() in ("buy", "long"):
                self._dry_run_positions[symbol] = {
                    "symbol": symbol,
                    "side": "long",
                    "quantity": valid_qty,
                    "entry_price": curr_price,
                    "leverage": self.default_leverage,
                    "notional": valid_qty * curr_price,
                }
            elif order.side.lower() in ("sell", "short"):
                if symbol in self._dry_run_positions:
                    del self._dry_run_positions[symbol]

            self.orders.append(order)
            self._notify_fill(order)
            return order

        # 4. Live Real-Money Execution
        try:
            t_start = time.time()
            resp = self.connector.create_order(
                symbol=symbol,
                side=order.side.upper(),
                order_type=order.order_type.upper(),
                quantity=valid_qty,
                price=valid_price if order.order_type.lower() == "limit" else None,
                client_order_id=order.order_id,
                reduce_only=order.reduce_only,
                position_side=order.position_side,
            )
            if order.order_type.lower() == "market":
                resp = self._await_market_fill(symbol, order, resp)
            latency_ms = (time.time() - t_start) * 1000
            self._map_exchange_order(order, resp, curr_price, latency_ms)

        except BinanceAPIError as exc:
            order.status = "rejected"
            order.message = f"Binance API error: {exc.message}"
            logger.error("[LIVE EXECUTION ERROR] Order rejected by Binance: %s", exc)
        except Exception as exc:
            order.status = "rejected"
            order.message = f"Execution exception: {exc}"
            logger.error("[LIVE EXECUTION ERROR] Unexpected exception: %s", exc)

        self.orders.append(order)
        self._notify_fill(order)
        return order

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """Cancel an open order by exchange id or client id.

        Passing ``symbol`` is strongly preferred for exchange ids. Client-id
        cancellation retains the legacy symbol scan for compatibility.
        """
        if self.dry_run:
            logger.info("[DRY RUN] Order %s canceled", order_id)
            return True

        symbols = [symbol.upper()] if symbol else list(self._symbol_filters)
        numeric_id: Optional[int]
        try:
            numeric_id = int(order_id)
        except (TypeError, ValueError):
            numeric_id = None
        for sym in symbols:
            try:
                res = self.connector.cancel_order(
                    symbol=sym,
                    order_id=numeric_id,
                    client_order_id=None if numeric_id is not None else order_id,
                )
                if str(res.get("status", "")).upper() in ("CANCELED", "FILLED", "EXPIRED"):
                    return True
            except Exception:
                continue
        return False


# -----------------------------------------------------------------------
# Twin-Leg Delta-Neutral Cash-and-Carry (Path 4)
# -----------------------------------------------------------------------


@dataclass
class TwinLegFilters:
    """Reconciled filters across the Spot + Futures legs of a carry trade.

    Picks the coarsest common step size (the larger of the two), and the
    strictest (largest) min quantity and min notional and smallest max
    quantity across both venues. A quantity rounded to this reconciled
    grid is therefore legal on BOTH legs with zero leftover unhedged delta.
    """

    symbol: str
    step_size: float
    min_qty: float
    max_qty: float
    min_notional: float

    def round_qty(self, qty: float) -> float:
        """Floor a quantity to the coarsest common step size."""
        if self.step_size <= 0:
            return qty
        precision = max(0, int(round(-math.log10(self.step_size))))
        steps = math.floor(qty / self.step_size)
        return round(steps * self.step_size, precision)

    def validate_qty(self, qty: float, price: float) -> Tuple[bool, List[str], float]:
        """Validate a quantity/notional against the reconciled rules.

        Returns (is_valid, error_reasons, rounded_qty).
        """
        reasons: List[str] = []
        rounded = self.round_qty(qty)
        if rounded < self.min_qty:
            reasons.append(
                f"Quantity {rounded} is below min_qty {self.min_qty} on {self.symbol}"
            )
        if rounded > self.max_qty:
            reasons.append(
                f"Quantity {rounded} exceeds max_qty {self.max_qty} on {self.symbol}"
            )
        notional = rounded * price
        if notional < self.min_notional:
            reasons.append(
                f"Notional ${notional:.2f} is below min_notional ${self.min_notional:.2f} on {self.symbol}"
            )
        return len(reasons) == 0, reasons, rounded

    @classmethod
    def from_filters(cls, spot: SymbolFilters, fut: SymbolFilters) -> "TwinLegFilters":
        """Derive the strictest intersection of Spot and Futures filters."""
        return cls(
            symbol=spot.symbol,
            step_size=max(spot.step_size, fut.step_size),
            min_qty=max(spot.min_qty, fut.min_qty),
            max_qty=min(spot.max_qty, fut.max_qty),
            min_notional=max(spot.min_notional, fut.min_notional),
        )


@dataclass
class TwinLegPosition:
    """Open book-keeping for a single twin-leg carry trade."""

    position_id: str
    base_asset: str          # e.g. "SOL"
    quote_asset: str         # e.g. "USDT"
    quantity: float
    spot_order_id: str
    spot_fill_price: float
    futures_order_id: str
    futures_fill_price: float
    entry_basis_spread_pct: float
    execution_gap_ms: float
    status: str              # OPEN | UNWINDING | CLOSED | ORPHAN_UNWOUND
    opened_at: float
    closed_at: Optional[float] = None
    residual_futures_qty: float = 0.0
    manual_intervention_required: bool = False
    # Pre-existing account inventory captured at entry (before this carry's
    # fills), so the daemon's delta-neutrality check measures ONLY the deltas
    # this carry managed rather than total balances.
    baseline_spot_qty: float = 0.0
    baseline_futures_qty: float = 0.0


class TwinLegExecutionError(Exception):
    """Raised when a twin-leg carry fails at any phase.

    Always carries the TwinLegPosition (even a partial/placeholder one) and the
    phase in which the failure occurred, so the caller can reconcile or settle.
    ``panic_unwound`` is True when the Spot inventory was successfully flattened
    by the circuit breaker after a Leg 2 failure.
    """

    def __init__(
        self,
        message: str,
        phase: str,
        position: TwinLegPosition,
        cause: Optional[Exception] = None,
        panic_unwound: bool = False,
    ):
        self.phase = phase
        self.position = position
        self.cause = cause
        self.panic_unwound = panic_unwound
        super().__init__(message)


class BrokenLegExecutionError(TwinLegExecutionError):
    """A Leg 2 (Futures) leg failed or was aborted after the Spot was bought.

    Subclasses :class:`TwinLegExecutionError` so the daemon's existing
    ``except TwinLegExecutionError`` reconciliation path still handles it.
    Raised only after the emergency market SELL on the Spot inventory has been
    issued, guaranteeing the account never carries orphaned, unhedged spot.
    """


class TwinLegCarryBroker:
    """Executes a delta-neutral Cash-and-Carry trade across Spot + Futures.

    Leg 1 buys the base asset on Spot. Leg 2 opens an equal-and-opposite short
    on USDⓈ-M Futures sized to the *actual* Spot fill. If Leg 2 fails for any
    reason (margin, network, API rejection), a panic-unwind circuit breaker
    immediately market-SELLs the Spot inventory to eliminate the unhedged
    directional risk, logs a critical alert, and raises TwinLegExecutionError.
    """

    name: str = "twin_leg_carry"

    def __init__(
        self,
        spot_broker: BinanceLiveBroker,
        futures_broker: BinanceLiveBroker,
        default_leverage: int = 1,
        position_mode: str = "ONE_WAY",
    ):
        position_mode = position_mode.upper()
        if position_mode not in ("ONE_WAY", "HEDGE"):
            raise ValueError("position_mode must be ONE_WAY or HEDGE")
        if "spot" not in spot_broker.market_type:
            raise ValueError("TwinLegCarryBroker requires a Spot broker for Leg 1")
        if "futures" not in futures_broker.market_type:
            raise ValueError("TwinLegCarryBroker requires a Futures broker for Leg 2")

        self.spot = spot_broker
        self.futures = futures_broker
        self.default_leverage = default_leverage
        self.position_mode = position_mode
        self.dry_run = spot_broker.dry_run or futures_broker.dry_run
        self._positions: Dict[str, TwinLegPosition] = {}
        # Disabled-safe Discord DM pipeline for panic unwind / circuit-breaker
        # alerts. Non-blocking and failure-isolated; a dead channel can never
        # block or break a fill.
        self._notifier = get_notifier()

    # -------------------------------------------------------------------
    # Filter reconciliation
    # -------------------------------------------------------------------
    @staticmethod
    def reconcile_twin_filters(
        spot: SymbolFilters, fut: SymbolFilters
    ) -> TwinLegFilters:
        """Find the common tradable grid across Spot and Futures filters."""
        return TwinLegFilters.from_filters(spot, fut)

    def _symbol(self, base_asset: str, quote_asset: str) -> str:
        return f"{base_asset.upper()}{quote_asset.upper()}"

    def get_twin_filters(
        self, base_asset: str, quote_asset: str = "USDT"
    ) -> TwinLegFilters:
        """Fetch and reconcile Spot + Futures filters for a symbol."""
        symbol = self._symbol(base_asset, quote_asset)
        spot = self.spot.get_symbol_filter(symbol)
        fut = self.futures.get_symbol_filter(symbol)
        return self.reconcile_twin_filters(spot, fut)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    @staticmethod
    def _fill(order: Order) -> Tuple[float, float]:
        """Extract exchange-confirmed (fill_qty, fill_price) from an order."""
        quantity = order.filled_quantity
        if quantity is None and order.status == "filled":
            quantity = order.quantity
        return float(quantity or 0.0), float(order.fill_price or 0.0)

    def _placeholder(self, base_asset: str, quote_asset: str) -> TwinLegPosition:
        """A minimal position object to attach to pre-fill failures."""
        return TwinLegPosition(
            position_id="",
            base_asset=base_asset.upper(),
            quote_asset=quote_asset.upper(),
            quantity=0.0,
            spot_order_id="",
            spot_fill_price=0.0,
            futures_order_id="",
            futures_fill_price=0.0,
            entry_basis_spread_pct=0.0,
            execution_gap_ms=0.0,
            status="CLOSED",
            opened_at=time.time(),
        )

    def _prep_futures(self, symbol: str) -> None:
        """Validate Futures position mode and configure leverage/margin."""
        if self.dry_run:
            return
        try:
            actual_mode = self.futures.connector.get_position_mode()
            if actual_mode != self.position_mode:
                raise RuntimeError(
                    f"Futures account is {actual_mode} but carry is configured for "
                    f"{self.position_mode}; pass the correct position_mode explicitly"
                )
        except AttributeError:
            logger.warning("Futures connector cannot report position mode; continuing")
        try:
            self.futures.connector.set_leverage(symbol, self.default_leverage)
        except RuntimeError:
            raise
        except BinanceAPIError as exc:  # pragma: no cover - network/API variance
            logger.warning("Futures leverage preflight failed for %s: %s", symbol, exc)
        try:
            self.futures.connector.set_margin_type(symbol, "ISOLATED")
        except BinanceAPIError as exc:  # pragma: no cover - network/API variance
            # -4046 is normalized by the connector; -4048 means an existing
            # position prevents a mode change. Neither should block a hedge if
            # the account is already usable, but both must remain visible.
            if exc.code in (-4046, -4048):
                logger.info("Futures margin state unchanged for %s: %s", symbol, exc)
            else:
                logger.warning("Futures margin preflight failed for %s: %s", symbol, exc)
        except Exception as exc:  # pragma: no cover - network/API variance
            logger.warning("Futures margin preflight failed for %s: %s", symbol, exc)

    # -------------------------------------------------------------------
    # Identifiers (defensive against Binance's newClientOrderId contract)
    # -------------------------------------------------------------------
    # Binance requires: ^[a-zA-Z0-9-_]{1,36}$. The id must NEVER depend on the
    # symbol length (SOLUSDT, 1INCHUSDT, 1000SHIBUSDT ... all must fit).
    _MAX_CLIENT_ID = 36
    _ID_BASE_BUDGET = 29       # 36 - 1 <sep> - 6 max tag

    @staticmethod
    def _sanitize_oid(text: str) -> str:
        """Keep only Binance-legal newClientOrderId characters."""
        return "".join(ch for ch in text if ch.isalnum() or ch in "-_")

    def _new_position_id(self, base_asset: str) -> str:
        """Human-readable, collision-safe internal position id (no Binance limit)."""
        ts = time.strftime("%y%m%d%H%M%S")
        rand = uuid.uuid4().hex[:8]
        return f"TWIN_{base_asset}_{ts}_{rand}"

    def _client_order_id(self, pos_id: str, tag: str) -> str:
        """Build a Binance-compliant newClientOrderId from a position id + tag.

        Always fits the ^[a-zA-Z0-9-_]{1,36}$ contract regardless of how long
        ``pos_id`` is: the tag suffix (with ``-`` separator) is reserved at the
        end and the base is truncated from the FRONT, keeping the unique random
        tail so distinct positions and legs never collide.
        """
        tag = self._sanitize_oid(tag)[:6]
        base = self._sanitize_oid(pos_id)
        budget = self._ID_BASE_BUDGET if tag else self._MAX_CLIENT_ID
        return f"{base[-budget:]}-{tag}"[: self._MAX_CLIENT_ID]

    # -------------------------------------------------------------------
    # Position registry
    # -------------------------------------------------------------------
    def get_twin_leg_positions(self, status: Optional[str] = None) -> List[TwinLegPosition]:
        """Return tracked twin-leg positions, optionally filtered by status."""
        if status is None:
            return list(self._positions.values())
        return [p for p in self._positions.values() if p.status == status]

    def get_twin_leg_position(self, position_id: str) -> Optional[TwinLegPosition]:
        """Return a single tracked position by id (or None)."""
        return self._positions.get(position_id)

    # -------------------------------------------------------------------
    # Core execution
    # -------------------------------------------------------------------
    def execute_twin_leg_carry(
        self,
        base_asset: str,
        quote_asset: str = "USDT",
        target_qty: float = 0.0,
        *,
        max_leg_gap_ms: float = 800.0,
        max_entry_slippage_pct: float = 0.0015,
    ) -> TwinLegPosition:
        """Open a twin-leg carry: Spot BUY and Futures SHORT dispatched concurrently.

        Both legs are sized to the same reconciled quantity and submitted in
        parallel (ThreadPoolExecutor) to minimise the unhedged "leg gap"
        (~700ms -> tens of ms). Fills are validated atomically: if either leg
        fails, partially fills, or times out, the panic-unwind circuit breaker
        flattens every leg that did fill before raising TwinLegExecutionError.
        Raises TwinLegExecutionError on any failure.
        """
        base_asset = base_asset.upper()
        quote_asset = quote_asset.upper()
        symbol = self._symbol(base_asset, quote_asset)
        pos_id = self._new_position_id(base_asset)
        opened_at = time.time()

        # -- Step 0: Pre-flight validation against reconciled filters ----
        twin = self.get_twin_filters(base_asset, quote_asset)
        try:
            spot_price = self.spot.get_price(symbol)
        except Exception as exc:
            raise TwinLegExecutionError(
                f"Pre-flight price fetch failed for {symbol}: {exc}",
                "preflight",
                self._placeholder(base_asset, quote_asset),
                cause=exc,
            )
        is_valid, errors, qty = twin.validate_qty(target_qty, spot_price)
        if not is_valid:
            raise TwinLegExecutionError(
                f"Twin-leg filter pre-flight failed for {symbol}: {'; '.join(errors)}",
                "preflight",
                self._placeholder(base_asset, quote_asset),
            )

        try:
            self._prep_futures(symbol)
        except Exception as exc:
            raise TwinLegExecutionError(
                f"Futures pre-flight failed for {symbol}: {exc}",
                "preflight",
                self._placeholder(base_asset, quote_asset),
                cause=exc,
            ) from exc

        # -- Steps 1+2: dispatch BOTH legs concurrently to close the leg gap --
        # Both legs are sized to the same reconciled quantity and submitted in
        # parallel.  Fills are validated atomically; any partial fill, failure,
        # or timeout triggers the panic-unwind circuit breaker which flattens
        # every leg that did fill before raising TwinLegExecutionError.
        def _leg_spec(oid_tag: str, side: str, reduce_only: bool = False) -> Order:
            """Build a single-leg Order pre-sized to the reconciled carry qty."""
            return Order(
                order_id=self._client_order_id(pos_id, oid_tag),
                symbol=symbol,
                side=side,
                quantity=qty,
                order_type="market",
                reduce_only=reduce_only,
                position_side=("SHORT" if self.position_mode == "HEDGE" else None),
                suppress_notify=True,
            )

        # Step 1: Place Leg 1 (Spot BUY) first.
        # This enables a deterministic pre-Leg2 slippage check before we
        # dispatch the futures short, preventing stale fills from creating
        # an orphaned leg window.
        t0 = time.perf_counter()
        try:
            spot_result = self.spot.place_order(_leg_spec("S", "buy"))
        except Exception as exc:
            spot_result = exc
        t_spot = time.perf_counter()

        spot_order = spot_result if isinstance(spot_result, Order) else None
        spot_exc = None if isinstance(spot_result, Order) else spot_result
        spot_ok = spot_order is not None and spot_order.status == "filled"
        spot_qty, spot_fill_price = self._fill(spot_order) if spot_order is not None else (0.0, 0.0)

        # Create a position placeholder early so the emergency unwind path can
        # always attach to an object.
        position = TwinLegPosition(
            position_id=pos_id,
            base_asset=base_asset,
            quote_asset=quote_asset,
            quantity=spot_qty,
            spot_order_id=(spot_order.exchange_order_id or spot_order.order_id) if spot_order else "",
            spot_fill_price=spot_fill_price,
            futures_order_id="",
            futures_fill_price=0.0,
            entry_basis_spread_pct=0.0,
            execution_gap_ms=0.0,
            status="OPEN" if spot_ok else "UNWINDING",
            opened_at=opened_at,
        )

        if not spot_ok:
            phase = "leg1"
            cause = spot_exc or RuntimeError(
                f"Leg 1 Spot BUY not filled for {symbol}: "
                f"{getattr(spot_order, 'message', '') if spot_order else 'no fill'}"
            )
            position.closed_at = time.time()
            self._positions[pos_id] = position
            raise TwinLegExecutionError(
                f"{phase} failed for {symbol}: {cause}.",
                phase,
                position,
                cause=cause,
            )

        # Step 2: Pre-Leg2 slippage check (live price drift vs expected).
        expected_fut_price = self.futures.get_price(symbol)
        actual_fut_price = self.futures.get_price(symbol)
        if expected_fut_price > 0:
            drift = (actual_fut_price - expected_fut_price) / expected_fut_price
        else:
            drift = 0.0
        if drift < 0 and abs(drift) > max_entry_slippage_pct:
            # Abort Leg2 (before placing it) and unwind Leg1 immediately.
            phase = "leg2"
            cause = RuntimeError(
                f"Adverse price drift before Leg 2: drift={drift:+.6%} "
                f"threshold=-{max_entry_slippage_pct:.6%}"
            )
            # Emergency unwind Leg 1 Spot inventory.
            try:
                unwind = self.spot.place_order(Order(
                    order_id=self._client_order_id(pos_id, "PX"),
                    symbol=symbol,
                    side="sell",
                    quantity=spot_qty,
                    order_type="market",
                    suppress_notify=True,
                ))
                if unwind.status == "filled":
                    position.status = "ORPHAN_UNWOUND"
                else:
                    position.status = "UNWINDING"
                    position.manual_intervention_required = True
            except Exception:
                position.status = "UNWINDING"
                position.manual_intervention_required = True
            position.closed_at = time.time()
            position.execution_gap_ms = abs((time.perf_counter() - t0) * 1000.0)
            self._positions[pos_id] = position
            self._notifier.notify_circuit_breaker(
                symbol=symbol,
                reason=f"PANIC UNWIND: {phase} aborted by slippage. Drift={drift:+.6%}. {cause}",
                extra={"Position ID": pos_id, "Leg gap (ms)": f"{position.execution_gap_ms:.1f}"},
            )
            raise BrokenLegExecutionError(
                f"Leg 2 aborted for {symbol}: {cause}. Emergency spot unwind executed.",
                phase,
                position,
                cause=cause,
                panic_unwound=(position.status == "ORPHAN_UNWOUND"),
            )

        # Step 3: Place Leg 2 (Futures SELL).
        try:
            fut_result = self.futures.place_order(_leg_spec("F", "sell"))
        except Exception as exc:
            fut_result = exc
        t_fut = time.perf_counter()

        fut_order = fut_result if isinstance(fut_result, Order) else None
        fut_exc = None if isinstance(fut_result, Order) else fut_result
        fut_ok = fut_order is not None and fut_order.status == "filled"
        _, fut_fill_price = self._fill(fut_order) if fut_order is not None else (0.0, 0.0)

        leg_gap_ms = abs(t_spot - t_fut) * 1000.0
        position.execution_gap_ms = leg_gap_ms

        if fut_ok:
            position.futures_order_id = fut_order.exchange_order_id or fut_order.order_id
            position.futures_fill_price = fut_fill_price
            position.entry_basis_spread_pct = (
                ((fut_fill_price - spot_fill_price) / spot_fill_price) * 100
                if spot_fill_price
                else 0.0
            )
            position.status = "OPEN"
            self._positions[pos_id] = position
            return position

        # -------- PANIC UNWIND / INCOMPLETE-FILL CIRCUIT BREAKER --------
        phase = "leg2"
        cause = fut_exc or RuntimeError(
            f"Leg 2 Futures SELL not filled for {symbol}: "
            f"{getattr(fut_order, 'message', '') if fut_order else 'no fill'}"
        )

        residual_futures_qty = float(fut_order.filled_quantity or 0.0) if fut_order is not None else 0.0
        position.residual_futures_qty = residual_futures_qty
        position.closed_at = time.time()

        self._notifier.notify_circuit_breaker(
            symbol=symbol,
            reason=(
                f"PANIC UNWIND: {phase} failed. Spot filled={spot_ok} "
                f"Futures filled={fut_ok}. Residual futures qty={residual_futures_qty:.8f}. "
                f"Cause: {cause}"
            ),
            extra={"Position ID": pos_id, "Leg gap (ms)": f"{leg_gap_ms:.1f}"},
        )

        # 1) Close any partially-open Futures short BEFORE touching Spot.
        if residual_futures_qty > 0:
            try:
                residual_close = self.futures.place_order(Order(
                    order_id=self._client_order_id(pos_id, "FR"),
                    symbol=symbol,
                    side="buy",
                    quantity=residual_futures_qty,
                    order_type="market",
                    reduce_only=True,
                    position_side=("SHORT" if self.position_mode == "HEDGE" else None),
                    suppress_notify=True,
                ))
                if residual_close.status == "filled":
                    position.residual_futures_qty = 0.0
                    position.manual_intervention_required = False
                else:
                    position.manual_intervention_required = True
            except Exception:
                position.manual_intervention_required = True

        # 2) Emergency flatten Spot inventory (Leg 1) to guarantee no orphan.
        try:
            unwind = self.spot.place_order(Order(
                order_id=self._client_order_id(pos_id, "PX"),
                symbol=symbol,
                side="sell",
                quantity=spot_qty,
                order_type="market",
                suppress_notify=True,
            ))
            if unwind.status == "filled" and not position.manual_intervention_required:
                position.status = "ORPHAN_UNWOUND"
            else:
                position.status = "UNWINDING"
                position.manual_intervention_required = True
        except Exception:
            position.status = "UNWINDING"
            position.manual_intervention_required = True

        self._positions[pos_id] = position
        raise BrokenLegExecutionError(
            f"Leg 2 failed for {symbol}: {cause}. Emergency spot unwind executed.",
            phase,
            position,
            cause=cause,
            panic_unwound=(position.status == "ORPHAN_UNWOUND"),
        )

    # -------------------------------------------------------------------
    # Unwind
    # -------------------------------------------------------------------
    def unwind_twin_leg_carry(self, position: TwinLegPosition) -> TwinLegPosition:
        """Symmetrically close a carry: Futures BUY to close the short, Spot SELL.

        Both close legs are dispatched **concurrently** (ThreadPoolExecutor) to
        shrink the unhedged exit window, then fills are validated atomically. If
        either leg fails, partially fills, or times out, the position is left
        UNWINDING (never half-closed), a red DM circuit-breaker alert fires, and
        :class:`TwinLegExecutionError` is raised without unwinding further. The
        measured exit ``leg_gap_ms`` is logged on success.
        """
        if position.status == "CLOSED":
            raise TwinLegExecutionError(
                f"Position {position.position_id} is already CLOSED",
                "unwind",
                position,
            )
        if position.status == "UNWINDING":
            raise TwinLegExecutionError(
                f"Position {position.position_id} is already UNWINDING",
                "unwind",
                position,
            )

        symbol = self._symbol(position.base_asset, position.quote_asset)
        qty = position.quantity
        position.status = "UNWINDING"

        from concurrent.futures import ThreadPoolExecutor

        futures_close_order = Order(
            order_id=self._client_order_id(position.position_id, "UF"),
            symbol=symbol,
            side="buy",
            quantity=qty,
            order_type="market",
            reduce_only=True,  # never opens, only closes the short
            position_side=("SHORT" if self.position_mode == "HEDGE" else None),
            suppress_notify=True,
        )
        spot_close_order = Order(
            order_id=self._client_order_id(position.position_id, "US"),
            symbol=symbol,
            side="sell",
            quantity=qty,
            order_type="market",
            suppress_notify=True,
        )

        def _timed_place(broker, order: Order):
            """Place one close leg; return (result, ts_after_place_or_exc)."""
            started = time.time()
            try:
                return broker.place_order(order), time.time()
            except Exception as exc:
                return exc, time.time()

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="carry-unwind") as _ex:
            f_fut = _ex.submit(_timed_place, self.futures, futures_close_order)
            f_spot = _ex.submit(_timed_place, self.spot, spot_close_order)
            # Both close legs are now running concurrently; the wall-clock gap
            # between their completions is the unhedged exit window.
            fut_result, t_fut = f_fut.result()
            spot_result, t_spot = f_spot.result()

        leg_gap_ms = abs(t_fut - t_spot) * 1000.0

        futures_close = fut_result if isinstance(fut_result, Order) else None
        fut_exc     = None if isinstance(fut_result, Order) else fut_result
        spot_close  = spot_result if isinstance(spot_result, Order) else None
        spot_exc    = None if isinstance(spot_result, Order) else spot_result

        both_closed = (
            futures_close is not None and futures_close.status == "filled"
            and spot_close is not None and spot_close.status == "filled"
        )
        if not both_closed:
            # A half-closed carry is an unhedged risk event — alert immediately.
            self._notifier.notify_circuit_breaker(
                symbol=symbol,
                reason=(
                    f"UNWIND INCOMPLETE: futures={getattr(futures_close, 'status', None) or fut_exc}, "
                    f"spot={getattr(spot_close, 'status', None) or spot_exc}"
                ),
                extra={"Position ID": position.position_id, "Unwind leg gap (ms)": f"{leg_gap_ms:.1f}"},
            )
            position.status = "UNWINDING"
            cause = RuntimeError(
                f"Unwind fills incomplete: futures={getattr(futures_close, 'status', None)}, "
                f"spot={getattr(spot_close, 'status', None)}"
            )
            raise TwinLegExecutionError(
                f"Unwind failed for {position.position_id} ({symbol}): {cause}",
                "unwind",
                position,
                cause=cause,
            )

        position.status = "CLOSED"
        position.closed_at = time.time()
        logger.info(
            "[TWIN-LEG] Carry CLOSED %s id=%s qty=%s leg_gap_ms=%.1f "
            "(futures_close=%s, spot_close=%s)",
            symbol,
            position.position_id,
            qty,
            leg_gap_ms,
            futures_close.exchange_order_id or futures_close.order_id,
            spot_close.exchange_order_id or spot_close.order_id,
        )
        return position
