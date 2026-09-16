"""Discord Bot DM notification pipeline.

Delivers trade alerts to the operator's personal Discord DMs using the
standard Discord REST API (no gateway / discord.py — no continuous websocket
connection needed). Every dispatch is:

- **Non-blocking**: network I/O is offloaded to a small background
  ``ThreadPoolExecutor``; public methods only submit and return immediately.
- **Failure-isolated**: all HTTP is wrapped in ``try/except Exception`` —
  timeouts, rate limits (429), and network errors are logged at WARNING and
  dropped cleanly. A Discord outage can never interrupt trading.
- **Safe-by-default**: if ``DISCORD_BOT_TOKEN`` or ``DISCORD_USER_ID`` is
  missing the notifier disables itself and every call becomes a silent no-op
  (never raises).

Flow:
1. Step A — fetch-or-cache the operator DM channel: ``POST /users/@me/channels``
   with body ``{"recipient_id": <user_id>}``. The returned channel ``id`` is
   cached so subsequent notifications need only a single API call.
2. Step B — post the embed: ``POST /channels/{dm_channel_id}/messages`` with
   body ``{"embeds": [<embed>]}``.
"""

from __future__ import annotations

# Defensive module-level hydration: load the root .env immediately at import
# time so the notifier is never constructed from a bare environment even when
# the CLI entry point (main.py) is bypassed or dotenv is loaded late. Pairs
# with the entry-point hydration in main.py for fully hands-off operation.
from dotenv import load_dotenv

load_dotenv()

import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from ..logging_config import get_logger

logger = get_logger("trading")

_API_BASE = "https://discord.com/api/v10"
_ENV_TOKEN = "DISCORD_BOT_TOKEN"
_ENV_USER_ID = "DISCORD_USER_ID"


class DiscordDMNotifier:
    """REST-only Discord DM alert client with non-blocking, isolated dispatch."""

    # ------------------------------------------------------------------
    # Candle completion notifications
    # ------------------------------------------------------------------
    def notify_candle_close(
        self,
        symbol: str,
        timeframe: str,
        ohlc_dict: Dict[str, Any],
        trend_bias: Optional[str] = None,
    ) -> None:
        """Blue embed — evaluation candle close.

        Called by live candle processing when an exchange kline marks the
        current candle as completed (the "x=True" bar).

        The method is best-effort: submit only, never blocks or raises.
        """
        open_p = ohlc_dict.get("open")
        high_p = ohlc_dict.get("high")
        low_p = ohlc_dict.get("low")
        close_p = ohlc_dict.get("close")
        ts = ohlc_dict.get("timestamp")

        title = f"🕯️ Candle Close — {symbol}"
        color = self.COLOR_BLUE

        fields_payload: Dict[str, Any] = {
            "Timeframe": timeframe,
            "Open": f"${open_p:,.4f}" if open_p is not None else "—",
            "High": f"${high_p:,.4f}" if high_p is not None else "—",
            "Low": f"${low_p:,.4f}" if low_p is not None else "—",
            "Close": f"${close_p:,.4f}" if close_p is not None else "—",
        }
        if trend_bias:
            fields_payload["Trend Bias"] = trend_bias
        if ts is not None:
            # Accept either seconds epoch or ISO-ish strings
            try:
                if isinstance(ts, (int, float)):
                    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                    fields_payload["Candle Time"] = dt.isoformat()
                else:
                    fields_payload["Candle Time"] = str(ts)
            except Exception:
                fields_payload["Candle Time"] = str(ts)

        self._dispatch(title, color, fields=_fields(fields_payload))

    # ------------------------------------------------------------------
    # Trade lifecycle notifications (public API)
    # ------------------------------------------------------------------
    # These new methods provide richer trade-entry/trade-exit embeds.
    # For backwards-compatibility, notify_trade_open/notify_trade_close
    # delegate to them (so existing callers automatically get the new
    # embeds without duplicated messages).

    # Rich embed colors (Discord decimal color codes).
    COLOR_GREEN = 0x00D084  # 53380  — entry / carry open / positive funding
    COLOR_BLUE = 0x2A9D8F   # 2792847 – standard exit / carry unwind
    COLOR_RED = 0xE74C3C    # 15158332 – panic unwind / circuit breaker / margin

    def __init__(
        self,
        token: Optional[str] = None,
        user_id: Optional[str] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        """Initialize from env vars unless overridden.

        Args:
            token: Bot token; defaults to ``DISCORD_BOT_TOKEN``.
            user_id: Target user's snowflake id; defaults to ``DISCORD_USER_ID``.
            executor: Optional shared executor (test injection). A daemon
                ``ThreadPoolExecutor(max_workers=2)`` is created when omitted.
        """
        self._token = token if token is not None else os.getenv(_ENV_TOKEN, "").strip()
        self._user_id = user_id if user_id is not None else os.getenv(_ENV_USER_ID, "").strip()
        self._dm_channel_id: Optional[str] = None
        self._enabled = bool(self._token and self._user_id)
        if not self._enabled:
            logger.debug(
                "Discord notifier disabled: missing %s/%s",
                _ENV_TOKEN,
                _ENV_USER_ID,
            )
        self._executor = executor

    def is_enabled(self) -> bool:
        """True when a bot token and target user id are configured for dispatch.

        Provided as a public, stable query so callers (e.g. the carry daemon)
        can surface disabled-notification warnings without touching internals.
        """
        return self._enabled

    # ------------------------------------------------------------------
    # Public event dispatchers (submit only — never block or raise)
    # ------------------------------------------------------------------
    def notify_trade_entry(
        self,
        symbol: str,
        strategy: str,
        side: str,
        qty: float,
        spot_fill_price: float,
        futures_fill_price: Optional[float] = None,
        allocated_capital_usdt: Optional[float] = None,
        target_apr_pct: Optional[float] = None,
        order_id: Optional[str] = None,
        entry_timestamp: Optional[datetime] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Green embed — trade entry.

        Used for both:
        - directional trades (futures_fill_price may be None)
        - carry twin-leg positions (spot_fill_price + futures_fill_price)
        """
        is_carry = futures_fill_price is not None
        title = f"🟢 CARRY ARBITRAGE — {symbol}" if is_carry else f"🟢 ENTRY — {symbol}"
        leg = side

        fields_payload: Dict[str, Any] = {
            "Symbol & Strategy": f"{symbol} · {strategy}",
            "Side": leg,
            "Quantity": f"{_num(qty)} {_base(symbol)}",
            "Spot Fill Price": f"${spot_fill_price:,.4f}",
        }
        if futures_fill_price is not None:
            fields_payload["Futures Fill Price"] = f"${futures_fill_price:,.4f}"

        if allocated_capital_usdt is not None:
            fields_payload["Allocated Capital (USDT)"] = f"${allocated_capital_usdt:,.2f}"
        if target_apr_pct is not None:
            fields_payload["Target APR (%)"] = f"{target_apr_pct:+.4f}%"
        if order_id is not None:
            fields_payload["Order ID / Position ID"] = order_id
        if entry_timestamp is not None:
            # Discord embeds prefer explicit timestamps.
            fields_payload["Entry Timestamp"] = entry_timestamp.isoformat()

        self._dispatch(title, self.COLOR_GREEN, fields=_fields(fields_payload, extra), is_carry=is_carry)

    def notify_trade_exit(
        self,
        symbol: str,
        position_id: Optional[str],
        strategy: str,
        side: str,
        qty: float,
        spot_fill_price: Optional[float] = None,
        futures_fill_price: Optional[float] = None,
        gross_pnl_usdt: Optional[float] = None,
        net_pnl_usdt: Optional[float] = None,
        funding_harvested_usdt: Optional[float] = None,
        roi_pct: Optional[float] = None,
        duration: Optional[str] = None,
        exit_reason: Optional[str] = None,
        entry_timestamp: Optional[datetime] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Trade exit embed (green WIN / red LOSS)."""
        is_carry = futures_fill_price is not None or funding_harvested_usdt is not None

        outcome = "WIN" if (net_pnl_usdt is not None and net_pnl_usdt >= 0) else "LOSS"
        title = (
            f"🟢 {outcome} — {symbol}" if outcome == "WIN" else f"🔴 {outcome} — {symbol}"
        )
        color = self.COLOR_GREEN if outcome == "WIN" else self.COLOR_RED

        fields_payload: Dict[str, Any] = {
            "Symbol & Strategy": f"{symbol} · {strategy}",
            "Position ID": position_id or "—",
            "Side": side,
            "Quantity": f"{_num(qty)} {_base(symbol)}",
        }
        if spot_fill_price is not None:
            fields_payload["Spot Fill Price"] = f"${spot_fill_price:,.4f}"
        if futures_fill_price is not None:
            fields_payload["Futures Fill Price"] = f"${futures_fill_price:,.4f}"

        if gross_pnl_usdt is not None:
            fields_payload["Gross PnL (USDT)"] = f"{gross_pnl_usdt:+,.4f}"
        if net_pnl_usdt is not None:
            fields_payload["Net PnL (USDT)"] = f"{net_pnl_usdt:+,.4f}"
        if funding_harvested_usdt is not None:
            fields_payload["Funding Harvested (USDT)"] = f"{funding_harvested_usdt:+,.4f}"
        if roi_pct is not None:
            fields_payload["ROI (%)"] = f"{roi_pct:+.3f}%"
        if duration is not None:
            fields_payload["Hold Time"] = duration
        if exit_reason is not None:
            fields_payload["Exit Reason"] = exit_reason

        if entry_timestamp is not None:
            fields_payload["Entry Timestamp"] = entry_timestamp.isoformat()

        # Ensure we keep the old payload format style.
        self._dispatch(title, color, fields=_fields(fields_payload, extra), is_carry=is_carry)

    # ------------------------------------------------------------------
    # Backwards-compatible wrappers
    # ------------------------------------------------------------------
    def notify_trade_open(
        self,
        symbol: str,
        side: str,
        qty: float,
        fill_price: float,
        order_id: Optional[str] = None,
        is_carry: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Green embed — entry / carry open (legacy behavior)."""
        title = f"🟢 CARRY OPEN — {symbol}" if is_carry else f"🟢 ENTRY — {symbol}"
        fields = _fields(
            {
                "Leg": side,
                "Quantity": f"{_num(qty)} {_base(symbol)}",
                "Fill Price": f"${fill_price:,.4f}",
                "Order ID": order_id,
            },
            extra,
        )
        self._dispatch(title, self.COLOR_GREEN, fields, is_carry)

    def notify_trade_close(
        self,
        symbol: str,
        side: str,
        qty: float,
        fill_price: float,
        realized_pnl: Optional[float] = None,
        duration: Optional[str] = None,
        is_carry: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Blue embed — standard exit / carry unwind (legacy behavior)."""
        title = f"🔵 CARRY UNWIND — {symbol}" if is_carry else f"🔵 EXIT — {symbol}"
        values: Dict[str, Any] = {
            "Leg": side,
            "Quantity": f"{_num(qty)} {_base(symbol)}",
            "Fill Price": f"${fill_price:,.4f}",
        }
        if realized_pnl is not None:
            values["Realized PnL (USDT)"] = f"{realized_pnl:+,.4f}"
        if duration is not None:
            values["Duration"] = duration
        if is_carry:
            values["Action"] = "Position closed / unwound"
        self._dispatch(title, self.COLOR_BLUE, fields=_fields(values, extra), is_carry=is_carry)


    def notify_funding_settlement(
        self,
        symbol: str,
        payment_usdt: float,
        funding_rate_pct: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Green embed — confirmed 8h funding settlement on an open carry."""
        fields = _fields({
            "Funding Rate": f"{funding_rate_pct:+.6f}%",
            "Payout (USDT)": f"{payment_usdt:+.6f}",
        }, extra)
        self._dispatch(f"🟢 FUNDING SETTLEMENT — {symbol}", self.COLOR_GREEN, fields)

    def notify_circuit_breaker(
        self,
        symbol: str,
        reason: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Red embed — panic unwind / circuit breaker / margin warning."""
        fields = _fields({
            "Reason": reason,
        }, extra)
        self._dispatch(f"🔴 CIRCUIT BREAKER — {symbol}", self.COLOR_RED, fields)

    def notify_financial_audit(
        self,
        available_free_margin_usdt: float,
        allocated_capital_usdt: float,
        net_unrealized_pnl_usdt: float,
        harvested_funding_usdt: float,
        open_positions: int = 0,
        margin_safety_buffer_pct: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Blue embed — periodic / lifecycle portfolio financial audit.

        Shows cash/collateral vs. capital invested (notional), realized (funding
        harvested) vs. unrealized PnL, the health/liquidation buffer, and the
        live open-book size + per-pair holding durations — so financial status
        is visible while positions hold.
        """
        fields = _fields({
            "Available Free Margin": f"${available_free_margin_usdt:,.2f}",
            "Capital Allocated": f"${allocated_capital_usdt:,.2f}",
            "Net Unrealized PnL": f"{net_unrealized_pnl_usdt:+,.2f}",
            "Harvested Funding": f"{harvested_funding_usdt:+,.4f}",
        }, extra)
        if margin_safety_buffer_pct is not None:
            fields.append({
                "name": "Margin Safety Buffer",
                "value": f"{margin_safety_buffer_pct:.2f}%",
                "inline": True,
            })
        fields.append({"name": "Open Positions", "value": str(open_positions), "inline": True})
        self._dispatch("💳 FINANCIAL AUDIT — PORTFOLIO", self.COLOR_BLUE, fields)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _dispatch(
        self,
        title: str,
        color: int,
        fields: List[Dict[str, Any]],
        is_carry: bool = False,
    ) -> None:
        """Submit one embed send to the executor (or no-op when disabled)."""
        if not self._enabled:
            logger.debug("Discord notification dropped (notifier disabled): %s", title)
            return
        embed: Dict[str, Any] = {
            "title": title,
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if fields:
            embed["fields"] = fields
        self._executor_or_new().submit(self._post_embed, embed)

    def _executor_or_new(self) -> ThreadPoolExecutor:
        if self._executor is None:
            # Daemon threads: never block interpreter shutdown.
            self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="discord-dm")
        return self._executor

    def _post_embed(self, embed: Dict[str, Any]) -> None:
        """Best-effort send. Never raises — failures are logged (with traceback)
        and dropped, so a bad token/payload is visible to the operator instead of
        silently swallowed."""
        try:
            channel_id = self._ensure_dm_channel()
            url = f"{_API_BASE}/channels/{channel_id}/messages"
            self._request("POST", url, json={"embeds": [embed]})
        except Exception as exc:
            logger.warning(
                "Discord DM send failed for %r: %s",
                embed.get("title"), exc, exc_info=True,
            )

    def _ensure_dm_channel(self) -> str:
        """Return (and cache) the operator DM channel id."""
        if self._dm_channel_id:
            return self._dm_channel_id
        url = f"{_API_BASE}/users/@me/channels"
        res = self._request("POST", url, json={"recipient_id": self._user_id})
        channel_id = str(res.get("id", ""))
        if not channel_id:
            raise RuntimeError("Discord did not return a DM channel id")
        self._dm_channel_id = channel_id
        return channel_id

    def _request(self, method: str, url: str, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """HTTPS call with a unified Authorization header and 429 handling."""
        headers = {"Authorization": f"Bot {self._token}", "Content-Type": "application/json"}
        resp = requests.request(method, url, headers=headers, json=json, timeout=10.0)
        if resp.status_code == 429 and "retry_after" in resp.text:
            logger.warning(
                "Discord rate limit (429): retry_after=%.1fs — notification dropped: %s",
                resp.json().get("retry_after", 0.0),
                url,
            )
            return {}
        resp.raise_for_status()
        return resp.json() if resp.content else {}


def get_notifier() -> DiscordDMNotifier:
    """Lazy process-wide singleton (mirrors ``get_config`` in config/settings.py).

    The singleton may first be created during module import — e.g. when
    ``live_broker.py`` is imported by ``_carry_build_broker`` *before* the CLI
    calls ``load_dotenv`` — at which point ``DISCORD_BOT_TOKEN``/``DISCORD_USER_ID``
    are not yet in the environment, leaving it silently disabled for the whole
    run (the "zero Discord notifications with positions opening" symptom). If the
    cached instance is disabled but credentials are now present, rebuild it so
    notifications actually go out.
    """
    global _notifier
    if _notifier is None:
        _notifier = DiscordDMNotifier()
    elif not _notifier.is_enabled() and (
        os.getenv(_ENV_TOKEN, "").strip() and os.getenv(_ENV_USER_ID, "").strip()
    ):
        logger.warning(
            "Discord notifier was created disabled (env loaded late); recreating "
            "now that DISCORD_BOT_TOKEN/DISCORD_USER_ID are present."
        )
        _notifier = DiscordDMNotifier(executor=_notifier._executor)
    return _notifier


_notifier: Optional[DiscordDMNotifier] = None


def reset_notifier() -> None:
    """Clear the singleton (tests)."""
    global _notifier
    _notifier = None


def _fields(prelude: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Build ordered Discord embed fields from a prelude dict plus optional extras."""
    ordered: List[Dict[str, Any]] = []
    for key, value in prelude.items():
        if value is not None and value != "":
            ordered.append({"name": key, "value": str(value), "inline": True})
    for key, value in (extra or {}).items():
        if value is not None and value != "":
            ordered.append({"name": str(key), "value": str(value), "inline": True})
    return ordered


def _base(symbol: str) -> str:
    """Best-effort base asset derived from a trailing-USDT symbol."""
    return symbol[:-4] if str(symbol).endswith("USDT") else str(symbol)


def _num(value: float) -> str:
    """Render a quantity with sensible precision."""
    return f"{value:,.8f}".rstrip("0").rstrip(".") if value else f"{value:,.8f}"