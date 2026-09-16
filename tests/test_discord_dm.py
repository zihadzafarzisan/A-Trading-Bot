"""Hermetic tests for the Discord DM notification pipeline (no real network).

Verifies:
1. Missing credentials -> notifier disables itself; every dispatch is a no-op, never raises.
2. DM channel is fetched once then cached (single API call afterwards).
3. Embed payload/color correct per event; Authorization header uses the bot token.
4. Dispatch is non-blocking (fire-and-forget onto a background executor).
5. Network failures / 429 rate limits are caught and only logged; never raise.
"""

import json
import logging
import time

import pytest

from crypto_quant.notifications.discord_dm import (
    DiscordDMNotifier,
    _API_BASE,
    reset_notifier,
)
from crypto_quant import notifications  # module import safety (no side effects)

API_BASE = _API_BASE


class FakeResponse:
    """Minimal stand-in for a ``requests.Response``."""

    def __init__(self, status_code=200, json_body=None, text=""):
        self.status_code = status_code
        self._json = json_body if json_body is not None else {}
        self.text = text
        self.content = (
            json.dumps(self._json).encode() if self._json else text.encode()
        )

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class SyncExecutor:
    """Runs submitted callables synchronously for deterministic tests."""

    def __init__(self):
        self.submitted = 0

    def submit(self, fn, *args, **kwargs):
        self.submitted += 1
        fn(*args, **kwargs)


@pytest.fixture(autouse=True)
def _isolate():
    reset_notifier()
    yield
    reset_notifier()


def _make_notifier(monkeypatch, token="tok-123", user_id="user-456",
                   executor=None):
    calls = {"requests": []}
    dm_env = {"__name__": "discord_dm", "json": json}

    fake = FakeResponse(status_code=200, json_body={"id": "channel-987"})

    def fake_request(method, url, headers=None, json=None, timeout=None):
        calls["requests"].append((method, url, headers, json, timeout))
        if url.endswith("/users/@me/channels"):
            return FakeResponse(status_code=200, json_body={"id": "channel-987"})
        return fake

    monkeypatch.setattr(
        "crypto_quant.notifications.discord_dm.requests.request", fake_request
    )
    return DiscordDMNotifier(
        token=token, user_id=user_id, executor=executor or SyncExecutor()
    ), calls


# --- 1. Missing credentials -> disabled, safe no-op --------------------------
def test_missing_credentials_disables_and_noops(monkeypatch):
    n, calls = _make_notifier(monkeypatch, token="", user_id="")  # both missing
    assert n._enabled is False
    # Every dispatcher must return without touching the network or raising.
    n.notify_trade_open("BTCUSDT", "LONG", 0.001, 50000.0)
    n.notify_trade_close("BTCUSDT", "LONG", 0.001, 51000.0, realized_pnl=1.0)
    n.notify_funding_settlement("SOLUSDT", 0.5, 0.01)
    n.notify_circuit_breaker("SOLUSDT", "margin breach")
    # No HTTP was attempted (disabled notifiers short-circuit before I/O).
    assert calls["requests"] == []


def test_missing_user_id_only_disables(monkeypatch):
    n, _ = _make_notifier(monkeypatch, token="tok", user_id="")
    assert n._enabled is False


# --- 2. DM channel caching ---------------------------------------------------
def test_dm_channel_fetched_once_then_cached(monkeypatch):
    n, calls = _make_notifier(monkeypatch)
    n.notify_trade_open("BTCUSDT", "LONG", 0.001, 50000.0)
    n.notify_funding_settlement("BTCUSDT", 1.2, 0.01)

    channel_calls = [c for c in calls["requests"]
                     if "/users/@me/channels" in c[1]]
    message_calls = [c for c in calls["requests"] if "/messages" in c[1]]
    assert len(channel_calls) == 1, "DM channel should be fetched exactly once"
    assert len(message_calls) == 2
    assert n._dm_channel_id == "channel-987"
    # Every message call hits the cached channel and carries the bot token.
    for _, url, headers, _, _ in message_calls:
        assert url == f"{API_BASE}/channels/channel-987/messages"
        assert headers.get("Authorization") == "Bot tok-123"


# --- 3. Embed payload / colors / fields --------------------------------------
def test_trade_open_embed_green(monkeypatch):
    n, calls = _make_notifier(monkeypatch)
    n.notify_trade_open(
        "SOLUSDT", "CARRY", 0.1, 150.25,
        order_id="TWIN_1", is_carry=True,
        extra={"Predicted APR %": "+12.34%"},
    )
    _, _, _, body, _ = calls["requests"][-1]
    embed = body["embeds"][0]
    assert embed["color"] == DiscordDMNotifier.COLOR_GREEN
    assert "CARRY OPEN" in embed["title"] and "SOLUSDT" in embed["title"]
    names = {f["name"]: f["value"] for f in embed["fields"]}
    assert names["Quantity"] == "0.1 SOL"
    assert names["Fill Price"] == "$150.2500"
    assert names["Order ID"] == "TWIN_1"
    assert names["Predicted APR %"] == "+12.34%"


def test_trade_close_embed_blue(monkeypatch):
    n, calls = _make_notifier(monkeypatch)
    n.notify_trade_close("BTCUSDT", "LONG", 0.001, 51000.0,
                         realized_pnl=1.23, duration="4.0h")
    embed = calls["requests"][-1][3]["embeds"][0]
    assert embed["color"] == DiscordDMNotifier.COLOR_BLUE
    assert "EXIT" in embed["title"]
    names = {f["name"]: f["value"] for f in embed["fields"]}
    assert names["Realized PnL (USDT)"] == "+1.2300"
    assert names["Duration"] == "4.0h"


def test_funding_settlement_embed_green(monkeypatch):
    n, calls = _make_notifier(monkeypatch)
    n.notify_funding_settlement("SOLUSDT", 0.5, 0.01234)
    embed = calls["requests"][-1][3]["embeds"][0]
    assert embed["color"] == DiscordDMNotifier.COLOR_GREEN
    assert "FUNDING" in embed["title"].upper()
    names = {f["name"]: f["value"] for f in embed["fields"]}
    assert names["Funding Rate"] == "+0.012340%"
    assert names["Payout (USDT)"] == "+0.500000"


def test_circuit_breaker_embed_red(monkeypatch):
    n, calls = _make_notifier(monkeypatch)
    n.notify_circuit_breaker("SOLUSDT", "margin safety buffer breached")
    embed = calls["requests"][-1][3]["embeds"][0]
    assert embed["color"] == DiscordDMNotifier.COLOR_RED
    assert "CIRCUIT BREAKER" in embed["title"].upper()
    assert embed["fields"][0]["value"] == "margin safety buffer breached"


# --- 4. Non-blocking (fire-and-forget onto background executor) --------------
def test_dispatch_is_non_blocking(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    entered = threading.Event()
    release = threading.Event()

    def blocking_request(method, url, headers=None, json=None, timeout=None):
        entered.set()
        release.wait(timeout=5)

    monkeypatch.setattr(
        "crypto_quant.notifications.discord_dm.requests.request", blocking_request
    )
    n = DiscordDMNotifier(
        token="tok", user_id="user-456", executor=ThreadPoolExecutor(max_workers=2)
    )
    t0 = time.monotonic()
    n.notify_trade_open("BTCUSDT", "LONG", 0.001, 50000.0)
    elapsed = time.monotonic() - t0
    # The network work is delegated to another thread: dispatch returns instantly.
    assert elapsed < 0.25
    assert entered.wait(timeout=2), "background worker should be running the send"
    release.set()
    n._executor_or_new().shutdown(wait=True)


# --- 5. Failure isolation -----------------------------------------------------
def test_connection_error_is_contained(monkeypatch, caplog):
    def boom(method, url, headers=None, json=None, timeout=None):
        raise ConnectionError("network down")

    monkeypatch.setattr(
        "crypto_quant.notifications.discord_dm.requests.request", boom
    )
    n = DiscordDMNotifier(token="tok", user_id="user-456", executor=SyncExecutor())
    with caplog.at_level(logging.WARNING):
        # Must not raise even though transport throws.
        n.notify_trade_open("BTCUSDT", "LONG", 0.001, 50000.0)
    assert "send failed" in caplog.text.lower()


def test_429_rate_limit_is_contained(monkeypatch, caplog):
    call = {"hit": False}

    def ratelimit(method, url, headers=None, json=None, timeout=None):
        if url.endswith("/users/@me/channels"):
            return FakeResponse(status_code=200, json_body={"id": "c1"})
        call["hit"] = True
        return FakeResponse(status_code=429, text='{"retry_after": 1.5, "message": "rate limited"}')

    monkeypatch.setattr(
        "crypto_quant.notifications.discord_dm.requests.request", ratelimit
    )
    n = DiscordDMNotifier(token="tok", user_id="user-456", executor=SyncExecutor())
    with caplog.at_level(logging.WARNING):
        n.notify_trade_open("BTCUSDT", "LONG", 0.001, 50000.0)
    assert call["hit"] is True
    assert "429" in caplog.text


# --- Import safety / singleton ------------------------------------------------
def test_import_side_effects_none(monkeypatch):
    # Importing the package must be safe even with no env credentials.
    assert notifications.DiscordDMNotifier is DiscordDMNotifier
    assert getattr(notifications, "get_notifier", None) is not None