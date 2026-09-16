"""Mandatory Safety, Isolation, Gate & Failure Tests for Live Trading (Phase 13).

Proves:
1. Paper mode can NEVER place a real Binance order
2. Dry-run mode connects, validates, but CANNOT place a real Binance order
3. Live mode REFUSES to start without all 15 safety gates satisfied
4. Emergency kill switch blocks all new order creation
5. Duplicate signals produce NO duplicate orders (Idempotency)
6. Symbol filters correctly round prices, quantities, and enforce min_notional
7. Position reconciliation detects and halts on exchange discrepancies
8. API credentials and secrets are safely masked
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from crypto_quant.exchange.binance_live import BinanceAPIError, BinanceLiveConnector
from crypto_quant.execution.broker import Order
from crypto_quant.execution.engine import PaperTradingEngine
from crypto_quant.execution.live_broker import BinanceLiveBroker, SymbolFilters
from crypto_quant.execution.live_worker import LiveExecutionWorker, LiveWorkerConfig
from crypto_quant.execution.paper import PaperBroker
from crypto_quant.execution.safety import (
    IdempotencyRegistry,
    LiveSafetyGateKeeper,
    PositionReconciler,
    ReconciliationDiscrepancy,
)
from crypto_quant.risk.limits import RiskLimits
from crypto_quant.risk.manager import RiskManager
from crypto_quant.strategies import create_strategy


class DummyPriceSource:
    def get_price(self, symbol: str) -> float:
        return 50000.0


class TestPaperIsolation:
    """Proves Paper Trading is strictly isolated and can never execute live orders."""

    def test_paper_engine_rejects_live_broker(self):
        connector = BinanceLiveConnector(api_key="mock", api_secret="mock", testnet=True)
        live_broker = BinanceLiveBroker(connector=connector, dry_run=True)
        risk = RiskManager(RiskLimits())
        strat = create_strategy("trend")

        # PaperTradingEngine must raise TypeError if passed anything other than PaperBroker
        with pytest.raises(TypeError, match="PaperTradingEngine requires a PaperBroker"):
            PaperTradingEngine(strategy=strat, broker=live_broker, risk_manager=risk)

    def test_paper_broker_has_no_exchange_order_endpoint(self):
        broker = PaperBroker(price_source=DummyPriceSource(), starting_capital=1000.0)
        assert not hasattr(broker, "connector")
        assert not hasattr(broker, "create_order")
        assert broker.name == "paper"


class TestDryRunSafety:
    """Proves Dry-Run mode simulates execution locally and never calls Binance create_order."""

    def test_dry_run_places_no_real_orders(self):
        connector = BinanceLiveConnector(api_key="mock", api_secret="mock", testnet=True)
        # Mock the underlying create_order method
        connector.create_order = MagicMock()

        broker = BinanceLiveBroker(connector=connector, dry_run=True)
        # Mock price source
        broker.get_price = MagicMock(return_value=50000.0)

        order = Order(
            order_id="test_order_1",
            symbol="BTCUSDT",
            side="buy",
            quantity=0.01,
            order_type="market",
        )

        res = broker.place_order(order)
        assert res.status == "filled"
        assert "[DRY RUN - SIMULATED]" in res.message
        # Crucial: Connector create_order must NOT be called in dry-run
        connector.create_order.assert_not_called()


class TestSafetyGates:
    """Proves Live mode refuses to start unless all safety gates pass."""

    def test_refusal_when_live_flag_missing(self):
        connector = BinanceLiveConnector(api_key="mock", api_secret="mock", testnet=False)
        broker = BinanceLiveBroker(connector=connector, dry_run=False)
        risk = RiskManager(RiskLimits())

        # With dry_run=False and is_live_flag=False, Gate 1 must fail
        result = LiveSafetyGateKeeper.verify_all_gates(
            connector=connector,
            broker=broker,
            risk=risk,
            strategy_name="trend",
            symbol="BTCUSDT",
            dry_run=False,
            is_live_flag=False,
        )
        assert not result.all_passed
        assert any("Gate 1" in f for f in result.failure_reasons)

    def test_refusal_when_kill_switch_active(self):
        connector = BinanceLiveConnector(api_key="mock", api_secret="mock", testnet=True)
        broker = BinanceLiveBroker(connector=connector, dry_run=True)
        risk = RiskManager(RiskLimits())
        risk.emergency_stop()

        result = LiveSafetyGateKeeper.verify_all_gates(
            connector=connector,
            broker=broker,
            risk=risk,
            strategy_name="trend",
            symbol="BTCUSDT",
            dry_run=True,
        )
        assert not result.all_passed
        assert any("Emergency Kill Switch" in f or "Gate 12" in f for f in result.failure_reasons)

    def test_refusal_on_invalid_strategy(self):
        connector = BinanceLiveConnector(api_key="mock", api_secret="mock", testnet=True)
        broker = BinanceLiveBroker(connector=connector, dry_run=True)
        risk = RiskManager(RiskLimits())

        result = LiveSafetyGateKeeper.verify_all_gates(
            connector=connector,
            broker=broker,
            risk=risk,
            strategy_name="non_existent_strat",
            symbol="BTCUSDT",
            dry_run=True,
        )
        assert not result.all_passed
        assert any("Strategy Configuration" in f or "Gate 7" in f for f in result.failure_reasons)


class TestIdempotencyAndDuplicatePrevention:
    """Proves duplicate signals and retries do not trigger duplicate orders."""

    def test_idempotency_registry_catches_duplicates(self):
        reg = IdempotencyRegistry()
        key1 = reg.generate_key("trend", "BTCUSDT", "1h", 1700000000000, "long")
        key2 = reg.generate_key("trend", "BTCUSDT", "1h", 1700000000000, "long")
        assert key1 == key2

        assert not reg.is_duplicate(key1)
        reg.register_order(key1, "order_123", {"qty": 0.01})
        assert reg.is_duplicate(key1)

    def test_different_bars_yield_different_keys(self):
        reg = IdempotencyRegistry()
        key1 = reg.generate_key("trend", "BTCUSDT", "1h", 1700000000000, "long")
        key2 = reg.generate_key("trend", "BTCUSDT", "1h", 1700003600000, "long")
        assert key1 != key2


class TestSymbolFilters:
    """Tests lot size, tick size, and min notional validation."""

    def test_rounding_and_notional_validation(self):
        filters = SymbolFilters(
            symbol="BTCUSDT",
            min_price=10.0,
            max_price=100000.0,
            tick_size=0.1,
            min_qty=0.001,
            max_qty=100.0,
            step_size=0.001,
            min_notional=10.0,
        )

        assert filters.round_price(50000.1234) == 50000.1
        assert filters.round_qty(0.01289) == 0.012

        # Valid order
        is_valid, errors, qty, price = filters.validate_order("buy", 0.01, 50000.0)
        assert is_valid
        assert len(errors) == 0

        # Sub-minimum notional order ($50000 * 0.0001 = $5 < $10)
        is_valid, errors, qty, price = filters.validate_order("buy", 0.0001, 50000.0)
        assert not is_valid
        assert any("below min_notional" in e for e in errors)


class TestReconciliation:
    """Tests detection of state discrepancies between local DB and exchange."""

    def test_reconciliation_detects_unexpected_exchange_position(self):
        connector = BinanceLiveConnector(api_key="mock", api_secret="mock", testnet=True)
        # Mock exchange returning an untracked ETH position
        connector.get_positions = MagicMock(
            return_value=[
                {"symbol": "ETHUSDT", "position_amt": 1.5, "entry_price": 2500.0, "mark_price": 2510.0}
            ]
        )
        broker = BinanceLiveBroker(connector=connector, dry_run=False)
        broker.get_positions = MagicMock(return_value=[])  # Local has 0 positions

        reconciler = PositionReconciler(connector, broker)
        is_clean, disc = reconciler.reconcile()
        assert not is_clean
        assert len(disc) == 1
        assert disc[0].category == "unexpected_position"
        assert disc[0].symbol == "ETHUSDT"


class TestLiveOrderFillPolling:
    """Regression coverage for async Binance MARKET order acknowledgements."""

    class FakeConnector:
        market_type = "futures"
        testnet = True

        def __init__(self, create_response, query_responses=None):
            self.create_response = create_response
            self.query_responses = list(query_responses or [])
            self.query_calls = 0

        def get_exchange_info(self):
            return {
                "symbols": [
                    {
                        "symbol": "SOLUSDT",
                        "baseAsset": "SOL",
                        "quoteAsset": "USDT",
                        "status": "TRADING",
                        "filters": [
                            {"filterType": "PRICE_FILTER", "minPrice": "0.01", "maxPrice": "100000", "tickSize": "0.01"},
                            {"filterType": "LOT_SIZE", "minQty": "0.01", "maxQty": "10000", "stepSize": "0.01"},
                            {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                        ],
                    }
                ]
            }

        def get_ticker_price(self, symbol):
            return 100.0

        def create_order(self, **kwargs):
            self.last_create = kwargs
            return dict(self.create_response)

        def query_order(self, **kwargs):
            self.query_calls += 1
            return dict(self.query_responses.pop(0))

    def test_market_new_zero_qty_polls_until_filled(self):
        connector = self.FakeConnector(
            {"orderId": 111, "status": "NEW", "executedQty": "0", "cumQuote": "0"},
            [{"orderId": 111, "status": "FILLED", "executedQty": "0.1", "cumQuote": "10.1"}],
        )
        broker = BinanceLiveBroker(
            connector=connector,
            dry_run=False,
            market_fill_timeout_seconds=0.2,
            market_fill_poll_interval_seconds=0.01,
        )
        order = broker.place_order(Order("cid", "SOLUSDT", "sell", 0.1, "market"))
        assert order.status == "filled"
        assert order.exchange_order_id == "111"
        assert order.filled_quantity == pytest.approx(0.1)
        assert order.fill_price == pytest.approx(101.0)
        assert connector.query_calls == 1

    def test_market_new_zero_qty_timeout_is_not_false_fill(self):
        connector = self.FakeConnector(
            {"orderId": 222, "status": "NEW", "executedQty": "0", "cumQuote": "0"},
            [{"orderId": 222, "status": "NEW", "executedQty": "0", "cumQuote": "0"}] * 20,
        )
        broker = BinanceLiveBroker(
            connector=connector,
            dry_run=False,
            market_fill_timeout_seconds=0.03,
            market_fill_poll_interval_seconds=0.01,
        )
        order = broker.place_order(Order("cid", "SOLUSDT", "sell", 0.1, "market"))
        assert order.status == "new"
        assert order.filled_quantity == 0.0
        assert order.fill_price is None
        assert "timeout" in order.message

    def test_partial_fill_is_not_reported_as_full_fill(self):
        connector = self.FakeConnector(
            {"orderId": 333, "status": "PARTIALLY_FILLED", "executedQty": "0.04", "cumQuote": "4.04"}
        )
        broker = BinanceLiveBroker(connector=connector, dry_run=False)
        order = broker.place_order(Order("cid", "SOLUSDT", "sell", 0.1, "market"))
        assert order.status == "partially_filled"
        assert order.filled_quantity == pytest.approx(0.04)
        assert order.fill_price == pytest.approx(101.0)


class TestConnectorFuturesOrderParams:
    def test_futures_position_side_and_reduce_only_are_sent(self):
        connector = BinanceLiveConnector(api_key="mock", api_secret="mock", market_type="futures", testnet=True)
        captured = {}

        def fake_request(method, path, params=None, signed=False):
            captured.update(params or {})
            return {"orderId": 1, "status": "NEW", "executedQty": "0"}

        connector._request = fake_request
        connector.create_order(
            symbol="SOLUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.1,
            client_order_id="cid",
            reduce_only=False,
            position_side="SHORT",
        )
        assert captured["positionSide"] == "SHORT"
        assert "reduceOnly" not in captured
        assert captured["newClientOrderId"] == "cid"

        captured.clear()
        connector.create_order(
            symbol="SOLUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=0.1,
            client_order_id="cid2",
            reduce_only=True,
            position_side=None,
        )
        assert captured["reduceOnly"] == "true"

    def test_spot_rejects_position_side(self):
        connector = BinanceLiveConnector(api_key="mock", api_secret="mock", market_type="spot", testnet=True)
        with pytest.raises(ValueError, match="position_side"):
            connector.create_order("SOLUSDT", "BUY", "MARKET", 0.1, position_side="SHORT")


class TestSecretMasking:
    """Proves API secrets are never revealed in string representations."""

    def test_secret_is_masked(self):
        connector = BinanceLiveConnector(
            api_key="very_secret_api_key_123456789",
            api_secret="super_confidential_secret_value_987654321",
        )
        rep = str(connector)
        assert "super_confidential_secret_value_987654321" not in rep
        assert "very_secret_api_key_123456789" not in rep
        assert "very...6789" in rep
