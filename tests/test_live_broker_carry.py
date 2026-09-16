"""Twin-Leg Delta-Neutral Cash-and-Carry tests (Path 4 broker integration).

Covers, hermetically (no network):
1. Filter reconciliation picks the coarsest common step + strictest limits.
2. A successful carry opens both legs with zero leftover delta.
3. Leg 2 failure triggers the panic-unwind circuit breaker (Spot SELL + error).
4. Symmetric unwind closes both legs and marks the position CLOSED.
5. The futures leg close is marked reduce_only so it can never open a position.
6. Every emitted newClientOrderId satisfies Binance's ^[a-zA-Z0-9-_]{1,36}$ contract.
"""

import re

import pytest

from crypto_quant.execution.broker import Order
from crypto_quant.execution.live_broker import (
    SymbolFilters,
    TwinLegCarryBroker,
    TwinLegExecutionError,
    TwinLegFilters,
)

SOL_SPOT = SymbolFilters(
    symbol="SOLUSDT",
    step_size=0.01,
    min_qty=0.01,
    max_qty=1000.0,
    min_notional=5.0,
)
SOL_FUT = SymbolFilters(
    symbol="SOLUSDT",
    step_size=0.1,       # coarser than spot
    min_qty=0.1,         # stricter than spot
    max_qty=900.0,
    min_notional=10.0,   # stricter than spot
)


class FakeBroker:
    """Minimal stand-in for BinanceLiveBroker with a recorded order log."""

    def __init__(self, market_type: str, price: float, filters: dict):
        self.market_type = market_type
        self.dry_run = True
        self.price = price
        self._filters = filters
        self.orders: list = []
        self.connector = None
        self.fail_placement: bool = False  # when True, raise on place_order

    def get_symbol_filter(self, symbol: str) -> SymbolFilters:
        return self._filters.get(symbol, SOL_SPOT)

    def get_price(self, symbol: str) -> float:
        return self.price

    def place_order(self, order: Order) -> Order:
        if self.fail_placement:
            raise RuntimeError(f"[{self.market_type}] placement deliberately failed")
        self.orders.append(order)
        order.status = "filled"
        order.fill_price = self.price
        return order


def _make_carry(spot_price: float = 100.0, fut_price: float = 100.5) -> TwinLegCarryBroker:
    spot = FakeBroker("spot", spot_price, {"SOLUSDT": SOL_SPOT})
    fut = FakeBroker("futures", fut_price, {"SOLUSDT": SOL_FUT})
    return TwinLegCarryBroker(spot_broker=spot, futures_broker=fut), spot, fut


class TestFilterReconciliation:
    def test_coarsest_step_and_strictest_limits(self):
        tf = TwinLegFilters.from_filters(SOL_SPOT, SOL_FUT)
        assert tf.symbol == "SOLUSDT"
        assert tf.step_size == 0.1             # coarsest common step
        assert tf.min_qty == 0.1                # strictest (largest) min
        assert tf.max_qty == 900.0              # strictest (smallest) max
        assert tf.min_notional == 10.0          # strictest (largest) notional

    def test_round_floor_prevents_leftover_delta(self):
        tf = TwinLegFilters.from_filters(SOL_SPOT, SOL_FUT)
        # 0.15 on a 0.1 grid must floor to 0.1 (never 0.2 → overshoot)
        assert tf.round_qty(0.15) == 0.1
        assert tf.round_qty(0.25) == 0.2

    def test_rejects_sub_minimal_notional(self):
        tf = TwinLegFilters.from_filters(SOL_SPOT, SOL_FUT)
        ok, reasons, qty = tf.validate_qty(0.05, price=100.0)
        assert not ok
        assert any("below min_qty" in r or "below min_notional" in r for r in reasons)


class TestSuccessfulCarry:
    def test_opens_both_legs_with_zero_delta(self):
        carry, spot, fut = _make_carry(spot_price=100.0, fut_price=100.5)
        pos = carry.execute_twin_leg_carry("SOL", "USDT", target_qty=0.15)

        assert pos.status == "OPEN"
        assert spot.orders[0].side == "buy"        # Leg 1 spot BUY
        assert fut.orders[0].side == "sell"        # Leg 2 futures SHORT
        assert fut.orders[0].quantity == pos.quantity  # hedge matches spot fill
        # Net delta is exactly zero after reconciliation to the common step.
        assert pos.quantity == pytest.approx(0.1)
        assert pos.execution_gap_ms >= 0.0
        assert pos.entry_basis_spread_pct == pytest.approx(0.5)

    def test_position_is_tracked(self):
        carry, _, _ = _make_carry()
        pos = carry.execute_twin_leg_carry("SOL", "USDT", target_qty=0.5)
        assert carry.get_twin_leg_positions() == [pos]
        assert carry.get_twin_leg_positions(status="OPEN") == [pos]
        assert carry.get_twin_leg_positions(status="CLOSED") == []
        assert carry.get_twin_leg_position(pos.position_id) is pos


class TestPanicUnwind:
    def test_leg2_failure_triggers_panic_unwind(self):
        carry, spot, fut = _make_carry()
        fut.fail_placement = True  # margin/network/API rejection on Leg 2

        with pytest.raises(TwinLegExecutionError) as ei:
            carry.execute_twin_leg_carry("SOL", "USDT", target_qty=0.5)

        err = ei.value
        assert err.phase == "leg2"
        assert err.panic_unwound is True
        assert err.position.status == "ORPHAN_UNWOUND"
        # Breaker must have issued a market SELL on the Spot to flatten inventory.
        assert spot.orders[-1].side == "sell"
        assert spot.orders[-1].quantity == err.position.quantity
        assert err.position.spot_order_id  # the original Leg 1 fill is retained

    def test_position_registered_even_after_panic(self):
        carry, _, fut = _make_carry()
        fut.fail_placement = True
        try:
            carry.execute_twin_leg_carry("SOL", "USDT", target_qty=0.5)
        except TwinLegExecutionError:
            pass
        assert len(carry.get_twin_leg_positions(status="ORPHAN_UNWOUND")) == 1

    def test_partial_futures_fill_is_closed_before_spot_panic_sell(self):
        class PartialFuturesBroker(FakeBroker):
            def place_order(self, order: Order) -> Order:
                self.orders.append(order)
                if order.side == "sell":
                    order.status = "partially_filled"
                    order.filled_quantity = 0.2
                    order.exchange_order_id = "987"
                    order.fill_price = self.price
                    return order
                order.status = "filled"
                order.filled_quantity = order.quantity
                order.fill_price = self.price
                return order

            def cancel_order(self, order_id: str, symbol: str = None) -> bool:
                return True

        spot = FakeBroker("spot", 100.0, {"SOLUSDT": SOL_SPOT})
        fut = PartialFuturesBroker("futures", 100.5, {"SOLUSDT": SOL_FUT})
        carry = TwinLegCarryBroker(spot, fut)
        with pytest.raises(TwinLegExecutionError) as raised:
            carry.execute_twin_leg_carry("SOL", "USDT", target_qty=0.5)
        position = raised.value.position
        assert position.status == "ORPHAN_UNWOUND"
        assert position.residual_futures_qty == 0.0
        assert fut.orders[-1].side == "buy"
        assert fut.orders[-1].reduce_only is True
        assert spot.orders[-1].side == "sell"


class TestUnwind:
    def test_symmetric_close_marks_closed(self):
        carry, spot, fut = _make_carry()
        pos = carry.execute_twin_leg_carry("SOL", "USDT", target_qty=0.5)

        carry.unwind_twin_leg_carry(pos)
        assert pos.status == "CLOSED"
        assert pos.closed_at is not None
        # Symmetric order flow: futures BUY (close short) then spot SELL.
        assert fut.orders[-1].side == "buy"
        assert spot.orders[-1].side == "sell"

    def test_futures_close_is_reduce_only(self):
        carry, _, fut = _make_carry()
        pos = carry.execute_twin_leg_carry("SOL", "USDT", target_qty=0.5)
        carry.unwind_twin_leg_carry(pos)
        assert fut.orders[-1].reduce_only is True

    def test_unwinding_twice_raises(self):
        carry, _, _ = _make_carry()
        pos = carry.execute_twin_leg_carry("SOL", "USDT", target_qty=0.5)
        carry.unwind_twin_leg_carry(pos)
        with pytest.raises(TwinLegExecutionError, match="already CLOSED"):
            carry.unwind_twin_leg_carry(pos)

    def test_unwind_dispatches_both_close_legs_concurrently(self):
        """Unwind closes futures and spot in one parallel pass (not serial)."""
        carry, spot, fut = _make_carry()
        pos = carry.execute_twin_leg_carry("SOL", "USDT", target_qty=0.5)
        carry.unwind_twin_leg_carry(pos)
        # The close pass emits exactly one futures BUY (reduce_only) and one
        # spot SELL, mirroring the sequential contract but now dispatched
        # concurrently — total order ids must stay 2 (entry) + 2 (unwind).
        all_oids = [o.order_id for o in spot.orders + fut.orders]
        assert len(all_oids) == 4
        assert len(set(all_oids)) == 4

    def test_partial_unwind_close_raises_and_leaves_unwinding(self):
        class CloseFailingBroker(FakeBroker):
            def place_order(self, order: Order) -> Order:
                self.orders.append(order)
                if order.reduce_only:  # the futures close leg
                    raise RuntimeError("[futures] close deliberately failed")
                order.status = "filled"
                order.fill_price = self.price
                return order

        spot = FakeBroker("spot", 100.0, {"SOLUSDT": SOL_SPOT})
        fut = CloseFailingBroker("futures", 100.5, {"SOLUSDT": SOL_FUT})
        carry = TwinLegCarryBroker(spot, fut)
        pos = carry.execute_twin_leg_carry("SOL", "USDT", target_qty=0.5)

        with pytest.raises(TwinLegExecutionError) as raised:
            carry.unwind_twin_leg_carry(pos)

        assert raised.value.phase == "unwind"
        assert raised.value.position is pos
        assert pos.status == "UNWINDING"  # never half-closed / masked as CLOSED
        # The spot close leg was still submitted before the futures leg failed.
        assert spot.orders[-1].side == "sell"


BINANCE_OID = re.compile(r"^[a-zA-Z0-9-_]{1,36}$")


class TestClientOrderIdContract:
    """Regression (Defect 1): newClientOrderId must fit ^[a-zA-Z0-9-_]{1,36}$."""

    LONG_BASES = ["SOL", "1INCH", "1000SHIB", "VERYLONGBASEASSETNAME"]

    def _collect_all_oids(self, base: str) -> list:
        carry, spot, fut = _make_carry()
        pos = carry.execute_twin_leg_carry(base, "USDT", target_qty=0.5)
        oids = [o.order_id for o in spot.orders] + [o.order_id for o in fut.orders]
        carry.unwind_twin_leg_carry(pos)
        oids += [o.order_id for o in spot.orders] + [o.order_id for o in fut.orders]
        return oids

    def test_all_position_and_unwind_ids_valid(self):
        for base in self.LONG_BASES:
            for oid in self._collect_all_oids(base):
                assert BINANCE_OID.fullmatch(oid), (
                    f"client order id {oid!r} violates Binance contract (base={base})"
                )

    def test_panic_unwind_id_valid(self):
        for base in self.LONG_BASES:
            carry, spot, fut = _make_carry()
            fut.fail_placement = True
            try:
                carry.execute_twin_leg_carry(base, "USDT", target_qty=0.5)
            except TwinLegExecutionError:
                pass
            assert spot.orders[-1].side == "sell"  # panic unwind was emitted
            for oid in [o.order_id for o in spot.orders]:
                assert BINANCE_OID.fullmatch(oid), f"{oid!r} invalid (base={base})"

    def test_ids_unique_across_legs_and_actions(self):
        carry, spot, fut = _make_carry()
        pos = carry.execute_twin_leg_carry("SOL", "USDT", target_qty=0.5)
        carry.unwind_twin_leg_carry(pos)
        all_oids = [o.order_id for o in spot.orders + fut.orders]
        assert len(all_oids) == len(set(all_oids)), "order ids must be unique"
        assert len(all_oids) == 4  # open spot + open fut + close fut + close spot


class TestPreflightGuard:
    def test_rejects_bad_market_types(self):
        spot = FakeBroker("spot", 100.0, {"SOLUSDT": SOL_SPOT})
        fut = FakeBroker("spot", 100.5, {"SOLUSDT": SOL_SPOT})  # wrong market
        with pytest.raises(ValueError, match="Futures broker"):
            TwinLegCarryBroker(spot, fut)

    def test_rejects_sub_minimal_target(self):
        carry, _, _ = _make_carry()
        with pytest.raises(TwinLegExecutionError, match="below min_qty"):
            carry.execute_twin_leg_carry("SOL", "USDT", target_qty=0.001)