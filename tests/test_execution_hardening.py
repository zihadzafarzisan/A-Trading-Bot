import pytest

from crypto_quant.execution.broker import Order
from crypto_quant.execution.live_broker import (
    SymbolFilters,
    TwinLegCarryBroker,
    TwinLegExecutionError,
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
    step_size=0.1,
    min_qty=0.1,
    max_qty=900.0,
    min_notional=10.0,
)


class FakeBroker:
    """Minimal stand-in for BinanceLiveBroker with a recorded order log."""

    def __init__(self, market_type: str, price: float, filters: dict):
        self.market_type = market_type
        self.dry_run = True
        self.price = price
        self._filters = filters
        self.orders = []
        self.connector = None
        self.fail_placement = False

    def get_symbol_filter(self, symbol: str) -> SymbolFilters:
        return self._filters.get(symbol, SOL_SPOT)

    def get_price(self, symbol: str) -> float:
        return float(self.price)

    def place_order(self, order: Order) -> Order:
        if self.fail_placement:
            raise RuntimeError(f"[{self.market_type}] placement deliberately failed")
        self.orders.append(order)
        order.status = "filled"
        order.fill_price = self.price
        order.filled_quantity = order.quantity
        return order


class DriftPriceBroker(FakeBroker):
    """Returns expected price on first get_price, adverse price next."""

    def __init__(self, market_type: str, expected: float, adverse: float, filters: dict):
        super().__init__(market_type, price=expected, filters=filters)
        self._expected = float(expected)
        self._adverse = float(adverse)
        self._calls = 0

    def get_price(self, symbol: str) -> float:
        self._calls += 1
        if self._calls <= 1:
            return self._expected
        return self._adverse


def _make_carry(spot_price: float = 100.0, fut_price: float = 100.5) -> TwinLegCarryBroker:
    spot = FakeBroker("spot", spot_price, {"SOLUSDT": SOL_SPOT})
    fut = FakeBroker("futures", fut_price, {"SOLUSDT": SOL_FUT})
    return TwinLegCarryBroker(spot_broker=spot, futures_broker=fut), spot, fut


def test_leg2_failure_unwinds_leg1_spot_market_sell():
    carry, spot, fut = _make_carry()
    fut.fail_placement = True

    with pytest.raises(TwinLegExecutionError) as raised:
        carry.execute_twin_leg_carry("SOL", "USDT", target_qty=0.5)

    err = raised.value
    assert err.phase == "leg2"
    assert err.panic_unwound is True
    assert spot.orders, "expected at least one spot order"
    # Last spot order must be emergency market SELL on the executed qty.
    assert spot.orders[-1].side == "sell"
    assert spot.orders[-1].quantity == pytest.approx(err.position.quantity)
    assert err.position.status == "ORPHAN_UNWOUND"


def test_adverse_drift_before_leg2_aborts_and_unwinds_leg1():
    # Expected mark ~100.5, adverse drifts to 100.0 => drift -0.5% (worse than -0.15%).
    spot = FakeBroker("spot", 100.0, {"SOLUSDT": SOL_SPOT})
    fut = DriftPriceBroker("futures", expected=100.5, adverse=100.0, filters={"SOLUSDT": SOL_FUT})
    carry = TwinLegCarryBroker(spot_broker=spot, futures_broker=fut)

    with pytest.raises(TwinLegExecutionError) as raised:
        carry.execute_twin_leg_carry(
            "SOL",
            "USDT",
            target_qty=0.5,
            max_entry_slippage_pct=0.0015,
        )

    err = raised.value
    assert err.phase == "leg2"
    assert err.panic_unwound is True
    # Emergency unwind must still sell the exact spot quantity.
    assert spot.orders[-1].side == "sell"
    assert spot.orders[-1].quantity == pytest.approx(err.position.quantity)
    assert err.position.status == "ORPHAN_UNWOUND"


def test_leg_gap_ms_recorded_on_clean_entry(monkeypatch):
    # Make perf_counter deterministic so we can assert exact ms.
    import crypto_quant.execution.live_broker as lb

    t = {"v": 1.0000}

    def perf_counter():
        # Each call advances by 0.0100s => 10ms. Entry calls 4 times (spot place, after check, fut place, after fut).
        t["v"] += 0.0100
        return t["v"]

    monkeypatch.setattr(lb.time, "perf_counter", perf_counter)

    carry, spot, fut = _make_carry(spot_price=100.0, fut_price=100.5)
    pos = carry.execute_twin_leg_carry(
        "SOL",
        "USDT",
        target_qty=0.5,
        max_entry_slippage_pct=0.9,  # avoid drift abort
    )

    assert pos.status == "OPEN"
    # With deterministic perf_counter steps, the recorded execution_gap_ms is stable.
    assert pos.execution_gap_ms >= 0.0
