"""Hermetic unit tests for the autonomous multi-pair cash-and-carry allocator.

These tests drive the *scanner-driven discovery + capital-based lot sizing*
architecture: positions are admitted through ``FundingScanner`` opportunities
rather than a hand-set ``target_qty``. A controlled fake scanner/provider
injects ranked candidates, and the per-pair capital budget
(``allocation_per_pair_usdt / spot_price``) is reconciled against the exchange
filters — any candidate that cannot clear ``min_qty``/``min_notional`` after
step-size rounding is skipped in favour of the next-ranked pair.

Verification invariants preserved from the single-pair suite: APR admission
gating, per-pair baseline delta tracking, margin safety checks, funding
settlement accounting, and clean (all-pairs) shutdown. Added: simultaneous
multi-pair tracking with isolated baselines, selective unwind of an
underperforming pair while healthy pairs stay open, and capital-discipline
skipping of too-small lots.
"""

from datetime import datetime, timezone

import pytest

from crypto_quant.db.connection import DatabaseManager
from crypto_quant.db.models import CarryFundingPaymentRecord, CarryPositionRecord
from crypto_quant.execution.carry_daemon import CarryDaemonConfig, CarryHarvesterDaemon
from crypto_quant.execution.live_broker import TwinLegFilters, TwinLegPosition
from crypto_quant.research.funding_scanner import CarryOpportunity

# Shared reconciled filters across both Spot + Futures legs. step_size=0.01,
# min_qty=0.01, min_notional=5 — mirror Binance defaults.
DEFAULT_TWIN = TwinLegFilters(
    symbol="*", step_size=0.01, min_qty=0.01, max_qty=1000.0, min_notional=5.0
)


def _premium(funding_rate: float = 0.0002, mark: float = 100.0, next_ms: int = 1):
    """A premium-index dict whose predicted 8h funding survives the exit hurdle."""
    return {
        "lastFundingRate": str(funding_rate),
        "markPrice": str(mark),
        "nextFundingTime": next_ms,
    }


def _opp(
    symbol: str = "SOLUSDT",
    base: str = "SOL",
    spot_price: float = 100.0,
    net_apr_pct: float = 25.0,
    predicted_funding_rate: float = 0.0001,
    basis_spread_pct: float = 0.1,
) -> CarryOpportunity:
    """A scanner candidate with a healthy net APR (%) for admission."""
    return CarryOpportunity(
        symbol=symbol,
        spot_price=spot_price,
        futures_price=spot_price * (1.0 + basis_spread_pct / 100.0),
        basis_spread_pct=basis_spread_pct,
        predicted_funding_rate=predicted_funding_rate,
        gross_apr_pct=net_apr_pct + 5.0,
        net_apr_pct=net_apr_pct,
        next_funding_time=datetime.now(timezone.utc),
        volume_24h_usdt=100_000_000.0,
    )


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeConnector:
    """Per-symbol balances / positions / premium with a shared fallback."""

    DEFAULT_PREMIUM = _premium()

    def __init__(self, premium=None, account=None):
        self.account = account or {
            "totalMarginBalance": "100", "totalMaintMargin": "10",
        }
        self.balances: dict = {}        # base -> {"total": float}
        self.positions: dict = {}       # symbol -> position_amt (short is negative)
        self._premium: dict = {}
        p = premium if premium is not None else dict(self.DEFAULT_PREMIUM)
        if "lastFundingRate" in p:      # shared premium for all symbols
            self._premium["*"] = p
        else:                           # keyed by symbol (e.g. "SOLUSDT")
            self._premium = dict(p)

    def _request(self, method, path, params=None):
        assert path == "/fapi/v1/premiumIndex"
        symbol = (params or {}).get("symbol") or ""
        return dict(
            self._premium.get(symbol) or self._premium.get("*") or self.DEFAULT_PREMIUM
        )

    def get_funding_rate_history(self, symbol, limit=1):
        p = self._request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol})
        return [{"fundingRate": p.get("lastFundingRate", "0")}]

    def get_funding_income_history(self, symbol, start_time=None, limit=100):
        return [{"time": start_time + 5 * 60 * 1000, "income": "0.01"}]

    def get_positions(self, symbol=None):
        # Mirrors the real connector: no symbol -> every OPEN position (size>0);
        # a symbol -> just that position's row (0.0 counts when explicitly asked).
        if symbol is None:
            return [
                {"position_amt": amt, "symbol": sym, "side": "short" if amt < 0 else "long"}
                for sym, amt in self.positions.items() if abs(amt) > 0
            ]
        return [{"position_amt": self.positions.get(symbol, 0.0)}]

    def get_account_info(self):
        return dict(self.account)

    def get_balances(self):
        return dict(self.balances)


class FakeLeg:
    def __init__(self, connector):
        self.connector = connector


class FakeScanner:
    """Controlled scanner: returns a fixed ranked list of opportunities."""

    def __init__(self, *opportunities):
        self.opportunities = list(opportunities)

    def scan(self, provider=None):
        return list(self.opportunities)


class FakeCarryBroker:
    """Twin-leg broker with a reconciled filter table and a fill simulator."""

    def __init__(self, futures_connector=None, spot_connector=None, filters=None):
        self.futures = FakeLeg(futures_connector or FakeConnector())
        self.spot = FakeLeg(spot_connector or FakeConnector())
        self.filters = filters or DEFAULT_TWIN
        self.executed = []          # [(base, quote, qty), ...]
        self.unwound = []           # [position_id, ...]
        self._seq: dict = {}

    def get_twin_filters(self, base, quote="USDT"):
        return self.filters

    def _position(self, base, quote, quantity):
        n = self._seq.get(base, 0) + 1
        self._seq[base] = n
        return TwinLegPosition(
            position_id=f"TWIN-{base}-{quote}-{n}",
            base_asset=base,
            quote_asset=quote,
            quantity=quantity,
            spot_order_id=f"spot-{n}",
            spot_fill_price=100.0,
            futures_order_id=f"future-{n}",
            futures_fill_price=100.2,
            entry_basis_spread_pct=0.2,
            execution_gap_ms=100.0,
            status="OPEN",
            opened_at=1_700_000_000.0,
        )

    def _simulate_fill(self, base, quote, quantity, futures_short=None):
        """Advance the account as if the carry filled: +qty spot, -qty short.

        The connectors report the account's TOTAL inventory, which already holds
        pre-existing balances. Mirrors what the daemon's baseline-allocated delta
        check expects: on top of an existing baseline, the fill adds ``quantity``
        to spot and opens an equal (negative) short on futures.
        """
        futures_short = quantity if futures_short is None else futures_short
        symbol = f"{base}{quote}"
        spot_bal = self.spot.connector.balances.get(base, {}).get("total", 0.0)
        self.spot.connector.balances[base] = {"total": spot_bal + quantity}
        self.futures.connector.positions[symbol] = (
            self.futures.connector.positions.get(symbol, 0.0) - futures_short
        )

    def execute_twin_leg_carry(self, base, quote, quantity):
        # The real broker reconciles BOTH legs to the exchanged grid before
        # submitting: floor to the coarsest common step_size. Mirrored here so
        # fills land even (a 0.1 carry on a 0.01 grid stays exactly -0.10).
        price = 100.0
        _, _, qty = self.filters.validate_qty(quantity, price)
        self.executed.append((base, quote, qty))
        self._simulate_fill(base, quote, qty)
        return self._position(base, quote, qty)

    def unwind_twin_leg_carry(self, position):
        if position.position_id in self.unwound:
            return position
        self.unwound.append(position.position_id)
        # The real unwind closes the exact hedged quantity on both legs,
        # restoring the account to its pre-open baseline (spot SELL + futures
        # reduce_only BUY that lifts the short back toward zero).
        symbol = f"{position.base_asset}{position.quote_asset}"
        opened_spot = position.quantity
        current_fut = self.futures.connector.positions.get(symbol, 0.0)
        # The carry added spot and subtracted futures relative to baseline; a
        # successful unwind reverts both, leaving the pre-existing baseline.
        self.spot.connector.balances[position.base_asset] = {
            "total": self.spot.connector.balances.get(
                position.base_asset, {}
            ).get("total", 0.0) - opened_spot
        }
        self.futures.connector.positions[symbol] = current_fut + opened_spot
        position.status = "CLOSED"
        position.closed_at = 1_700_000_100.0
        return position


class UnderHedgedCarryBroker(FakeCarryBroker):
    """A carry whose Futures short fills SHORT of the ordered quantity."""

    def __init__(self, *args, shortfall: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.shortfall = shortfall

    def execute_twin_leg_carry(self, base, quote, quantity):
        self.executed.append((base, quote, quantity))
        opened_short = max(0.0, quantity - self.shortfall)
        self._simulate_fill(base, quote, quantity, futures_short=opened_short)
        return self._position(base, quote, quantity)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def make_daemon(
    db=None,
    carry=None,
    allocation=50.0,
    opportunities=None,
    **overrides,
):
    """Build a daemon wired to a controlled scanner for the given candidates."""
    config_kwargs = dict(
        allocation_per_pair_usdt=allocation,
        min_apr_threshold=0.10,
        dry_run=True,
    )
    config_kwargs.update(overrides)
    config = CarryDaemonConfig(**config_kwargs)
    carry = carry or FakeCarryBroker()
    opps = opportunities if opportunities is not None else [_opp()]
    scanner = FakeScanner(*opps)
    daemon = CarryHarvesterDaemon(config, carry, db, scanner=scanner)
    # Hermetic by default: the daemon constructs the live notifier from env, so
    # with real DISCORD_* creds present in .env it would dispatch REAL embeds as
    # positions open/unwind — non-hermetic and rate-limit-hammering. Replace it
    # with a recording no-op unless a test explicitly installs its own notifier.
    daemon._notifier = RecordingNotifier()
    return daemon


# ---------------------------------------------------------------------------
# Scanner-driven admission + capital sizing
# ---------------------------------------------------------------------------


def test_apri_admission_opens_and_persists_capital_sized_lot():
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    carry = FakeCarryBroker()
    # allocation 50 / spot 100 -> target 0.5, reconciled to 0.5.
    daemon = make_daemon(db, carry, allocation=50.0, opportunities=[_opp()])

    daemon.tick()

    assert carry.executed == [("SOL", "USDT", 0.5)]
    assert set(daemon.active_positions) == {"SOLUSDT"}
    session = db.get_session()
    try:
        row = session.get(CarryPositionRecord, "TWIN-SOL-USDT-1")
        assert row is not None
        assert row.status == "OPEN"
        assert row.quantity == pytest.approx(0.5)
        assert row.symbol == "SOLUSDT"
        assert row.leg_gap_ms == pytest.approx(100.0)
    finally:
        session.close()


def test_rate_at_or_below_threshold_does_not_enter():
    carry = FakeCarryBroker()
    daemon = make_daemon(carry=carry, opportunities=[_opp(net_apr_pct=5.0)])

    daemon.tick()

    assert carry.executed == []
    assert daemon.active_positions == {}


def test_capital_discipline_skips_small_lot_then_tries_next_candidate():
    """A candidate that can't clear min_qty after step rounding is skipped; the
    next-ranked pair is admitted instead (no automatic lot bumping)."""
    carry = FakeCarryBroker()
    # BTC at $50,000 with a $50 budget -> target 0.001 < min_qty 0.01 -> skip.
    # SOL at $100 with the same $50 budget -> target 0.5 -> admitted.
    candidates = [
        _opp(symbol="BTCUSDT", base="BTC", spot_price=50_000.0, net_apr_pct=40.0),
        _opp(symbol="SOLUSDT", base="SOL", spot_price=100.0, net_apr_pct=25.0),
    ]
    daemon = make_daemon(carry=carry, allocation=50.0, opportunities=candidates)

    daemon.tick()

    assert carry.executed == [("SOL", "USDT", 0.5)]
    assert set(daemon.active_positions) == {"SOLUSDT"}


# ---------------------------------------------------------------------------
# Delta / margin health invariants
# ---------------------------------------------------------------------------


def test_margin_buffer_breach_unwinds_and_persists_closed():
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    # A healthy account at entry (90% buffer clears the pre-flight margin
    # gatekeeper); maintenance then rises so the live buffer collapses.
    futures = FakeConnector(account={"totalMarginBalance": "100", "totalMaintMargin": "10"})
    carry = FakeCarryBroker(futures_connector=futures)
    daemon = make_daemon(db, carry, min_margin_ratio=0.20)

    daemon.tick()  # opens (entry margin ratio 90% >= 25% pre-flight hurdle)
    futures.account["totalMaintMargin"] = "90"  # live buffer drops to 10%
    daemon.tick()  # health sees a 10% buffer (< 20% min) and exits

    assert carry.unwound == ["TWIN-SOL-USDT-1"]
    assert daemon.active_positions == {}
    session = db.get_session()
    try:
        assert session.get(CarryPositionRecord, "TWIN-SOL-USDT-1").status == "CLOSED"
    finally:
        session.close()


def test_pre_existing_inventory_does_not_false_unwind():
    """Regression: the health check must measure ALLOCATED deltas (current minus
    entry baseline), never absolute balances, so pre-existing inventory never
    trips a false emergency unwind."""
    futures = FakeConnector()
    futures.positions["SOLUSDT"] = -0.3
    spot = FakeConnector()
    spot.balances["SOL"] = {"total": 4.3}
    carry = FakeCarryBroker(futures_connector=futures, spot_connector=spot)
    # allocation 10 / spot 100 -> a 0.1 carry opened on top of pre-existing stock.
    daemon = make_daemon(carry=carry, allocation=10.0)

    daemon.tick()  # opens a 0.1 carry on pre-existing inventory
    daemon.tick()  # health sees allocated deltas balance (spot +0.1 / fut -0.1)

    assert carry.unwound == []
    assert "SOLUSDT" in daemon.active_positions


def test_genuine_under_hedged_short_still_unwinds():
    """A real shortfall (futures short far smaller than the carry) must still trip."""
    futures = FakeConnector()
    futures.positions["SOLUSDT"] = -0.3
    spot = FakeConnector()
    spot.balances["SOL"] = {"total": 4.3}
    carry = UnderHedgedCarryBroker(
        futures_connector=futures, spot_connector=spot, shortfall=0.06
    )
    daemon = make_daemon(carry=carry, allocation=10.0)

    daemon.tick()  # opens a 0.1 carry but the hedge only lands -0.04
    daemon.tick()  # health sees the short is under-hedged beyond tolerance

    assert carry.unwound == ["TWIN-SOL-USDT-1"]
    assert daemon.active_positions == {}


# ---------------------------------------------------------------------------
# Funding settlement accounting
# ---------------------------------------------------------------------------


def test_funding_settlement_persists_once_per_position(monkeypatch):
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    now_ms = 1_700_000_000_000
    futures = FakeConnector(premium=_premium(funding_rate=0.0002, next_ms=now_ms))
    carry = FakeCarryBroker(futures_connector=futures)
    daemon = make_daemon(db, carry)
    monkeypatch.setattr("crypto_quant.execution.carry_daemon.time.time", lambda: now_ms / 1000)

    daemon.tick()  # opens
    daemon.tick()  # captures current nextFundingTime
    futures._premium["*"]["nextFundingTime"] = now_ms + 8 * 60 * 60 * 1000
    daemon.tick()  # rollover records the confirmed prior settlement
    daemon.tick()  # does not duplicate settlement

    session = db.get_session()
    try:
        payments = session.query(CarryFundingPaymentRecord).all()
        assert len(payments) == 1
        assert payments[0].position_id == "TWIN-SOL-USDT-1"
        assert payments[0].funding_payment_usdt == pytest.approx(0.01)
    finally:
        session.close()


class _DelayedIncomeConnector(FakeConnector):
    """Income stays empty until ``income_available`` flips — Binance batch
    clearing can lag up to ~1-5 min after nextFundingTime rolls forward."""

    def __init__(self, *args, income_available=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.income_available = income_available or {"bool": False}

    def get_funding_income_history(self, symbol, start_time=None, limit=100):
        if not self.income_available["bool"]:
            return []
        return [{"time": start_time + 5 * 60 * 1000, "income": "0.005"}]


def test_funding_rollover_retries_until_income_and_never_advances_on_empty(monkeypatch):
    """Defect 1 regression: an empty /fapi/v1/income poll must NOT advance
    ``_last_seen_next_funding_ms``; the daemon keeps re-querying on later health
    ticks (30s cadence) and captures the settlement the moment income appears."""
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    now_ms = 1_700_000_000_000
    futures = _DelayedIncomeConnector(premium=_premium(funding_rate=0.0002, next_ms=now_ms))
    carry = FakeCarryBroker(futures_connector=futures)
    daemon = make_daemon(db, carry)
    monkeypatch.setattr("crypto_quant.execution.carry_daemon.time.time", lambda: now_ms / 1000)

    daemon.tick()  # opens
    daemon.tick()  # baseline current nextFundingTime
    futures._premium["*"]["nextFundingTime"] = now_ms + 8 * 60 * 60 * 1000
    daemon.tick()  # rollover detected; income empty -> marker must NOT advance
    assert daemon._last_seen_next_funding_ms["SOLUSDT"] == now_ms
    assert "SOLUSDT" in daemon._pending_funding_retry_until

    futures.income_available["bool"] = True
    daemon.tick()  # next health tick re-queries and now confirms the settlement
    assert daemon._last_seen_next_funding_ms["SOLUSDT"] == now_ms + 8 * 60 * 60 * 1000
    session = db.get_session()
    try:
        payments = session.query(CarryFundingPaymentRecord).all()
        assert len(payments) == 1
        assert payments[0].funding_payment_usdt == pytest.approx(0.005)
    finally:
        session.close()


def test_funding_rollover_gives_up_after_retry_window_without_capturing(monkeypatch):
    """After the 15-minute retry window elapses with no income, the daemon
    stops auto-retrying (no ledger row) so the operator can backfill via
    `carry reconcile-funding`."""
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    now_ms = 1_700_000_000_000
    clock = {"t": now_ms / 1000}
    futures = _DelayedIncomeConnector(premium=_premium(funding_rate=0.0002, next_ms=now_ms))
    carry = FakeCarryBroker(futures_connector=futures)
    daemon = make_daemon(db, carry)
    monkeypatch.setattr("crypto_quant.execution.carry_daemon.time.time", lambda: clock["t"])

    daemon.tick()
    daemon.tick()
    futures._premium["*"]["nextFundingTime"] = now_ms + 8 * 60 * 60 * 1000
    daemon.tick()  # rollover detected, income stays empty -> retry armed
    assert daemon._last_funding_settlement_ms.get("SOLUSDT") is None

    clock["t"] += 900 + 60  # advance past the 15-minute retry window
    daemon.tick()  # window expired -> give up: mark settled, no auto-capture

    assert daemon._last_funding_settlement_ms["SOLUSDT"] == now_ms
    assert daemon._last_seen_next_funding_ms["SOLUSDT"] == now_ms + 8 * 60 * 60 * 1000
    assert "SOLUSDT" not in daemon._pending_funding_retry_until
    session = db.get_session()
    try:
        assert session.query(CarryFundingPaymentRecord).count() == 0
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Multi-pair portfolio behaviour
# ---------------------------------------------------------------------------


def test_multipair_tracks_two_pairs_with_isolated_baselines():
    """Two pairs open together; each keeps its OWN entry baselines so the delta
    check for SOL never sees ETH's inventory."""
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    futures = FakeConnector()
    futures.positions["SOLUSDT"] = -3.0   # pre-existing SOL short
    spot = FakeConnector()
    spot.balances["SOL"] = {"total": 55.0}  # pre-existing SOL spot
    carry = FakeCarryBroker(futures_connector=futures, spot_connector=spot)
    candidates = [
        _opp(symbol="SOLUSDT", base="SOL", spot_price=100.0, net_apr_pct=30.0),
        _opp(symbol="ETHUSDT", base="ETH", spot_price=3000.0, net_apr_pct=28.0),
    ]
    daemon = make_daemon(db, carry, max_active_pairs=2, allocation=50.0, opportunities=candidates)

    daemon.tick()
    daemon.tick()  # discovery is full; health must not false-unwind either pair

    assert set(daemon.active_positions) == {"SOLUSDT", "ETHUSDT"}
    sol = daemon.active_positions["SOLUSDT"]
    eth = daemon.active_positions["ETHUSDT"]
    # SOL's baselines capture its own pre-existing +3/-6 inventory.
    assert sol.baseline_spot_qty == pytest.approx(55.0)
    assert sol.baseline_futures_qty == pytest.approx(-3.0)
    # ETH had no pre-existing inventory, so its baselines stay zero.
    assert eth.baseline_spot_qty == pytest.approx(0.0)
    assert eth.baseline_futures_qty == pytest.approx(0.0)
    assert carry.unwound == []  # healthy, fully-hedged pairs stay open


def test_selective_unwind_of_underperforming_pair_only():
    """Funding collapses on one pair; only that pair unwinds, healthy ones stay."""
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    sol_premium = _premium(funding_rate=0.0002, next_ms=1)   # healthy
    eth_premium = _premium(funding_rate=0.0002, next_ms=1)
    futures = FakeConnector(premium={"SOLUSDT": sol_premium, "ETHUSDT": eth_premium})
    carry = FakeCarryBroker(futures_connector=futures)
    candidates = [
        _opp(symbol="SOLUSDT", base="SOL", spot_price=100.0, net_apr_pct=30.0),
        _opp(symbol="ETHUSDT", base="ETH", spot_price=3000.0, net_apr_pct=28.0),
    ]
    daemon = make_daemon(db, carry, max_active_pairs=2, allocation=50.0, opportunities=candidates)

    daemon.tick()
    assert set(daemon.active_positions) == {"SOLUSDT", "ETHUSDT"}

    # Let SOL's predicted funding collapse below the 2% exit hurdle; ETH stays put.
    sol_premium["lastFundingRate"] = "0.00001"  # annualized ~1.1% < 2% exit
    daemon.tick()

    assert carry.unwound == ["TWIN-SOL-USDT-1"]
    assert set(daemon.active_positions) == {"ETHUSDT"}


def test_capacity_limited_by_max_active_pairs():
    """No more than max_active_pairs positions are ever opened."""
    carry = FakeCarryBroker()
    candidates = [
        _opp(symbol="SOLUSDT", base="SOL", spot_price=100.0, net_apr_pct=30.0),
        _opp(symbol="ETHUSDT", base="ETH", spot_price=3000.0, net_apr_pct=28.0),
        _opp(symbol="DOGEUSDT", base="DOGE", spot_price=0.16, net_apr_pct=26.0),
    ]
    daemon = make_daemon(carry=carry, max_active_pairs=2, allocation=50.0, opportunities=candidates)

    daemon.tick()
    daemon.tick()

    assert len(daemon.active_positions) == 2


def test_shutdown_unwinds_all_active_pairs_and_marks_closed():
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    carry = FakeCarryBroker()
    candidates = [
        _opp(symbol="SOLUSDT", base="SOL", spot_price=100.0, net_apr_pct=30.0),
        _opp(symbol="ETHUSDT", base="ETH", spot_price=3000.0, net_apr_pct=28.0),
    ]
    daemon = make_daemon(db, carry, max_active_pairs=2, allocation=50.0, opportunities=candidates)
    daemon.tick()

    daemon.shutdown()

    assert sorted(carry.unwound) == ["TWIN-ETH-USDT-1", "TWIN-SOL-USDT-1"]
    assert daemon.active_positions == {}
    session = db.get_session()
    try:
        for pid in ("TWIN-SOL-USDT-1", "TWIN-ETH-USDT-1"):
            assert session.get(CarryPositionRecord, pid).status == "CLOSED"
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Pre-flight risk gatekeeper (Phase 1 Step 1.1)
# ---------------------------------------------------------------------------


def test_preflight_blocks_duplicate_base_asset():
    """Rule A: a candidate whose base asset is already active is rejected."""
    carry = FakeCarryBroker()
    daemon = make_daemon(carry=carry)
    daemon._active_positions["SOLUSDT"] = carry._position("SOL", "USDT", 0.5)

    assert daemon._check_preflight_risk("SOLUSDT", 40.0) is False
    # A genuinely different asset on a healthy, under-cap book still passes.
    assert daemon._check_preflight_risk("ETHUSDT", 40.0) is True
    assert carry.executed == []


def test_preflight_blocks_when_total_deployed_cap_reached():
    """Rule B: opening past max_portfolio_usd (deployed + new) is rejected."""
    carry = FakeCarryBroker()
    daemon = make_daemon(carry=carry, max_portfolio_usd=70.0)
    # A 0.5 SOL carry deploys ~$50 (0.5 x futures 100.2) on the book.
    daemon._active_positions["SOLUSDT"] = carry._position("SOL", "USDT", 0.5)

    assert daemon._check_preflight_risk("ETHUSDT", 30.0) is False  # 50 + 30 > 70
    assert daemon._check_preflight_risk("ETHUSDT", 10.0) is True   # 50 + 10 <= 70


def test_preflight_blocks_on_margin_buffer_below_entry_floor():
    """Rule C: an entry into a <25% futures margin buffer is rejected."""
    thin = FakeConnector(account={"totalMarginBalance": "100", "totalMaintMargin": "80"})
    daemon = make_daemon(carry=FakeCarryBroker(futures_connector=thin))
    assert daemon._futures_margin_ratio() == pytest.approx(0.20)  # sanity
    assert daemon._check_preflight_risk("SOLUSDT", 40.0) is False

    healthy = FakeConnector(account={"totalMarginBalance": "100", "totalMaintMargin": "10"})
    daemon2 = make_daemon(carry=FakeCarryBroker(futures_connector=healthy))
    assert daemon2._check_preflight_risk("SOLUSDT", 40.0) is True


def test_preflight_admits_valid_candidate_all_green():
    """A candidate clearing all three hurdles is admitted and opened."""
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    carry = FakeCarryBroker()
    daemon = make_daemon(db, carry, allocation=50.0, opportunities=[_opp()])

    assert daemon._check_preflight_risk("SOLUSDT", 50.0) is True
    daemon.tick()

    assert carry.executed == [("SOL", "USDT", 0.5)]
    assert set(daemon.active_positions) == {"SOLUSDT"}


def test_preflight_rejection_skips_candidate_without_raising():
    """Discovery safely skips a gatekeeper-rejected pair (no exception, next
    candidate still considered)."""
    carry = FakeCarryBroker()
    # One healthy candidate: only the account margin triage matters, but the
    # whole first slot is gated by an over-tight margin floor -> nothing opens.
    daemon = make_daemon(carry=carry, entry_margin_ratio_min=0.25)
    daemon._futures_margin_ratio = lambda: 0.10  # force a thin buffer however

    # Calling the gatekeeper from discovery must not raise.
    daemon.tick()
    assert carry.executed == []
    assert daemon.active_positions == {}


# ---------------------------------------------------------------------------
# Discord notification robustness + financial audit + pre-trade safeguards
# ---------------------------------------------------------------------------


class RecordingNotifier:
    """Records every lifecycle call without any network I/O."""

    def __init__(self):
        self.calls = []

    def is_enabled(self):
        return True

    def notify_trade_open(self, **kw):
        self.calls.append(("open", kw))

    def notify_trade_close(self, **kw):
        self.calls.append(("close", kw))

    def notify_trade_entry(self, **kw):
        self.calls.append(("entry", kw))

    def notify_trade_exit(self, **kw):
        self.calls.append(("exit", kw))

    def notify_financial_audit(self, **kw):
        self.calls.append(("audit", kw))

    def notify_funding_settlement(self, **kw):
        self.calls.append(("funding", kw))

    def notify_circuit_breaker(self, **kw):
        self.calls.append(("circuit", kw))


def test_notifier_rebuilds_when_env_loads_late(monkeypatch):
    """Root-cause fix: the singleton is created disabled during import (before
    ``load_dotenv``); ``get_notifier`` must rebuild it once creds appear so the
    run isn't left with zero notifications."""
    from crypto_quant.notifications import discord_dm as dm

    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    monkeypatch.delenv("DISCORD_USER_ID", raising=False)
    dm.reset_notifier()
    notifier = dm.get_notifier()
    assert notifier.is_enabled() is False

    monkeypatch.setenv("DISCORD_BOT_TOKEN", "bot-token")
    monkeypatch.setenv("DISCORD_USER_ID", "user-123")
    rebuilt = dm.get_notifier()
    assert rebuilt.is_enabled() is True
    assert rebuilt is not notifier  # a fresh instance, no longer the dead singleton

    dm.reset_notifier()
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    monkeypatch.delenv("DISCORD_USER_ID", raising=False)


def test_pre_trade_funding_hurdle_rejects_nonpaying_carry():
    carry = FakeCarryBroker()
    daemon = make_daemon(carry=carry, opportunities=[_opp(predicted_funding_rate=0.0000)])
    daemon.tick()
    assert carry.executed == []
    assert daemon.active_positions == {}


def test_pre_trade_adverse_basis_rejected():
    carry = FakeCarryBroker()
    daemon = make_daemon(carry=carry, opportunities=[_opp(basis_spread_pct=-0.5)])
    daemon.tick()
    assert carry.executed == []
    assert daemon.active_positions == {}


def test_pre_trade_positive_basis_still_admitted():
    carry = FakeCarryBroker()
    daemon = make_daemon(carry=carry, opportunities=[_opp(basis_spread_pct=0.5)])
    daemon.tick()
    assert carry.executed == [("SOL", "USDT", 0.5)]
    assert set(daemon.active_positions) == {"SOLUSDT"}


def test_scale_to_available_margin_caps_budget():
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    futures = FakeConnector(account={"totalMarginBalance": "100", "totalMaintMargin": "10"})
    carry = FakeCarryBroker(futures_connector=futures)
    daemon = make_daemon(
        db, carry, allocation=50.0,
        scale_to_available_margin=True, available_margin_use_fraction=0.5,
    )
    daemon.tick()
    # budget = min(50, (100-10) * 0.5 = 45) = 45; spot 100 -> qty 0.45
    assert carry.executed == [("SOL", "USDT", 0.45)]


def test_trade_open_dispatches_notify_trade_entry_with_metrics():
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    daemon = make_daemon(db, allocation=50.0)
    rec = RecordingNotifier()
    daemon._notifier = rec
    daemon.tick()

    entries = [c for c in rec.calls if c[0] == "entry"]
    assert len(entries) == 1
    kw = entries[0][1]
    assert kw["symbol"] == "SOLUSDT"
    assert kw["strategy"] == "CARRY_ARBITRAGE"
    assert kw["side"] == "CARRY"
    assert kw["qty"] == pytest.approx(0.5)
    extra = kw["extra"]
    # per-position capital + audit metrics (equity, remaining free margin)
    assert "Capital Invested (USDT)" in extra
    assert "Equity (USDT)" in extra
    assert "Available Free Margin (USDT)" in extra


def test_unwind_dispatches_notify_trade_exit_with_metrics():
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    # Healthy entry buffer (90% clears the pre-flight gatekeeper); maintenance
    # then spikes so the live buffer (10%) breaches the health-check minimum.
    futures = FakeConnector(account={"totalMarginBalance": "100", "totalMaintMargin": "10"})
    carry = FakeCarryBroker(futures_connector=futures)
    daemon = make_daemon(db, carry, min_margin_ratio=0.20)
    rec = RecordingNotifier()
    daemon._notifier = rec
    daemon.tick()  # opens (entry margin ratio 90% >= 25% pre-flight hurdle)
    futures.account["totalMaintMargin"] = "90"
    daemon.tick()  # 10% buffer < 20% min -> emergency unwind -> exit embed

    exits = [c for c in rec.calls if c[0] == "exit"]
    assert len(exits) == 1
    extra = exits[0][1]["extra"]
    for field in ("Reason", "Net PnL (USDT)", "Net PnL %", "Capital Returned (USDT)",
                  "Funding Harvested (USDT)", "Realized Basis PnL (USDT)",
                  "Available Free Margin (USDT)"):
        assert field in extra


def test_funding_settlement_dispatches_notify_funding(monkeypatch):
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    now_ms = 1_700_000_000_000
    futures = FakeConnector(premium=_premium(funding_rate=0.0002, next_ms=now_ms))
    carry = FakeCarryBroker(futures_connector=futures)
    daemon = make_daemon(db, carry)
    rec = RecordingNotifier()
    daemon._notifier = rec
    monkeypatch.setattr("crypto_quant.execution.carry_daemon.time.time", lambda: now_ms / 1000)

    daemon.tick()  # open
    daemon.tick()  # baseline next funding
    futures._premium["*"]["nextFundingTime"] = now_ms + 8 * 60 * 60 * 1000
    daemon.tick()  # rollover -> confirmed payment -> funding embed

    fundings = [c for c in rec.calls if c[0] == "funding"]
    assert len(fundings) == 1
    assert fundings[0][1]["payment_usdt"] == pytest.approx(0.01)
    assert fundings[0][1]["symbol"] == "SOLUSDT"


def test_audit_metrics_basis_and_net_pnl():
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    daemon = make_daemon(db, allocation=50.0)
    daemon.tick()  # open qty 0.5 @ futures 100.2
    m = daemon._audit_metrics()
    assert m["equity_usdt"] == pytest.approx(100.0)          # fake futures margin balance
    assert m["allocated_capital_usdt"] == pytest.approx(0.5 * 100.2)
    # FakeConnector has no live mark/spot -> net unrealized 0; net PnL = -fees
    assert m["net_unrealized_pnl_usdt"] == pytest.approx(0.0)
    assert m["estimated_fees_usdt"] == pytest.approx(2 * 0.0004 * 0.5 * 100.2)
    assert m["net_pnl_usdt"] == pytest.approx(-2 * 0.0004 * 0.5 * 100.2)


def test_adverse_basis_divergence_stable_basis_stays_open():
    carry = FakeCarryBroker()
    daemon = make_daemon(carry=carry, allocation=50.0, opportunities=[_opp()])

    # Set stable live spot + futures marks by monkeypatching daemon live
    # price readers to avoid any network calls.
    daemon._live_spot = lambda symbol: 100.0
    daemon._live_mark = lambda symbol: 100.2

    daemon.tick()  # opens
    assert "SOLUSDT" in daemon.active_positions

    # Keep basis divergence within threshold.
    daemon._live_spot = lambda symbol: 100.0
    daemon._live_mark = lambda symbol: 100.2 + 0.0001
    daemon.tick()

    assert "SOLUSDT" in daemon.active_positions
    assert carry.unwound == []


def test_adverse_basis_divergence_triggers_graceful_unwind_with_exit_reason():
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    carry = FakeCarryBroker()

    futures = carry.futures.connector
    # Entry basis from FakeCarryBroker fill: spot_fill_price=100.0,
    # futures_fill_price=100.2 -> entry_basis = 0.2%.
    daemon = make_daemon(db, carry=carry, allocation=50.0, opportunities=[_opp()])
    rec = RecordingNotifier()
    daemon._notifier = rec

    # Open with stable basis near entry.
    daemon._live_spot = lambda symbol: 100.0
    daemon._live_mark = lambda symbol: 100.2
    daemon.tick()
    assert "SOLUSDT" in daemon.active_positions

    # Widen futures-over-spot basis adversely: live basis +1.0% vs entry 0.2%
    # -> divergence 0.8% = 0.008 > max_adverse_basis_pct default 0.0075.
    daemon._live_spot = lambda symbol: 100.0
    daemon._live_mark = lambda symbol: 101.0

    daemon.tick()

    assert carry.unwound == ["TWIN-SOL-USDT-1"]
    exits = [c for c in rec.calls if c[0] == "exit"]
    assert len(exits) == 1
    assert exits[0][1]["exit_reason"] == "ADVERSE_BASIS_STOP"

    session = db.get_session()
    try:
        row = session.get(CarryPositionRecord, "TWIN-SOL-USDT-1")
        assert row is not None
        assert row.status == "CLOSED"
    finally:
        session.close()


def test_financial_audit_dispatch_sends_card(monkeypatch):
    """Defect 2 regression: the hourly heartbeat must dispatch the financial-audit
    card (capital vs. cash, realized vs. unrealized, health buffer), not just
    compute the metrics internally."""
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    daemon = make_daemon(db, allocation=50.0)
    rec = RecordingNotifier()
    daemon._notifier = rec
    monkeypatch.setattr("crypto_quant.execution.carry_daemon.time.time", lambda: 1_700_000_000.0)

    daemon.tick()  # open qty 0.5 @ futures 100.2
    daemon._dispatch_financial_audit()

    audits = [c for c in rec.calls if c[0] == "audit"]
    assert len(audits) == 1
    card = audits[0][1]
    # Every metric the audit card must surface is present (Defect 2 fields).
    for field in (
        "available_free_margin_usdt",
        "allocated_capital_usdt",
        "net_unrealized_pnl_usdt",
        "harvested_funding_usdt",
        "open_positions",
    ):
        assert field in card
    assert card["allocated_capital_usdt"] == pytest.approx(0.5 * 100.2)
    assert card["open_positions"] == 1
    # Health buffer and the active-pair summary ride along on the card.
    assert card["margin_safety_buffer_pct"] == pytest.approx((100.0 - 10.0) / 100.0 * 100.0)
    assert any(k.startswith("Active Pairs") for k in card.get("extra", {}))


# ---------------------------------------------------------------------------
# Tiered margin circuit breakers (Phase 1 Step 1.3)
# ---------------------------------------------------------------------------


def test_config_validates_circuit_breaker_ordering():
    """The tier invariant must hold: emergency < soft warning < entry floor."""
    CarryDaemonConfig()  # defaults 0.10 < 0.15 < 0.25 pass
    with pytest.raises(ValueError):
        CarryDaemonConfig(emergency_delever_ratio=0.12, soft_margin_ratio_warning=0.10)
    with pytest.raises(ValueError):
        CarryDaemonConfig(soft_margin_ratio_warning=0.30)  # must stay under entry floor 0.25
    with pytest.raises(ValueError):
        CarryDaemonConfig(emergency_delever_ratio=0.03)  # must be > 0.05


def test_soft_margin_breaker_pauses_entries_without_unwinding():
    """Level 1: a buffer in the soft zone (between 10% and 15%) pauses ALL new
    admissions and never unwinds an open pair."""
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    futures = FakeConnector(account={"totalMarginBalance": "100", "totalMaintMargin": "10"})
    carry = FakeCarryBroker(futures_connector=futures)
    # min_margin_ratio sits BELOW the soft zone (0.08 < 0.10 emergency floor) so
    # the existing per-position margin guard stays silent and only the tiered
    # Level 1 soft breaker reacts — isolating the soft path for this test.
    daemon = make_daemon(
        db, carry, min_margin_ratio=0.08,
        opportunities=[_opp(symbol="SOLUSDT", base="SOL")],
    )

    daemon.tick()  # opens SOL at a healthy 90% buffer
    assert set(daemon.active_positions) == {"SOLUSDT"}

    # Drop the account buffer to 12% — inside the soft zone (< 15%, > 10%).
    futures.account["totalMaintMargin"] = "88"
    daemon.tick()

    assert carry.unwound == []                    # soft breaker never unwinds
    assert "SOLUSDT" in daemon.active_positions   # the open pair stays put
    assert daemon._margin_soft_pause is True      # the latch engages

    # With spare capacity on the book, discovery must skip ALL new entries.
    daemon.scanner.opportunities = [_opp(symbol="ETHUSDT", base="ETH", net_apr_pct=30.0)]
    daemon.tick()
    assert set(daemon.active_positions) == {"SOLUSDT"}              # ETH was paused
    assert carry.executed == [("SOL", "USDT", 0.5)]

    # Buffer recovers -> the soft breaker releases and admissions resume.
    futures.account["totalMaintMargin"] = "10"
    daemon.tick()
    assert daemon._margin_soft_pause is False
    assert set(daemon.active_positions) == {"SOLUSDT", "ETHUSDT"}


def test_emergency_delever_below_hard_breaker_unwinds_target_with_exit_reason():
    """Level 2: a buffer under 10% auto-deleverages the active pair and marks
    the exit embed with exit_reason=EMERGENCY_DELEVER."""
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    rec = RecordingNotifier()
    futures = FakeConnector(account={"totalMarginBalance": "100", "totalMaintMargin": "10"})
    carry = FakeCarryBroker(futures_connector=futures)
    daemon = make_daemon(db, carry, opportunities=[_opp(symbol="SOLUSDT", base="SOL")])
    daemon._notifier = rec

    daemon.tick()  # opens SOL at a healthy 90% buffer
    assert "SOLUSDT" in daemon.active_positions

    # Collapse the account buffer to 5% — past the 10% hard circuit breaker.
    futures.account["totalMaintMargin"] = "95"
    daemon.tick()

    assert carry.unwound == ["TWIN-SOL-USDT-1"]
    assert daemon.active_positions == {}
    exits = [c for c in rec.calls if c[0] == "exit"]
    assert len(exits) == 1
    assert exits[0][1]["exit_reason"] == "EMERGENCY_DELEVER"
    # An urgent account-level circuit-breaker card was also dispatched.
    circuits = [c for c in rec.calls if c[0] == "circuit"]
    assert any(c[1]["symbol"] == "ACCOUNT" for c in circuits)


def test_pick_delever_target_selects_largest_deployed_notional():
    """The hard breeder deleverages the pair with the largest deployed notional
    — the biggest collateral at risk — first."""
    carry = FakeCarryBroker()
    daemon = make_daemon(carry=carry)
    small = carry._position("SOL", "USDT", 0.5)   # notional 0.5 * 100.2 = 50.1
    big = carry._position("ETH", "USDT", 2.0)     # notional 2.0 * 100.2 = 200.4
    daemon._active_positions = {"SOLUSDT": small, "ETHUSDT": big}

    assert daemon._pick_delever_target() is big
    assert daemon._pick_delever_target().position_id == "TWIN-ETH-USDT-1"