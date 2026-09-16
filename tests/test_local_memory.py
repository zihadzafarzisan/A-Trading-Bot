"""Hermetic unit tests for the Phase 2 Step 2.2 local memory gate.

Covers the :class:`LocalMemoryGate` admission rules (missing-file fallback,
mtime-driven reload, ``BLACKLIST`` / ``MAX_ENTRY_BASIS_DISCOUNT`` / ``MIN_APR``
blocks, and scope matching) plus a ``carry_daemon.tick()`` integration check
that a candidate matching a memory rule is skipped before any order is placed.
No network calls anywhere — the daemon is wired to in-memory fakes.
"""

import json
import os
from datetime import datetime, timezone

import pytest

from crypto_quant.execution.carry_daemon import CarryDaemonConfig, CarryHarvesterDaemon
from crypto_quant.execution.live_broker import TwinLegFilters
from crypto_quant.research.funding_scanner import CarryOpportunity
from crypto_quant.risk.local_memory import LocalMemoryGate

# ---------------------------------------------------------------------------
# Rule-list builders
# ---------------------------------------------------------------------------


def _rule(**kw) -> dict:
    return {
        "type": kw.pop("type"),
        "symbol": kw.pop("symbol", "*"),
        "strategy": kw.pop("strategy", "CARRY_ARBITRAGE"),
        "active": kw.pop("active", True),
        "rationale": kw.pop("rationale", "test rule"),
        **kw,
    }


def write_rules(tmp_path, rules) -> str:
    """Persist ``rules`` to a temp learnings file and return its path (str)."""
    path = tmp_path / "learnings.json"
    path.write_text(json.dumps(rules), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Gate unit tests
# ---------------------------------------------------------------------------


def test_missing_learnings_file_admits_all(tmp_path):
    gate = LocalMemoryGate(learnings_path=str(tmp_path / "does_not_exist.json"))

    admitted, reason = gate.evaluate_candidate("SOLUSDT", "CARRY_ARBITRAGE", {})
    assert admitted is True
    assert reason == "All memory rules passed"


def test_mtime_reload_picks_up_disk_updates(tmp_path):
    path = tmp_path / "learnings.json"
    # First state: a MIN_APR 30 rule that the 25% candidate does NOT meet.
    path.write_text(json.dumps([_rule(type="MIN_APR", min_apr=30.0)]), encoding="utf-8")
    gate = LocalMemoryGate(learnings_path=str(path))

    admitted, _ = gate.evaluate_candidate(
        "SOLUSDT", "CARRY_ARBITRAGE", {"net_apr": 25.0}
    )
    assert admitted is False  # 25% < min 30% -> blocked by the initial file

    # Rewrite the file with a looser rule and bump the mtime so reload is forced.
    path.write_text(
        json.dumps([_rule(type="MIN_APR", min_apr=10.0, rationale="min APR 10")]),
        encoding="utf-8",
    )
    os.utime(path, ns=(path.stat().st_mtime_ns + 1_000_000_000,) * 2)

    # The reloaded rule is now met -> the same candidate is admitted.
    admitted, reason = gate.evaluate_candidate(
        "SOLUSDT", "CARRY_ARBITRAGE", {"net_apr": 25.0}
    )
    assert admitted is True
    assert reason == "All memory rules passed"

    # And the freshly-loaded rule rejects an under-yield candidate.
    admitted, reason = gate.evaluate_candidate(
        "SOLUSDT", "CARRY_ARBITRAGE", {"net_apr": 5.0}
    )
    assert admitted is False
    assert "min APR 10" in reason


def test_blacklist_blocks_symbol_unconditionally(tmp_path):
    path = write_rules(
        tmp_path,
        [_rule(type="BLACKLIST", symbol="SOLUSDT", rationale="bad runs on SOL")],
    )
    gate = LocalMemoryGate(learnings_path=path)

    admitted, reason = gate.evaluate_candidate("SOLUSDT", "CARRY_ARBITRAGE", {})
    assert admitted is False
    assert "bad runs on SOL" in reason

    # Case-insensitive symbol matching.
    admitted, _ = gate.evaluate_candidate("solusdt", "CARRY_ARBITRAGE", {})
    assert admitted is False


def test_max_entry_basis_discount_blocks_shortfall(tmp_path):
    path = write_rules(
        tmp_path,
        [_rule(type="MAX_ENTRY_BASIS_DISCOUNT", max_discount_pct=-0.5,
               rationale="adverse entry basis")],
    )
    gate = LocalMemoryGate(learnings_path=path)
    admit, _ = gate.evaluate_candidate(
        "SOLUSDT", "CARRY_ARBITRAGE", {"basis_spread_pct": 0.1}
    )
    assert admit is True   # 0.1% > -0.5%

    admit, reason = gate.evaluate_candidate(
        "SOLUSDT", "CARRY_ARBITRAGE", {"basis_spread_pct": -1.2}
    )
    assert admit is False
    assert "adverse entry basis" in reason


def test_min_apr_blocks_low_yield(tmp_path):
    path = write_rules(
        tmp_path,
        [_rule(type="MIN_APR", min_apr=20.0, rationale="yield too thin")],
    )
    gate = LocalMemoryGate(learnings_path=path)
    admit, _ = gate.evaluate_candidate(
        "SOLUSDT", "CARRY_ARBITRAGE", {"net_apr": 25.0}
    )
    assert admit is True

    admit, reason = gate.evaluate_candidate(
        "SOLUSDT", "CARRY_ARBITRAGE", {"net_apr": 12.0}
    )
    assert admit is False
    assert "yield too thin" in reason


def test_non_matching_rules_admit(tmp_path):
    rules = [
        _rule(type="BLACKLIST", symbol="SOLUSDT", strategy="OTHER_STRATEGY"),
        _rule(type="BLACKLIST", symbol="BTCUSDT"),
        _rule(type="MIN_APR", min_apr=20.0),
    ]
    gate = LocalMemoryGate(learnings_path=write_rules(tmp_path, rules))

    # Different strategy -> neither BLACKLIST applies; MIN_APR at 25% passes.
    admitted, reason = gate.evaluate_candidate(
        "SOLUSDT", "DIFFERENT", {"net_apr": 25.0}
    )
    assert admitted is True
    assert reason == "All memory rules passed"


def test_disabled_rule_is_inert(tmp_path):
    rules = [
        _rule(type="BLACKLIST", symbol="SOLUSDT", active=False),
    ]
    gate = LocalMemoryGate(learnings_path=write_rules(tmp_path, rules))

    admitted, _ = gate.evaluate_candidate("SOLUSDT", "CARRY_ARBITRAGE", {})
    assert admitted is True


def test_malformed_json_degrades_to_allow_all(tmp_path):
    path = tmp_path / "learnings.json"
    path.write_text("{ not valid json", encoding="utf-8")
    gate = LocalMemoryGate(learnings_path=str(path))

    admitted, _ = gate.evaluate_candidate("SOLUSDT", "CARRY_ARBITRAGE", {})
    assert admitted is True


# ---------------------------------------------------------------------------
# carry_daemon integration
# ---------------------------------------------------------------------------


class _FakeConnector:
    """In-memory account view: healthy margin ratio, no open positions."""

    def __init__(self):
        self.balances: dict = {}
        self.positions: dict = {}

    def get_account_info(self):
        return {"totalMarginBalance": "100", "totalMaintMargin": "10"}

    def get_balances(self):
        return dict(self.balances)

    def get_positions(self, symbol):
        return [{"position_amt": self.positions.get(symbol, 0.0)}]

    def get_funding_rate_history(self, symbol, limit=1):
        return [{"fundingRate": "0.0002"}]

    def get_funding_income_history(self, symbol, start_time=None, limit=100):
        return [{"time": start_time + 5 * 60 * 1000, "income": "0.01"}]

    def _request(self, method, path, params=None):
        return {"lastFundingRate": "0.0002", "markPrice": "100", "nextFundingTime": 1}


class _FakeLeg:
    def __init__(self, connector):
        self.connector = connector


class _FakeBroker:
    """Twin-leg broker: fills record into ``executed``; no exchange I/O."""

    def __init__(self):
        self.futures = _FakeLeg(_FakeConnector())
        self.spot = _FakeLeg(_FakeConnector())
        self.filters = TwinLegFilters(
            symbol="*", step_size=0.01, min_qty=0.01, max_qty=1000.0, min_notional=5.0
        )
        self.executed: list = []

    def get_twin_filters(self, base, quote="USDT"):
        return self.filters

    def execute_twin_leg_carry(self, base, quote, quantity):
        self.executed.append((base, quote, quantity))
        # Fake position — never reached when a memory rule blocks admission.
        from crypto_quant.execution.live_broker import TwinLegPosition
        return TwinLegPosition(
            position_id=f"TWIN-{base}-{quote}-1",
            base_asset=base, quote_asset=quote, quantity=quantity,
            spot_order_id="s1", spot_fill_price=100.0,
            futures_order_id="f1", futures_fill_price=100.2,
            entry_basis_spread_pct=0.2, execution_gap_ms=100.0,
            status="OPEN", opened_at=1_700_000_000.0,
        )


class _FakeScanner:
    def __init__(self, *opps):
        self.opps = list(opps)

    def scan(self, provider=None):
        return list(self.opps)


class _RecordingNotifier:
    def __init__(self):
        self.calls: list = []

    def is_enabled(self):
        return True

    def notify_trade_entry(self, **kw):
        self.calls.append(("entry", kw))

    def notify_financial_audit(self, **kw):
        self.calls.append(("audit", kw))


def _opp(symbol="SOLUSDT", base="SOL"):
    return CarryOpportunity(
        symbol=symbol,
        spot_price=100.0,
        futures_price=100.1,
        basis_spread_pct=0.1,
        predicted_funding_rate=0.0001,
        gross_apr_pct=30.0,
        net_apr_pct=25.0,
        next_funding_time=datetime.now(timezone.utc),
        volume_24h_usdt=100_000_000.0,
    )


def _make_daemon(tmp_path, rules, opportunities):
    """Wire a daemon whose memory gate is controlled by ``rules`` on disk."""
    learnings = write_rules(tmp_path, rules)
    config = CarryDaemonConfig(
        allocation_per_pair_usdt=50.0,
        min_apr_threshold=0.10,
        learnings_path=learnings,
        dry_run=True,
    )
    broker = _FakeBroker()
    daemon = CarryHarvesterDaemon(
        config,
        broker,
        scanner=_FakeScanner(*opportunities),
    )
    daemon._notifier = _RecordingNotifier()
    return daemon, broker


def test_memory_rule_skips_candidate_in_tick(tmp_path):
    daemon, broker = _make_daemon(
        tmp_path,
        [_rule(type="BLACKLIST", symbol="SOLUSDT", rationale="blacklisted in tick")],
        [_opp()],
    )

    daemon.tick()

    assert broker.executed == []
    assert set(daemon.active_positions) == set()


def test_admission_proceeds_without_memory_block(tmp_path):
    daemon, broker = _make_daemon(tmp_path, [], [_opp()])

    daemon.tick()

    assert broker.executed == [("SOL", "USDT", 0.5)]
    assert set(daemon.active_positions) == {"SOLUSDT"}