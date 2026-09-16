"""Hermetic unit tests for Phase 3 Step 3.2 — Automated Ledger-to-Binance
Balance Reconciliation.

Drives ``CarryHarvesterDaemon._reconcile_exchange_state`` against the shared
hermetic broker/connector fakes, so nothing touches the network:

* Ghost positions (open locally, 0 on the exchange) are cleaned up and
  persisted as ``DESYNC_CLOSED``.
* Delta divergence between the spot and futures legs logs a warning and, past
  5% relative drift, dispatches a circuit-breaker alert.
* Unmanaged exchange futures positions trigger an alert log without raising
  through the daemon loop.
* Reconciliation fires from the health loop on the ``reconcile_interval_seconds``
  cadence.
"""

import logging

from crypto_quant.db.connection import DatabaseManager
from crypto_quant.db.models import CarryPositionRecord
from crypto_quant.execution.carry_daemon import CarryDaemonConfig

# Reuse the hermetic fakes/harness from the carry daemon suite.
from test_carry_daemon import (
    FakeCarryBroker,
    RecordingNotifier,
    _opp,
    make_daemon,
)


def test_ghost_position_marked_desync_closed_and_persisted():
    """A pair locally tracked as OPEN but with a 0-size futures position on the
    exchange is retired as DESYNC_CLOSED, dropped from the book and persisted."""
    db = DatabaseManager(in_memory=True)
    db.create_tables()
    carry = FakeCarryBroker()
    daemon = make_daemon(db, carry, allocation=50.0, opportunities=[_opp()])

    daemon.tick()  # opens a 0.5 carry: spot +0.5, futures short -0.5
    assert "SOLUSDT" in daemon.active_positions

    # The exchange no longer holds the futures leg -> ghost.
    carry.futures.connector.positions["SOLUSDT"] = 0.0
    daemon._reconcile_exchange_state()

    assert "SOLUSDT" not in daemon.active_positions
    session = db.get_session()
    try:
        row = session.get(CarryPositionRecord, "TWIN-SOL-USDT-1")
        assert row is not None
        assert row.status == "DESYNC_CLOSED"
        assert row.closed_at is not None
    finally:
        session.close()


def test_delta_divergence_logs_warning_and_flags_discrepancy(caplog):
    """A spot/futures leg that drifts apart beyond delta_tolerance logs a warning
    and (past 5% relative drift) dispatches a circuit-breaker alert — without
    unwinding or removing the pair."""
    carry = FakeCarryBroker()
    daemon = make_daemon(carry=carry, allocation=50.0, opportunities=[_opp()])
    rec = RecordingNotifier()
    daemon._notifier = rec

    daemon.tick()  # opens 0.5 carry: spot +0.5, futures short -0.5

    # Futures short collapses to -0.1 -> allocated residual |0.5 + (-0.1)| = 0.4.
    carry.futures.connector.positions["SOLUSDT"] = -0.1
    with caplog.at_level(logging.WARNING):
        daemon._reconcile_exchange_state()

    # Not a ghost (the futures leg still exists), so the pair stays managed.
    assert "SOLUSDT" in daemon.active_positions
    assert any("Delta drift on SOLUSDT" in r.getMessage() for r in caplog.records)
    cells = [c for c in rec.calls if c[0] == "circuit"]
    assert any("RECONCILIATION_DELTA_DRIFT" in c[1]["reason"] for c in cells)
    # The alert names the pair so the operator can locate the unhedged leg.
    assert any(c[1]["symbol"] == "SOLUSDT" for c in cells)


def test_delta_divergence_within_tolerance_stays_quiet(caplog):
    """A healthy, still-hedged carry (no drift) produces no warning and no alert."""
    carry = FakeCarryBroker()
    daemon = make_daemon(carry=carry, allocation=50.0, opportunities=[_opp()])
    rec = RecordingNotifier()
    daemon._notifier = rec

    daemon.tick()  # hedged: spot +0.5, futures -0.5 -> residual 0

    with caplog.at_level(logging.WARNING):
        daemon._reconcile_exchange_state()

    assert "SOLUSDT" in daemon.active_positions
    assert not any("Delta drift" in r.getMessage() for r in caplog.records)
    assert not any(c[0] == "circuit" for c in rec.calls)  # no severe escalation/escalation alert found


def test_unmanaged_futures_position_triggers_alert_without_raising(caplog):
    """An open exchange futures position the daemon never opened is surfaced as
    an alert log, and the reconciliation loop keeps running (no exception)."""
    carry = FakeCarryBroker()
    # A foreign short the daemon did not create.
    carry.futures.connector.positions["DOGEUSDT"] = -2.0
    daemon = make_daemon(carry=carry)  # no positions opened

    with caplog.at_level(logging.WARNING):
        daemon._reconcile_exchange_state()  # must not raise

    assert any(
        "Unmanaged futures position detected for DOGEUSDT" in r.getMessage()
        for r in caplog.records
    )
    assert daemon.active_positions == {}  # book untouched; daemon never trades it


def test_reconcile_fires_from_health_check_on_interval():
    """Reconciliation runs on the reconcile_interval_seconds cadence from the
    health loop (not on every tick)."""
    daemon = make_daemon()
    calls: list[int] = []
    daemon._reconcile_exchange_state = lambda: calls.append(1)

    daemon.tick()  # freshly started => not yet due
    assert calls == []

    daemon._last_reconcile_at = 0.0  # force the interval to have elapsed
    daemon.tick()
    assert len(calls) == 1


def test_reconcile_interval_config_default_and_validation():
    """reconcile_interval_seconds defaults to 300s and must be positive."""
    config = CarryDaemonConfig()
    assert config.reconcile_interval_seconds == 300.0
    try:
        CarryDaemonConfig(reconcile_interval_seconds=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive reconcile_interval_seconds must be rejected")