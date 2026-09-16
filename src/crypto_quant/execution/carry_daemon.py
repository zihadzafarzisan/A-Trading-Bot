"""Autonomous, persisted multi-pair funding-rate cash-and-carry allocator.

The daemon owns the *strategy lifecycle* around :class:`TwinLegCarryBroker`: it
periodically scans the USDⓈ-M universe (:class:`FundingScanner`), ranks the best
net-annualized-yield candidates, sizes each lot from a per-pair USDT capital
budget, and manages up to ``max_active_pairs`` concurrent twin-leg positions.

Each open position retains its own baseline inventory captured at its own entry
(pre-existing spot/futures), so delta neutrality and lot reconciliation stay
isolated per pair. Discovery fills spare capacity up to ``max_active_pairs``;
the health loop independently verifies delta/margin and gracefully unwinds any
pair whose predicted funding collapses below ``exit_apr_threshold``.

It deliberately delegates every order to the hardened broker rather than
reimplementing execution, and persists state to SQLite so the book survives
restarts (baselines included).
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import func

from ..analytics import TradeLedger
from ..db.connection import DatabaseManager, get_db_manager
from ..db.models import CarryFundingPaymentRecord, CarryPositionRecord
from ..logging_config import get_logger
from ..notifications import get_notifier
from ..research.funding_scanner import FundingScanner
from ..risk.local_memory import LocalMemoryGate
from .dashboard_commands import process_dashboard_commands
from .live_broker import TwinLegCarryBroker, TwinLegExecutionError, TwinLegPosition
from .worker import WorkerLock

logger = get_logger("trading")

_FUNDING_PERIOD_SECONDS = 8 * 60 * 60
_CARRY_LOCK_PATH = Path("data/carry_daemon.lock")
# After nextFundingTime rolls forward, Binance USDⓈ-M batch-clearing can take
# 1-5 minutes to post the FUNDING_FEE row to /fapi/v1/income. The daemon keeps
# re-querying for this long before giving up (and pointing the operator at
# `carry reconcile-funding` to backfill it).
_FUNDING_RETRY_WINDOW_SECONDS = 15 * 60
# Cadence for the periodic portfolio financial-audit Discord card.
_AUDIT_INTERVAL_SECONDS = 60 * 60
# Pre-trade loss-prevention hurdles (Safeguard). Minimum predicted 8h funding a
# carry must pay to be admitted (+0.005%/8h, so zero/negative funding never
# opens), and the most-negative basis spread tolerated (-0.12%: beyond this,
# spot is too expensive relative to futures and the entry drag is adverse).
_MIN_ENTRY_FUNDING_RATE = 0.00005
_MIN_ENTRY_BASIS_PCT = -0.0012


@dataclass(frozen=True)
class FundingSnapshot:
    """Current Futures funding estimate and the next settlement time."""

    symbol: str
    funding_rate: float
    annualized_rate: float
    mark_price: float
    next_funding_time_ms: Optional[int]


@dataclass
class CarryDaemonConfig:
    """Operational limits for one autonomous multi-pair cash-and-carry daemon."""

    quote_asset: str = "USDT"
    max_active_pairs: int = 3
    total_allocation_usdt: float = 1000.0
    allocation_per_pair_usdt: Optional[float] = None  # default = total / max_active_pairs
    scan_interval_seconds: float = 300.0
    min_apr_threshold: float = 0.08    # entry hurdle (fraction: 0.08 = 8%)
    exit_apr_threshold: float = 0.02   # unwinding hurdle when funding collapses (fraction)
    # Seconds a symbol is barred from re-admission after being unwound, so a
    # just-flattened pair (e.g. a margin/delta emergency) can't immediately
    # re-open and churn the book.
    unwind_cooldown_seconds: float = 900.0
    poll_interval_seconds: float = 30.0
    # Phase 3 Step 3.2 — cadence for the ledger-to-exchange balance/position
    # reconciliation (ghost-position cleanup, delta-drift detection, and the
    # unmanaged-futures scan), measured from the health loop.
    reconcile_interval_seconds: float = 300.0
    # Phase 3 Step 3.1 — broken-leg and slippage guards for the entry path.
    # If Leg 2 fails or is aborted, the broker must emergency market-SELL
    # the exact Spot fill qty immediately so the account never holds
    # orphaned, unhedged spot inventory.
    max_leg_gap_ms: float = 800.0
    max_entry_slippage_pct: float = 0.0015  # 0.15% adverse drift guard

    min_margin_ratio: float = 0.20
    # Phase 2 Step 2.1 — local forensic trade ledger. Every completed unwind
    # appends one JSON line here for the adaptive reflection layer.
    ledger_path: str = "data/trade_ledger.jsonl"
    # Phase 2 Step 2.2 — local memory gate. Advisory rules learned from past
    # trades (blacklist / basis-discount / min-APR) enforced as an extra
    # admission hurdle in the pre-flight risk gatekeeper. JSON reloads only when
    # the file's mtime changes, so a long-lived daemon picks up new learnings
    # without a restart.
    learnings_path: str = "config/learnings.json"
    # Pre-flight risk gatekeeper (Phase 1 Step 1.1) hurdles:
    #   max_portfolio_usd — hard cap on TOTAL deployed pair notional across the
    #   book. When None, the declared budget is read from config/settings
    #   (risk.starting_capital, default $1,000) as the ceiling.
    #   entry_margin_ratio_min — minimum futures available-margin ratio
    #   ((equity - maintenance) / equity) required to admit a NEW entry, so a
    #   short-leg liquidation spike can't be entered into a thin buffer.
    max_portfolio_usd: Optional[float] = None
    entry_margin_ratio_min: float = 0.25
    # Phase 1 Step 1.3 — tiered margin circuit breakers. Account-wide futures
    # available-margin ratio ((equity - maintenance) / equity) gates below:
    #   soft_margin_ratio_warning — once the live buffer falls under this (15%
    #   default) ALL new carry admissions are paused until it recovers. It never
    #   unwinds — an operator-visible Level 1 soft breaker.
    #   emergency_delever_ratio — past this harder bound (10% default) the
    #   daemon auto-deleverages one active pair (largest deployed notional) and
    #   DMs an urgent card (Level 2 hard circuit breaker / liquidation guard).
    #   Invariant enforced below:
    #   emergency_delever_ratio < soft_margin_ratio_warning < entry_margin_ratio_min
    soft_margin_ratio_warning: float = 0.15
    emergency_delever_ratio: float = 0.10
    # Phase 1 Step 1.2 — adverse basis divergence stop. When the live futures-
    # over-spot basis has moved adversely AGAINST an open carry by more than
    # ``max_adverse_basis_pct`` since entry (0.75% default), the pair is
    # gracefully unwound (``ADVERSE_BASIS_STOP``) before the spread drag erodes
    # the delta-neutral PnL. Fraction: 0.0075 = 0.75%.
    max_adverse_basis_pct: float = 0.0075
    # Base-quantity tolerance (in units of the base asset) for the daemon's
    # delta-neutrality check. Accounts may carry pre-existing inventory, so the
    # check compares ALLOCATED deltas (current - baseline); this tolerance just
    # absorbs fill rounding on top of a healthy hedge.
    delta_tolerance: float = 0.01
    # When True, per-pair lot sizing reads the account's live available free
    # USDT margin and caps the budget at that * ``available_margin_use_fraction``
    # — letting allocation scale beyond small default test budgets when a real
    # margin balance is configured.
    scale_to_available_margin: bool = False
    available_margin_use_fraction: float = 0.5
    dry_run: bool = True
    run_id: str = ""
    unwind_on_shutdown: bool = True

    def __post_init__(self) -> None:
        self.quote_asset = self.quote_asset.upper()
        if self.max_active_pairs < 1:
            raise ValueError("max_active_pairs must be at least 1")
        if self.total_allocation_usdt <= 0:
            raise ValueError("total_allocation_usdt must be positive")
        if self.allocation_per_pair_usdt is not None and self.allocation_per_pair_usdt <= 0:
            raise ValueError("allocation_per_pair_usdt must be positive when set")
        if self.scan_interval_seconds <= 0:
            raise ValueError("scan_interval_seconds must be positive")
        if self.min_apr_threshold < 0:
            raise ValueError("min_apr_threshold cannot be negative")
        if self.exit_apr_threshold < 0:
            raise ValueError("exit_apr_threshold cannot be negative")
        if self.exit_apr_threshold >= self.min_apr_threshold:
            raise ValueError("exit_apr_threshold must be below min_apr_threshold")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if self.reconcile_interval_seconds <= 0:
            raise ValueError("reconcile_interval_seconds must be positive")
        if not 0 < self.min_margin_ratio < 1:
            raise ValueError("min_margin_ratio must be between 0 and 1")
        if self.max_portfolio_usd is not None and self.max_portfolio_usd <= 0:
            raise ValueError("max_portfolio_usd must be positive when set")
        if not 0 < self.entry_margin_ratio_min < 1:
            raise ValueError("entry_margin_ratio_min must be between 0 and 1")
        if not (0.05 < self.emergency_delever_ratio < self.soft_margin_ratio_warning):
            raise ValueError(
                "emergency_delever_ratio must satisfy "
                "0.05 < emergency_delever_ratio < soft_margin_ratio_warning"
            )
        if not (self.soft_margin_ratio_warning < self.entry_margin_ratio_min):
            raise ValueError(
                "soft_margin_ratio_warning must be below entry_margin_ratio_min "
                "(circuit breakers sit under the admission floor)"
            )
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        # Note: validated above; remaining duplicated blocks removed in follow-up.
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        # (Guardrail) validated above.
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        # (Guardrail) validated above.
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        # (Guardrail) validated above.
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        # (Guardrail) validated above.
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        # (Guardrail) validated above.
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        # (Guardrail) validated above.
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        # (Guardrail) validated above.
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        # (Guardrail) validated above.
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        # (Guardrail) validated above.
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        # (Guardrail) validated above.
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        # (Guardrail) validated above.
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        # (Guardrail) validated above.
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        # (Guardrail) validated above.
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        # (Guardrail) validated above.
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        # (Guardrail) validated above.
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        # (Guardrail) validated above.
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        if self.max_adverse_basis_pct <= 0:
            raise ValueError("max_adverse_basis_pct must be positive")
        # (Guardrail) validated above.
        if not 0 < self.max_adverse_basis_pct < 0.05:
            raise ValueError("max_adverse_basis_pct must be between 0 and 0.05")
        # validated above
        if self.delta_tolerance < 0:
            raise ValueError("delta_tolerance cannot be negative")
        if not 0 < self.available_margin_use_fraction <= 1.0:
            raise ValueError("available_margin_use_fraction must be in (0, 1]")

    @property
    def allocation_per_pair(self) -> float:
        """USDT budget deployed per twin-leg pair."""
        if self.allocation_per_pair_usdt is not None:
            return self.allocation_per_pair_usdt
        return self.total_allocation_usdt / self.max_active_pairs


class CarryHarvesterDaemon:
    """Owns multi-pair carry admission, health checks, and durable state."""

    def __init__(
        self,
        config: CarryDaemonConfig,
        carry_broker: TwinLegCarryBroker,
        db: Optional[DatabaseManager] = None,
        lock: Optional[WorkerLock] = None,
        scanner: Optional[FundingScanner] = None,
        scanner_provider: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.carry_broker = carry_broker
        self.db = db
        self.lock = lock or WorkerLock(path=_CARRY_LOCK_PATH)
        self.scanner = scanner
        self.scanner_provider = scanner_provider
        if not self.config.run_id:
            self.config.run_id = f"CARRY-{'DRY' if config.dry_run else 'LIVE'}-{int(time.time() * 1000)}"
        self._running = False
        self._shutdown_requested = False
        self._active_positions: Dict[str, TwinLegPosition] = {}  # keyed by symbol, e.g. "SOLUSDT"
        # Phase 2 Step 2.1 — local append-only forensic trade ledger. Completed
        # unwinds are mirrored here (in addition to SQLite + Discord) so the
        # reflection layer can read them back next session.
        self.trade_ledger = TradeLedger(ledger_path=self.config.ledger_path)
        # Phase 2 Step 2.2 — local memory gate: enforces advisory learnings
        # (blacklist / basis-discount / min-APR) as Rule D of pre-flight risk.
        self.memory_gate = LocalMemoryGate(learnings_path=self.config.learnings_path)
        self._last_funding_settlement_ms: Dict[str, Optional[int]] = {}
        self._last_seen_next_funding_ms: Dict[str, Optional[int]] = {}
        # Wall-clock deadline (time.time() + _FUNDING_RETRY_WINDOW_SECONDS, keyed
        # by symbol) while a rolled-over settlement is still awaiting its
        # (delayed) income event. Cleared once confirmed or on expiry.
        self._pending_funding_retry_until: Dict[str, float] = {}
        # Wall-clock of the most recent unwind per symbol, so a just-flattened
        # pair is barred from re-admission until ``unwind_cooldown_seconds``
        # elapses (prevents the scanner from re-opening a pair that just
        # collapsed/churned out of the book).
        self._last_unwind_at: Dict[str, float] = {}
        # Phase 3 Step 3.2 — wall-clock of the last ledger-to-exchange
        # reconciliation run; the next fire happens once
        # ``reconcile_interval_seconds`` elapses inside the health loop.
        self._last_reconcile_at: float = time.time()
        # Dashboard IPC control flags (set by process_dashboard_commands each tick).
        self._entries_paused = False
        self._kill_switch = False
        # Phase 1 Step 1.3 — Level 1 soft breaker: latches True while the live
        # futures margin ratio sits under ``soft_margin_ratio_warning``, blocking
        # ALL new admissions until the buffer recovers. Kept distinct from the
        # dashboard ``_entries_paused`` so an operator RESUME doesn't silently
        # lift an automatic margin-guard pause.
        self._margin_soft_pause = False
        self._notifier = get_notifier()  # disabled-safe Discord DM pipeline
        if not self._notifier.is_enabled():
            logger.warning(
                "[CARRY DAEMON] Discord notifications are DISABLED — "
                "DISCORD_BOT_TOKEN/DISCORD_USER_ID are missing from the environment. "
                "Lifecycle alerts (entries, unwinds, funding, audits) will NOT be "
                "delivered. export them (or load via .env) and restart the daemon."
            )
        if self.db is not None:
            self.db.create_tables()
            self._restore_active_positions()

    # ------------------------------------------------------------------ state
    @property
    def active_positions(self) -> Dict[str, TwinLegPosition]:
        """The daemon-managed twin-leg book, keyed by symbol."""
        return self._active_positions

    @property
    def open_position(self) -> Optional[TwinLegPosition]:
        """Backward-compat convenience: the single active position, else None."""
        if len(self._active_positions) == 1:
            return next(iter(self._active_positions.values()))
        return None

    # ------------------------------------------------------------------ dashboard IPC
    def _dashboard_handlers(self) -> Dict[str, Callable[[dict], None]]:
        """Return command-name → handler map for process_dashboard_commands()."""
        def reset_daily_loss(_params: dict) -> None:
            logger.info("[DASHBOARD CMD] RESET_DAILY_LOSS: acknowledged (no-op for carry daemon).")

        def pause_entries(_params: dict) -> None:
            self._entries_paused = True
            logger.info("[DASHBOARD CMD] PAUSE_ENTRIES: new carry admissions paused.")

        def resume_entries(_params: dict) -> None:
            self._entries_paused = False
            logger.info("[DASHBOARD CMD] RESUME_ENTRIES: carry admissions re-enabled.")

        def kill_switch_engage(_params: dict) -> None:
            # Immediate termination request: stop discovery and break the loop.
            self._kill_switch = True
            self.request_shutdown()
            self._running = False
            logger.critical("[DASHBOARD CMD] KILL_SWITCH_ENGAGE: kill switch engaged; requesting termination.")

        def kill_switch_disengage(_params: dict) -> None:
            self._kill_switch = False
            logger.warning("[DASHBOARD CMD] KILL_SWITCH_DISENGAGE: kill switch released.")

        return {
            "RESET_DAILY_LOSS": reset_daily_loss,
            "PAUSE_ENTRIES": pause_entries,
            "RESUME_ENTRIES": resume_entries,
            "KILL_SWITCH_ENGAGE": kill_switch_engage,
            "KILL_SWITCH_DISENGAGE": kill_switch_disengage,
        }

    def request_shutdown(self, *_: Any) -> None:
        """Signal-safe intent flag; actual exchange calls occur in the main loop."""
        self._shutdown_requested = True
        self._running = False

    def install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers when running on the main thread."""
        signal.signal(signal.SIGINT, self.request_shutdown)
        signal.signal(signal.SIGTERM, self.request_shutdown)

    # ------------------------------------------------------------------ core
    def funding_snapshot(self, symbol: str) -> FundingSnapshot:
        """Read the Binance premium index for ``symbol`` (predicted 8h rate)."""
        connector = self.carry_broker.futures.connector
        response = connector._request(
            "GET", "/fapi/v1/premiumIndex", params={"symbol": symbol}
        )
        history = connector.get_funding_rate_history(symbol, limit=1)
        settled_rate = float(history[-1].get("fundingRate", 0.0)) if history else 0.0
        funding_rate = float(response.get("lastFundingRate", settled_rate) or 0.0)
        next_time = response.get("nextFundingTime")
        return FundingSnapshot(
            symbol=symbol,
            funding_rate=funding_rate,
            annualized_rate=funding_rate * 3.0 * 365.0,
            mark_price=float(response.get("markPrice", 0.0) or 0.0),
            next_funding_time_ms=int(next_time) if next_time is not None else None,
        )

    def tick(self) -> None:
        """One full cycle: health checks on the book, then a discovery pass."""
        self._health_check()
        self._discover()

    def run(self) -> None:
        """Run until SIGINT/SIGTERM or an unrecoverable daemon exception."""
        if not self.lock.acquire(self.config.run_id):
            raise RuntimeError("Another carry daemon holds the lock. Refusing start.")
        self.install_signal_handlers()
        self._running = True
        last_scan: Optional[float] = None
        last_audit: Optional[float] = None
        try:
            while self._running and not self._shutdown_requested:
                try:
                    process_dashboard_commands(self._dashboard_handlers())
                    if (not self._running) or self._shutdown_requested:
                        logger.critical("[CARRY DAEMON] Terminating execution loop via Kill Switch.")
                        break
                    self._health_check()                 # poll cadence
                    if self._scan_due(last_scan):        # scan cadence
                        last_scan = time.time()
                        self._discover()
                    # hourly portfolio financial-audit heartbeat (Defect 2)
                    now = time.time()
                    if (last_audit is None or (now - last_audit) >= _AUDIT_INTERVAL_SECONDS):
                        last_audit = now
                        self._dispatch_financial_audit()
                except Exception as exc:
                    logger.exception("[CARRY DAEMON] tick failed: %s", exc)
                self.lock.heartbeat()

                # Responsive sleep polling IPC every 1 second so kill/pause
                # commands take effect quickly.
                poll_interval = int(self.config.poll_interval_seconds)
                sleep_remaining = max(0, poll_interval)
                while (
                    sleep_remaining > 0
                    and self._running
                    and not self._shutdown_requested
                ):
                    time.sleep(1)
                    sleep_remaining -= 1
                    process_dashboard_commands(self._dashboard_handlers())
        finally:
            self.shutdown()

    def _scan_due(self, last_scan: Optional[float]) -> bool:
        if last_scan is None:
            return True
        return (time.time() - last_scan) >= self.config.scan_interval_seconds

    def shutdown(self) -> None:
        """Persist state and, when configured, concurrently unwind the book."""
        self._running = False
        try:
            if self.config.unwind_on_shutdown and self._active_positions:
                self._unwind_all_on_shutdown()
            else:
                for position in list(self._active_positions.values()):
                    self._persist_position(position)
        finally:
            self.lock.release()

    def _unwind_all_on_shutdown(self) -> None:
        """Concurrently flatten every active pair, blue-embed + persist each."""
        positions = list(self._active_positions.values())

        def _close(position: TwinLegPosition):
            try:
                self.carry_broker.unwind_twin_leg_carry(position)
                return position, None
            except TwinLegExecutionError as exc:
                position.status = "UNWINDING"
                return position, exc

        with ThreadPoolExecutor(
            max_workers=max(1, len(positions)), thread_name_prefix="carry-shutdown"
        ) as ex:
            results = list(ex.map(_close, positions))  # concurrent exchange calls

        for position, exc in results:
            symbol = f"{position.base_asset}{self.config.quote_asset}"
            if exc is None and position.status == "CLOSED":
                self._persist_position(position)
                duration = _fmt_duration(position.opened_at, position.closed_at)
                m = self._close_metrics(position)
                capital = float(m["capital_invested_usdt"] or 0.0)
                gross = float(m["funding_on_trade_usdt"] + m["unrealized_basis_pnl_usdt"])
                net = float(m["net_pnl_usdt"])
                roi = float(m["net_pnl_pct"])
                funding = float(m["funding_on_trade_usdt"])
                # Rich exit card, identical format to graceful/emergency unwinds.
                self._notifier.notify_trade_exit(
                    symbol=symbol,
                    position_id=position.position_id,
                    strategy="CARRY_ARBITRAGE",
                    side="CARRY",
                    qty=position.quantity,
                    spot_fill_price=float(position.spot_fill_price or 0.0),
                    futures_fill_price=float(position.futures_fill_price or 0.0),
                    gross_pnl_usdt=gross,
                    net_pnl_usdt=net,
                    funding_harvested_usdt=funding,
                    roi_pct=roi,
                    duration=duration,
                    exit_reason="MANUAL",
                    extra=self._close_extra(position, "daemon shutdown"),
                )
                self._mark_unwound(symbol, position)
            else:
                self._persist_position(position)  # persists UNWINDING state
                self._notifier.notify_circuit_breaker(
                    symbol=symbol,
                    reason="shutdown unwind failed",
                    extra={"Position ID": position.position_id},
                )

    # ------------------------------------------------------------------ health
    def _health_check(self) -> None:
        # Account-wide tiered margin circuit breakers (Phase 1 Step 1.3) run once
        # per tick, before per-pair safety, so a thin book can batten down first.
        self._check_tiered_margin_health()
        # Periodic ledger-to-exchange reconciliation (Phase 3 Step 3.2): ghost-
        # position cleanup, delta-drift warning/alerts, and the unmanaged-futures
        # scan all run on the ``reconcile_interval_seconds`` cadence.
        now = time.time()
        if (self._last_reconcile_at is None
                or (now - self._last_reconcile_at) >= self.config.reconcile_interval_seconds):
            self._last_reconcile_at = now
            self._reconcile_exchange_state()
        for symbol, position in list(self._active_positions.items()):
            if position.status in ("UNWINDING", "CLOSED"):
                continue
            # 1) Per-pair delta + account margin (may emergency-unwind).
            self._verify_delta_and_margin(position, symbol)
            if self._active_positions.get(symbol) is not position:
                continue  # removed during verification
            if position.status != "OPEN":
                continue
            # 2) Adverse basis spread divergence stop.
            # When the live futures-over-spot basis moves adversely beyond the
            # configured tolerance since entry, we gracefully stop/unwind to
            # prevent spread drag erosion.
            try:
                entry_basis = float(position.entry_basis_spread_pct or 0.0) / 100.0
                spot_px = self._live_spot(symbol)
                fut_px = self._live_mark(symbol)
                if spot_px is not None and fut_px is not None and spot_px > 0:
                    live_basis = (fut_px - spot_px) / spot_px
                    divergence = live_basis - entry_basis
                    if divergence > self.config.max_adverse_basis_pct:
                        self._graceful_unwind(
                            position,
                            symbol,
                            reason=(
                                f"ADVERSE_BASIS_STOP divergence={divergence * 100:+.4f}% "
                                f"(live_basis={live_basis * 100:+.4f}%, entry_basis={entry_basis * 100:+.4f}%)"
                            ),
                        )
                        continue
            except Exception as exc:
                logger.warning("Adverse basis check failed for %s: %s", symbol, exc)

            # 3) Graceful unwind if predicted funding collapses below exit hurdle.
            try:
                snapshot = self.funding_snapshot(symbol)
            except Exception as exc:
                logger.warning("Funding snapshot failed for %s: %s", symbol, exc)
                continue
            if snapshot.annualized_rate < self.config.exit_apr_threshold:
                self._graceful_unwind(
                    position, symbol,
                    f"funding below exit threshold ({snapshot.annualized_rate * 100:+.4f}% "
                    f"vs {self.config.exit_apr_threshold * 100:.2f}%)",
                )
                continue
            # 4) Bookkeeping for a settled funding period.
            self._record_funding_if_settled(symbol, snapshot, position)

    def _check_tiered_margin_health(self) -> None:
        """Account-wide tiered margin circuit breakers (Phase 1 Step 1.3).

        Runs once per health tick on the whole futures account, independent of
        the per-pair safety loop in :meth:`_verify_delta_and_margin`.

        Level 1 (soft): the live available-margin ratio falls below
        ``soft_margin_ratio_warning`` (15% default) -> latch
        ``_margin_soft_pause`` so discovery skips ALL new entries until the
        buffer recovers. Never unwinds.
        Level 2 (hard): the ratio drops below ``emergency_delever_ratio`` (10%
        default) -> auto-deleverage the active pair with the largest deployed
        notional (the biggest collateral at risk) and DM an urgent card.
        """
        ratio = self._futures_margin_ratio()

        # Level 1 — soft breaker: pause admissions, keep the book open.
        if ratio < self.config.soft_margin_ratio_warning:
            self._margin_soft_pause = True
            logger.warning(
                "[MARGIN ALERT] Low margin ratio: %.2f%%. Pausing new entries.",
                ratio * 100,
            )
        else:
            self._margin_soft_pause = False

        # Level 2 — hard circuit breaker: auto-deleverage the riskiest pair.
        if ratio < self.config.emergency_delever_ratio:
            logger.critical(
                "[CIRCUIT BREAKER] Critical margin ratio %.2f%%. "
                "Initiating auto-deleveraging.",
                ratio * 100,
            )
            target = self._pick_delever_target()
            if target is None:
                return
            symbol = f"{target.base_asset}{self.config.quote_asset}"
            # Urgent account-level alert before the per-position unwind card lands.
            self._notifier.notify_circuit_breaker(
                symbol="ACCOUNT",
                reason=f"EMERGENCY_DELEVERAGING margin_ratio={ratio:.2%}",
                extra={
                    "Auto-Deleverage Target": symbol,
                    "Position ID": target.position_id,
                },
            )
            try:
                self._emergency_unwind(
                    target,
                    symbol,
                    reason=f"EMERGENCY_DELEVERAGING margin_ratio={ratio:.2%}",
                )
            except Exception as exc:
                logger.exception(
                    "[CIRCUIT BREAKER] Auto-deleveraging failed for %s: %s",
                    symbol, exc,
                )

    # ------------------------------------------------------------------ reconciliation
    # Phase 3 Step 3.2 — Automated Ledger-to-Binance Balance Reconciliation.
    #
    # Runs periodically from the health loop (``reconcile_interval_seconds``).
    # Everything here is best effort: an exchange/network failure degrades to a
    # warning and never raises through the daemon tick.
    def _reconcile_exchange_state(self) -> None:
        """Reconcile the local carry book against live Binance balances/positions.

        For every locally-managed pair:

        * Ghost position — the exchange reports a futures position size of
          exactly 0.0 for a pair our book still marks OPEN. The pair is marked
          ``DESYNC_CLOSED``, dropped from the active book, and persisted, with a
          ``CRITICAL`` log.
        * Delta divergence — the ALLOCATED spot and futures legs (current minus
          each pair's entry baseline, exactly like ``_verify_delta_and_margin``)
          have drifted apart beyond ``delta_tolerance``: a warning is logged and,
          past 5% relative drift, a circuit-breaker alert is dispatched.

        Finally, the unmanaged-futures scan (:meth:`_reconcile_unmanaged_positions`)
        warns on any open exchange position the daemon never opened.
        """
        for symbol, position in list(self._active_positions.items()):
            if position.status not in ("OPEN", "UNWINDING"):
                continue
            base = position.base_asset
            try:
                spot_total = self._raw_spot_qty(base)
                futures_total = self._raw_futures_qty(symbol)  # signed (short is negative)
            except Exception as exc:
                logger.warning("[RECONCILIATION] balance fetch failed for %s: %s", symbol, exc)
                continue

            # Ghost position: the exchange holds no futures position for this
            # pair at all, yet the local book still tracks it as OPEN. Treat the
            # ledger as authoritative and retire the stale pair.
            if abs(futures_total) == 0.0:
                position.status = "DESYNC_CLOSED"
                position.closed_at = time.time()
                self._active_positions.pop(symbol, None)
                self._persist_position(position)
                logger.critical(
                    "[RECONCILIATION] Ghost position detected for %s. Marked DESYNC_CLOSED.",
                    symbol,
                )
                continue

            # Delta divergence: measured on the deltas THIS pair manages (current
            # minus entry baseline), so untouched pre-existing inventory never
            # tripped a false alert.
            spot_allocated = spot_total - position.baseline_spot_qty
            futures_allocated = futures_total - position.baseline_futures_qty
            residual = abs(spot_allocated + futures_allocated)
            if residual > self.config.delta_tolerance:
                logger.warning(
                    "[RECONCILIATION] Delta drift on %s: Spot=%+.4f, Fut=%+.4f",
                    symbol, spot_allocated, futures_allocated,
                )
                denominator = max(abs(spot_allocated), abs(futures_allocated), 1e-9)
                if (residual / denominator) > 0.05:  # >5% relative drift circuit breaker
                    self._notifier.notify_circuit_breaker(
                        symbol=symbol,
                        reason=(
                            f"RECONCILIATION_DELTA_DRIFT spot={spot_allocated:+.4f} "
                            f"fut={futures_allocated:+.4f}"
                        ),
                        extra={"Position ID": position.position_id},
                    )
        self._reconcile_unmanaged_positions()

    def _reconcile_unmanaged_positions(self) -> None:
        """Warn on any open futures position the exchange holds that the daemon
        never opened (absent from ``_active_positions``).

        Purely diagnostic — the daemon does not trade positions it did not
        create, it just surfaces them so the operator can intervene before an
        unhedged exposure sits unmanaged.
        """
        try:
            exchange_positions = self.carry_broker.futures.connector.get_positions()
        except Exception as exc:
            logger.warning("[RECONCILIATION] full position scan failed: %s", exc)
            return
        managed = set(self._active_positions.keys())
        for pos in exchange_positions:
            symbol = pos.get("symbol")
            qty = float(pos.get("position_amt", 0.0) or 0.0)
            if symbol and abs(qty) > 0 and symbol not in managed:
                logger.warning(
                    "[RECONCILIATION ALERT] Unmanaged futures position detected for %s (qty=%+.4f).",
                    symbol, qty,
                )

    def _pick_delever_target(self) -> Optional[TwinLegPosition]:
        """The active pair to auto-deleverage first under the hard breaker.

        Selects the position with the largest deployed futures notional
        (``qty x futures_fill_price``) — the biggest collateral at risk — so a
        single unwind relieves the most margin pressure per deleverage.
        """
        candidates = [
            p for p in self._active_positions.values() if p.status == "OPEN"
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda p: float(p.quantity or 0.0) * float(p.futures_fill_price or 0.0),
        )

    def _verify_delta_and_margin(self, position: TwinLegPosition, symbol: str) -> None:
        tol = self.config.delta_tolerance
        base = position.base_asset
        # The delta-neutrality check is scoped to what THIS daemon manages:
        # allocated = current total minus the baseline that pre-existed entry.
        spot_allocated = self._raw_spot_qty(base) - position.baseline_spot_qty
        futures_allocated = self._raw_futures_qty(symbol) - position.baseline_futures_qty
        hedge_size = -futures_allocated
        spot_present = spot_allocated + tol >= position.quantity
        hedge_present = hedge_size + tol >= position.quantity
        delta_neutral = abs(spot_allocated + futures_allocated) < tol

        if not (spot_present and hedge_present and delta_neutral):
            self._emergency_unwind(
                position, symbol,
                f"delta mismatch: spot_allocated={spot_allocated:+.4f}, "
                f"futures_allocated={futures_allocated:+.4f}, expected={position.quantity:.4f}, "
                f"(baseline spot={position.baseline_spot_qty:.4f}, "
                f"baseline futures={position.baseline_futures_qty:+.4f})",
            )
            return
        account = self.carry_broker.futures.connector.get_account_info()
        margin_balance = float(account.get("totalMarginBalance", account.get("totalWalletBalance", 0.0)) or 0.0)
        maintenance = float(account.get("totalMaintMargin", 0.0) or 0.0)
        margin_ratio = (margin_balance - maintenance) / margin_balance if margin_balance > 0 else 0.0
        if margin_ratio < self.config.min_margin_ratio:
            self._emergency_unwind(
                position, symbol,
                f"margin safety buffer breached: ratio={margin_ratio:.6f}, min={self.config.min_margin_ratio:.6f}",
            )

    # ----------------------------------------------------------------- risk
    # Pre-flight risk gatekeeper (Phase 1 Step 1.1 — Risk Management &
    # Capital Guardrails). Runs immediately BEFORE any order is placed and, on
    # rejection, logs a warning and returns False WITHOUT raising — the
    # discovery loop simply skips to the next candidate pair.
    def _check_preflight_risk(
        self,
        symbol: str,
        alloc_amount: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Gatekeeper: admit a carry entry only if every guardrail is green.

        Rule A  rejects when a position on the same base asset is already
                active (or in an opening state) in the book;
        Rule B  rejects when the total deployed notional PLUS the new pair's
                ``alloc_amount`` would exceed ``max_portfolio_usd``;
        Rule C  rejects when the futures available-margin ratio
                (``available_margin / total_margin_equity``) falls below
                ``entry_margin_ratio_min`` (25% default), guarding the short
                leg against a liquidation spike;
        Rule D  rejects when the adaptive local-memory gate (:class:`LocalMemoryGate`)
                — advisory learnings from past trades (blacklist / basis-discount
                / min-APR) — blocks the candidate. ``context`` supplies the live
                metrics these thresholds compare against and is optional (a
                candidate is admitted by default when no context is supplied,
                unless a ``BLACKLIST`` rule matches).

        Returns True only when all pass; otherwise logs
        ``[RISK GATEKEEPER] Rejected {symbol} entry: {reason}``.
        """
        base = self._base_asset(symbol)

        # Rule D — adaptive local-memory gate (Phase 2 Step 2.2). Most
        # authoritative block: a historical blacklist must outrank every other
        # guardrail, so it is evaluated first.
        admitted, reason = self.memory_gate.evaluate_candidate(
            symbol, "CARRY_ARBITRAGE", context or {}
        )
        if not admitted:
            self._reject_entry(symbol, f"MEMORY RULE: {reason}")
            return False

        # Rule A — duplicate base asset already active / opening.
        for existing in self._active_positions.values():
            if existing.base_asset == base:
                self._reject_entry(
                    symbol,
                    f"duplicate asset {base}: {existing.position_id} is "
                    f"{existing.status} (qty {existing.quantity:g})",
                )
                return False

        # Rule B — max portfolio allocation (total deployed + new notional).
        current = self._deployed_capital_usd()
        cap = self._max_portfolio_usd()
        if current + alloc_amount > cap:
            self._reject_entry(
                symbol,
                f"portfolio allocation cap ${cap:,.2f} reached "
                f"(${current:,.2f} deployed + ${alloc_amount:,.2f} new)",
            )
            return False

        # Rule C — futures margin-ratio buffer (protects the short leg).
        ratio = self._futures_margin_ratio()
        if ratio < self.config.entry_margin_ratio_min:
            self._reject_entry(
                symbol,
                f"futures margin ratio {ratio:.2%} < "
                f"{self.config.entry_margin_ratio_min:.0%} minimum",
            )
            return False

        return True

    def _reject_entry(self, symbol: str, reason: str) -> None:
        """Log a clear, structured gatekeeper rejection without raising."""
        logger.warning("[RISK GATEKEEPER] Rejected %s entry: %s", symbol, reason)

    def _base_asset(self, symbol: str) -> str:
        """Base asset of a ``SYMBOLQUOTE`` (e.g. ``SOLUSDT`` -> ``SOL``)."""
        if symbol.endswith(self.config.quote_asset):
            return symbol[: -len(self.config.quote_asset)].upper()
        return symbol.upper()

    def _deployed_capital_usd(self) -> float:
        """Total short-leg notional currently deployed across the book (USDT)."""
        return sum(
            float(p.quantity or 0.0) * float(p.futures_fill_price or 0.0)
            for p in self._active_positions.values()
        )

    def _max_portfolio_usd(self) -> float:
        """The portfolio allocation ceiling (USDT) for Rule B.

        Explicit ``config.max_portfolio_usd`` wins when set; otherwise the
        declared budget is read from config/settings (``risk.starting_capital``,
        default $1,000), falling back to the daemon's own total allocation.
        """
        if self.config.max_portfolio_usd is not None and self.config.max_portfolio_usd > 0:
            return float(self.config.max_portfolio_usd)
        try:
            from ..config.settings import get_config as _get_settings
            settings_cap = float(_get_settings().risk.starting_capital)
            if settings_cap > 0:
                return settings_cap
        except Exception:
            pass  # settings unavailable -> fall back to the daemon budget
        return self.config.total_allocation_usdt

    def _futures_equity(self) -> float:
        """Live futures total margin equity (USDT), 0.0 if unavailable."""
        try:
            account = self.carry_broker.futures.connector.get_account_info()
            return float(
                account.get("totalMarginBalance", account.get("totalWalletBalance", 0.0)) or 0.0
            )
        except Exception as exc:
            logger.warning("[CARRY RISK] futures equity unavailable: %s", exc)
            return 0.0

    def _futures_margin_ratio(self) -> float:
        """Available-margin ratio ``(equity - maintenance) / equity`` for Rule C."""
        try:
            account = self.carry_broker.futures.connector.get_account_info()
            margin = float(
                account.get("totalMarginBalance", account.get("totalWalletBalance", 0.0)) or 0.0
            )
            maintenance = float(account.get("totalMaintMargin", 0.0) or 0.0)
            return (margin - maintenance) / margin if margin > 0 else 0.0
        except Exception as exc:
            logger.warning("[CARRY RISK] margin ratio unavailable: %s", exc)
            return 0.0

    # ----------------------------------------------------------------- discovery
    def _discover(self) -> None:
        """Fill spare pair capacity with the highest-ranked scanner candidates."""
        if self.scanner is None or self._shutdown_requested:
            return
        # Dashboard control gates: pause/kill prevent new carry admissions
        if self._entries_paused or self._kill_switch or self._margin_soft_pause:
            return
        if len(self._active_positions) >= self.config.max_active_pairs:
            return
        remaining = self.config.max_active_pairs - len(self._active_positions)
        try:
            opportunities = self.scanner.scan(self.scanner_provider)
        except Exception as exc:
            logger.warning("Funding scan failed: %s", exc)
            return

        for opp in opportunities:
            if remaining <= 0:
                break
            symbol, base = self._opportunity_symbol_and_base(opp)
            if symbol in self._active_positions:
                continue
            if symbol in self._last_unwind_at and (
                (time.time() - self._last_unwind_at[symbol])
                < self.config.unwind_cooldown_seconds
            ):
                logger.debug("Skipping %s (unwind cooldown active)", symbol)
                continue
            if (float(getattr(opp, "net_apr_pct", 0.0)) / 100.0) <= self.config.min_apr_threshold:
                continue
            # Pre-trade loss prevention (Safeguard): reject adversarially priced
            # or non-paying carries before any order is placed.
            if not self._passes_entry_hurdles(symbol, opp):
                continue
            try:
                ok, reasons, qty = self._size_lot(base, symbol, opp)
            except Exception as exc:
                logger.warning("Lot sizing failed for %s: %s", symbol, exc)
                continue
            if not ok:
                logger.debug(
                    "Skipping %s (capital discipline): %s", symbol, "; ".join(reasons)
                )
                continue
            # Pre-flight risk gatekeeper (Phase 1 Step 1.1): reject the entry
            # before any order is placed if it would duplicate the book, blow
            # the portfolio cap, or open into a thin futures margin buffer.
            # Safe rejection: logs and moves to the next candidate, never raises.
            spot_price = float(getattr(opp, "spot_price", 0.0) or 0.0)
            futures_price = float(getattr(opp, "futures_price", 0.0) or 0.0)
            alloc_amount = qty * (futures_price if futures_price > 0 else spot_price)
            # Candidate metrics for the local-memory gate (Rule D): basis spread
            # and net APR, both in percent (mirrors the scanner's own units).
            memory_context = {
                "basis_spread_pct": float(getattr(opp, "basis_spread_pct", 0.0) or 0.0),
                "net_apr": float(getattr(opp, "net_apr_pct", 0.0) or 0.0),
            }
            if not self._check_preflight_risk(symbol, alloc_amount, context=memory_context):
                continue
            try:
                self._open_pair(opp, base, qty, symbol)
            except TwinLegExecutionError as exc:
                logger.error("Entry failed for %s (skipping): %s", symbol, exc)
                continue
            remaining -= 1

    def _opportunity_symbol_and_base(self, opp):
        """Return ``(symbol, base)`` for an opportunity's perp ``symbol``.

        Handles both ``"SOL/USDT:USDT"`` (scanner perp form) and ``"SOLUSDT"``.
        """
        raw = str(getattr(opp, "symbol", "") or "")
        if "/" in raw:
            base = raw.split("/")[0].upper()
        else:
            upper = raw.upper()
            quote = self.config.quote_asset
            base = upper[: -len(quote)] if upper.endswith(quote) else upper
        return f"{base}{self.config.quote_asset}", base

    def _passes_entry_hurdles(self, symbol: str, opp) -> bool:
        """Pre-trade loss-prevention gates on the scanner-advertised economics.

        Rejects a carry that either (1) pays no meaningful funding
        (``predicted_funding_rate <= +0.005%`` per 8h, so negative or ~zero
        funding never gets opened) or (2) carries materially adverse basis
        (spot too expensive relative to futures, ``basis_spread_pct < -0.12%``),
        which would drag the entry PnL down before any funding is earned.
        """
        predicted_funding = float(getattr(opp, "predicted_funding_rate", 0.0) or 0.0)
        if predicted_funding <= _MIN_ENTRY_FUNDING_RATE:
            logger.debug(
                "Skipping %s (predicted funding %+.6f%%/8h <= %+.6f%% hurdle)",
                symbol, predicted_funding * 100, _MIN_ENTRY_FUNDING_RATE * 100,
            )
            return False
        basis_pct = float(getattr(opp, "basis_spread_pct", 0.0) or 0.0)
        if basis_pct < _MIN_ENTRY_BASIS_PCT:
            logger.debug(
                "Skipping %s (adverse entry basis %+.4f%% < %+.4f%%)",
                symbol, basis_pct, _MIN_ENTRY_BASIS_PCT,
            )
            return False
        return True

    def _size_lot(self, base: str, symbol: str, opp) -> tuple:
        """Return (ok, reasons, rounded_qty) for the per-pair capital budget.

        When ``scale_to_available_margin`` is set, the budget is capped by the
        account's live available free USDT margin (times
        ``available_margin_use_fraction``) so allocation can scale beyond small
        default tests; otherwise the static ``allocation_per_pair`` applies.
        """
        spot_price = float(getattr(opp, "spot_price", 0.0) or 0.0)
        if spot_price <= 0:
            return False, ["spot price unavailable"], 0.0
        budget = self.config.allocation_per_pair
        reasons: List[str] = []
        if self.config.scale_to_available_margin:
            available = self._available_usdt_margin()
            budget = min(budget, available * self.config.available_margin_use_fraction)
            if budget <= 0:
                return False, ["no available margin to scale into"], 0.0
            reasons.append(
                f"margin-scaled ${budget:,.2f} (avail ${available:,.2f}"
                f" x {self.config.available_margin_use_fraction:.0%})"
            )
        target_qty = budget / spot_price
        twin = self.carry_broker.get_twin_filters(base, self.config.quote_asset)
        ok, reasons2, qty = twin.validate_qty(target_qty, spot_price)
        if not ok:
            return False, reasons2, 0.0
        return True, reasons, qty

    def _open_pair(self, opp, base: str, qty: float, symbol: str) -> None:
        # Snapshot pre-existing inventory BEFORE the carry fills so delta
        # neutrality is measured on the deltas THIS pair manages, never against
        # total account balances that start non-zero.
        baseline_spot = self._raw_spot_qty(base)
        baseline_futures = self._raw_futures_qty(symbol)
        try:
            position = self.carry_broker.execute_twin_leg_carry(
                base,
                self.config.quote_asset,
                qty,
                max_leg_gap_ms=float(self.config.max_leg_gap_ms),
                max_entry_slippage_pct=float(self.config.max_entry_slippage_pct),
            )
        except TypeError:
            # Backward compatibility for test fakes / older broker
            # implementations that don't accept the new guard kwargs.
            position = self.carry_broker.execute_twin_leg_carry(
                base, self.config.quote_asset, qty
            )
        position.baseline_spot_qty = baseline_spot
        position.baseline_futures_qty = baseline_futures
        self._active_positions[symbol] = position
        self._persist_position(position)
        logger.info(
            "[CARRY DAEMON] OPEN %s id=%s qty=%s net_apr=%.4f%% alloc=$%.2f",
            symbol, position.position_id, position.quantity,
            float(getattr(opp, "net_apr_pct", 0.0)), self.config.allocation_per_pair,
        )
        extra = {
            "Spot Buy": f"${position.spot_fill_price:,.4f}",
            "Futures Short": f"${position.futures_fill_price:,.4f}",
            "Initial Basis Spread %": f"{position.entry_basis_spread_pct:+.4f}%",
            "Net APR %": f"{float(getattr(opp, 'net_apr_pct', 0.0)):+.4f}%",
            "Capital Invested (USDT)": (
                f"${float(position.quantity or 0.0) * float(position.futures_fill_price or 0.0):,.2f}"
            ),
        }
        extra.update(self._audit_extra())  # equity + remaining free + book PnL
        # Full trade-lifecycle DM (entry)
        try:
            if hasattr(self._notifier, "notify_trade_entry"):
                self._notifier.notify_trade_entry(
                    symbol=symbol,
                    strategy="CARRY_ARBITRAGE",
                    side="CARRY",
                    qty=position.quantity,
                    spot_fill_price=float(position.spot_fill_price or 0.0),
                    futures_fill_price=float(position.futures_fill_price or 0.0),
                    allocated_capital_usdt=float(self.config.allocation_per_pair),
                    target_apr_pct=float(getattr(opp, "net_apr_pct", 0.0) or 0.0),
                    order_id=position.position_id,
                    extra=extra,
                )
        except Exception as exc:
            logger.warning("[CARRY DM] trade_entry dispatch failed: %s", exc)


    # ------------------------------------------------------------------ raw
    def _raw_spot_qty(self, base: str) -> float:
        """Total base-asset spot balance on the account (all inventory)."""
        spot_balances = self.carry_broker.spot.connector.get_balances()
        return float(spot_balances.get(base, {}).get("total", 0.0))

    def _raw_futures_qty(self, symbol: str) -> float:
        """Total base-asset futures open position (all inventory, short is negative)."""
        futures_positions = self.carry_broker.futures.connector.get_positions(symbol)
        return sum(float(p.get("position_amt", 0.0)) for p in futures_positions)

    # ------------------------------------------------------------------ unwinds
    def _mark_unwound(self, symbol: str, position: TwinLegPosition) -> None:
        """Record the wall-clock of a successful flatten for cooldown gating."""
        self._last_unwind_at[symbol] = time.time()
        self._active_positions.pop(symbol, None)
        logger.info(
            "[CARRY DAEMON] RECONCILE CLOSED %s id=%s", symbol, position.position_id
        )

    def _graceful_unwind(self, position: TwinLegPosition, symbol: str, reason: str) -> None:
        """Close a pair cleanly (blue embed). The pair is healthy but funding collapsed."""
        logger.info("[CARRY DAEMON] GRACEFUL UNWIND %s: %s", position.position_id, reason)
        try:
            self.carry_broker.unwind_twin_leg_carry(position)
        except TwinLegExecutionError as exc:
            position.status = "UNWINDING"
            self._persist_position(position)
            self._notifier.notify_circuit_breaker(
                symbol=symbol, reason=f"UNWIND FAILED: {reason}",
                extra={"Position ID": position.position_id},
            )
            raise
        self._persist_position(position)
        if position.status == "CLOSED":
            close_extra = self._close_extra(position, reason)
            duration = _fmt_duration(position.opened_at, position.closed_at)

            # Full trade-lifecycle DM (exit)
            try:
                if hasattr(self._notifier, "notify_trade_exit"):
                    m = self._close_metrics(position)
                    capital = float(m["capital_invested_usdt"] or 0.0)
                    gross = float(m["funding_on_trade_usdt"] + m["unrealized_basis_pnl_usdt"])
                    net = float(m["net_pnl_usdt"])
                    roi = float(m["net_pnl_pct"])
                    funding = float(m["funding_on_trade_usdt"])
                    outcome_reason = (
                        "EMERGENCY_DELEVER"
                        if "emergency_deleveraging" in str(reason).lower()
                        else "ADVERSE_BASIS_STOP"
                        if "adverse_basis_stop" in str(reason).lower()
                        else "YIELD_COMPRESSION"
                        if "funding below" in str(reason).lower()
                        else "MANUAL"
                        if "shutdown" in str(reason).lower()
                        else "MANUAL"
                        if "manual" in str(reason).lower()
                        else "MANUAL"
                    )

                    self._notifier.notify_trade_exit(
                        symbol=symbol,
                        position_id=position.position_id,
                        strategy="CARRY_ARBITRAGE",
                        side="CARRY",
                        qty=position.quantity,
                        spot_fill_price=float(position.spot_fill_price or 0.0),
                        futures_fill_price=float(position.futures_fill_price or 0.0),
                        gross_pnl_usdt=gross,
                        net_pnl_usdt=net,
                        funding_harvested_usdt=funding,
                        roi_pct=roi,
                        duration=duration,
                        exit_reason=outcome_reason,
                        extra=close_extra,
                    )
            except Exception as exc:
                logger.warning("[CARRY DM] trade_exit dispatch failed: %s", exc)

            self._record_closed_trade(position, symbol, reason)
            self._mark_unwound(symbol, position)

    def _emergency_unwind(self, position: TwinLegPosition, symbol: str, reason: str) -> None:
        """Flatten a pair that breached delta/margin invariants (red circuit breaker)."""
        logger.critical("[CARRY DAEMON] EMERGENCY UNWIND %s: %s", position.position_id, reason)
        self._notifier.notify_circuit_breaker(
            symbol=symbol, reason=reason, extra={"Position ID": position.position_id},
        )
        try:
            self.carry_broker.unwind_twin_leg_carry(position)
        except TwinLegExecutionError:
            position.status = "UNWINDING"
            self._persist_position(position)
            raise
        self._persist_position(position)
        if position.status == "CLOSED":
            close_extra = self._close_extra(position, reason)
            duration = _fmt_duration(position.opened_at, position.closed_at)

            # Full trade-lifecycle DM (exit)
            try:
                if hasattr(self._notifier, "notify_trade_exit"):
                    m = self._close_metrics(position)
                    capital = float(m["capital_invested_usdt"] or 0.0)
                    gross = float(m["funding_on_trade_usdt"] + m["unrealized_basis_pnl_usdt"])
                    net = float(m["net_pnl_usdt"])
                    roi = float(m["net_pnl_pct"])
                    funding = float(m["funding_on_trade_usdt"])
                    outcome_reason = (
                        "EMERGENCY_DELEVER"
                        if "emergency_deleveraging" in str(reason).lower()
                        else "ADVERSE_BASIS_STOP"
                        if "adverse_basis_stop" in str(reason).lower()
                        else "YIELD_COMPRESSION"
                        if "funding below" in str(reason).lower()
                        else "MANUAL"
                        if "shutdown" in str(reason).lower()
                        else "MANUAL"
                        if "manual" in str(reason).lower()
                        else "MANUAL"
                    )

                    self._notifier.notify_trade_exit(
                        symbol=symbol,
                        position_id=position.position_id,
                        strategy="CARRY_ARBITRAGE",
                        side="CARRY",
                        qty=position.quantity,
                        spot_fill_price=float(position.spot_fill_price or 0.0),
                        futures_fill_price=float(position.futures_fill_price or 0.0),
                        gross_pnl_usdt=gross,
                        net_pnl_usdt=net,
                        funding_harvested_usdt=funding,
                        roi_pct=roi,
                        duration=duration,
                        exit_reason=outcome_reason,
                        extra=close_extra,
                    )
            except Exception as exc:
                logger.warning("[CARRY DM] trade_exit dispatch failed: %s", exc)

            self._record_closed_trade(position, symbol, reason)
            self._mark_unwound(symbol, position)

    # ------------------------------------------------------------------ funding
    def _record_funding_if_settled(
        self, symbol: str, snapshot: FundingSnapshot, position: TwinLegPosition
    ) -> None:
        if snapshot.next_funding_time_ms is None:
            return
        previous_next = self._last_seen_next_funding_ms.get(symbol)
        if previous_next is None:
            # First observation of premiumIndex for this symbol: baseline only.
            self._last_seen_next_funding_ms[symbol] = snapshot.next_funding_time_ms
            return
        # premiumIndex rolls nextFundingTime forward only after settlement; a
        # forward advance means the period that ended at ``previous_next`` settled.
        if snapshot.next_funding_time_ms <= previous_next:
            return
        settlement_ms = previous_next
        if self._last_funding_settlement_ms.get(symbol) == settlement_ms:
            # Already recorded (or deliberately given up on); advance the marker
            # so this stale rollover is not re-detected every tick.
            self._last_seen_next_funding_ms[symbol] = snapshot.next_funding_time_ms
            return
        payment = self._confirmed_funding_payment(symbol, settlement_ms)
        if payment is None:
            # No confirmed income event yet: Binance batch-clearing posts to
            # /fapi/v1/income up to ~1-5 minutes AFTER nextFundingTime advances.
            # Do NOT advance _last_seen_next_funding_ms on an empty poll — keep
            # re-entering this branch on subsequent health ticks and re-querying
            # income until the 15-minute retry window expires.
            if self._retry_window_expired(symbol):
                # Give up on auto-capture: stop re-detecting, and tell the
                # operator the reconciliation command will backfill it.
                self._last_funding_settlement_ms[symbol] = settlement_ms
                self._last_seen_next_funding_ms[symbol] = snapshot.next_funding_time_ms
                logger.warning(
                    "Funding settlement rolled for %s (settlement_ms=%s) but no "
                    "confirmed income event landed within the %ds retry window; "
                    "run `carry reconcile-funding` to backfill the ledger.",
                    symbol, settlement_ms, _FUNDING_RETRY_WINDOW_SECONDS,
                )
            return
        # Confirmed — persist with the exact schema, record the settlement,
        # advance the marker and dispatch the green payout embed.
        self._persist_funding(position.position_id, snapshot, payment, settlement_ms)
        self._last_funding_settlement_ms[symbol] = settlement_ms
        self._last_seen_next_funding_ms[symbol] = snapshot.next_funding_time_ms
        self._pending_funding_retry_until.pop(symbol, None)
        self._notifier.notify_funding_settlement(
            symbol=symbol,
            payment_usdt=payment,
            funding_rate_pct=snapshot.funding_rate * 100,
            extra={"Position ID": position.position_id},
        )

    def _retry_window_expired(self, symbol: str) -> bool:
        """True once the post-rollover income retry window elapses.

        Starts a ``_FUNDING_RETRY_WINDOW_SECONDS`` timer on first empty poll and
        returns False (keep retrying) until the window passes, then True (give
        up for auto-capture). Idempotent per settlement.
        """
        now = time.time()
        deadline = self._pending_funding_retry_until.get(symbol)
        if deadline is None:
            self._pending_funding_retry_until[symbol] = now + _FUNDING_RETRY_WINDOW_SECONDS
            return False
        if now >= deadline:
            self._pending_funding_retry_until.pop(symbol, None)
            return True
        return False

    def _confirmed_funding_payment(self, symbol: str, settlement_ms: int) -> Optional[float]:
        """Get the account-confirmed funding credit/debit nearest settlement."""
        connector = self.carry_broker.futures.connector
        start = settlement_ms - 5 * 60 * 1000
        events = connector.get_funding_income_history(symbol, start_time=start)
        candidates = [
            event for event in events
            if abs(int(event.get("time", 0)) - settlement_ms) <= 10 * 60 * 1000
        ]
        if not candidates:
            return None
        return sum(float(event.get("income", 0.0) or 0.0) for event in candidates)

    # ------------------------------------------------------------------ audit
    def _audit_metrics(self) -> Dict[str, float]:
        """Return the current portfolio financial-audit numerics.

        Defensive: any exchange/DB failure degrades a single metric to zero
        rather than blocking a lifecycle event. Gives the operator cash vs.
        capital, realized (funding) vs. unrealized (basis) PnL, and the health
        buffer — on entry, on exit, and on the hourly heartbeat.
        """
        metrics: Dict[str, float] = {
            "available_free_margin_usdt": 0.0,
            "allocated_capital_usdt": 0.0,
            "net_unrealized_pnl_usdt": 0.0,
            "unrealized_basis_pct": 0.0,
            "equity_usdt": 0.0,
            "margin_safety_buffer_pct": 0.0,
            "harvested_funding_usdt": 0.0,
            "estimated_fees_usdt": 0.0,
            "net_pnl_usdt": 0.0,
        }
        try:
            account = self.carry_broker.futures.connector.get_account_info()
            margin_balance = float(
                account.get("totalMarginBalance", account.get("totalWalletBalance", 0.0)) or 0.0
            )
            maintenance = float(account.get("totalMaintMargin", 0.0) or 0.0)
            available_raw = account.get("availableBalance")
            metrics["available_free_margin_usdt"] = (
                float(available_raw) if available_raw is not None else max(0.0, margin_balance - maintenance)
            )
            metrics["equity_usdt"] = margin_balance
            if margin_balance > 0:
                metrics["margin_safety_buffer_pct"] = (
                    (margin_balance - maintenance) / margin_balance * 100.0
                )
        except Exception as exc:
            logger.warning("[CARRY AUDIT] available margin fetch failed: %s", exc)
        # Capital invested = per-pair carry notional (qty x futures fill).
        for position in self._active_positions.values():
            metrics["allocated_capital_usdt"] += (
                float(position.quantity or 0.0) * float(position.futures_fill_price or 0.0)
            )
        # Current net unrealized PnL across the delta-neutral book (marks both legs).
        try:
            for symbol, position in self._active_positions.items():
                mark = self._live_mark(symbol)
                spot = self._live_spot(symbol)
                qty = float(position.quantity or 0.0)
                if mark is None or spot is None:
                    continue
                metrics["net_unrealized_pnl_usdt"] += (
                    (spot - float(position.spot_fill_price)) * qty
                    + (float(position.futures_fill_price) - mark) * qty
                )
        except Exception as exc:
            logger.warning("[CARRY AUDIT] unrealized PnL fetch failed: %s", exc)
        # Cumulative harvested funding from the durable ledger.
        if self.db is not None:
            try:
                session = self.db.get_session()
                try:
                    harvested = session.query(
                        func.coalesce(func.sum(CarryFundingPaymentRecord.funding_payment_usdt), 0.0)
                    ).scalar() or 0.0
                    metrics["harvested_funding_usdt"] = float(harvested)
                finally:
                    session.close()
            except Exception as exc:
                logger.warning("[CARRY AUDIT] harvested funding fetch failed: %s", exc)
        allocated = metrics["allocated_capital_usdt"]
        if allocated > 0:
            metrics["unrealized_basis_pct"] = metrics["net_unrealized_pnl_usdt"] / allocated * 100.0
            # Estimated 2-leg round-trip taker fees on the deployed notional.
            metrics["estimated_fees_usdt"] = 2.0 * 0.0004 * allocated
        metrics["net_pnl_usdt"] = (
            metrics["harvested_funding_usdt"]
            + metrics["net_unrealized_pnl_usdt"]
            - metrics["estimated_fees_usdt"]
        )
        return metrics

    def _available_usdt_margin(self) -> float:
        """Live available free USDT margin on the Futures account (for sizing)."""
        try:
            account = self.carry_broker.futures.connector.get_account_info()
            raw = account.get("availableBalance")
            if raw is not None:
                return float(raw)
            margin_balance = float(account.get("totalMarginBalance", 0.0) or 0.0)
            maintenance = float(account.get("totalMaintMargin", 0.0) or 0.0)
            return max(0.0, margin_balance - maintenance)
        except Exception as exc:
            logger.warning("[CARRY SIZING] available margin unavailable: %s", exc)
            return 0.0

    def _live_mark(self, symbol: str) -> Optional[float]:
        """Live futures mark price, or None if unavailable."""
        try:
            return float(self.carry_broker.futures.connector.get_mark_price(symbol))
        except Exception:
            return None

    def _live_spot(self, symbol: str) -> Optional[float]:
        """Live spot ticker price, or None if unavailable."""
        try:
            return float(self.carry_broker.spot.connector.get_ticker_price(symbol))
        except Exception:
            return None

    def _active_pairs_summary(self) -> List[str]:
        """Compact ``SYMBOL (duration)`` lines for the live audit heartbeat."""
        lines: List[str] = []
        for symbol, position in self._active_positions.items():
            duration = _fmt_duration(position.opened_at, time.time())
            lines.append(f"{symbol} {position.quantity:g} ({duration})")
        return lines

    def _audit_extra(self) -> Dict[str, str]:
        """Formatted audit fields appended to entry/exit embeds (shared labels)."""
        m = self._audit_metrics()
        return {
            "Equity (USDT)": f"${m['equity_usdt']:,.2f}",
            "Available Free Margin (USDT)": f"${m['available_free_margin_usdt']:,.2f}",
            "Capital Allocated (USDT)": f"${m['allocated_capital_usdt']:,.2f}",
            "Net Unrealized PnL (USDT)": f"{m['net_unrealized_pnl_usdt']:+,.2f}",
            "Unrealized Basis %": f"{m['unrealized_basis_pct']:+,.3f}%",
            "Harvested Funding (USDT)": f"{m['harvested_funding_usdt']:+,.4f}",
            "Estimated Fees (USDT)": f"${m['estimated_fees_usdt']:,.4f}",
            "Net PnL (USDT)": f"{m['net_pnl_usdt']:+,.4f}",
        }

    def _basis_divergence_pct_for_close(self, position: TwinLegPosition) -> float:
        """Compute live adverse basis divergence % against this position's entry."""
        try:
            symbol = f"{position.base_asset}{self.config.quote_asset}"
            entry_basis = float(position.entry_basis_spread_pct or 0.0) / 100.0
            spot_px = self._live_spot(symbol)
            fut_px = self._live_mark(symbol)
            if spot_px is None or fut_px is None or spot_px <= 0:
                return 0.0
            live_basis = (fut_px - spot_px) / spot_px
            divergence = live_basis - entry_basis
            return divergence * 100.0
        except Exception:
            return 0.0

    def _close_metrics(self, position: TwinLegPosition) -> Dict[str, float]:
        """Per-position exit PnL breakdown (realized funding + basis - fees)."""
        qty = float(position.quantity or 0.0)
        symbol = f"{position.base_asset}{self.config.quote_asset}"
        capital_invested = qty * float(position.futures_fill_price or 0.0)
        funding_on_trade = 0.0
        if self.db is not None:
            try:
                session = self.db.get_session()
                try:
                    funding_on_trade = float(session.query(
                        func.coalesce(func.sum(CarryFundingPaymentRecord.funding_payment_usdt), 0.0)
                    ).filter(CarryFundingPaymentRecord.position_id == position.position_id).scalar() or 0.0)
                finally:
                    session.close()
            except Exception as exc:
                logger.warning("[CARRY CLOSE] funding attribution failed: %s", exc)
        mark = self._live_mark(symbol)
        spot = self._live_spot(symbol)
        unrealized_basis = 0.0
        if mark is not None and spot is not None:
            unrealized_basis = (
                (spot - float(position.spot_fill_price)) * qty
                + (float(position.futures_fill_price) - mark) * qty
            )
        estimated_fees = 2.0 * 0.0004 * capital_invested
        net_pnl = funding_on_trade + unrealized_basis - estimated_fees
        return {
            "capital_invested_usdt": capital_invested,
            "funding_on_trade_usdt": funding_on_trade,
            "unrealized_basis_pnl_usdt": unrealized_basis,
            "unrealized_basis_pct": (unrealized_basis / capital_invested * 100.0) if capital_invested else 0.0,
            "estimated_fees_usdt": estimated_fees,
            "net_pnl_usdt": net_pnl,
            "net_pnl_pct": (net_pnl / capital_invested * 100.0) if capital_invested else 0.0,
        }

    def _ledger_close_payload(
        self, position: TwinLegPosition, symbol: str, reason: str
    ) -> Dict[str, Any]:
        """Forensic recap of a completed unwind for the local trade ledger.

        Carries the required ledger fields (``trade_id``/``symbol``/``strategy``/
        ``exit_reason``/``net_pnl_usd``) plus the close-metric decomposition so
        the reflection layer can attribute PnL without re-deriving it.
        """
        m = self._close_metrics(position)
        return {
            "trade_id": position.position_id,
            "symbol": symbol,
            "strategy": "CARRY_ARBITRAGE",
            "exit_reason": reason,
            "net_pnl_usd": m["net_pnl_usdt"],
            "status": position.status,
            "opened_at": position.opened_at,
            "closed_at": position.closed_at,
            "quantity": position.quantity,
            "spot_fill_price": position.spot_fill_price,
            "futures_fill_price": position.futures_fill_price,
            "entry_basis_spread_pct": position.entry_basis_spread_pct,
            "capital_invested_usdt": m["capital_invested_usdt"],
            "funding_on_trade_usdt": m["funding_on_trade_usdt"],
            "unrealized_basis_pnl_usdt": m["unrealized_basis_pnl_usdt"],
            "estimated_fees_usdt": m["estimated_fees_usdt"],
            "net_pnl_pct": m["net_pnl_pct"],
        }

    def _record_closed_trade(self, position: TwinLegPosition, symbol: str, reason: str) -> None:
        """Append a closed trade to the local ledger, best-effort (never blocks
        or crashes the unwind: a disk hiccup must not abort a completed close)."""
        try:
            self.trade_ledger.record_closed_trade(
                self._ledger_close_payload(position, symbol, reason)
            )
        except Exception as exc:
            logger.warning("[CARRY LEDGER] record_closed_trade failed: %s", exc)

    def _close_extra(self, position: TwinLegPosition, reason: str) -> Dict[str, str]:
        """Formatted exit extra: capital returned, funding, basis, net PnL + updated audit."""
        m = self._close_metrics(position)
        extra = {
            "Status": "CLOSED",
            "Reason": reason,
            # If this unwind was triggered by adverse basis stop, include the
            # divergence percentage in the operator-visible audit payload.
            "Adverse Basis Divergence %": self._basis_divergence_pct_for_close(position),
            "Capital Returned (USDT)": f"${m['capital_invested_usdt']:,.2f}",
            "Funding Harvested (USDT)": f"{m['funding_on_trade_usdt']:+,.4f}",
            "Realized Basis PnL (USDT)": f"{m['unrealized_basis_pnl_usdt']:+,.4f}",
            "Basis PnL %": f"{m['unrealized_basis_pct']:+,.3f}%",
            "Net PnL (USDT)": f"{m['net_pnl_usdt']:+,.4f}",
            "Net PnL %": f"{m['net_pnl_pct']:+,.2f}%",
        }
        # Updated free balance & equity for the operator's post-trade view.
        extra.update(self._audit_extra())
        return extra

    def _dispatch_financial_audit(self) -> None:
        """Send the hourly portfolio financial-audit heartbeat card (non-blocking)."""
        try:
            m = self._audit_metrics()
            self._notifier.notify_financial_audit(
                available_free_margin_usdt=m["available_free_margin_usdt"],
                allocated_capital_usdt=m["allocated_capital_usdt"],
                net_unrealized_pnl_usdt=m["net_unrealized_pnl_usdt"],
                harvested_funding_usdt=m["harvested_funding_usdt"],
                open_positions=len(self._active_positions),
                margin_safety_buffer_pct=m["margin_safety_buffer_pct"],
                extra={
                    "Active Pairs": "\n".join(self._active_pairs_summary()) or "—",
                    "Account Equity (USDT)": f"${m['equity_usdt']:,.2f}",
                },
            )
        except Exception as exc:
            logger.warning("[CARRY DAEMON] audit embed dispatch failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------ persist
    def _persist_position(self, position: Optional[TwinLegPosition]) -> None:
        if self.db is None or position is None or not position.position_id:
            return
        symbol = f"{position.base_asset}{self.config.quote_asset}"
        session = self.db.get_session()
        try:
            row = session.get(CarryPositionRecord, position.position_id)
            if row is None:
                row = CarryPositionRecord(position_id=position.position_id)
                session.add(row)
            row.symbol = symbol
            row.quantity = position.quantity
            row.spot_fill_price = position.spot_fill_price
            row.futures_fill_price = position.futures_fill_price
            row.entry_basis_spread_pct = position.entry_basis_spread_pct
            row.leg_gap_ms = position.execution_gap_ms
            row.baseline_spot_qty = position.baseline_spot_qty
            row.baseline_futures_qty = position.baseline_futures_qty
            row.status = position.status
            row.opened_at = datetime.fromtimestamp(position.opened_at, tz=timezone.utc)
            row.closed_at = (
                datetime.fromtimestamp(position.closed_at, tz=timezone.utc)
                if position.closed_at is not None else None
            )
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Failed persisting carry position %s", position.position_id)
            raise
        finally:
            session.close()

    def _persist_funding(
        self, position_id: str, snapshot: FundingSnapshot, payment: float, settlement_ms: int
    ) -> None:
        if self.db is None:
            return
        session = self.db.get_session()
        try:
            timestamp = datetime.fromtimestamp(settlement_ms / 1000, tz=timezone.utc)
            exists = session.query(CarryFundingPaymentRecord).filter_by(
                position_id=position_id, timestamp=timestamp
            ).first()
            if exists is None:
                session.add(CarryFundingPaymentRecord(
                    position_id=position_id,
                    symbol=snapshot.symbol,
                    funding_rate=snapshot.funding_rate,
                    funding_payment_usdt=payment,
                    mark_price=snapshot.mark_price,
                    timestamp=timestamp,
                ))
                session.commit()
        except Exception:
            session.rollback()
            logger.exception("Failed persisting funding for %s", position_id)
            raise
        finally:
            session.close()

    def _restore_active_positions(self) -> None:
        """Rehydrate every persisted OPEN position back into the active book."""
        assert self.db is not None
        session = self.db.get_session()
        try:
            rows = session.query(CarryPositionRecord).filter(
                CarryPositionRecord.status == "OPEN",
            ).order_by(CarryPositionRecord.opened_at.desc()).all()
            for row in rows:
                symbol = row.symbol
                base = (symbol[:-len(self.config.quote_asset)] if symbol.endswith(self.config.quote_asset) else symbol)
                base = base.upper()
                if not base:
                    continue
                self._active_positions[symbol] = TwinLegPosition(
                    position_id=row.position_id,
                    base_asset=base,
                    quote_asset=self.config.quote_asset,
                    quantity=float(row.quantity),
                    spot_order_id="",
                    spot_fill_price=float(row.spot_fill_price),
                    futures_order_id="",
                    futures_fill_price=float(row.futures_fill_price),
                    entry_basis_spread_pct=float(row.entry_basis_spread_pct),
                    execution_gap_ms=float(row.leg_gap_ms or 0.0),
                    status=row.status,
                    opened_at=row.opened_at.replace(tzinfo=timezone.utc).timestamp(),
                    closed_at=row.closed_at.replace(tzinfo=timezone.utc).timestamp() if row.closed_at else None,
                    baseline_spot_qty=float(row.baseline_spot_qty or 0.0),
                    baseline_futures_qty=float(row.baseline_futures_qty or 0.0),
                )
                self._last_seen_next_funding_ms[symbol] = None
        finally:
            session.close()


def _fmt_duration(opened_at: float, closed_at: Optional[float]) -> str:
    """Human-readable duration between two UNIX timestamps."""
    if opened_at is None or closed_at is None:
        return "N/A"
    hours = max(0.0, (closed_at - opened_at) / 3600.0)
    if hours < 1.0:
        return f"{hours * 60:.0f}m"
    if hours < 24.0:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous multi-pair cash-and-carry allocator")
    parser.add_argument("--max-pairs", type=int, default=3)
    parser.add_argument("--total-usdt", type=float, default=1000.0)
    parser.add_argument("--per-pair-usdt", type=float, default=None)
    parser.add_argument("--scan-interval", type=float, default=300.0)
    parser.add_argument("--min-apr", type=float, default=0.08)
    parser.add_argument("--exit-apr", type=float, default=0.02)
    parser.add_argument("--poll", type=float, default=30.0)
    parser.add_argument("--dry", action="store_true", help="Simulate execution; no exchange orders")
    parser.add_argument("--once", action="store_true", help="Run exactly one monitoring tick")
    parser.add_argument("--sample", action="store_true", help="Use offline sample data (no network)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from ..exchange.binance_live import BinanceLiveConnector
    from ..research.funding_scanner import FundingScanner, sample_provider
    from .live_broker import BinanceLiveBroker

    load_dotenv(override=True)
    spot = BinanceLiveConnector(os.getenv("BINANCE_SPOT_TESTNET_KEY", ""), os.getenv("BINANCE_SPOT_TESTNET_SECRET", ""), "spot", True, recv_window=60000)
    futures = BinanceLiveConnector(os.getenv("BINANCE_FUTURES_TESTNET_KEY", ""), os.getenv("BINANCE_FUTURES_TESTNET_SECRET", ""), "futures", True, recv_window=60000)
    carry = TwinLegCarryBroker(
        BinanceLiveBroker(spot, dry_run=args.dry),
        BinanceLiveBroker(futures, dry_run=args.dry),
    )
    config = CarryDaemonConfig(
        max_active_pairs=args.max_pairs,
        total_allocation_usdt=args.total_usdt,
        allocation_per_pair_usdt=args.per_pair_usdt,
        scan_interval_seconds=args.scan_interval,
        min_apr_threshold=args.min_apr,
        exit_apr_threshold=args.exit_apr,
        poll_interval_seconds=args.poll,
        dry_run=args.dry,
    )
    daemon = CarryHarvesterDaemon(
        config,
        carry,
        get_db_manager(),
        scanner=FundingScanner(),
        scanner_provider=sample_provider() if args.sample else None,
    )
    if args.once:
        daemon.tick()
        daemon.shutdown()
    else:
        daemon.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())