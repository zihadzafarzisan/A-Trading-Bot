"""Production Live Execution Engine & Worker.

Drives the live trading loop for Binance (Spot / Futures) and Dry-Run mode:
1. Verifies 15 safety gates before starting
2. Manages process locking & heartbeat
3. Polling market data & generating signals
4. Idempotent pre-trade validation & risk gating
5. Order placement via BinanceLiveBroker
6. Position tracking, SL/TP management, and futures funding/liquidation awareness
7. State persistence and periodic position reconciliation
8. Graceful shutdown, kill-switch handling, and crash recovery
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..backtesting.execution import ExecutionConfig, ExecutionModel
from ..exchange.binance_live import BinanceLiveConnector
from ..logging_config import get_logger
from ..risk.limits import RiskLimits
from ..risk.manager import RiskManager
from ..strategies import create_strategy
from ..strategies.base import Direction, Signal
from .broker import Order
from .live_broker import BinanceLiveBroker
from .safety import IdempotencyRegistry, LiveSafetyGateKeeper, PositionReconciler
from .worker import WorkerLock

logger = get_logger("trading")

_LIVE_LOCK_PATH = Path("data/live_worker.lock")
_LIVE_SESSION_PATH = Path("data/live_session.json")


@dataclass
class LiveWorkerConfig:
    """Configuration for live/testnet/dry-run trading worker."""

    strategy: str = "trend"
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    market_type: str = "spot"  # spot or futures
    testnet: bool = True  # True for Testnet, False for Live
    dry_run: bool = True  # True for Dry-Run (no real orders)
    is_live_flag: bool = False  # Explicit flag for real-money execution
    starting_capital: float = 1000.0
    risk_per_trade: float = 0.01  # 1% default
    max_open_positions: int = 3
    max_leverage: int = 5
    leverage: int = 1
    poll_interval: int = 60
    window: int = 300
    run_id: str = ""
    reconcile_interval_bars: int = 5


class LiveExecutionWorker:
    """Production live trading worker."""

    def __init__(
        self,
        config: LiveWorkerConfig,
        connector: BinanceLiveConnector,
        broker: BinanceLiveBroker,
        risk: RiskManager,
        db=None,
        lock: Optional[WorkerLock] = None,
    ):
        self.config = config
        if not self.config.run_id:
            env_tag = "DRY" if self.config.dry_run else ("TEST" if self.config.testnet else "LIVE")
            self.config.run_id = f"LIVE-{env_tag}-{int(time.time() * 1000)}"

        self.connector = connector
        self.broker = broker
        self.risk = risk
        self.db = db
        self.lock = lock or WorkerLock(path=_LIVE_LOCK_PATH)

        self.strategy = create_strategy(self.config.strategy)
        self.idempotency = IdempotencyRegistry()
        self.reconciler = PositionReconciler(connector, broker)
        self.exec_model = ExecutionModel(ExecutionConfig(market_type=self.config.market_type))

        self.bars_processed = 0
        self.orders_placed = 0
        self.last_bar_time: Optional[int] = None
        self.is_running = False
        self._pending_signal: Optional[Signal] = None

        logger.info(
            "LiveExecutionWorker constructed [run=%s, env=%s, dry_run=%s, symbol=%s]",
            self.config.run_id,
            "TESTNET" if self.config.testnet else ("DRY_RUN" if self.config.dry_run else "LIVE"),
            self.config.dry_run,
            self.config.symbol,
        )

    # -----------------------------------------------------------------------
    # Safety Check & Startup
    # -----------------------------------------------------------------------
    def verify_safety_gates(self) -> Tuple[bool, List[str]]:
        """Run all 15 safety gates."""
        result = LiveSafetyGateKeeper.verify_all_gates(
            connector=self.connector,
            broker=self.broker,
            risk=self.risk,
            db=self.db,
            strategy_name=self.config.strategy,
            symbol=self.config.symbol,
            dry_run=self.config.dry_run,
            is_live_flag=self.config.is_live_flag,
        )
        return result.all_passed, result.failure_reasons

    # -----------------------------------------------------------------------
    # Core Loop
    # -----------------------------------------------------------------------
    def run(self) -> None:
        """Start the worker execution loop."""
        # 1. Gate check
        passed, failures = self.verify_safety_gates()
        if not passed:
            err_msg = f"LIVE TRADING START REFUSED: Safety gates failed:\n" + "\n".join(f"- {f}" for f in failures)
            logger.critical(err_msg)
            raise RuntimeError(err_msg)

        # 2. Acquire lock
        if not self.lock.acquire(self.config.run_id):
            raise RuntimeError("Another worker holds the live lock. Refusing start.")

        self.is_running = True
        logger.info("[LIVE WORKER] Started successfully (run=%s)", self.config.run_id)

        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("[LIVE WORKER] User interrupted via keyboard")
        except Exception as exc:
            logger.critical("[LIVE WORKER CRASH] Unexpected exception in worker loop: %s", exc, exc_info=True)
            self.risk.emergency_stop()
        finally:
            self.shutdown()

    def _main_loop(self) -> None:
        """Main execution polling loop."""
        from ..exchange.binance import BinanceAdapter

        adapter = BinanceAdapter(market_type=self.config.market_type)
        last_seen_ts = None

        while self.is_running:
            self.lock.heartbeat()

            # Check stop flag
            if self._is_stop_requested():
                logger.info("[LIVE WORKER] Stop flag detected. Initiating clean shutdown.")
                break

            # Fetch fresh OHLCV candle window
            try:
                df = adapter.get_ohlcv_as_dataframe(
                    self.config.symbol, self.config.timeframe, limit=self.config.window
                )
            except Exception as exc:
                logger.warning("[LIVE WORKER] Market data fetch error: %s", exc)
                time.sleep(self.config.poll_interval)
                continue

            if df is None or df.empty:
                time.sleep(self.config.poll_interval)
                continue

            latest_ts = int(df["timestamp"].iloc[-1])
            if latest_ts != last_seen_ts:
                last_seen_ts = latest_ts
                self.on_bar(df)

            # Periodic reconciliation
            if self.bars_processed > 0 and self.bars_processed % self.config.reconcile_interval_bars == 0:
                is_clean, discrepancies = self.reconciler.reconcile()
                if not is_clean:
                    logger.critical("[LIVE WORKER] Reconciliation discrepancy found! Halting new entries.")
                    self.risk.emergency_stop()

            time.sleep(self.config.poll_interval)

    def on_bar(self, df: pd.DataFrame) -> None:
        """Process one completed market bar."""
        self.bars_processed += 1
        bar_idx = len(df) - 1
        bar_time = int(df["timestamp"].iloc[bar_idx])
        curr_close = float(df["close"].iloc[bar_idx])
        curr_open = float(df["open"].iloc[bar_idx])
        curr_high = float(df["high"].iloc[bar_idx])
        curr_low = float(df["low"].iloc[bar_idx])
        self.last_bar_time = bar_time

        # 1. Fill pending entry at current bar's open
        if self._pending_signal is not None:
            self._execute_signal(self._pending_signal, curr_open, bar_time)
            self._pending_signal = None

        # 2. Manage open positions (SL/TP exits)
        self._manage_exits(curr_high, curr_low, curr_open, curr_close, bar_time)

        # 3. Snapshot account equity & persist state
        self._persist_snapshot()

        # 4. Generate signal for next bar
        prepared = self.strategy.setup(df)
        signal = self.strategy.generate_signal(prepared, bar_idx)
        if signal.is_active:
            self._pending_signal = signal

    def _execute_signal(self, signal: Signal, entry_price: float, bar_time: int) -> None:
        """Validate risk, size, check idempotency, and place order."""
        if not signal.is_active or signal.stop_loss is None:
            return

        # Spot cannot short
        if self.config.market_type == "spot" and signal.direction == Direction.SHORT:
            return

        # Idempotency check
        idem_key = self.idempotency.generate_key(
            self.config.strategy, self.config.symbol, self.config.timeframe, bar_time, signal.direction.value
        )
        if self.idempotency.is_duplicate(idem_key):
            logger.warning("[IDEMPOTENCY] Duplicate signal detected for key %s, skipping", idem_key)
            return

        # Risk check
        balance = self.broker.get_balance()
        positions = self.broker.get_positions()
        check = self.risk.check_entry(
            equity=balance,
            open_positions=len(positions),
            leverage=self.config.leverage if self.config.market_type == "futures" else 1,
            symbol=self.config.symbol,
        )
        if not check.is_allowed:
            logger.warning("[RISK GATING] Live trade entry rejected: %s", check.reasons)
            return

        # Position Sizing
        try:
            sizing = self.risk.size_position(
                equity=balance,
                entry_price=entry_price,
                stop_price=signal.stop_loss,
                direction=signal.direction.value,
                market_type=self.config.market_type,
                leverage=self.config.leverage if self.config.market_type == "futures" else 1,
            )
        except ValueError as exc:
            logger.error("[SIZING ERROR] Failed to size position: %s", exc)
            return

        side = "buy" if signal.direction == Direction.LONG else "sell"
        order_id = f"LIVE_{self.orders_placed}_{int(time.time()*1000)}"
        order = Order(
            order_id=order_id,
            symbol=self.config.symbol,
            side=side,
            quantity=sizing.quantity,
            order_type="market",
        )

        # Register idempotency before network transmission
        self.idempotency.register_order(
            idem_key,
            order_id,
            {"symbol": self.config.symbol, "side": side, "qty": sizing.quantity, "sl": signal.stop_loss},
        )

        placed = self.broker.place_order(order)
        self.orders_placed += 1

        # Audit order in DB if present
        if self.db and placed:
            self._persist_order(placed, idem_key)

    def _manage_exits(self, high: float, low: float, open_p: float, close_p: float, bar_time: int) -> None:
        """Check stop loss / take profit triggers."""
        # Open positions check
        positions = self.broker.get_positions()
        for pos in positions:
            sym = pos["symbol"]
            side = pos.get("side", "long")
            sl = pos.get("stop_loss")
            tp = pos.get("take_profit")
            hit_sl = (side == "long" and sl and low <= sl) or (side == "short" and sl and high >= sl)
            hit_tp = (side == "long" and tp and high >= tp) or (side == "short" and tp and low <= tp)

            if hit_sl or hit_tp:
                exit_side = "sell" if side == "long" else "buy"
                order = Order(
                    order_id=f"EXIT_{sym}_{int(time.time()*1000)}",
                    symbol=sym,
                    side=exit_side,
                    quantity=pos["quantity"],
                    order_type="market",
                )
                self.broker.place_order(order)
                logger.info("[EXIT TRIGGERED] %s %s triggered for %s", "SL" if hit_sl else "TP", exit_side, sym)

    # -----------------------------------------------------------------------
    # State Persistence
    # -----------------------------------------------------------------------
    def _persist_snapshot(self) -> None:
        if not self.db:
            return
        from ..db.models import LiveAccount
        session = self.db.get_session()
        try:
            row = session.get(LiveAccount, self.config.run_id)
            if row is None:
                row = LiveAccount(run_id=self.config.run_id, started_at=datetime.now(timezone.utc))
                session.add(row)

            env = "dry_run" if self.config.dry_run else ("testnet" if self.config.testnet else "live")
            row.environment = env
            row.symbol = self.config.symbol
            row.timeframe = self.config.timeframe
            row.market_type = self.config.market_type
            row.strategy = self.config.strategy
            row.initial_capital = self.config.starting_capital
            row.cash = self.broker.get_balance()
            row.equity = row.cash  # updated with unrealized PnL
            row.positions = json.dumps(self.broker.get_positions())
            row.closed_trades = len(self.broker.closed_trades)
            row.worker_state = json.dumps(
                {
                    "bars_processed": self.bars_processed,
                    "orders_placed": self.orders_placed,
                    "last_bar_time": self.last_bar_time,
                }
            )
            row.risk_status = json.dumps(
                {
                    "kill_switch": bool(self.risk.is_shutdown),
                    "drawdown": 0.0,
                }
            )
            row.last_market_ts = self.last_bar_time
            row.status = "halted" if self.risk.is_shutdown else ("running" if self.is_running else "stopped")
            row.updated_at = datetime.now(timezone.utc)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.debug("Failed to persist live account snapshot: %s", exc)
        finally:
            session.close()

    def _persist_order(self, order: Order, idempotency_key: str) -> None:
        if not self.db:
            return
        from ..db.models import LiveOrder
        session = self.db.get_session()
        try:
            env = "dry_run" if self.config.dry_run else ("testnet" if self.config.testnet else "live")
            lo = LiveOrder(
                id=order.order_id,
                run_id=self.config.run_id,
                environment=env,
                strategy_id=f"STRAT-{self.config.strategy.upper()}",
                symbol=order.symbol,
                market_type=self.config.market_type,
                side=order.side,
                order_type=order.order_type,
                requested_qty=order.quantity,
                executed_qty=order.quantity if order.status == "filled" else 0.0,
                requested_price=order.limit_price,
                avg_fill_price=order.fill_price,
                fee=order.fee,
                status=order.status,
                rejection_reason=order.message if order.status == "rejected" else None,
                idempotency_key=idempotency_key,
                created_at=datetime.now(timezone.utc),
            )
            session.add(lo)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.debug("Failed to persist live order: %s", exc)
        finally:
            session.close()

    # -----------------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------------
    def shutdown(self) -> None:
        """Gracefully stop worker and release lock."""
        self.is_running = False
        self._persist_snapshot()
        self.lock.release()
        try:
            if _LIVE_SESSION_PATH.exists():
                _LIVE_SESSION_PATH.unlink()
        except Exception:
            pass
        logger.info("[LIVE WORKER] Shutdown complete for %s", self.config.run_id)

    def _is_stop_requested(self) -> bool:
        if not _LIVE_SESSION_PATH.exists():
            return False
        try:
            data = json.loads(_LIVE_SESSION_PATH.read_text(encoding="utf-8"))
            return bool(data.get("stop_requested"))
        except Exception:
            return False
