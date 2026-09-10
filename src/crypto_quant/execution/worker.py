"""Live/paper trading worker (orchestration layer).

Thin process + lifecycle layer on top of the existing ``PaperTradingEngine``.
It does NOT re-implement entry/sizing/SL-TP logic — that all lives in
``execution/engine.py`` through the ``RiskManager`` and ``PaperBroker``. This
worker only:

  - builds strategy / broker / risk manager / engine from config,
  - drives two data modes (replay over stored history, realtime over Binance),
  - persists the paper account snapshot every bar,
  - runs the standalone monitoring agent,
  - manages the worker lifecycle and process lock (duplicate/stale/graceful stop).

PAPER-ONLY SAFETY: the worker only ever constructs a ``PaperBroker`` and will
pass it to a ``PaperTradingEngine`` that rejects any non-paper broker. There is
no code path from here to a live order.

Usage as script:
    python -m crypto_quant worker --symbol BTCUSDT --strategy trend \
        --mode replay --start 2025-01-01 --end 2025-03-01
    python -m crypto_quant worker --mode realtime --leading-run 1

Usage as import:
    worker = LiveTradingWorker(WorkerConfig(mode="replay", ...), db=db)
    worker.run_replay()
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..logging_config import get_logger
from ..strategies import create_strategy
from ..risk.limits import RiskLimits
from ..risk.manager import RiskManager
from .broker import AdapterPriceSource, CallbackPriceSource, PriceSource
from .engine import DataframePriceSource, PaperTradingConfig, PaperTradingEngine
from .paper import PaperBroker
from .persistence import load_account_snapshot, restore_broker_state, save_account_snapshot
from .standalone_agent import StandaloneAgent

logger = get_logger("trading")

_PAPER_LOCK_PATH = Path("data/paper_worker.lock")


# ---------------------------------------------------------------------------
# Configuration & state
# ---------------------------------------------------------------------------
@dataclass
class WorkerConfig:
    """Configuration for the paper trading worker (never live)."""

    strategy: str = "trend"
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    market_type: str = "spot"
    starting_capital: float = 1000.0
    risk_per_trade: float = 0.01
    max_open_positions: int = 3
    max_leverage: float = 5.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    max_holding_bars: Optional[int] = None
    poll_interval: int = 60           # seconds between realtime polls
    leverage: int = 1                 # futures leverage (spot is 1x)
    auto_emergency_stop: bool = False  # kill switch on daily-loss breach
    mode: str = "replay"              # 'replay' | 'realtime'
    start: Optional[str] = None       # replay start (YYYY-MM-DD or ms)
    end: Optional[str] = None         # replay end
    run_id: str = ""                  # session id; auto-generated if empty
    resume: Optional[str] = None      # db run_id to restore account from
    window: int = 300                 # warmup bars for realtime signal generation
    persist: bool = True              # whether to persist trades/snapshots to DB


@dataclass
class WorkerState:
    """Mutable, persisted-able state for the worker loop."""

    is_running: bool = False
    bars_processed: int = 0
    signals_generated: int = 0
    orders_placed: int = 0
    last_bar_time: Optional[int] = None
    last_equity: float = 0.0
    consecutive_errors: int = 0
    last_seen: str = ""


# ---------------------------------------------------------------------------
# Process lock (duplicate prevention + stale detection)
# ---------------------------------------------------------------------------
class WorkerLock:
    """A simple PID + heartbeat lock preventing two workers on one portfolio.

    Duplicate prevention: if the lock exists and its heartbeat is fresh, another
    worker is live → refuse. Stale detection: if the heartbeat is old, the owner
    died (or was killed) → reclaim. The owner refreshes the heartbeat each loop
    and releases (deletes) the lock on graceful shutdown.
    """

    def __init__(self, path: Path = _PAPER_LOCK_PATH, heartbeat_s: float = 30.0):
        self.path = Path(path)
        self.heartbeat_s = heartbeat_s
        self._owned = False

    def acquire(self, run_id: str = "") -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if self.path.exists() and data:
            last = float(data.get("last_seen", 0.0))
            if now - last < self.heartbeat_s:
                # A live owner holds the lock -> duplicate worker.
                logger.error(
                    "[PAPER MODE] duplicate worker blocked: lock owned by pid=%s run=%s",
                    data.get("pid"), data.get("run_id"))
                return False
            # stale lock -> reclaim
            logger.warning("[PAPER MODE] reclaiming stale worker lock (owner pid=%s)",
                           data.get("pid"))
        self.path.write_text(json.dumps({
            "pid": os.getpid(),
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_seen": now,
        }, indent=2), encoding="utf-8")
        self._owned = True
        return True

    def heartbeat(self) -> None:
        """Refresh the heartbeat so other starts see this worker as live."""
        if not self._owned:
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
            data["last_seen"] = time.time()
            if "pid" not in data:
                data["pid"] = os.getpid()
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def release(self) -> None:
        if self._owned:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass
            self._owned = False

    @classmethod
    def current(cls, path: Path = _PAPER_LOCK_PATH) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------
class LiveTradingWorker:
    """Orchestrates the paper pipeline by delegating to the PaperTradingEngine."""

    def __init__(
        self,
        config: Optional[WorkerConfig] = None,
        db=None,
        monitoring: Optional[StandaloneAgent] = None,
        adapter=None,
        lock: Optional[WorkerLock] = None,
    ):
        self.config = config or WorkerConfig()
        if not self.config.run_id:
            self.config.run_id = f"PAPER-{int(time.time() * 1000)}"
        self.db = db
        self.monitor = monitoring
        self.adapter = adapter
        self.lock = lock or WorkerLock()

        self.strategy, self.broker, self.risk, self.engine = self._build()
        self.state = WorkerState()
        logger.info("[PAPER MODE] worker initialized run=%s mode=%s symbol=%s %s "
                    "strategy=%s market=%s",
                    self.config.run_id, self.config.mode, self.config.symbol,
                    self.config.timeframe, self.config.strategy, self.config.market_type)

    # ------------------------------------------------------------ construction
    def _build(self):
        strat_obj = create_strategy(self.config.strategy)
        limits = RiskLimits(
            starting_capital=self.config.starting_capital,
            risk_per_trade=self.config.risk_per_trade,
            max_open_positions=self.config.max_open_positions,
            max_leverage=self.config.max_leverage,
        )
        risk = RiskManager(limits)
        market = self.config.market_type
        cfg = PaperTradingConfig(
            symbol=self.config.symbol,
            market_type=market,
            timeframe=self.config.timeframe,
            starting_capital=self.config.starting_capital,
            fee_rate=self.config.fee_rate,
            slippage=self.config.slippage,
            max_holding_bars=self.config.max_holding_bars,
            leverage=self.config.leverage,
            auto_emergency_stop=self.config.auto_emergency_stop,
        )

        if self.config.mode == "realtime":
            source = DataframePriceSource(_empty_frame())
            # Realtime polls fetch fresh windows; the price source is swapped in
            # poll_once() so the engine sees the new closes.
            self._realtime_frame = []
        else:
            df = self._load_replay_data(cfg)
            if df is None or len(df) < 2:
                raise ValueError(
                    f"[PAPER MODE] not enough stored data for {self.config.symbol} "
                    f"{self.config.timeframe}. Run `data download` first.")
            source = DataframePriceSource(df)
            self._replay_df = df

        broker = PaperBroker(
            price_source=source,
            starting_capital=self.config.starting_capital,
            fee_rate=self.config.fee_rate,
            slippage=self.config.slippage,
            market_type=market,
        )

        # Optional restart recovery: restore a prior account from the DB.
        if self.config.resume and self.db is not None:
            row = load_account_snapshot(self.db, self.config.resume, any_status=True)
            if row is not None:
                restore_broker_state(broker, row)
                logger.info("[PAPER MODE] recovered account run=%s equity=%.2f",
                            row.run_id, row.equity)

        engine = PaperTradingEngine(
            strategy=strat_obj, broker=broker, risk_manager=risk,
            config=cfg, db=self.db if self.config.persist else None,
        )
        return strat_obj, broker, risk, engine

    def _load_replay_data(self, cfg):
        from ..data.repository import MarketDataRepository
        start_ms = _parse_date_ms(self.config.start)
        end_ms = _parse_date_ms(self.config.end)
        repo = MarketDataRepository(self.db)
        df = repo.load(self.config.symbol, self.config.timeframe,
                       start_ms, end_ms, market_type=cfg.market_type)
        if df is None:
            return None
        return df.reset_index(drop=True)

    # ------------------------------------------------------------ MODE A: replay
    def run_replay(self) -> Any:
        """Deterministically replay stored bars through the paper pipeline."""
        logger.info("[PAPER MODE] starting replay run=%s (%s -> %s)",
                    self.config.run_id, self.config.start, self.config.end)
        result = self.engine.run_bars(self._replay_df)
        self.state.is_running = True
        self.state.bars_processed = len(self._replay_df)
        self.state.last_equity = float(self.broker.equity)
        self.state.last_bar_time = (
            int(self._replay_df["timestamp"].iloc[-1])
            if len(self._replay_df) else None)
        self.state.is_running = False
        # Persist final account snapshot + risk events.
        self._snapshot(status="stopped")
        if self.db is not None:
            self._persist_risk()
        logger.info("[PAPER MODE] replay finished run=%s trades=%d equity=%.2f",
                    self.config.run_id, result.n_trades, result.final_equity)
        return result

    # ------------------------------------------------------------ MODE B: realtime
    def _ensure_adapter(self):
        if self.adapter is not None:
            return self.adapter
        from ..exchange.binance import BinanceAdapter
        self.adapter = BinanceAdapter(market_type=self.config.market_type)
        return self.adapter

    def _fetch_frame(self) -> Optional[Any]:
        """Pull the latest completed-bar window from the real feed."""
        import pandas as pd
        adapter = self._ensure_adapter()
        try:
            df = adapter.get_ohlcv_as_dataframe(
                self.config.symbol, self.config.timeframe, limit=self.config.window)
        except Exception as exc:  # noqa: BLE001
            self.state.consecutive_errors += 1
            logger.warning("[PAPER MODE] feed error (%d): %s",
                           self.state.consecutive_errors, exc)
            return None
        if df is None or df.empty:
            self.state.consecutive_errors += 1
            return None
        self.state.consecutive_errors = 0
        return df.reset_index(drop=True)

    def on_new_bar(self, df) -> Dict[str, Any]:
        """Process one newly completed bar from the feed through the engine."""
        src = self.broker.price_source
        src.df = df
        src.set_index(len(df) - 1)
        # Live: the engine queues the current signal for its next on_bar so both
        # replay and realtime share the exact same risk-gated execution path.
        prepared = self.strategy.setup(df)
        signal = self.strategy.generate_signal(prepared, len(df) - 1)
        self.engine.set_live_signal(signal)
        bar = {
            "open": float(df["open"].iloc[-1]),
            "high": float(df["high"].iloc[-1]),
            "low": float(df["low"].iloc[-1]),
            "close": float(df["close"].iloc[-1]),
            "timestamp": int(df["timestamp"].iloc[-1]),
        }
        self.engine.on_bar(bar)
        self.state.bars_processed += 1
        self.state.last_bar_time = bar["timestamp"]
        self.state.last_equity = float(self.broker.equity)
        self._snapshot()
        return bar

    def run_loop(self) -> None:
        """Realtime polling loop with graceful stop / Ctrl+C handling."""
        return self._realtime_loop()

    def _realtime_loop(self) -> None:
        logger.info("[PAPER MODE] realtime loop run=%s poll=%ss", self.config.run_id,
                    self.config.poll_interval)
        adapter = self._ensure_adapter()
        last_ts = None
        try:
            df = self._fetch_frame()
            if df is not None:
                last_ts = int(df["timestamp"].iloc[-1])
            while True:
                time.sleep(self.config.poll_interval)
                self.lock.heartbeat()
                df = self._fetch_frame()
                if df is None or self.state.consecutive_errors > 3:
                    if self.state.consecutive_errors > 3:
                        logger.critical("[PAPER MODE] data feed disconnected (feed alert)")
                    continue
                ts = int(df["timestamp"].iloc[-1])
                if ts == last_ts:
                    continue
                last_ts = ts
                try:
                    self.on_new_bar(df)
                except Exception as exc:  # noqa: BLE001
                    logger.error("[PAPER MODE] bar processing error (does not stop loop): %s", exc)
                if self._monitor_step():
                    pass
                if self._stop_flagged():
                    logger.info("[PAPER MODE] stop requested; finalizing")
                    break
        except KeyboardInterrupt:
            logger.info("[PAPER MODE] interrupted by user")
        finally:
            self._shutdown(adapter=adapter, reason="live_stop")

    # ------------------------------------------------------------ lifecycle
    def run(self) -> Any:
        """Execute the configured mode and block until done/stopped."""
        if self.config.mode == "replay":
            if not self.lock.acquire(self.config.run_id):
                raise RuntimeError("another worker is holding the lock; refusing to start")
            try:
                return self.run_replay()
            finally:
                self.lock.release()
        return self.run_loop()

    def stop(self) -> None:
        """Signal a graceful stop for a background worker."""
        self._write_stop_flag()

    def _shutdown(self, adapter=None, reason: str = "session_end") -> None:
        """Finalize: close positions, persist, release lock, close adapter."""
        try:
            if self.broker.positions:
                self.broker.close_all(reason=reason)
                self.broker.mark_positions(int(time.time() * 1000))
            self.risk.update_equity(self.broker.equity)
            result = self.engine._result()
            if self.db is not None and self.config.persist:
                self.engine._persist(result)
                self._persist_risk()
            self._snapshot(status="stopped", last_market_ts=self.state.last_bar_time)
            logger.info("[PAPER MODE] finalized run=%s trades=%d equity=%.2f",
                        self.config.run_id, result.n_trades, result.final_equity)
        finally:
            self.lock.release()
            if adapter is not None:
                try:
                    adapter.close()
                except Exception:  # noqa: BLE001
                    pass
            try:
                self._clear_stop_flag()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------ persistence
    def _snapshot(self, status: str = "running",
                  last_market_ts: Optional[int] = None) -> None:
        if self.db is None or not self.config.persist:
            return
        save_account_snapshot(
            self.db,
            run_id=self.config.run_id,
            config=self.engine.config,
            broker=self.broker,
            risk=self.risk,
            strategy=self.config.strategy,
            worker_state={
                "bars_processed": self.state.bars_processed,
                "signals": self.state.signals_generated,
                "orders": self.state.orders_placed,
                "last_bar_time": self.state.last_bar_time,
            },
            status=status,
            last_market_ts=last_market_ts if last_market_ts is not None else self.state.last_bar_time,
        )

    def _persist_risk(self) -> None:
        if not self.config.persist:
            return
        from .persistence import persist_risk_events
        persist_risk_events(self.db, self.risk)

    # ------------------------------------------------------------ monitoring
    def _monitor_step(self) -> bool:
        if self.monitor is None:
            return False
        try:
            alerts = self.monitor.run_once()
        except Exception as exc:  # noqa: BLE001
            logger.debug("monitor step failed: %s", exc)
            return False
        if alerts:
            for a in alerts:
                logger.warning("[PAPER MODE][monitor][%s] %s", a.level, a.message)
        return bool(alerts)

    # ------------------------------------------------------------ stop signal
    def _stop_flag_path(self) -> Path:
        return Path("data/paper_session.json")

    def _stop_flagged(self) -> bool:
        try:
            data = json.loads(self._stop_flag_path().read_text(encoding="utf-8"))
            return bool(data.get("stop_requested"))
        except (OSError, json.JSONDecodeError):
            return False

    def _write_stop_flag(self) -> None:
        p = self._stop_flag_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except (OSError, json.JSONDecodeError):
            data = {}
        data["stop_requested"] = True
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _clear_stop_flag(self) -> None:
        p = self._stop_flag_path()
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _empty_frame():
    import pandas as pd
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


def _parse_date_ms(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    if value.isdigit():
        return int(value)
    from datetime import datetime as _dt
    try:
        return int(_dt.strptime(value, "%Y-%m-%d").timestamp() * 1000)
    except ValueError:
        return None


def load_strategy_from_registry(registry, strategy_name: str, symbol: str,
                                timeframe: str):
    """Back-compat shim: load a strategy from the registry (kept for imports)."""
    try:
        from ..strategies import create_strategy
        return create_strategy(strategy_name)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load strategy '%s': %s", strategy_name, exc)
        return None


# ---------------------------------------------------------------------------
# Script / subprocess entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    """Run the paper worker (used directly or as a detached subprocess)."""
    import argparse
    parser = argparse.ArgumentParser(description="Paper trading worker (never live)")
    parser.add_argument("--strategy", type=str, default="trend")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--market", type=str, choices=["spot", "futures"], default="spot")
    parser.add_argument("--mode", type=str, choices=["replay", "realtime"], default="replay")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--leverage", type=int, default=1)
    parser.add_argument("--auto-emergency-stop", action="store_true", default=False)
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args(argv)

    from ..db.connection import get_db_manager
    db = get_db_manager()
    db.create_tables()

    cfg = WorkerConfig(
        strategy=args.strategy, symbol=args.symbol, timeframe=args.timeframe,
        market_type=args.market, mode=args.mode, start=args.start, end=args.end,
        starting_capital=args.capital, risk_per_trade=args.risk_per_trade,
        max_open_positions=args.max_positions, leverage=args.leverage,
        auto_emergency_stop=args.auto_emergency_stop,
        poll_interval=args.poll, run_id=args.run_id, resume=args.resume,
    )
    worker = LiveTradingWorker(config=cfg, db=db)
    worker.run()


if __name__ == "__main__":
    main()