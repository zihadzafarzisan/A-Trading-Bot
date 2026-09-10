"""Live trading worker.

Orchestrates the live trading pipeline (spec #57):

    Market Data -> Signal Engine -> Risk Engine -> Live Execution
               -> Portfolio -> Database -> Dashboard

Architecture: single-threaded poll loop with configurable interval.
Supports both PaperBroker (dry-run) and LiveBroker (go-live).

Usage as script:
    python worker.py --symbol BTCUSDT --strategy momentum --dry-run

Usage as import:
    worker = LiveTradingWorker(strategy=..., broker=..., risk_manager=...)
    worker.run_loop()
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..logging_config import get_logger
from ..risk.manager import RiskManager
from ..strategies.base import BaseStrategy, Direction, Signal
from .broker import Order, PriceSource, AdapterPriceSource, CallbackPriceSource
from .paper import PaperBroker
from .standalone_agent import StandaloneAgent, StandaloneAgentConfig

logger = get_logger("trading")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class WorkerConfig:
    """Configuration for the live trading worker."""

    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    market_type: str = "spot"
    starting_capital: float = 1000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    max_holding_bars: Optional[int] = None
    poll_interval: int = 60  # seconds between market data polls
    dry_run: bool = True  # default to paper trading


@dataclass
class WorkerState:
    """Mutable state for the worker loop."""

    is_running: bool = False
    bars_processed: int = 0
    signals_generated: int = 0
    orders_placed: int = 0
    last_bar_time: Optional[int] = None


# ---------------------------------------------------------------------------
# Live Trading Worker
# ---------------------------------------------------------------------------
class LiveTradingWorker:
    """Orchestrates the live trading pipeline.

    Connects to market data feed, runs strategies, and executes via broker.
    Supports single strategy or registry-based loading.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        broker,
        risk_manager: RiskManager,
        config: Optional[WorkerConfig] = None,
        exchange_adapter=None,
        monitor: Optional[StandaloneAgent] = None,
    ):
        """Initialize the worker.

        Args:
            strategy: The strategy generating signals.
            broker: PaperBroker or LiveBroker instance.
            risk_manager: RiskManager gating every entry.
            config: Worker configuration.
            exchange_adapter: ExchangeAdapter for market data.
            monitor: Optional monitoring agent for alerts.
        """
        self.config = config or WorkerConfig()
        self.strategy = strategy
        self.broker = broker
        self.risk = risk_manager
        self.exchange_adapter = exchange_adapter
        self.monitor = monitor

        # State
        self.state = WorkerState()

        # Internal
        self._pending_signal: Optional[Signal] = None
        self._price_history: List[Dict[str, float]] = []
        self._max_history: int = 1000

    # ------------------------------------------------------------------ public API
    def run_once(self) -> Optional[Dict[str, Any]]:
        """Run a single iteration of the trading loop.

        Returns:
            Dict with iteration results, or None if no action taken.
        """
        if self.state.is_running:
            logger.warning("Worker already running")
            return None

        self.state.is_running = True
        result = None

        try:
            # 1. Fetch market data
            bar = self._fetch_bar()
            if bar is None:
                return None

            # 2. Manage existing positions (SL/TP)
            self._manage_exits(bar)

            # 3. Generate signal
            signal = self._generate_signal(bar)

            # 4. Process pending entry from previous bar
            if self._pending_signal is not None:
                entry_result = self._enter(self._pending_signal, bar["open"], bar["timestamp"])
                self._pending_signal = None

            # 5. Queue signal for next bar
            if signal is not None and signal.is_active:
                self._pending_signal = signal
                self.state.signals_generated += 1

            # 6. Snapshot equity
            self.broker.mark_positions(bar["timestamp"])

            self.state.bars_processed += 1
            self.state.last_bar_time = bar["timestamp"]

            result = {
                "bar": bar,
                "signal": signal.to_dict() if signal else None,
                "equity": self.broker.equity,
                "positions": self.broker.get_positions(),
            }

        except Exception as e:
            logger.error("Worker iteration failed: %s", e)
        finally:
            self.state.is_running = False

        return result

    def run_loop(self) -> None:
        """Run continuous trading loop."""
        logger.info(
            "Starting live trading worker (symbol=%s, strategy=%s, dry_run=%s, interval=%ds)",
            self.config.symbol,
            self.strategy.name,
            self.config.dry_run,
            self.config.poll_interval,
        )

        try:
            while True:
                result = self.run_once()
                if result:
                    logger.debug(
                        "Iteration %d: equity=%.2f, positions=%d",
                        self.state.bars_processed,
                        result["equity"],
                        len(result["positions"]),
                    )
                time.sleep(self.config.poll_interval)
        except KeyboardInterrupt:
            logger.info("Worker stopped by user")
            self._cleanup()

    def stop(self) -> None:
        """Stop the worker gracefully."""
        logger.info("Stopping worker...")
        self._cleanup()

    # ------------------------------------------------------------------ market data
    def _fetch_bar(self) -> Optional[Dict[str, float]]:
        """Fetch latest market data bar."""
        if self.exchange_adapter is None:
            logger.error("No exchange adapter configured")
            return None

        try:
            df = self.exchange_adapter.get_ohlcv_as_dataframe(
                self.config.symbol, self.config.timeframe, limit=1
            )
            if df is None or df.empty:
                logger.warning("No data returned for %s", self.config.symbol)
                return None

            bar = {
                "open": float(df["open"].iloc[-1]),
                "high": float(df["high"].iloc[-1]),
                "low": float(df["low"].iloc[-1]),
                "close": float(df["close"].iloc[-1]),
                "volume": float(df["volume"].iloc[-1]),
                "timestamp": int(df["timestamp"].iloc[-1]) if "timestamp" in df.columns else int(time.time() * 1000),
            }

            # Update price history
            self._price_history.append(bar)
            if len(self._price_history) > self._max_history:
                self._price_history.pop(0)

            return bar

        except Exception as e:
            logger.error("Failed to fetch bar for %s: %s", self.config.symbol, e)
            return None

    # ------------------------------------------------------------------ strategy
    def _generate_signal(self, bar: Dict[str, float]) -> Optional[Signal]:
        """Generate trading signal from current bar."""
        try:
            # Build DataFrame from price history for strategy
            import pandas as pd
            if len(self._price_history) < 2:
                return None

            df = pd.DataFrame(self._price_history)
            if "timestamp" not in df.columns:
                df["timestamp"] = range(len(df))

            prepared = self.strategy.setup(df)
            signal = self.strategy.generate_signal(prepared, len(prepared) - 1)
            return signal

        except Exception as e:
            logger.error("Signal generation failed: %s", e)
            return None

    # ------------------------------------------------------------------ execution
    def _enter(self, signal: Signal, bar_open: float, bar_time: int) -> Optional[Dict[str, Any]]:
        """Risk-gate, size, and place an entry order."""
        if not signal.is_active:
            return None

        if self.config.market_type == "spot" and signal.direction == Direction.SHORT:
            return None

        if signal.stop_loss is None:
            return None

        equity = self.broker.equity

        # Risk check
        check = self.risk.check_entry(
            equity=equity,
            open_positions=len(self.broker.get_positions()),
            leverage=1,
            symbol=self.config.symbol,
        )
        if not check.is_allowed:
            logger.debug("Entry rejected: %s", check.reasons)
            return None

        # Position sizing
        try:
            sizing = self.risk.size_position(
                equity=equity,
                entry_price=bar_open,
                stop_price=signal.stop_loss,
                direction=signal.direction.value,
                market_type=self.config.market_type,
                leverage=1,
            )
        except ValueError:
            return None

        # Place order
        side = "buy" if signal.direction == Direction.LONG else "sell"
        order = Order(
            order_id=f"live_{self.state.orders_placed}",
            symbol=self.config.symbol,
            side=side,
            quantity=sizing.quantity,
            order_type="market",
        )
        self.broker.place_order(order)
        self.state.orders_placed += 1

        # Attach stop/take levels to position
        if order.status == "filled":
            positions = self.broker.get_positions()
            for pos in positions:
                if pos.get("symbol") == self.config.symbol:
                    pos["stop_loss"] = signal.stop_loss
                    pos["take_profit"] = signal.take_profit
                    break

        return {"order": order.order_id, "side": side, "quantity": sizing.quantity}

    def _manage_exits(self, bar: Dict[str, float]) -> None:
        """Close positions whose stop/take was touched this bar."""
        positions = self.broker.get_positions()
        for pos in list(positions):
            symbol = pos.get("symbol")
            sl = pos.get("stop_loss")
            tp = pos.get("take_profit")
            side = pos.get("side")
            quantity = pos.get("quantity", 0)

            if side not in ("long", "short"):
                continue

            hit_sl = (side == "long" and sl is not None and bar["low"] <= sl) or \
                     (side == "short" and sl is not None and bar["high"] >= sl)
            hit_tp = (side == "long" and tp is not None and bar["high"] >= tp) or \
                     (side == "short" and tp is not None and bar["low"] <= tp)

            fill = None
            reason = None
            if hit_sl and hit_tp:
                fill, reason = sl, "sl"
            elif hit_sl:
                fill = min(bar["open"], sl) if side == "long" else max(bar["open"], sl)
                reason = "sl"
            elif hit_tp:
                fill = max(bar["open"], tp) if side == "long" else min(bar["open"], tp)
                reason = "tp"

            if fill is not None:
                # Update price source for fill
                if hasattr(self.broker, "price_source"):
                    self.broker.price_source.set_price(fill)

                order = Order(
                    order_id=f"live_exit_{self.state.orders_placed}",
                    symbol=symbol,
                    side="sell" if side == "long" else "buy",
                    quantity=quantity,
                    order_type="market",
                )
                self.broker.place_order(order)
                self.state.orders_placed += 1

                # Record exit reason
                if self.broker.closed_trades:
                    self.broker.closed_trades[-1]["exit_reason"] = reason
                    self.broker.closed_trades[-1]["exit_time"] = bar.get("timestamp")

    # ------------------------------------------------------------------ cleanup
    def _cleanup(self) -> None:
        """Clean up resources on shutdown."""
        logger.info(
            "Worker stats: bars=%d, signals=%d, orders=%d",
            self.state.bars_processed,
            self.state.signals_generated,
            self.state.orders_placed,
        )


# ---------------------------------------------------------------------------
# Registry support
# ---------------------------------------------------------------------------
def load_strategy_from_registry(
    registry,
    strategy_name: str,
    symbol: str,
    timeframe: str,
) -> Optional[BaseStrategy]:
    """Load a strategy from the strategy registry.

    Args:
        registry: StrategyRegistry instance.
        strategy_name: Name of the strategy to load.
        symbol: Trading symbol.
        timeframe: Timeframe for the strategy.

    Returns:
        Configured strategy instance, or None if not found.
    """
    try:
        strategy_cls = registry.get(strategy_name)
        if strategy_cls is None:
            logger.error("Strategy '%s' not found in registry", strategy_name)
            return None
        return strategy_cls(symbol=symbol, timeframe=timeframe)
    except Exception as e:
        logger.error("Failed to load strategy '%s': %s", strategy_name, e)
        return None


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
def main():
    """Run live trading worker."""
    import argparse

    parser = argparse.ArgumentParser(description="Live trading worker")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--strategy", type=str, default="momentum", help="Strategy name")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds")
    parser.add_argument("--capital", type=float, default=1000.0, help="Starting capital")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Paper trading mode (default)")
    parser.add_argument("--live", action="store_true", help="Live trading mode (use with caution)")
    parser.add_argument("--monitor", action="store_true", help="Enable monitoring agent")
    args = parser.parse_args()

    # Import here to avoid circular imports
    from ..strategies.registry import StrategyRegistry
    from ..risk.manager import RiskManager
    from .paper import PaperBroker
    from .broker import AdapterPriceSource

    # Determine broker based on mode
    dry_run = not args.live

    # Create adapter (placeholder - in production, pass real adapter)
    adapter = None  # Would be BinanceAdapter in production

    # Create price source
    price_source = AdapterPriceSource(adapter) if adapter else None

    # Create broker
    broker = PaperBroker(
        price_source=price_source,
        initial_capital=args.capital,
        fee_rate=0.001,
        slippage=0.0005,
    )

    # Load strategy
    registry = StrategyRegistry()
    strategy = load_strategy_from_registry(
        registry, args.strategy, args.symbol, args.timeframe
    )
    if strategy is None:
        logger.error("Failed to load strategy, exiting")
        return

    # Create risk manager
    risk_manager = RiskManager()

    # Create worker config
    config = WorkerConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        starting_capital=args.capital,
        poll_interval=args.interval,
        dry_run=dry_run,
    )

    # Create worker
    worker = LiveTradingWorker(
        strategy=strategy,
        broker=broker,
        risk_manager=risk_manager,
        config=config,
        exchange_adapter=adapter,
    )

    # Optionally attach monitoring agent
    if args.monitor:
        monitor_config = StandaloneAgentConfig(
            symbols=[args.symbol],
            timeframe=args.timeframe,
        )
        monitor = StandaloneAgent(
            exchange_adapter=adapter,
            price_source=price_source,
            broker=broker,
            config=monitor_config,
        )
        worker.monitor = monitor

    # Run
    mode = "DRY-RUN" if dry_run else "LIVE"
    logger.info("Starting worker in %s mode", mode)
    worker.run_loop()


if __name__ == "__main__":
    main()
