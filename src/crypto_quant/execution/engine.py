"""Paper trading engine.

Orchestrates the paper-trading pipeline (spec #35):

    Market Data -> Signal Engine -> Risk Engine -> Paper Execution
               -> Portfolio -> Database -> Dashboard

The engine can run in two modes:
- replay(df): deterministically processes historical bars (testable, identical
  timing to the backtest engine).
- on_bar(i): process a single completed bar from a live feed.

In both modes the SAME code path runs: risk-gated paper orders through the
PaperBroker. No real order is ever placed.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from ..backtesting.execution import ExecutionModel
from ..data.repository import TIMEFRAME_MS
from ..logging_config import get_logger
from ..risk.manager import RiskManager
from ..strategies.base import BaseStrategy, Direction, Signal
from .broker import Order, PriceSource
from .paper import PaperBroker

logger = get_logger("trading")


@dataclass
class PaperTradingConfig:
    """Configuration for a paper trading session."""

    symbol: str = "BTCUSDT"
    market_type: str = "spot"
    timeframe: str = "1h"
    starting_capital: float = 1000.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    max_holding_bars: Optional[int] = None
    leverage: int = 1                 # futures leverage (spot is always 1x)
    auto_emergency_stop: bool = False  # engage kill switch when the daily loss limit is breached
    funding_bars: int = 8              # bars per funding interval (8 per 8h on 1h bars)


@dataclass
class PaperSessionResult:
    """Summary of a paper trading session."""

    symbol: str
    timeframe: str
    n_trades: int
    n_orders: int
    n_rejected: int
    starting_capital: float
    final_equity: float
    realized_pnl: float
    win_rate: float
    trades: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    risk_events: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "timeframe": self.timeframe,
            "n_trades": self.n_trades, "n_orders": self.n_orders,
            "n_rejected": self.n_rejected,
            "starting_capital": self.starting_capital,
            "final_equity": round(self.final_equity, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "win_rate": self.win_rate,
        }


class DataframePriceSource(PriceSource):
    """Price source that reads from a DataFrame (replay/testing).

    The engine advances the index per bar and can override the fill price for
    specific orders (entry at open, stop/take at their levels).
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._i = 0
        self._override: Optional[float] = None

    def set_index(self, i: int) -> None:
        self._i = i
        self._override = None

    def set_price(self, price: float) -> None:
        self._override = price

    def get_price(self, symbol: str) -> float:
        if self._override is not None:
            return self._override
        return float(self.df["close"].iloc[self._i])


class PaperTradingEngine:
    """Runs risk-gated paper trading against a price feed."""

    def __init__(
        self,
        strategy: BaseStrategy,
        broker: PaperBroker,
        risk_manager: RiskManager,
        config: Optional[PaperTradingConfig] = None,
        db=None,
    ):
        """Initialize the engine.

        Args:
            strategy: The strategy generating signals.
            broker: A PaperBroker (never a LiveBroker).
            risk_manager: The RiskManager gating every entry.
            config: Session configuration.
            db: Optional DatabaseManager to persist paper trades.
        """
        # PAPER-ONLY SAFETY: the engine can only ever be driven by a PaperBroker.
        # Any other broker is rejected at construction. This structurally guards
        # the paper pipeline from placing a real order through a live execution path.
        if not isinstance(broker, PaperBroker):
            raise TypeError(
                "PaperTradingEngine requires a PaperBroker (paper mode can never "
                "execute live orders). Got %r" % type(broker).__name__
            )

        self.strategy = strategy
        self.broker = broker
        self.risk = risk_manager
        self.config = config or PaperTradingConfig()
        self.db = db
        from ..backtesting.execution import ExecutionConfig
        self.exec = ExecutionModel(ExecutionConfig(market_type=self.config.market_type))
        self.equity_curve: List[Dict[str, Any]] = []
        self._pending: Optional[Signal] = None
        self._prepared: Optional[pd.DataFrame] = None
        self._bar_index = 0
        self._fed_trade_ids: set = set()
        self.warnings: List[str] = []

    # ------------------------------------------------------------ replay
    def run_bars(self, df: pd.DataFrame) -> PaperSessionResult:
        """Replay historical bars through the paper pipeline.

        Args:
            df: OHLCV DataFrame (sorted ascending).

        Returns:
            PaperSessionResult with trades and equity curve.
        """
        if not isinstance(self.broker.price_source, DataframePriceSource):
            raise ValueError("run_bars requires a DataframePriceSource broker")

        prepared = self.strategy.setup(df)
        self._prepared = prepared
        source: DataframePriceSource = self.broker.price_source  # type: ignore

        opens = prepared["open"].to_numpy()
        highs = prepared["high"].to_numpy()
        lows = prepared["low"].to_numpy()
        closes = prepared["close"].to_numpy()
        times = prepared["timestamp"].to_numpy()
        n = len(prepared)

        for i in range(n):
            self._bar_index = i
            source.set_index(i)

            # 1) Fill pending entry at this bar's open
            if self._pending is not None:
                self._enter(self._pending, opens[i], times[i])
                self._pending = None

            # 2) Manage open positions (SL/TP / liquidation) using this bar's range
            self._manage_exits(highs[i], lows[i], opens[i], closes[i], times[i])

            # 3) Mark positions, snapshot equity, and feed the risk engine so the
            #    daily/weekly loss and drawdown limits stay live (not just configured).
            self.broker.mark_positions(int(times[i]))
            self.equity_curve.append({
                "time": int(times[i]), "equity": self.broker.equity,
            })
            self.risk.update_equity(self.broker.equity)
            self._feed_risk_from_closed(int(times[i]))
            self._maybe_auto_stop()

            # 4) Generate the signal for the next bar
            if i + 1 < n:
                self._pending = self.strategy.generate_signal(prepared, i)

        # Close any remaining positions at the last price
        if self.broker.positions:
            source.set_price(closes[-1])
            self.broker.close_all(reason="session_end")
            self.broker.mark_positions(int(times[-1]))
            self.equity_curve.append({"time": int(times[-1]), "equity": self.broker.equity})
            self.risk.update_equity(self.broker.equity)
            self._feed_risk_from_closed(int(times[-1]))
            self._maybe_auto_stop()

        result = self._result()
        if self.db:
            self._persist(result)
        return result

    # ------------------------------------------------------------ live step
    def on_bar(self, bar: Dict[str, float]) -> None:
        """Process a single completed bar (for live feeds).

        Args:
            bar: dict with keys open, high, low, close, timestamp.
        """
        source = self.broker.price_source
        if not isinstance(source, DataframePriceSource):
            raise ValueError("on_bar requires a DataframePriceSource broker")
        source.set_price(float(bar["close"]))
        source._i += 1
        self._bar_index += 1

        if self._pending is not None:
            self._enter(self._pending, float(bar["open"]), int(bar["timestamp"]))
            self._pending = None

        self._manage_exits(float(bar["high"]), float(bar["low"]),
                           float(bar["open"]), float(bar["close"]),
                           int(bar["timestamp"]))
        self.broker.mark_positions(int(bar["timestamp"]))
        self.equity_curve.append({"time": int(bar["timestamp"]), "equity": self.broker.equity})
        self.risk.update_equity(self.broker.equity)
        self._feed_risk_from_closed(int(bar["timestamp"]))
        self._maybe_auto_stop()

    def set_live_signal(self, signal: Signal) -> None:
        """Accept a strategy signal produced externally (live mode)."""
        self._pending = signal

    # ------------------------------------------------------------ internals
    def _enter(self, signal: Signal, bar_open: float, bar_time: int) -> None:
        """Risk-gate, size, and place a paper entry order."""
        if not signal.is_active:
            return
        if self.config.market_type == "spot" and signal.direction == Direction.SHORT:
            return
        if signal.stop_loss is None:
            return

        equity = self.broker.equity
        check = self.risk.check_entry(
            equity=equity,
            open_positions=len(self.broker.positions),
            leverage=self.config.leverage,
            symbol=self.config.symbol,
        )
        if not check.is_allowed:
            logger.debug("Paper entry rejected: %s", check.reasons)
            return

        try:
            leverage = 1 if self.config.market_type == "spot" else self.config.leverage
            sizing = self.risk.size_position(
                equity=equity, entry_price=bar_open, stop_price=signal.stop_loss,
                direction=signal.direction.value,
                market_type=self.config.market_type, leverage=leverage,
            )
        except ValueError:
            return

        side = "buy" if signal.direction == Direction.LONG else "sell"
        order = Order(
            order_id=f"paper_{len(self.broker.orders)}",
            symbol=self.config.symbol, side=side,
            quantity=sizing.quantity, order_type="market",
        )
        self.broker.place_order(order)
        # Attach stop/take / leverage / funding metadata to the new position so
        # SL/TP exits, liquidation, funding, and persistence all carry it forward.
        if order.status == "filled":
            pos = self.broker.positions.get(self.config.symbol)
            if pos is not None:
                pos["stop_loss"] = signal.stop_loss
                pos["take_profit"] = signal.take_profit
                pos["leverage"] = sizing.leverage
                pos["funding"] = 0.0
                pos["entry_bar"] = self._bar_index
                pos["timeframe"] = self.config.timeframe

    def _manage_exits(self, high: float, low: float, open_p: float, close_p: float, bar_time: int) -> None:
        """Close positions whose stop/take/liquidation level was touched this bar."""
        for symbol in list(self.broker.positions.keys()):
            pos = self.broker.positions[symbol]
            sl = pos.get("stop_loss")
            tp = pos.get("take_profit")
            side = pos["side"]

            # Futures liquidation (isolated-margin estimate) is checked first and
            # wins: if the position's margin is wiped intrabar it is closed at the
            # liquidation price regardless of stop/take.
            if self.config.market_type == "futures":
                lev = float(pos.get("leverage") or 1)
                if lev > 0:
                    liq = self.exec.liquidation_price(float(pos["entry_price"]), side, lev)
                    if liq and liq > 0:
                        liq_hit = (side == "long" and low <= liq) or \
                                  (side == "short" and high >= liq)
                        if liq_hit:
                            self._do_exit(pos, liq, "liquidation", bar_time)
                            continue

            hit_sl = (side == "long" and sl is not None and low <= sl) or \
                     (side == "short" and sl is not None and high >= sl)
            hit_tp = (side == "long" and tp is not None and high >= tp) or \
                     (side == "short" and tp is not None and low <= tp)

            fill = None
            reason = None
            if hit_sl and hit_tp:
                fill, reason = sl, "sl"
            elif hit_sl:
                fill = min(open_p, sl) if side == "long" else max(open_p, sl)
                reason = "sl"
            elif hit_tp:
                fill = max(open_p, tp) if side == "long" else min(open_p, tp)
                reason = "tp"

            if fill is not None:
                self._do_exit(pos, fill, reason, bar_time)

    def _do_exit(self, pos: Dict[str, Any], fill: float, reason: str, bar_time: int) -> None:
        """Place a paper exit at ``fill`` and stamp the resulting closed trade."""
        self.broker.price_source.set_price(fill)  # type: ignore[attr-defined]
        order = Order(
            order_id=f"paper_exit_{len(self.broker.orders)}",
            symbol=pos["symbol"],
            side="sell" if pos["side"] == "long" else "buy",
            quantity=pos["quantity"], order_type="market",
        )
        self.broker.place_order(order)
        if self.broker.closed_trades:
            t = self.broker.closed_trades[-1]
            t["exit_reason"] = reason
            t["exit_time"] = int(bar_time)

    # ------------------------------------------------------------ risk feedback
    def _feed_risk_from_closed(self, now_ms: int) -> None:
        """Feed realized PnL (net of fees & futures funding) back to the RiskManager.

        This is what makes the daily/weekly loss and drawdown limits live during a
        paper run instead of merely configured: every closed trade flows through
        ``risk.record_trade_result`` so subsequent ``check_entry`` calls see the
        updated loss counters. Idempotent per closed trade.
        """
        for t in self.broker.closed_trades:
            tid = t.get("trade_id")
            if not tid or tid in self._fed_trade_ids:
                continue
            funding = 0.0
            if self.config.market_type == "futures" and now_ms and t.get("entry_time"):
                bar_ms = self._bar_ms()
                if bar_ms > 0:
                    holding = (now_ms - int(t["entry_time"])) / bar_ms
                    if holding > 0:
                        notional = float(t["entry_price"]) * float(t["quantity"])
                        funding = round(
                            self.exec.funding_cost(notional, max(1.0, holding),
                                                   self.config.funding_bars), 8
                        )
                        t["funding"] = funding
                        t["net_pnl"] = round((t.get("net_pnl") or 0.0) - funding, 8)
            self.risk.record_trade_result(t.get("net_pnl") or 0.0)
            self._fed_trade_ids.add(tid)

    def _maybe_auto_stop(self) -> None:
        """Engage the kill switch (and close positions) if the daily loss limit is breached."""
        if not self.config.auto_emergency_stop:
            return
        limits = self.risk.limits
        if self.risk.state.realized_pnl_day <= -limits.daily_loss_limit and not self.risk.is_shutdown:
            self.risk.emergency_stop(close_positions=True)
            self.warnings.append(
                f"auto emergency stop: daily loss limit hit "
                f"({self.risk.state.realized_pnl_day:.2f} <= -{limits.daily_loss_limit:.2f})"
            )
            logger.critical("[PAPER MODE] %s", self.warnings[-1])
            if self.broker.positions:
                self.broker.close_all(reason="emergency_stop")

    def _bar_ms(self) -> int:
        """Milliseconds per bar for the configured timeframe."""
        return TIMEFRAME_MS.get(self.config.timeframe, 3_600_000)

    # ------------------------------------------------------------ results
    def _result(self) -> PaperSessionResult:
        trades = self.broker.closed_trades
        pnls = [t["net_pnl"] for t in trades if t.get("net_pnl") is not None]
        wins = sum(1 for p in pnls if p > 0)
        rejected = sum(1 for o in self.broker.orders if o.status == "rejected")
        unrealized = sum(self.broker._unrealized(p, self.broker.get_price(p["symbol"]))
                         for p in self.broker.positions.values())
        return PaperSessionResult(
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            n_trades=len(trades),
            n_orders=len(self.broker.orders),
            n_rejected=rejected,
            starting_capital=self.config.starting_capital,
            final_equity=self.broker.equity,
            realized_pnl=sum(pnls),
            win_rate=wins / len(pnls) if pnls else 0.0,
            trades=trades,
            equity_curve=self.equity_curve,
            risk_events=self.risk.recent_events(),
            warnings=self.warnings,
        )

    def _persist(self, result: PaperSessionResult) -> None:
        """Persist paper trades to the execution_trades table."""
        from ..db.models import ExecutionTrade, Strategy
        session = self.db.get_session()
        try:
            # Ensure a strategy row exists (use the strategy type as the id)
            sid = f"STRAT-{self.strategy.strategy_type.upper()[:8]}"
            if session.query(Strategy).filter(Strategy.id == sid).first() is None:
                session.add(Strategy(id=sid, name=self.strategy.name,
                                     type=self.strategy.strategy_type,
                                     parameters=str(self.strategy.params), version=1))
            for t in result.trades:
                session.add(ExecutionTrade(
                    id=t["trade_id"],
                    strategy_id=sid,
                    execution_mode="paper",
                    symbol=t["symbol"],
                    market_type=self.config.market_type,
                    timeframe=self.config.timeframe,
                    direction=t["direction"],
                    entry_time=int(t["entry_time"] or 0),
                    exit_time=int(t["exit_time"] or 0),
                    entry_price=float(t["entry_price"]),
                    exit_price=float(t["exit_price"]),
                    quantity=float(t["quantity"]),
                    leverage=int(t.get("leverage") or 1),
                    stop_loss=float(t["stop_loss"]) if t.get("stop_loss") else None,
                    take_profit=float(t["take_profit"]) if t.get("take_profit") else None,
                    fees=float(t.get("fees") or 0.0),
                    funding=float(t.get("funding") or 0.0),
                    gross_pnl=float(t.get("gross_pnl") or 0.0),
                    status="closed",
                    net_pnl=float(t["net_pnl"]),
                    exit_reason=t.get("exit_reason"),
                ))
            session.commit()
            logger.info("Persisted %d paper trades", len(result.trades))
        finally:
            session.close()