"""Base strategy framework.

Defines the abstract strategy interface shared by every strategy family, along
with the data structures used to express trade intentions (Signal) and risk
(sizing + stop/take specification).

Design contract
---------------
- A strategy precomputes indicator columns once via ``setup()``.
- On each closed bar ``i`` it returns a ``Signal`` describing an *intent* for
  the NEXT bar. Because all indicators are shifted by one bar (see the
  indicators package), decisions at bar ``i`` never use current/future data.
- Entry fill, stop/take monitoring, fees, and sizing are handled by the
  execution/backtest layer — a strategy only expresses *what* and *where* the
  stops are, not the accounting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..logging_config import get_logger

logger = get_logger("strategies")


class Direction(str, Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"
    NONE = "none"


class StopType(str, Enum):
    """Mechanism used to set a dynamic stop loss."""
    ATR = "atr"
    PERCENT = "percent"
    SWING = "swing"
    FIXED = "fixed"


class ExitReason(str, Enum):
    """Reasons a position may be closed."""
    TAKE_PROFIT = "tp"
    STOP_LOSS = "sl"
    SIGNAL = "signal"
    TIME = "time"


@dataclass
class Signal:
    """A trading intention for the next bar."""

    direction: Direction = Direction.NONE
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Whether this signal calls for a new entry."""
        return self.direction in (Direction.LONG, Direction.SHORT)

    @property
    def direction_str(self) -> str:
        return self.direction.value


@dataclass
class StopLossSpec:
    """How a stop loss is derived."""

    stop_type: StopType = StopType.ATR
    atr_multiplier: float = 2.0      # for ATR
    percent: float = 0.03            # for PERCENT (0.03 = 3%)
    swing_lookback: int = 20         # for SWING


@dataclass
class TakeProfitSpec:
    """How a take profit is derived."""

    mode: str = "rr"                 # 'rr' | 'atr' | 'fixed'
    risk_reward_ratio: float = 2.0   # for 'rr'
    atr_multiplier: float = 3.0      # for 'atr'
    fixed_pct: float = 0.10          # for 'fixed'


class BaseStrategy(ABC):
    """Abstract base class for all strategies."""

    # ---- metadata (override in subclasses) ----
    name: str = "base"
    strategy_type: str = "base"
    description: str = ""
    supports_spot: bool = True
    supports_futures: bool = True
    supports_short: bool = False

    @classmethod
    def default_timeframes(cls) -> List[str]:
        """Timeframes best suited to this strategy family."""
        return ["1h", "4h", "1d"]

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        stop_spec: Optional[StopLossSpec] = None,
        tp_spec: Optional[TakeProfitSpec] = None,
    ):
        """Initialize strategy with parameters.

        Args:
            params: Strategy-specific parameters (validated by subclass).
            stop_spec: Optional stop-loss mechanism override.
            tp_spec: Optional take-profit specification override.
        """
        self.params: Dict[str, Any] = dict(params or {})
        self.stop_spec = stop_spec or StopLossSpec()
        self.tp_spec = tp_spec or TakeProfitSpec()
        self.validate_params()
        logger.debug("Initialized %s with params=%s", self.name, self.params)

    # ------------------------------------------------------------ abstraction
    @abstractmethod
    def validate_params(self) -> None:
        """Validate self.params, raising ValueError on invalid input."""

    @abstractmethod
    def setup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Precompute indicator columns on a copy of df.

        Returns the enriched DataFrame used by entry_signal(). Indicators must
        be shifted (see indicators package) so they carry no look-ahead.
        """

    @abstractmethod
    def entry_signal(self, df: pd.DataFrame, i: int, ctx: Optional[Dict] = None) -> tuple[Direction, str]:
        """Return entry (direction, reason) at closed bar i.

        Implementations read precomputed indicator columns from ``df`` at row
        ``i`` only. Return Direction.NONE with reason "" when no valid setup.
        """

    # ---------------------------------------------------- stop/take defaults
    def compute_stop_loss(
        self,
        df: pd.DataFrame,
        i: int,
        direction: Direction,
        entry_price: float,
    ) -> Optional[float]:
        """Derive a stop-loss price for an entry priced at entry_price."""
        spec = self.stop_spec
        if spec.stop_type == StopType.PERCENT:
            if direction == Direction.LONG:
                return entry_price * (1 - spec.percent)
            return entry_price * (1 + spec.percent)

        if spec.stop_type == StopType.SWING:
            swing = self._swing_extreme(df, i, spec.swing_lookback, direction)
            if swing is not None:
                return swing

        # ATR-based (default) and FIXED both default to ATR behavior
        self._ensure_atr(df)
        atr_val = self._atr_at(df, i)
        if atr_val is None or atr_val == 0:
            return None
        offset = spec.atr_multiplier * atr_val
        if direction == Direction.LONG:
            return entry_price - offset
        return entry_price + offset

    def compute_take_profit(
        self,
        df: pd.DataFrame,
        i: int,
        direction: Direction,
        entry_price: float,
        stop_loss: Optional[float] = None,
    ) -> Optional[float]:
        """Derive a take-profit price based on the TP spec."""
        spec = self.tp_spec
        if spec.mode == "fixed":
            if direction == Direction.LONG:
                return entry_price * (1 + spec.fixed_pct)
            return entry_price * (1 - spec.fixed_pct)

        if spec.mode == "atr":
            self._ensure_atr(df)
            atr_val = self._atr_at(df, i)
            if atr_val is None:
                return None
            offset = spec.atr_multiplier * atr_val
            if direction == Direction.LONG:
                return entry_price + offset
            return entry_price - offset

        # 'rr' mode: use risk-reward ratio applied to stop distance
        if stop_loss is None:
            return None
        risk = abs(entry_price - stop_loss)
        reward = risk * spec.risk_reward_ratio
        if direction == Direction.LONG:
            return entry_price + reward
        return entry_price - reward

    # ------------------------------------------------------- fill-time anchoring
    def prepare_fill(
        self,
        df: pd.DataFrame,
        signal_bar_index: int,
        direction: Direction,
        actual_fill_price: float,
        sig: Optional[Signal] = None,
    ) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
        """Derive deterministic risk state for an entry at the given fill price.

        Called by the engine AFTER the actual fill price is known (post-slippage).
        Returns ``(stop_loss, take_profit, extra_state)``.

        Default (legacy) behavior: keep the stop/take that generate_signal()
        already computed, and carry no extra state. Strategies that must
        re-anchor stop/take to the real fill price (dynamic stop families)
        override this.
        """
        if sig is not None:
            return sig.stop_loss, sig.take_profit, {}
        return None, None, {}

    # ------------------------------------------------------------ stop management
    def update_stop(
        self,
        df: pd.DataFrame,
        bar_index: int,
        position,
    ) -> None:
        """Per-bar stop management; called after the close of ``bar_index``.

        Default no-op. Strategies with dynamic stops (ATR trailing, breakeven)
        override to set ``position.active_stop``, ``position.risk_r``, or
        ``position.breakeven_active``. Effective one bar delayed.
        """
        return

    # ------------------------------------------------------------ filters
    def apply_filters(self, df: pd.DataFrame, i: int, context: Optional[Dict] = None) -> bool:
        """Return True if the bar passes the strategy-level filters.

        Override to add e.g. volume or trend-strength filters. Default: pass.
        """
        return True

    # ---------------------------------------------------------- high-level API
    def generate_signal(self, df: pd.DataFrame, i: int, ctx: Optional[Dict] = None) -> Signal:
        """Return a full Signal for the next bar, evaluated at closed bar i.

        Combines entry conditions, filters, and stop/take derivation. The backtest
        layer calls this once per closed bar.
        """
        if not self.apply_filters(df, i, ctx):
            return Signal(direction=Direction.NONE)

        direction, reason = self.entry_signal(df, i, ctx)
        if direction == Direction.NONE:
            return Signal(direction=Direction.NONE)

        entry_price = float(df["close"].iloc[i])
        stop = self.compute_stop_loss(df, i, direction, entry_price)
        tp = self.compute_take_profit(df, i, direction, entry_price, stop)

        return Signal(
            direction=direction,
            stop_loss=stop,
            take_profit=tp,
            reason=reason,
        )

    # ------------------------------------------------------------ helpers
    def _ensure_atr(self, df: pd.DataFrame) -> None:
        """Attach an ATR-14 column to df if not already present.

        Ensures ATR-based stops work regardless of whether the strategy's
        setup() computed ATR itself. Idempotent and cached per DataFrame.
        """
        if "atr_14" not in df.columns:
            from ..indicators import volatility as vt
            df["atr_14"] = vt.atr(df["high"], df["low"], df["close"], 14, shift=0)

    def _atr_at(self, df: pd.DataFrame, i: int) -> Optional[float]:
        col = "atr_14"
        if col not in df.columns:
            return None
        val = df[col].iloc[i]
        import numpy as np
        if pd.isna(val) or np.isinf(val):
            return None
        return float(val)

    def _swing_extreme(
        self, df: pd.DataFrame, i: int, lookback: int, direction: Direction
    ) -> Optional[float]:
        """Recent swing low (long) or swing high (short) for stop placement."""
        if i < 1:
            return None
        start = max(0, i - lookback)
        if direction == Direction.LONG:
            return float(df["low"].iloc[start:i].min())
        return float(df["high"].iloc[start:i].max())

    # ------------------------------------------------------------ metadata
    def metadata(self) -> Dict[str, Any]:
        """Strategy metadata for experiment tracking."""
        return {
            "name": self.name,
            "type": self.strategy_type,
            "description": self.description,
            "supports_spot": self.supports_spot,
            "supports_futures": self.supports_futures,
            "supports_short": self.supports_short,
            "params": dict(self.params),
            "stop_spec": {
                "type": self.stop_spec.stop_type.value if hasattr(self.stop_spec.stop_type, "value") else str(self.stop_spec.stop_type),
                "atr_multiplier": self.stop_spec.atr_multiplier,
                "percent": self.stop_spec.percent,
                "swing_lookback": self.stop_spec.swing_lookback,
            },
            "tp_spec": {
                "mode": self.tp_spec.mode,
                "risk_reward_ratio": self.tp_spec.risk_reward_ratio,
            },
        }

    # ------------------------------------------------------------ param grid
    @classmethod
    def param_grid(cls) -> Dict[str, List[Any]]:
        """Default parameter grid for optimization (override in subclasses)."""
        return {}