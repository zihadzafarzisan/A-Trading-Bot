"""Strategy #10 V1.0 — Structural Liquidity Sweep & Mean Reversion.

Frozen-protocol implementation of ``PROTOCOL-STRATEGY-10-V1.0``
(``data/memo_strategy_10_v1_0_design.md``).

Specification baseline (do not change without a human-authorized protocol amendment)
------------------------------------------------------------------------------------
Core hypothesis: exploit institutional liquidity sweeps by **fading failed breakouts**
of N-bar swing highs/lows and capturing the mean-reversion snapback to the equilibrium
fair-value line (SMA-50 / Midpoint).

Entry engine (closed bar ``t``):
  - Swing levels strictly over *prior* lookback bars ``[t - lookback, t - 1]``:
      Swing_High_t = max(High[t-lookback .. t-1])     Shift(1) + rolling max (no look-ahead)
      Swing_Low_t  = min(Low [t-lookback .. t-1])     Shift(1) + rolling min (no look-ahead)
  - Sweep trigger:
      Bearish sweep (Short) : High_t > Swing_High_t
      Bullish sweep (Long)  : Low_t  < Swing_Low_t
  - Wick exhaustion ratios (``Delta_t = High_t - Low_t``, ``FLAT_RANGE_EPS=1e-8`` guard):
      W_upper,t = High_t - max(Open_t, Close_t)   R_upper,t = W_upper,t / Delta_t
      W_lower,t = min(Open_t, Close_t) - Low_t    R_lower,t = W_lower,t / Delta_t
      Short requires R_upper >= theta;  Long requires R_lower >= theta.
  - Strict range containment: Close_t < Swing_High_t (Short); Close_t > Swing_Low_t (Long).
  - Body-displacement asymmetry: Close_t <= Low_t + 0.5*Delta_t (Short);
                                  Close_t >= Low_t + 0.5*Delta_t (Long).
  - Minimum dynamic range (micro-doji suppression): Delta_t >= min_range_atr_mult * ATR(14)_t.
  - Fail-closed warm-up: no signal for ``t < max(lookback, 50, 14) + 1``; any NaN / non-finite /
    zero / short baseline fails closed to no-signal. Execution at ``open_{t+1}``.

Dynamic exit (V1.0):
  - Stop-loss beyond the sweep-bar extreme with adverse buffer:
      Short Stop_0 = High_t * (1 + 0.0010) ;   Long Stop_0 = Low_t * (1 - 0.0010)
  - Take-profit targets the equilibrium fair-value line:
      SMA50_t  (50-period SMA of close)  ||  Midpoint_t = (Swing_High_t + Swing_Low_t)/2
  - Breakeven ratchet (F-02): armed once unrealized PnL >= +1.0R (bar close), latched
    permanently, level ``Fill * (1 +/- 0.0020)`` covering round-trip fees (0.08%) +
    adverse slippage (0.10%) + funding drag (approx 0.02%). Enforced monotonic (never loosens).
  - Collision (invariant 5): same-bar High >= TP and Low <= Stop resolves STOP-FIRST
    (delegated to the frozen engine's ``resolve_collision``).

36-candidate grid (C00..C35): lookback x exhaustion_ratio x min_range_atr_mult x exit_mode.
  C00 anchor = (lookback=48, theta=0.45, min_range_atr_mult=0.5, exit_mode=SMA50).

DATA FIREWALL: no raw market data is required or accessed. All arithmetic is
deterministic and testable on synthetic fixtures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseStrategy, Direction

# --------------------------------------------------------------------------- #
# Frozen protocol constants
# --------------------------------------------------------------------------- #
WARM_UP_SMA: int = 50              # SMA(50) equilibrium line period
ATR_PERIOD: int = 14               # ATR(14) Wilder-RMA
STOP_BUFFER: float = 0.0010        # Stop_0 adverse buffer beyond the sweep extreme
BREAKEVEN_BUFFER: float = 0.0020   # Fill*(1 +/- 0.0020) ratchet level (F-02)
BREAKEVEN_TRIGGER_R: float = 1.0   # arm the breakeven ratchet at +1.0R
FLAT_RANGE_EPS: float = 1e-8       # flat / zero-range guard

# Grid domains
LOOKBACKS: Tuple[int, ...] = (24, 48, 72)
EXHAUSTION_RATIOS: Tuple[float, ...] = (0.35, 0.45, 0.55)
MIN_RANGE_ATR_MULTS: Tuple[float, ...] = (0.5, 0.75)
EXIT_MODES: Tuple[str, ...] = ("SMA50", "MIDPOINT")

_C00_ANCHOR: Tuple[int, float, float, str] = (48, 0.45, 0.5, "SMA50")


# --------------------------------------------------------------------------- #
# Pure entry-math helpers
# --------------------------------------------------------------------------- #
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range. Bar 0 has no prior close -> TR_0 = High_0 - Low_0."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    if len(tr) > 0:
        tr.iloc[0] = float(high.iloc[0] - low.iloc[0])
    return tr


def wilders_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = ATR_PERIOD) -> pd.Series:
    """``ATR(period)`` via Wilder's RMA: ``Sm[i] = Sm[i-1] + (TR - Sm[i-1])/period``.

    ``ewm(alpha=1/period, adjust=False, min_periods=period)`` matchifies the recursive
    Wilder recurrence while staying NaN until ``period`` bars (fail-closed warm-up).
    """
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def swing_high(high: pd.Series, lookback: int) -> pd.Series:
    """Current-bar-excluded trailing max high over the *prior* ``lookback`` bars.

    ``shift(1)`` then ``rolling(lookback).max`` -> value at ``t`` = ``max(High[t-lb .. t-1])``.
    """
    return high.shift(1).rolling(lookback, min_periods=lookback).max()


def swing_low(low: pd.Series, lookback: int) -> pd.Series:
    """Current-bar-excluded trailing min low over the *prior* ``lookback`` bars."""
    return low.shift(1).rolling(lookback, min_periods=lookback).min()


def warm_up_for(lookback: int) -> int:
    """Fail-closed bound (invariant 4): ``t < max(lookback, 50, 14) + 1``."""
    return max(int(lookback), WARM_UP_SMA, ATR_PERIOD) + 1


def safe_ratio(num: pd.Series, den: pd.Series, eps: float = FLAT_RANGE_EPS) -> pd.Series:
    """Vectorized safe division; flat / zero / negative ``den`` yield ``0.0`` (fail-closed)."""
    num_n = num.to_numpy(dtype=np.float64)
    den_n = den.to_numpy(dtype=np.float64)
    out = np.zeros_like(den_n, dtype=np.float64)
    np.divide(num_n, den_n, out=out, where=den_n > eps)
    return pd.Series(out, index=den.index)


# --------------------------------------------------------------------------- #
# Pure exit-math helpers
# --------------------------------------------------------------------------- #
def stop_0(bar_high: float, bar_low: float, direction: Direction,
           buffer: float = STOP_BUFFER) -> float:
    """Initial stop beyond the sweep-bar extreme with adverse buffer (protocol exit)."""
    if direction == Direction.LONG:
        return float(bar_low) * (1.0 - buffer)
    return float(bar_high) * (1.0 + buffer)


def take_profit_level(sma50: float, midpoint: float, exit_mode: str) -> float:
    """Equilibrium fair-value target: SMA50 or Midpoint (swept by exit_mode)."""
    if exit_mode == "SMA50":
        return float(sma50)
    return float(midpoint)


def breakeven_level(fill: float, direction: Direction, buffer: float = BREAKEVEN_BUFFER) -> float:
    """Ratchet level ``Fill * (1 +/- buffer)`` (long/short symmetric; F-02)."""
    if direction == Direction.LONG:
        return float(fill) * (1.0 + buffer)
    return float(fill) * (1.0 - buffer)


def ratchet_stop(prior_stop: float, rbe: float, direction: Direction) -> float:
    """Monotonic breakeven ratchet: long non-decreasing, short non-increasing.

    ``Stop_t = max(Stop_{t-1}, RBE)`` (long) / ``min(Stop_{t-1}, RBE)`` (short) — the ratchet
    can never loosen a stop already in force.
    """
    if direction == Direction.LONG:
        return max(prior_stop, rbe)
    return min(prior_stop, rbe)


def cost_covered(rbe_level: float, fill: float, direction: Direction) -> float:
    """Net protected gain at the ratchet level (positive => fees/slip/funding covered)."""
    if direction == Direction.LONG:
        return rbe_level - fill
    return fill - rbe_level


# --------------------------------------------------------------------------- #
# 36-candidate grid (lookback x theta x min_range x exit_mode)
# --------------------------------------------------------------------------- #
def candidate_grid() -> List[Dict[str, Any]]:
    """Return exactly 36 orthogonal parameter configs C00..C35.

    Covers the full ``3 x 3 x 2 x 2 = 36`` Cartesian product. The declared C00 anchor
    ``(48, 0.45, 0.5, SMA50)`` is emitted first, followed by the remaining 35 tuples in
    canonical declaration order (no gaps, no duplicates).
    """
    def cross() -> List[Tuple[int, float, float, str]]:
        return [
            (lb, th, mr, ex)
            for lb in LOOKBACKS
            for th in EXHAUSTION_RATIOS
            for mr in MIN_RANGE_ATR_MULTS
            for ex in EXIT_MODES
        ]

    flat = cross()
    ordered = [_C00_ANCHOR] + [t for t in flat if t != _C00_ANCHOR]
    grid: List[Dict[str, Any]] = []
    for i, (lb, th, mr, ex) in enumerate(ordered):
        grid.append({
            "id": "C%02d" % i,
            "lookback": lb,
            "exhaustion_ratio": th,
            "min_range_atr_mult": mr,
            "exit_mode": ex,
        })
    return grid


CANDIDATE_GRID: List[Dict[str, Any]] = candidate_grid()
CANDIDATE_TUPLES: List[Dict[str, Any]] = CANDIDATE_GRID


# --------------------------------------------------------------------------- #
# Strategy class
# --------------------------------------------------------------------------- #
class LiquiditySweepMeanReversionStrategy(BaseStrategy):
    """Structural Liquidity Sweep & Mean Reversion (PROTOCOL-STRATEGY-10-V1.0)."""

    name = "liquidity_sweep_mean_reversion"
    strategy_type = "liquidity_sweep_mean_reversion"
    description = (
        "Fade failed breakouts of N-bar swing highs/lows on a liquidity sweep; exhausting "
        "wick rejection filtering; ATR-14 micro-doji suppression; mean-reversion snapback "
        "to the SMA50/Midpoint equilibrium line with a fill-based stop and +1.0R breakeven "
        "ratchet covering round-trip friction."
    )
    supports_spot = True
    supports_futures = True
    supports_short = True

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        defaults = {
            "lookback": 48,
            "exhaustion_ratio": 0.45,
            "min_range_atr_mult": 0.5,
            "exit_mode": "SMA50",
        }
        merged = dict(defaults)
        merged.update(params or {})
        super().__init__(merged, **kwargs)

    # ---- parameter accessors ----
    @property
    def lookback(self) -> int:
        return int(self.params["lookback"])

    @property
    def exhaustion_ratio(self) -> float:
        return float(self.params["exhaustion_ratio"])

    @property
    def min_range_atr_mult(self) -> float:
        return float(self.params["min_range_atr_mult"])

    @property
    def exit_mode(self) -> str:
        return str(self.params["exit_mode"])

    @property
    def warm_up(self) -> int:
        return warm_up_for(self.lookback)

    # ---- validation ----
    def validate_params(self) -> None:
        p = self.params
        if int(p["lookback"]) not in LOOKBACKS:
            raise ValueError(f"lookback must be in {LOOKBACKS}")
        if float(p["exhaustion_ratio"]) not in EXHAUSTION_RATIOS:
            raise ValueError(f"exhaustion_ratio must be in {EXHAUSTION_RATIOS}")
        if float(p["min_range_atr_mult"]) not in MIN_RANGE_ATR_MULTS:
            raise ValueError(f"min_range_atr_mult must be in {MIN_RANGE_ATR_MULTS}")
        if str(p["exit_mode"]) not in EXIT_MODES:
            raise ValueError(f"exit_mode must be in {EXIT_MODES}")

    # ---- fail-closed finite guard ----
    @staticmethod
    def _finite(x) -> bool:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return False
        return np.isfinite(v)

    # ---- indicator columns (no look-ahead) ----
    def setup(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        lb = self.lookback
        theta = self.exhaustion_ratio
        mr_mult = self.min_range_atr_mult

        # Swing levels strictly over prior bars [t-lookback .. t-1] (shift(1) enforced).
        out["swing_high"] = swing_high(out["high"], lb)
        out["swing_low"] = swing_low(out["low"], lb)

        # ATR(14) Wilder-RMA and the SMA(50) equilibrium line.
        out["atr_14"] = wilders_atr(out["high"], out["low"], out["close"], ATR_PERIOD)
        out["sma_50"] = out["close"].rolling(WARM_UP_SMA, min_periods=WARM_UP_SMA).mean()

        # Midpoint fair-value target from the (past-only) swing levels.
        out["midpoint"] = (out["swing_high"] + out["swing_low"]) * 0.5

        # Candle components.
        out["body_top"] = out[["open", "close"]].max(axis=1)
        out["body_bot"] = out[["open", "close"]].min(axis=1)
        out["total_range"] = out["high"] - out["low"]
        out["upper_wick"] = out["high"] - out["body_top"]
        out["lower_wick"] = out["body_bot"] - out["low"]

        # Wick rejection ratios, vectorized with safe division (flat -> 0.0).
        out["upper_wick_ratio"] = safe_ratio(out["upper_wick"], out["total_range"])
        out["lower_wick_ratio"] = safe_ratio(out["lower_wick"], out["total_range"])

        # Dynamic minimum-range (micro-doji) filter and warm-up mask.
        range_ok = (
            (out["total_range"] > FLAT_RANGE_EPS)
            & (out["total_range"] >= mr_mult * out["atr_14"])
            & (out["atr_14"] > 0)
        )
        finite_ok = (
            np.isfinite(out["swing_high"].to_numpy(dtype=np.float64))
            & np.isfinite(out["swing_low"].to_numpy(dtype=np.float64))
            & np.isfinite(out["close"].to_numpy(dtype=np.float64))
            & np.isfinite(out["low"].to_numpy(dtype=np.float64))
            & np.isfinite(out["atr_14"].to_numpy(dtype=np.float64))
        )

        # Long — bullish sweep of the prior swing low, then rejection back inside.
        long_sweep = out["low"] < out["swing_low"]
        long_wick = out["lower_wick_ratio"] >= theta
        long_contain = out["close"] > out["swing_low"]
        long_upper = out["close"] >= out["low"] + 0.5 * out["total_range"]
        long_sig = finite_ok & range_ok & long_sweep & long_wick & long_contain & long_upper

        # Short — bearish sweep of the prior swing high, then rejection back inside.
        short_sweep = out["high"] > out["swing_high"]
        short_wick = out["upper_wick_ratio"] >= theta
        short_contain = out["close"] < out["swing_high"]
        short_lower = out["close"] <= out["low"] + 0.5 * out["total_range"]
        short_sig = finite_ok & range_ok & short_sweep & short_wick & short_contain & short_lower

        # Vectorized entry-signal column: 1 (Long) / -1 (Short) / 0 (Neutral).
        sig = np.where(long_sig.to_numpy(), 1.0,
                       np.where(short_sig.to_numpy(), -1.0, 0.0))
        warm = self.warm_up
        if len(sig) > 0:
            sig[:warm] = 0.0
        out["entry_signal"] = sig

        return out

    # ---- scalar entry conditions (single source of truth for entry_signal()) ----
    def _long_ok(self, df: pd.DataFrame, i: int) -> bool:
        if i < self.warm_up:
            return False
        low = df["low"].iloc[i]
        close = df["close"].iloc[i]
        swing_lo = df["swing_low"].iloc[i]
        delta = df["total_range"].iloc[i]
        atr = df["atr_14"].iloc[i]
        lwr = df["lower_wick_ratio"].iloc[i]
        if not (self._finite(swing_lo) and self._finite(delta) and self._finite(atr)):
            return False
        if delta <= FLAT_RANGE_EPS or atr <= 0:
            return False
        if not (delta >= self.min_range_atr_mult * atr):   # micro-doji suppression
            return False
        sweep = low < swing_lo
        wick = lwr >= self.exhaustion_ratio
        contain = close > swing_lo
        upper = close >= low + 0.5 * delta
        return sweep and wick and contain and upper

    def _short_ok(self, df: pd.DataFrame, i: int) -> bool:
        if i < self.warm_up:
            return False
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]
        close = df["close"].iloc[i]
        swing_hi = df["swing_high"].iloc[i]
        delta = df["total_range"].iloc[i]
        atr = df["atr_14"].iloc[i]
        uwr = df["upper_wick_ratio"].iloc[i]
        if not (self._finite(swing_hi) and self._finite(delta) and self._finite(atr)):
            return False
        if delta <= FLAT_RANGE_EPS or atr <= 0:
            return False
        if not (delta >= self.min_range_atr_mult * atr):
            return False
        sweep = high > swing_hi
        wick = uwr >= self.exhaustion_ratio
        contain = close < swing_hi
        lower = close <= low + 0.5 * delta
        return sweep and wick and contain and lower

    def entry_signal(self, df: pd.DataFrame, i: int, ctx: Optional[Dict] = None
                     ) -> Tuple[Direction, str]:
        """Entry direction at the close of bar ``i`` (uses only bars ``<= i``)."""
        if self._long_ok(df, i):
            return Direction.LONG, "sweep_reject_long"
        if self._short_ok(df, i):
            return Direction.SHORT, "sweep_reject_short"
        return Direction.NONE, ""

    # ---- stop / take ----
    def compute_stop_loss(self, df, i, direction, entry_price):
        """Initial stop beyond the sweep-bar extreme with adverse buffer."""
        if not np.isfinite(float(df["high"].iloc[i])) or not np.isfinite(float(df["low"].iloc[i])):
            return None
        return stop_0(float(df["high"].iloc[i]), float(df["low"].iloc[i]), direction)

    def compute_take_profit(self, df, i, direction, entry_price, stop_loss=None):
        """Take-profit is managed via ``prepare_fill``/``update_stop`` state, not here."""
        return None

    # ---- lifecycle: fill anchoring ----
    def prepare_fill(self, df: pd.DataFrame, signal_bar_index: int,
                     side: Direction, fill_price: float = None,
                     sig=None) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
        """Fill-anchored initial stop, dynamic take-profit target, and ratchet state.

        Returns ``(stop, tp, state)`` where:
          - ``stop``  = the initial ``Stop_0`` beyond the sweep extreme (adverse buffer).
          - ``tp``    = the equilibrium target (SMA50 or Midpoint) evaluated at the signal bar.
          - ``state`` = a dict seeding the +1.0R breakeven ratchet (``breakeven_armed=False``,
                        ``entry_price``, ``initial_r``, ``prior_stop``).
        """
        direction = Direction(side) if isinstance(side, str) else side
        hi = float(df["high"].iloc[signal_bar_index])
        lo = float(df["low"].iloc[signal_bar_index])
        if not (np.isfinite(hi) and np.isfinite(lo)):
            return None, None, {}
        stop = stop_0(hi, lo, direction)
        if fill_price is None:
            fill_price = float(df["open"].iloc[signal_bar_index + 1]) \
                if signal_bar_index + 1 < len(df) else float(df["close"].iloc[signal_bar_index])
        fill = float(fill_price)

        R = abs(fill - stop)
        swing_hi = float(df["swing_high"].iloc[signal_bar_index])
        swing_lo = float(df["swing_low"].iloc[signal_bar_index])
        sma50 = float(df["sma_50"].iloc[signal_bar_index])
        midpoint = float(df["midpoint"].iloc[signal_bar_index])
        tp = take_profit_level(sma50, midpoint, self.exit_mode)

        # §9.1 target-polarity fail-closed (FROZEN): the equilibrium target must lie on
        # the *profit* side of the fill or the entry is aborted. For a Long the target
        # must exceed the fill; for a Short it must sit below the fill. SMA50 (trend) and
        # Midpoint (range) can each land on the wrong side of the fill in a given regime;
        # a non-monotonic target is unreachable as profit and must fail closed to no-trade
        # rather than emit a target that truncates the position.
        if direction == Direction.LONG and float(fill) >= tp:
            return None, None, {}
        if direction == Direction.SHORT and float(fill) <= tp:
            return None, None, {}

        state = {
            "fill_price": fill,
            "entry_price": fill,
            "initial_r": float(R),
            "prior_stop": float(stop),
            "stop_direction": direction.value,
            "breakeven_armed": False,
            "break_even_level": float(breakeven_level(fill, direction)),
            "take_profit": float(tp),
            "signal_bar": int(signal_bar_index),
            "swing_high": swing_hi,
            "swing_low": swing_lo,
            "exit_mode": self.exit_mode,
        }
        return stop, tp, state

    # ---- lifecycle: dynamic stop (breakeven ratchet) ----
    def update_stop(self, df: pd.DataFrame, current_bar_index: int, position,
                    state: Optional[Dict[str, Any]] = None) -> None:
        """Arm and apply the +1.0R breakeven ratchet on each closed bar after entry.

        Evaluates the completed bar ``current_bar_index - 1`` (the bar whose close just
        printed) so no look-ahead occurs. Once realized-beyond ``+1.0R`` the ratchet latches
        permanently to ``Fill * (1 +/- 0.0020)`` and the effective stop is enforced monotonic
        (long non-decreasing / short non-increasing) — it can never loosen.
        """
        st = state if state is not None else getattr(position, "extra_state", None)
        if not st or st.get("entry_price") is None or st.get("initial_r") is None:
            return
        direction_val = st.get("stop_direction", getattr(position, "direction", "long"))
        direction = Direction(direction_val) if isinstance(direction_val, str) else direction_val
        entry = float(st["entry_price"])
        R = float(st["initial_r"])
        rbe = float(st.get("break_even_level", breakeven_level(entry, direction)))
        armed = bool(st.get("breakeven_armed", False))
        prior = float(st.get("prior_stop", stop_0(
            float(df["high"].iloc[max(0, current_bar_index - 1)]),
            float(df["low"].iloc[max(0, current_bar_index - 1)]),
            direction)))

        j = current_bar_index - 1          # the completed bar driving the decision
        if j < int(st.get("signal_bar", j)) + 1:
            # no bar has fully printed after fill yet — hold the initial stop
            if hasattr(position, "active_stop"):
                position.active_stop = prior
            return

        if j < len(df):
            close = float(df["close"].iloc[j])
            if direction == Direction.LONG:
                unreal = close - entry
            else:
                unreal = entry - close
            if not armed and unreal >= BREAKEVEN_TRIGGER_R * R:
                armed = True

        new_stop = ratchet_stop(prior, rbe, direction) if armed else prior
        st["prior_stop"] = float(new_stop)
        st["breakeven_armed"] = bool(armed)
        if hasattr(position, "active_stop"):
            position.active_stop = new_stop

    # ---- param grid (protocol §4) ----
    @classmethod
    def param_grid(cls) -> Dict[str, List[Any]]:
        return {
            "lookback": list(LOOKBACKS),
            "exhaustion_ratio": list(EXHAUSTION_RATIOS),
            "min_range_atr_mult": list(MIN_RANGE_ATR_MULTS),
            "exit_mode": list(EXIT_MODES),
        }


def fill_price(open_next: float, direction: Direction) -> float:
    """Fill at ``open_{t+1}`` (signal at close of ``t`` -> execute next-bar open)."""
    return float(open_next)