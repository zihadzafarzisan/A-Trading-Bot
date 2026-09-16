"""Strategy #9 V1.1 — Volatility Compression Breakout with Chandelier Trailing & Breakeven Ratchet.

Frozen-protocol implementation of ``PROTOCOL-STRATEGY-09-V1.1``
(fingerprint SHA-256 ``465cb1a43158acce4488777fcbce48b80abf630350ba37017fdd0f70f5e09e34``,
``data/protocol_strategy_09_v1_1.md``).

Specification baseline (do not change without a human-authorized protocol amendment)
-------------------------------------------------------------------------------------
Entry engine (preserved *verbatim* from V1.0 — this protocol freezes it unchanged):
  - Compression : ``BBWidth_t = 4*sd20_t / SMA20_t`` (population ddof=0) <= Type-7
    linear-interpolation ``q``-percentile of a *past-only* 120-bar baseline.
  - Breakout    : strict ``>``/``<`` cross of ``close_t`` against a *current-bar-
    excluded* trailing High/Low window of length ``breakout_lookback``. No extra
    freshness gate (matches V1.0 ``long_break_at``/``short_break_at``).
  - Expansion   : ``ATR_t = SMA(TR,14)`` >= ``expansion_multiplier * avgATR_t``,
    where ``avgATR_t`` is the mean over a *past-only* 10-bar baseline and
    ``expansion_multiplier = 1.1`` is a frozen single value (NOT the risk ATR multiple).
  - Trend filter: SMA-seeded ``EMA20 > EMA50`` (long) / ``EMA20 < EMA50`` (short).
  - Fail-closed : no signal for ``t < 139`` (0-indexed); any invalid / short / zero /
    non-finite baseline fails the respective condition to no-signal.
  - Execution   : signal at close of ``t``; fill at ``open_{t+1}`` with slip
    ``open*(1+0.0005)`` (long) / ``open*(1-0.0005)`` (short).

Dynamic exit (V1.1 redesign):
  - Take-profit removed (uncapped right tail).
  - Initial risk : ``Stop0 = Fill - a*ATR_{t0}`` (long) / ``Fill + a*ATR_{t0}`` (short).
  - Breakeven ratchet (F-02): armed at +1.5R unrealized (bar close), latched
    permanently, level ``Fill * (1 +/- 0.0020)`` (price-level invariant fraction).
  - Chandelier trail: ``CH_t = HH(window) - trail_atr_mult * ATR_t`` (long) /
    ``CH_t = LL(window) + trail_atr_mult * ATR_t`` (short), window from entry bar to ``t``.
  - STRICT RUNNING-MAX (F-01): long ``Stop_t = max(Stop_{t-1}, RBE, CH_t)``,
    short ``Stop_t = min(Stop_{t-1}, RBE, CH_t)`` — carried forward so the stop
    never recedes, even under an ATR spike.
  - Collision: conservative stop-first; gap fills at the adverse of open vs level.

This module ships the 36-candidate enumeration utility (C00..C35) and pure,
unit-testable helpers for the entry signal and the exit mechanics. It is deliberately
NOT registered into the central ``STRATEGY_REGISTRY``; registration (if desired) is a
separate, human-authorized integration step.

DATA FIREWALL: no raw market data is required or accessed. All arithmetic is
deterministic and testable on synthetic fixtures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseStrategy, Direction

# --------------------------------------------------------------------------- #
# Frozen protocol constants (protocol §3 / §4 / §5)
# --------------------------------------------------------------------------- #
WARM_UP: int = 139                     # 0-indexed first bar a signal may fire on
BB_PERIOD: int = 20                    # Bollinger window for compression
NUM_STD: float = 2.0                   # BB half-width multiplier (width = 4*sd)
SQUEEZE_BASELINE: int = 120            # past-only baseline for the q-percentile
ATR_PERIOD: int = 14                   # SMA(TR,14)
EXPANSION_BASELINE: int = 10           # past-only baseline for avgATR
EMA_FAST: int = 20
EMA_SLOW: int = 50
BREAKEVEN_TRIGGER_R: float = 1.5       # arm the ratchet at +1.5R unrealized
BREAKEVEN_BUFFER: float = 0.0020       # Fill*(1 +/- 0.0020) ratchet level (F-02)
SLIP_RATE: float = 0.0005              # adverse slippage per side (fill formula)

# Grid domains (protocol §5b)
BREAKOUT_LOOKBACKS = (20, 40)
SQUEEZE_PERCENTILES = (30, 40)
ATR_MULTIPLIERS = (2.0, 2.5, 3.0)
EXPANSION_MULTIPLIER: float = 1.1      # frozen, single value (protocol §5a)
EXIT_CONFIGS: Dict[str, Tuple[int, float]] = {          # lookback_trail -> trail_atr_mult
    "X1": (20, 3.0),
    "X2": (10, 3.0),
    "X3": (20, 2.0),
}
DOMINATED_EXIT = (10, 2.0)             # excluded (fastest whipsaw-prone pair)

_C00_ANCHOR = (20, 30, EXPANSION_MULTIPLIER, 2.5, "X1")   # protocol §5d


# --------------------------------------------------------------------------- #
# Pure entry-math helpers (V1.0 verbatim where the protocol freezes them)
# --------------------------------------------------------------------------- #
def type7_percentile(values: np.ndarray, q: float) -> float:
    """Hyndman-Fan Type 7 linear-interpolation quantile of ``values``.

    Matches ``numpy.quantile(..., method='linear')``. Implemented explicitly so the
    pipeline never depends on a library default shifting between versions.
    """
    a = np.sort(np.asarray(values, dtype=np.float64))
    n = len(a)
    if n == 0:
        return float("nan")
    h = (n - 1) * q
    lo = int(np.floor(h))
    hi = int(np.ceil(h))
    if lo == hi:
        return float(a[lo])
    frac = h - lo
    return float(a[lo] * (1.0 - frac) + a[hi] * frac)


def bb_width(close: pd.Series, period: int = BB_PERIOD, num_std: float = NUM_STD) -> pd.Series:
    """``BBWidth = (2*num_std*sd20)/SMA20`` with *population* std (ddof=0).

    With ``num_std=2.0`` the width equals ``4*sd20/SMA20`` as specified (V1.0 §4.1).
    Zero/negative ``SMA20`` leaves NaN so callers can fail closed.
    """
    sma = close.rolling(period, min_periods=period).mean()
    sd = close.rolling(period, min_periods=period).std(ddof=0)
    width = (2.0 * num_std * sd) / sma
    return width.where(np.isfinite(width))


def past_only_rolling_quantile(series: pd.Series, window: int, q: float) -> pd.Series:
    """Type-7 ``q``-percentile of the *past-only* ``window`` values before each bar.

    The current bar is excluded by shifting the series forward one row before the
    rolling summary, so a decision at bar ``t`` never reads bar ``t`` (V1.0 §4.2).
    """
    return (
        series.shift(1)
        .rolling(window, min_periods=window)
        .apply(lambda w: type7_percentile(np.asarray(w, dtype=np.float64), q), raw=True)
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range. Bar 0 has no prior close -> TR_0 = High_0 - Low_0 (V1.0 §4.4)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    if len(tr) > 0:
        tr.iloc[0] = float(high.iloc[0] - low.iloc[0])
    return tr


def sma_true_range_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                       period: int = ATR_PERIOD) -> pd.Series:
    """``ATR = SMA(TR, period)`` — simple moving average of true range, NOT Wilder-RMA.

    The value at bar ``t`` uses only bars ``<= t`` (closed-bar quantity, no look-ahead).
    """
    return true_range(high, low, close).rolling(period, min_periods=period).mean()


def past_only_mean(series: pd.Series, window: int) -> pd.Series:
    """Mean of the *past-only* ``window`` values before each bar (excludes current)."""
    return series.shift(1).rolling(window, min_periods=window).mean()


def sma_seeded_ema(close: pd.Series, period: int) -> pd.Series:
    """SMA-seeded recursive EMA (V1.0 §4.5).

    ``EMA_k = alpha*close_k + (1-alpha)*EMA_{k-1}`` with ``alpha = 2/(period+1)``,
    seeded from the simple mean of the first ``period`` closes. NaN until ``>= period``
    observations exist.
    """
    closes = close.to_numpy(dtype=float)
    n = len(closes)
    out = np.full(n, np.nan, dtype=float)
    if period <= 0 or n < period:
        return pd.Series(out, index=close.index)
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = float(np.mean(closes[:period]))
    for k in range(period, n):
        out[k] = alpha * float(closes[k]) + (1.0 - alpha) * out[k - 1]
    return pd.Series(out, index=close.index)


def breakout_window_extreme(df: pd.DataFrame, high_low: str, lookback: int) -> pd.Series:
    """Current-bar-excluded trailing window extreme (max high / min low)."""
    if high_low == "high":
        return df["high"].shift(1).rolling(lookback, min_periods=lookback).max()
    return df["low"].shift(1).rolling(lookback, min_periods=lookback).min()


# --------------------------------------------------------------------------- #
# Pure exit-math helpers (V1.1 redesign, protocol §4)
# --------------------------------------------------------------------------- #
def break_even_level(fill: float, direction: Direction, buffer: float = BREAKEVEN_BUFFER) -> float:
    """Ratchet level ``Fill * (1 +/- buffer)`` (long/short symmetric; F-02)."""
    if direction == Direction.LONG:
        return fill * (1.0 + buffer)
    return fill * (1.0 - buffer)


def chandelier_trail(direction: Direction, window_extreme: float, atr_t: float,
                     trail_atr_mult: float) -> float:
    """Raw Chandelier stop at bar ``t`` (protocol §4d).

    NOTE: this is the *raw* value (NOT yet monotonic). Combine it with the running-max
    ratchet in ``running_max_stop`` so an ATR spike cannot recede the effective stop.
    """
    if direction == Direction.LONG:
        return window_extreme - trail_atr_mult * atr_t
    return window_extreme + trail_atr_mult * atr_t


def running_max_stop(direction: Direction, prior: float, rbe_if_armed: float,
                     chandelier: float) -> float:
    """STRICT RUNNING-MAX effective stop (protocol §4e / finding F-01).

    Long  -> ``max(Stop_{t-1}, RBE_if_armed, CH_t)``  (always >= prior)
    Short -> ``min(Stop_{t-1}, RBE_if_armed, CH_t)`` (always <= prior)

    Because ``prior`` is carried forward, an ATR-driven drop of the raw Chandelier can
    never recede the effective stop.
    """
    if direction == Direction.LONG:
        return max(prior, rbe_if_armed, chandelier)
    return min(prior, rbe_if_armed, chandelier)


def resolve_collision(direction: Direction, open_price: float, high: float, low: float,
                      stop_level: float, alt_level: Optional[float] = None
                      ) -> Tuple[Optional[str], float]:
    """Conservative stop-first same-bar collision resolution (protocol §4f / invariant 3).

    Returns ``(outcome, fill_price)`` where ``outcome`` is ``"stop"`` (stopped out),
    ``"alt"`` (an alternative profit level touched), or ``None`` (no fill). On a
    simultaneous stop / competing-level touch the STOP wins (adverse/stop-first bias);
    a gap past the level fills at the worse of ``open`` vs the level.
    """
    if direction == Direction.LONG:
        stop_hit = low <= stop_level
        alt_hit = (alt_level is not None) and (high >= alt_level)
    else:
        stop_hit = high >= stop_level
        alt_hit = (alt_level is not None) and (low <= alt_level)

    if stop_hit:
        # gap adverse: if the bar gapped past the stop, fill at the (worse) open.
        if direction == Direction.LONG:
            fill = min(open_price, stop_level) if open_price < stop_level else stop_level
        else:
            fill = max(open_price, stop_level) if open_price > stop_level else stop_level
        return "stop", float(fill)
    if alt_hit:
        if direction == Direction.LONG:
            fill = max(open_price, alt_level) if open_price > alt_level else alt_level
        else:
            fill = min(open_price, alt_level) if open_price < alt_level else alt_level
        return "alt", float(fill)
    return None, float("nan")


# --------------------------------------------------------------------------- #
# 36-candidate grid (protocol §5b / §5c / §5d / §5e)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Candidate:
    """A single frozen candidate tuple of the 36-candidate grid."""
    id: str
    breakout_lookback: int
    squeeze_percentile: int
    expansion_multiplier: float
    atr_multiplier: float
    exit_config: str
    lookback_trail: int
    trail_atr_mult: float

    @property
    def params(self) -> Dict[str, Any]:
        return asdict(self)


def candidate_grid() -> List[Candidate]:
    """Return the canonical 36 candidates in declaration order C00..C35.

    Ordering follows the frozen protocol table: C00 is the anchor
    ``(20, 30, 1.1, 2.5, X1)`` first, then the remaining 35 tuples covering the full
    5-dimensional cross (lb x sq x exp x atr x exit) with no gaps or duplicates.
    The rows are grouped by ``breakout_lookback`` then ``squeeze_percentile``, and
    within a group by ``atr_multiplier`` then ``exit_config``.
    """
    def cross(lb, sq):
        return [
            (lb, sq, EXPANSION_MULTIPLIER, atr_m, ex)
            for atr_m in ATR_MULTIPLIERS
            for ex in ("X1", "X2", "X3")
        ]

    groups = [cross(lb, sq) for lb in BREAKOUT_LOOKBACKS for sq in SQUEEZE_PERCENTILES]
    flat = [t for g in groups for t in g]
    # C00 anchor is not lexicographically first (its atr=2.5 precedes the 2.0 rows),
    # so place the anchor first, then the remaining rows in declared order.
    ordered = [_C00_ANCHOR] + [t for t in flat if t != _C00_ANCHOR]
    cands = []
    for i, (lb, sq, exp, atr_m, ex) in enumerate(ordered):
        lt, tam = EXIT_CONFIGS[ex]
        cands.append(Candidate(
            id="C%02d" % i, breakout_lookback=lb, squeeze_percentile=sq,
            expansion_multiplier=exp, atr_multiplier=atr_m,
            exit_config=ex, lookback_trail=lt, trail_atr_mult=tam,
        ))
    return cands


CANDIDATE_GRID: List[Candidate] = candidate_grid()
CANDIDATE_TUPLES: Tuple[Candidate, ...] = tuple(CANDIDATE_GRID)


# --------------------------------------------------------------------------- #
# Dynamic exit manager (V1.1)
# --------------------------------------------------------------------------- #
class V1_1ExitManager:
    """Stateful per-position dynamic exit: combines initial ATR stop, +1.5R
    breakeven ratchet (latched to ``Fill*(1 +/- 0.0020)``), and a Chandelier trail
    into a single strict running-max effective stop (protocol F-01 / F-02).
    """

    def __init__(self, direction: Direction, fill: float, R: float, entry_bar: int,
                 lookback_trail: int, trail_atr_mult: float,
                 breakeven_buffer: float = BREAKEVEN_BUFFER):
        if direction not in (Direction.LONG, Direction.SHORT):
            raise ValueError("exit manager requires a LONG or SHORT position")
        self.direction = direction
        self.fill = float(fill)
        self.R = float(R)
        self.entry_bar = int(entry_bar)
        self.lookback_trail = int(lookback_trail)
        self.trail_atr_mult = float(trail_atr_mult)
        self.breakeven_buffer = float(breakeven_buffer)
        # Protocol §4b base case: Stop_{t0+1} = Stop0.
        self.stop0 = (self.fill - self.R) if direction == Direction.LONG else (self.fill + self.R)
        self.prior_stop = self.stop0
        self.rbe = break_even_level(self.fill, self.direction, self.breakeven_buffer)
        self.armed = False

    # ---- latching ----
    def _rbe_if_armed(self) -> float:
        if not self.armed:
            return -np.inf if self.direction == Direction.LONG else np.inf
        return self.rbe

    def _check_arm(self, close: float) -> None:
        """Arm (permanently) once unrealized PnL reaches +1.5R at a bar close (F-02)."""
        if self.armed:
            return
        if self.direction == Direction.LONG:
            unrealized = close - self.fill
        else:
            unrealized = self.fill - close
        if unrealized >= BREAKEVEN_TRIGGER_R * self.R:
            self.armed = True

    # ---- chandelier window ----
    def _window_extreme(self, df: pd.DataFrame, t: int) -> float:
        start = self.entry_bar
        if self.direction == Direction.LONG:
            return float(df["high"].iloc[start:t + 1].max())
        return float(df["low"].iloc[start:t + 1].min())

    # ---- scan one closed bar ----
    def scan_bar(self, df: pd.DataFrame, t: int) -> float:
        """Advance the running-max stop using closed bar ``t``.

        Uses only closed bars ``<= t`` and carries ``self.prior_stop`` forward so the
        effective stop is strictly monotone (F-01). Returns the effective stop.
        """
        window_extreme = self._window_extreme(df, t)
        atr_t = float(df["atr_14"].iloc[t])
        raw_ch = chandelier_trail(self.direction, window_extreme, atr_t, self.trail_atr_mult)
        self._check_arm(float(df["close"].iloc[t]))
        self.prior_stop = running_max_stop(
            self.direction, self.prior_stop, self._rbe_if_armed(), raw_ch
        )
        return self.prior_stop

    def effective_stop(self) -> float:
        """The strict running-max stop currently in force (to govern the next bar)."""
        return self.prior_stop


# --------------------------------------------------------------------------- #
# V1.1 strategy class
# --------------------------------------------------------------------------- #
class VolatilityCompressionBreakoutV1_1(BaseStrategy):
    """Volatility Compression Breakout with Chandelier trailing & breakeven ratchet
    (frozen PROTOCOL-STRATEGY-09-V1.1)."""

    name = "volatility_compression_breakout_v1_1"
    strategy_type = "volatility_compression_breakout"
    description = (
        "Frozen PROTOCOL-STRATEGY-09-V1.1: BB-width compression + breakout + ATR "
        "expansion(1.1) + SMA-seeded EMA20/50 filter entry; Chandelier running-max "
        "trail and Fill*(1+/-0.0020) breakeven ratchet exit (no take-profit)."
    )
    supports_spot = True
    supports_futures = True
    supports_short = True

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        defaults = {
            "breakout_lookback": 20,
            "squeeze_percentile": 30,
            "expansion_multiplier": EXPANSION_MULTIPLIER,
            "atr_multiplier": 2.5,
            "exit_config": "X1",
            "lookback_trail": None,     # derived from exit_config if unset
            "trail_atr_mult": None,     # derived from exit_config if unset
        }
        merged = dict(defaults)
        merged.update(params or {})
        if merged.get("exit_config") not in EXIT_CONFIGS:
            raise ValueError(
                f"unknown exit_config {merged.get('exit_config')!r}; "
                f"expected {list(EXIT_CONFIGS)}"
            )
        if merged.get("lookback_trail") is None or merged.get("trail_atr_mult") is None:
            lt, tam = EXIT_CONFIGS[merged["exit_config"]]
            merged["lookback_trail"] = lt
            merged["trail_atr_mult"] = tam
        super().__init__(merged, **kwargs)

    # ---- parameter accessors ----
    @property
    def breakout_lookback(self) -> int:
        return int(self.params["breakout_lookback"])

    @property
    def squeeze_percentile(self) -> int:
        return int(self.params["squeeze_percentile"])

    @property
    def atr_multiplier(self) -> float:
        return float(self.params["atr_multiplier"])

    @property
    def lookback_trail(self) -> int:
        return int(self.params["lookback_trail"])

    @property
    def trail_atr_mult(self) -> float:
        return float(self.params["trail_atr_mult"])

    # ---- validation ----
    def validate_params(self) -> None:
        p = self.params
        if p["breakout_lookback"] not in BREAKOUT_LOOKBACKS:
            raise ValueError(f"breakout_lookback must be in {BREAKOUT_LOOKBACKS}")
        if p["squeeze_percentile"] not in SQUEEZE_PERCENTILES:
            raise ValueError(f"squeeze_percentile must be in {SQUEEZE_PERCENTILES}")
        if abs(float(p["expansion_multiplier"]) - EXPANSION_MULTIPLIER) > 1e-9:
            raise ValueError(f"expansion_multiplier must be {EXPANSION_MULTIPLIER} (frozen)")
        if float(p["atr_multiplier"]) not in ATR_MULTIPLIERS:
            raise ValueError(f"atr_multiplier must be in {ATR_MULTIPLIERS}")
        lt, tam = int(p["lookback_trail"]), float(p["trail_atr_mult"])
        if (lt, tam) == DOMINATED_EXIT:
            raise ValueError(f"exit pair {DOMINATED_EXIT} is excluded (dominated) per protocol")
        valid_known = any(EXIT_CONFIGS[k] == (lt, tam) for k in EXIT_CONFIGS)
        if not valid_known:
            raise ValueError(f"exit pair ({lt}, {tam}) does not correspond to X1/X2/X3")

    # ---- indicator columns (no look-ahead) ----
    def setup(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["bb_width"] = bb_width(out["close"], BB_PERIOD, NUM_STD)
        out["squeeze_q"] = past_only_rolling_quantile(
            out["bb_width"], SQUEEZE_BASELINE, self.squeeze_percentile / 100.0
        )
        out["atr_14"] = sma_true_range_atr(out["high"], out["low"], out["close"], ATR_PERIOD)
        out["avg_atr"] = past_only_mean(out["atr_14"], EXPANSION_BASELINE)
        out["ema_fast"] = sma_seeded_ema(out["close"], EMA_FAST)
        out["ema_slow"] = sma_seeded_ema(out["close"], EMA_SLOW)
        lb = self.breakout_lookback
        out["breakout_hh"] = breakout_window_extreme(out, "high", lb)
        out["breakout_ll"] = breakout_window_extreme(out, "low", lb)
        return out

    # ---- fail-closed value guard ----
    @staticmethod
    def _finite(x) -> bool:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return False
        return np.isfinite(v)

    # ---- entry conditions (V1.0 verbatim) ----
    def _long_setup(self, df: pd.DataFrame, i: int) -> bool:
        if i < WARM_UP:
            return False
        bbw, sq = df["bb_width"].iloc[i], df["squeeze_q"].iloc[i]
        if not (self._finite(bbw) and self._finite(sq)):
            return False
        compression = bbw <= sq
        atr_v, avg = df["atr_14"].iloc[i], df["avg_atr"].iloc[i]
        if not (self._finite(atr_v) and self._finite(avg) and avg > 0):
            return False
        expansion = atr_v >= float(self.params["expansion_multiplier"]) * avg
        close = df["close"].iloc[i]
        hh = df["breakout_hh"].iloc[i]
        if not (self._finite(close) and self._finite(hh)):
            return False
        breakout = close > hh            # strict cross, current-bar-excluded (V1.0 verbatim)
        fast, slow = df["ema_fast"].iloc[i], df["ema_slow"].iloc[i]
        if not (self._finite(fast) and self._finite(slow)):
            return False
        trend = fast > slow
        return compression and breakout and expansion and trend

    def _short_setup(self, df: pd.DataFrame, i: int) -> bool:
        if i < WARM_UP:
            return False
        bbw, sq = df["bb_width"].iloc[i], df["squeeze_q"].iloc[i]
        if not (self._finite(bbw) and self._finite(sq)):
            return False
        compression = bbw <= sq
        atr_v, avg = df["atr_14"].iloc[i], df["avg_atr"].iloc[i]
        if not (self._finite(atr_v) and self._finite(avg) and avg > 0):
            return False
        expansion = atr_v >= float(self.params["expansion_multiplier"]) * avg
        close = df["close"].iloc[i]
        ll = df["breakout_ll"].iloc[i]
        if not (self._finite(close) and self._finite(ll)):
            return False
        breakout = close < ll            # strict cross, current-bar-excluded (V1.0 verbatim)
        fast, slow = df["ema_fast"].iloc[i], df["ema_slow"].iloc[i]
        if not (self._finite(fast) and self._finite(slow)):
            return False
        trend = fast < slow
        return compression and breakout and expansion and trend

    def entry_signal(self, df: pd.DataFrame, i: int, ctx: Optional[Dict] = None
                     ) -> Tuple[Direction, str]:
        """Entry direction at the close of bar ``i`` (uses only bars ``<= i``)."""
        if i < WARM_UP:
            return Direction.NONE, "warmup"
        if self._long_setup(df, i):
            return Direction.LONG, "vcb_long"
        if self._short_setup(df, i):
            return Direction.SHORT, "vcb_short"
        return Direction.NONE, ""

    # ---- sizing / fill helpers ----
    def risk_distance(self, atr_at_signal: float) -> float:
        """``R = atr_multiplier * ATR_{t0}`` at the signal bar."""
        return self.atr_multiplier * float(atr_at_signal)

    def fill_price(self, open_next: float, direction: Direction) -> float:
        """Fill at ``open_{t+1}`` with the frozen slip formula."""
        if direction == Direction.LONG:
            return float(open_next) * (1.0 + SLIP_RATE)
        return float(open_next) * (1.0 - SLIP_RATE)

    # ---- stop / take (no take-profit in V1.1) ----
    def compute_take_profit(self, df, i, direction, entry_price, stop_loss=None):
        # V1.1 removes the take-profit entirely — the right tail is uncapped (§4a).
        return None

    def compute_stop_loss(self, df, i, direction, entry_price):
        atr_v = df["atr_14"].iloc[i]
        if not self._finite(atr_v) or atr_v <= 0:
            return None
        offset = self.atr_multiplier * atr_v
        return entry_price - offset if direction == Direction.LONG else entry_price + offset

    def prepare_fill(
        self,
        df: pd.DataFrame,
        signal_bar_index: int,
        direction: Direction,
        actual_fill_price: float,
        sig=None,
    ) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
        """Fill-anchored initial stop; take-profit ``None``; carries exit-manager state.

        Returns ``(stop, None, state)`` where ``state`` seeds the strict running-max
        exit (protocol §4b base case ``Stop_{t0+1} = Stop0``) and the +1.5R ratchet.
        ``R = atr_multiplier * ATR_{t0}`` (ATR at the signal bar).
        """
        atr = float(df["atr_14"].iloc[signal_bar_index])
        if not np.isfinite(atr) or atr <= 0:
            return None, None, {}
        R = self.risk_distance(atr)
        if direction == Direction.LONG:
            stop = actual_fill_price - R
        else:
            stop = actual_fill_price + R
        state = {
            "fill_price": float(actual_fill_price),
            "risk_R": float(R),
            "atr_t0": float(atr),
            "signal_bar": int(signal_bar_index),
            "stop_direction": direction.value,
            "prior_stop": float(stop),
            "breakeven_armed": False,
        }
        return stop, None, state

    def update_stop(self, df: pd.DataFrame, bar_index: int, position) -> None:
        """Drive the strict running-max dynamic stop on each closed bar.

        The frozen engine calls this before the same-bar exit check; we evaluate the
        completed bar ``bar_index - 1`` so the stop computed at the close of ``t``
        governs exits from ``t+1`` onward (protocol §3 / §4e, no look-ahead). State is
        carried on ``position.extra_state`` as scalars (no full replay).
        """
        state = position.extra_state
        if not state or state.get("risk_R", None) is None or state.get("fill_price", None) is None:
            return
        direction = state.get("stop_direction", position.direction)
        d = Direction(direction) if isinstance(direction, str) else direction
        mgr = V1_1ExitManager(
            d, fill=state["fill_price"], R=state["risk_R"],
            entry_bar=int(position.entry_bar),
            lookback_trail=self.lookback_trail,
            trail_atr_mult=self.trail_atr_mult,
        )
        if state.get("breakeven_armed"):
            mgr.armed = True
        mgr.prior_stop = float(state.get("prior_stop", mgr.stop0))

        j = bar_index - 1                     # completed bar determining the stop
        if j < int(position.entry_bar):
            position.active_stop = mgr.stop0   # initial stop until the entry bar closes
            return
        mgr.scan_bar(df, j)
        position.active_stop = mgr.effective_stop()
        state["prior_stop"] = float(mgr.effective_stop())
        state["breakeven_armed"] = bool(mgr.armed)

    # ---- param grid (protocol §5) ----
    @classmethod
    def param_grid(cls) -> Dict[str, List[Any]]:
        return {
            "breakout_lookback": list(BREAKOUT_LOOKBACKS),
            "squeeze_percentile": list(SQUEEZE_PERCENTILES),
            "atr_multiplier": list(ATR_MULTIPLIERS),
            "exit_config": ["X1", "X2", "X3"],
        }