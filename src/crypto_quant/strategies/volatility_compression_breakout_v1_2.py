"""Strategy #9 V1.2 — Regime-Filtered Volatility Compression Breakout.

Frozen-protocol implementation of ``PROTOCOL-STRATEGY-09-V1.2``
(fingerprint SHA-256 ``17a255d50b72b255a3490af297799985cca9b075623456af363c5ffc28ad008e``,
``data/protocol_strategy_09_v1_2.md``).

Specification baseline (do not change without a human-authorized protocol amendment)
-------------------------------------------------------------------------------------
Entry engine (preserved *verbatim* from V1.1 — this protocol freezes it unchanged):
  - Compression : ``BBWidth_t = 4*sd20_t / SMA20_t`` (population ddof=0) <= Type-7
    linear-interpolation ``q``-percentile of a *past-only* 120-bar baseline, with
    ``squeeze_percentile = 30`` (fixed; V1.1 survivor convergence).
  - Breakout    : strict ``>``/``<`` cross of ``close_t`` against a *current-bar-
    excluded* trailing High/Low window of length ``breakout_lookback`` (20 or 40).
  - Expansion   : ``ATR_t = SMA(TR,14)`` >= ``expansion_multiplier * avgATR_t``,
    ``expansion_multiplier = 1.1`` (frozen single value, NOT the risk ATR multiple).
  - Trend filter: SMA-seeded ``EMA20 > EMA50`` (long) / ``EMA20 < EMA50`` (short).
  - Fail-closed : no signal for ``t < 139`` (0-indexed); any invalid / short / zero /
    non-finite baseline fails the respective condition to no-signal.

NEW regime gate (the V1.2 change — entry-side disable/enable only):
  - ``ADX(14)_t > adx_threshold`` with ``adx_threshold in {20, 25, 30}``.
  - Exact ADX is the classical **Wilder RMA** definition (protocol §3b / review-O1):
      TR[i]  = max(High-Low, |High-Close[i-1]|, |Low-Close[i-1]|)         (i>=1)
      +DM[i] = High[i]-High[i-1] if that > Low[i-1]-Low[i] and > 0 else 0
      -DM[i] = Low[i-1]-Low[i]  if that > High[i]-High[i-1] and > 0 else 0
    Wilder smoothing (factor 1/14), seeded with the sum of raw[1..14]:
      SmX[i] = SmX[i-1] - SmX[i-1]/14 + X[i]
      +DI = 100*Sm+DM/SmTR ; -DI = 100*Sm-DM/SmTR
      DX  = 100*|+DI - -DI|/(+DI + -DI)                    (fail-closed on zero divisor)
      ADX = Wilder RMA of DX, seeded with the mean of the first 14 DX values.
    Strictly backward-looking (bars <= t); fail-closed (NaN/non-finite => no signal).
  - The gate never modifies signal or exit math (protocol invariant 8, neutrality).

Dynamic exit (V1.1 verbatim — frozen):
  - Take-profit removed (uncapped right tail).
  - Initial risk : ``Stop0 = Fill - a*ATR_{t0}`` (long) / ``Fill + a*ATR_{t0}`` (short),
    with ``a = atr_multiplier = 2.5`` (fixed frozen constant — the 24-grid sweeps
    only {breakout_lookback, adx_threshold, exit_config}; the risk width is constant).
  - Breakeven ratchet (F-02): armed at +1.5R unrealized (bar close), latched
    permanently, level ``Fill * (1 +/- 0.0020)`` (price-level invariant fraction).
  - Chandelier trail: ``CH_t = HH(window) - trail_atr_mult * ATR_t`` (long) /
    ``CH_t = LL(window) + trail_atr_mult * ATR_t`` (short), window from entry to ``t``.
  - STRICT RUNNING-MAX (F-01): long ``Stop_t = max(Stop_{t-1}, RBE, CH_t)``,
    short ``Stop_t = min(Stop_{t-1}, RBE, CH_t)`` — carried forward so the stop
    never recedes, even under an ATR spike.
  - Collision: conservative stop-first; gap fills at the adverse of open vs level.

This module ships the 24-candidate enumeration utility (C00..C23) and pure,
unit-testable helpers for the entry signal, the Wilder-ADX regime gate, and the
exit mechanics. It is deliberately NOT registered into the central
``STRATEGY_REGISTRY``; registration (if desired) is a separate, human-authorized
integration step.

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
ATR_PERIOD: int = 14                   # SMA(TR,14) -> also the Wilder ADX period
EXPANSION_BASELINE: int = 10           # past-only baseline for avgATR
EMA_FAST: int = 20
EMA_SLOW: int = 50
BREAKEVEN_TRIGGER_R: float = 1.5       # arm the ratchet at +1.5R unrealized
BREAKEVEN_BUFFER: float = 0.0020       # Fill*(1 +/- 0.0020) ratchet level (F-02)
SLIP_RATE: float = 0.0005              # adverse slippage per side (fill formula)

# Regime gate (protocol §3b / §5b)
ADX_THRESHOLDS = (25, 20, 30)          # declared ordering; the set is {20, 25, 30}

# Fixed (signal / risk) dimensions — V1.2 sweeps ONLY lb x adx x exit (protocol §5)
SQUEEZE_PERCENTILE: int = 30           # fixed (V1.1 survivor convergence)
EXPANSION_MULTIPLIER: float = 1.1      # frozen, single value (protocol §5a)
RISK_ATR_MULTIPLIER: float = 2.5       # fixed risk width a in R = a*ATR (C00-anchor)

# Grid exit dimension (protocol §5c) — full 2x2 INCLUDING the (10, 2.0) pair
# that V1.1 had excluded as dominated (X4 is re-added in V1.2).
BREAKOUT_LOOKBACKS = (20, 40)
EXIT_CONFIGS: Dict[str, Tuple[int, float]] = {          # lookback_trail -> trail_atr_mult
    "X1": (20, 3.0),
    "X2": (10, 3.0),
    "X3": (20, 2.0),
    "X4": (10, 2.0),
}
EXIT_CONFIG_ORDER = ("X1", "X2", "X3", "X4")

_C00_ANCHOR = (20, SQUEEZE_PERCENTILE, EXPANSION_MULTIPLIER, ADX_THRESHOLDS[0], "X1")


# --------------------------------------------------------------------------- #
# Pure entry-math helpers (V1.1 verbatim where the protocol freezes them)
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
    """``BBWidth = (2*num_std*sd20)/SMA20`` with *population* std (ddof=0)."""
    sma = close.rolling(period, min_periods=period).mean()
    sd = close.rolling(period, min_periods=period).std(ddof=0)
    width = (2.0 * num_std * sd) / sma
    return width.where(np.isfinite(width))


def past_only_rolling_quantile(series: pd.Series, window: int, q: float) -> pd.Series:
    """Type-7 ``q``-percentile of the *past-only* ``window`` values before each bar."""
    return (
        series.shift(1)
        .rolling(window, min_periods=window)
        .apply(lambda w: type7_percentile(np.asarray(w, dtype=np.float64), q), raw=True)
    )


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


def sma_true_range_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                       period: int = ATR_PERIOD) -> pd.Series:
    """``ATR = SMA(TR, period)`` — simple moving average of true range, NOT Wilder-RMA."""
    return true_range(high, low, close).rolling(period, min_periods=period).mean()


def past_only_mean(series: pd.Series, window: int) -> pd.Series:
    """Mean of the *past-only* ``window`` values before each bar (excludes current)."""
    return series.shift(1).rolling(window, min_periods=window).mean()


def sma_seeded_ema(close: pd.Series, period: int) -> pd.Series:
    """SMA-seeded recursive EMA (V1.0 §4.5)."""
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
# Regime-gate helpers — Wilder ADX(14) (protocol §3b, review-O1 RESOLVED)
# --------------------------------------------------------------------------- #
def wilder_rma(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RMA (factor ``1/period``) with classic seeding.

    Recurrence ``out[k] = out[k-1] + (v[k] - out[k-1])/period`` (equivalently
    ``(out[k-1]*(period-1) + v[k])/period``), seeded with the mean of the first
    ``period`` raw values. NaN before ``period`` observations exist.
    """
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    out = np.full(n, np.nan, dtype=np.float64)
    if period <= 0 or n < period:
        return out
    out[period - 1] = float(np.mean(v[:period]))
    for k in range(period, n):
        out[k] = out[k - 1] + (float(v[k]) - out[k - 1]) / period
    return out


def wilder_adx(high, low, close, period: int = ATR_PERIOD) -> pd.Series:
    """Classical Wilder's ADX(``period``) as a Series (strictly backward-looking).

    Uses the standard Wilder convention with bar-1-based directional indices:
      TR, +DM, -DM are defined for bars ``i>=1`` (bar 0 carries 0/placeholder).
      Wilder smoothing is seeded with the *sum* of raw[1..period], then
      ``SmX[i] = SmX[i-1] - SmX[i-1]/period + X[i]``.
      +DI / -DI = 100*Sm±DM/SmTR ; DX = 100*|+DI - -DI|/(+DI + -DI).
      ADX = Wilder RMA of DX, seeded with the mean of the first `period` DX values.
    Fail-closed: NaN before the training windows and wherever any divisor is zero
    or non-finite — callers combine this with the fail-closed guard in the signal.
    """
    H = np.asarray(high, dtype=np.float64)
    L = np.asarray(low, dtype=np.float64)
    C = np.asarray(close, dtype=np.float64)
    n = len(H)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 2 * period:
        return pd.Series(out)

    tr = np.zeros(n); dm_p = np.zeros(n); dm_m = np.zeros(n)
    for i in range(1, n):
        up = H[i] - H[i - 1]
        dn = L[i - 1] - L[i]
        tr[i] = max(H[i] - L[i], abs(H[i] - C[i - 1]), abs(L[i] - C[i - 1]))
        dm_p[i] = up if (up > dn and up > 0.0) else 0.0
        dm_m[i] = dn if (dn > up and dn > 0.0) else 0.0

    # Wilder smoothing, seeded with the sum of raw[1..period] at index {period}.
    # Recurrence: Sm[i] = Sm[i-1] + (raw[i] - Sm[i-1])/period  (Wilder's running avg).
    s_tr = np.zeros(n); s_p = np.zeros(n); s_m = np.zeros(n)
    for i in range(period, n):
        if i == period:
            s_tr[i] = float(np.sum(tr[1:period + 1]))
            s_p[i] = float(np.sum(dm_p[1:period + 1]))
            s_m[i] = float(np.sum(dm_m[1:period + 1]))
        else:
            s_tr[i] = s_tr[i - 1] + (tr[i] - s_tr[i - 1]) / period
            s_p[i] = s_p[i - 1] + (dm_p[i] - s_p[i - 1]) / period
            s_m[i] = s_m[i - 1] + (dm_m[i] - s_m[i - 1]) / period

    # Directional Movement Index (DX).
    dx = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if s_tr[i] <= 0:
            continue
        dpi = 100.0 * s_p[i] / s_tr[i]
        dmi = 100.0 * s_m[i] / s_tr[i]
        denom = dpi + dmi
        if denom <= 0:
            continue
        dx[i] = 100.0 * abs(dpi - dmi) / denom

    # ADX = Wilder RMA of DX; first valid block is DX[period .. 2*period-1].
    first = 2 * period - 1
    if n >= first + 1:
        window = [dx[k] for k in range(period, first + 1)]
        if not any(map(np.isfinite, window)) or len([v for v in window if np.isfinite(v)]) < period:
            return pd.Series(out)
        seed = float(np.mean([dx[k] for k in range(period, first + 1)]))
        out[first] = seed
        for i in range(first + 1, n):
            if not np.isfinite(dx[i]):
                out[i] = np.nan
                continue
            out[i] = out[i - 1] + (dx[i] - out[i - 1]) / period
    return pd.Series(out)


# --------------------------------------------------------------------------- #
# Pure exit-math helpers (V1.1 verbatim, protocol §4)
# --------------------------------------------------------------------------- #
def break_even_level(fill: float, direction: Direction, buffer: float = BREAKEVEN_BUFFER) -> float:
    """Ratchet level ``Fill * (1 +/- buffer)`` (long/short symmetric; F-02)."""
    if direction == Direction.LONG:
        return fill * (1.0 + buffer)
    return fill * (1.0 - buffer)


def chandelier_trail(direction: Direction, window_extreme: float, atr_t: float,
                     trail_atr_mult: float) -> float:
    """Raw Chandelier stop at bar ``t`` (protocol §4d) — NOT yet monotonic."""
    if direction == Direction.LONG:
        return window_extreme - trail_atr_mult * atr_t
    return window_extreme + trail_atr_mult * atr_t


def running_max_stop(direction: Direction, prior: float, rbe_if_armed: float,
                     chandelier: float) -> float:
    """STRICT RUNNING-MAX effective stop (protocol §4e / finding F-01)."""
    if direction == Direction.LONG:
        return max(prior, rbe_if_armed, chandelier)
    return min(prior, rbe_if_armed, chandelier)


def resolve_collision(direction: Direction, open_price: float, high: float, low: float,
                      stop_level: float, alt_level: Optional[float] = None
                      ) -> Tuple[Optional[str], float]:
    """Conservative stop-first same-bar collision resolution (protocol §4f)."""
    if direction == Direction.LONG:
        stop_hit = low <= stop_level
        alt_hit = (alt_level is not None) and (high >= alt_level)
    else:
        stop_hit = high >= stop_level
        alt_hit = (alt_level is not None) and (low <= alt_level)

    if stop_hit:
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
# 24-candidate grid (protocol §5b / §5c / §5d / §5e)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Candidate:
    """A single frozen candidate tuple of the 24-candidate grid (C00..C23)."""
    id: str
    breakout_lookback: int
    squeeze_percentile: int
    expansion_multiplier: float
    adx_threshold: int
    atr_multiplier: float
    exit_config: str
    lookback_trail: int
    trail_atr_mult: float

    @property
    def params(self) -> Dict[str, Any]:
        return asdict(self)


def candidate_grid() -> List[Candidate]:
    """Return the canonical 24 candidates in declaration order C00..C23.

    Ordering follows the frozen protocol §5e: ``lb`` major, then
    ``adx_threshold in {25, 20, 30}``, then ``exit_config in {X1, X2, X3, X4}``.
    C00 is the declared anchor ``(lb=20, sq=30, exp=1.1, adx=25, X1)`` — the first
    tuple of the fully-ordered ``2 x 3 x 4`` Cartesian cross, with no gaps/duplicates.
    ``squeeze_percentile=30``, ``expansion_multiplier=1.1``, and
    ``atr_multiplier=2.5`` are fixed for every candidate.
    """
    cands: List[Candidate] = []
    i = 0
    for lb in BREAKOUT_LOOKBACKS:
        for adx in ADX_THRESHOLDS:
            for ex in EXIT_CONFIG_ORDER:
                lt, tam = EXIT_CONFIGS[ex]
                cands.append(Candidate(
                    id="C%02d" % i, breakout_lookback=lb,
                    squeeze_percentile=SQUEEZE_PERCENTILE,
                    expansion_multiplier=EXPANSION_MULTIPLIER,
                    adx_threshold=adx, atr_multiplier=RISK_ATR_MULTIPLIER,
                    exit_config=ex, lookback_trail=lt, trail_atr_mult=tam,
                ))
                i += 1
    return cands


CANDIDATE_GRID: List[Candidate] = candidate_grid()
CANDIDATE_TUPLES: Tuple[Candidate, ...] = tuple(CANDIDATE_GRID)


# --------------------------------------------------------------------------- #
# Dynamic exit manager (V1.1 verbatim -> V1.2)
# --------------------------------------------------------------------------- #
class V1_2ExitManager:
    """Stateful per-position dynamic exit: combines the initial ATR stop, the +1.5R
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

    def _window_extreme(self, df: pd.DataFrame, t: int) -> float:
        start = self.entry_bar
        if self.direction == Direction.LONG:
            return float(df["high"].iloc[start:t + 1].max())
        return float(df["low"].iloc[start:t + 1].min())

    def scan_bar(self, df: pd.DataFrame, t: int) -> float:
        """Advance the running-max stop using closed bar ``t`` (monotonic, F-01)."""
        window_extreme = self._window_extreme(df, t)
        atr_t = float(df["atr_14"].iloc[t])
        raw_ch = chandelier_trail(self.direction, window_extreme, atr_t, self.trail_atr_mult)
        self._check_arm(float(df["close"].iloc[t]))
        self.prior_stop = running_max_stop(
            self.direction, self.prior_stop, self._rbe_if_armed(), raw_ch
        )
        return self.prior_stop

    def effective_stop(self) -> float:
        return self.prior_stop


# --------------------------------------------------------------------------- #
# V1.2 strategy class
# --------------------------------------------------------------------------- #
class VolatilityCompressionBreakoutV1_2(BaseStrategy):
    """Regime-Filtered Volatility Compression Breakout (frozen PROTOCOL-STRATEGY-09-V1.2)."""

    name = "volatility_compression_breakout_v1_2"
    strategy_type = "volatility_compression_breakout"
    description = (
        "Frozen PROTOCOL-STRATEGY-09-V1.2: BB-width compression + breakout + ATR "
        "expansion(1.1) + SMA-seeded EMA20/50 filter entry, gated by a Wilder "
        "ADX(14)>threshold regime filter; Chandelier running-max trail and "
        "Fill*(1+/-0.0020) breakeven ratchet exit (no take-profit)."
    )
    supports_spot = True
    supports_futures = True
    supports_short = True

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        defaults = {
            "breakout_lookback": 20,
            "squeeze_percentile": SQUEEZE_PERCENTILE,
            "expansion_multiplier": EXPANSION_MULTIPLIER,
            "adx_threshold": 25,
            "atr_multiplier": RISK_ATR_MULTIPLIER,
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
    def adx_threshold(self) -> float:
        return float(self.params["adx_threshold"])

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
        if int(p["squeeze_percentile"]) != SQUEEZE_PERCENTILE:
            raise ValueError(f"squeeze_percentile must be {SQUEEZE_PERCENTILE} (fixed in V1.2)")
        if abs(float(p["expansion_multiplier"]) - EXPANSION_MULTIPLIER) > 1e-9:
            raise ValueError(f"expansion_multiplier must be {EXPANSION_MULTIPLIER} (frozen)")
        if int(p["adx_threshold"]) not in ADX_THRESHOLDS:
            raise ValueError(f"adx_threshold must be in {sorted(ADX_THRESHOLDS)} ({ADX_THRESHOLDS})")
        if abs(float(p["atr_multiplier"]) - RISK_ATR_MULTIPLIER) > 1e-9:
            raise ValueError(f"atr_multiplier must be {RISK_ATR_MULTIPLIER} (fixed in V1.2)")
        lt, tam = int(p["lookback_trail"]), float(p["trail_atr_mult"])
        valid_known = any(EXIT_CONFIGS[k] == (lt, tam) for k in EXIT_CONFIGS)
        if not valid_known:
            raise ValueError(f"exit pair ({lt}, {tam}) does not correspond to X1/X2/X3/X4")

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
        out["adx_14"] = wilder_adx(out["high"], out["low"], out["close"], ATR_PERIOD)
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

    # ---- regime gate (fail-closed) ----
    def _regime_ok(self, df: pd.DataFrame, i: int) -> bool:
        adx = df["adx_14"].iloc[i]
        if not self._finite(adx):
            return False
        return bool(adx > self.adx_threshold)     # strict >; == fails (fail-closed)

    # ---- entry conditions (V1.1 verbatim + ADX regime gate) ----
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
        if not compression or not breakout or not expansion or not trend:
            return False
        return self._regime_ok(df, i)    # NEW: ADX(14) > adx_threshold (regime gate)

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
        if not compression or not breakout or not expansion or not trend:
            return False
        return self._regime_ok(df, i)    # NEW: ADX(14) > adx_threshold (regime gate)

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
        """``R = atr_multiplier * ATR_{t0}`` at the signal bar (atr_multiplier fixed 2.5)."""
        return self.atr_multiplier * float(atr_at_signal)

    def fill_price(self, open_next: float, direction: Direction) -> float:
        """Fill at ``open_{t+1}`` with the frozen slip formula."""
        if direction == Direction.LONG:
            return float(open_next) * (1.0 + SLIP_RATE)
        return float(open_next) * (1.0 - SLIP_RATE)

    # ---- stop / take (no take-profit in V1.2) ----
    def compute_take_profit(self, df, i, direction, entry_price, stop_loss=None):
        # V1.2 removes the take-profit entirely — the right tail is uncapped (§4a).
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
        """Fill-anchored initial stop; take-profit ``None``; carries exit-manager state."""
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
        """Drive the strict running-max dynamic stop on each closed bar (no look-ahead)."""
        state = position.extra_state
        if not state or state.get("risk_R", None) is None or state.get("fill_price", None) is None:
            return
        direction = state.get("stop_direction", position.direction)
        d = Direction(direction) if isinstance(direction, str) else direction
        mgr = V1_2ExitManager(
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
            "squeeze_percentile": [SQUEEZE_PERCENTILE],
            "adx_threshold": sorted(ADX_THRESHOLDS),
            "exit_config": list(EXIT_CONFIG_ORDER),
            "expansion_multiplier": [EXPANSION_MULTIPLIER],
        }