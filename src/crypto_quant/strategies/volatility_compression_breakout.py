"""Volatility Compression Breakout (Strategy #9).

Implements PROTOCOL-STRATEGY-09-V1.0 exactly:

- Bollinger-width compression vs Type-7 percentile of a past-only 120-bar baseline
- Trailing high/low breakout with current-bar exclusion
- SMA-of-TR ATR expansion vs a past-only 10-bar baseline
- EMA20 vs EMA50 trend filter (SMA-seeded EMA)
- Fixed ATR stop / 2R target (no trailing, no FVG, no ER)

Signal at close of bar t; engine executes at open[t+1].
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..logging_config import get_logger
from .base import BaseStrategy, Direction, Signal, StopLossSpec, StopType, TakeProfitSpec

logger = get_logger("strategies")

# Frozen constants (not grid dimensions)
BB_PERIOD = 20
BB_STD_MULT = 2.0
BB_DDOF = 0  # population stddev
SQUEEZE_LOOKBACK = 120
EXPANSION_LOOKBACK = 10
FAST_EMA = 20
SLOW_EMA = 50
ATR_PERIOD = 14
RISK_REWARD_RATIO = 2.0
WARMUP_MIN_INDEX = 139  # 0-indexed; squeeze baseline is binding

GRID_BREAKOUT_LOOKBACK = (20, 40)
GRID_SQUEEZE_PERCENTILE = (20, 30, 40)
GRID_EXPANSION_MULTIPLIER = (1.1, 1.3)
GRID_ATR_MULTIPLIER = (1.8, 2.2, 2.6)

C00_PARAMS: Dict[str, Any] = {
    "candidate_id": "C00",
    "breakout_lookback": 20,
    "squeeze_percentile": 30,
    "expansion_multiplier": 1.1,
    "atr_multiplier": 2.2,
    "bb_period": BB_PERIOD,
    "squeeze_lookback": SQUEEZE_LOOKBACK,
    "expansion_lookback": EXPANSION_LOOKBACK,
    "fast_ema": FAST_EMA,
    "slow_ema": SLOW_EMA,
    "atr_period": ATR_PERIOD,
    "risk_reward_ratio": RISK_REWARD_RATIO,
}


def sma_seeded_ema(close: np.ndarray, period: int) -> np.ndarray:
    """EMA seeded with SMA of the first ``period`` closes, then recursive.

    Protocol §4.5: EMA_k = α·close_k + (1−α)·EMA_{k−1}, α = 2/(period+1).
    Valid only once ≥ period observations exist.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    if period <= 0 or n < period:
        return out
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = float(np.mean(close[:period]))
    for k in range(period, n):
        out[k] = alpha * float(close[k]) + (1.0 - alpha) * out[k - 1]
    return out


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True range. Bar 0 has no prior close → TR_0 = High_0 − Low_0."""
    n = len(close)
    tr = np.empty(n, dtype=float)
    tr[0] = float(high[0] - low[0])
    prev = close[:-1]
    h = high[1:]
    l = low[1:]
    tr[1:] = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    return tr


def sma_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = ATR_PERIOD) -> np.ndarray:
    """SMA of True Range over ``period`` bars (current bar included). Protocol §4.4."""
    tr = true_range(high, low, close)
    s = pd.Series(tr)
    return s.rolling(period, min_periods=period).mean().to_numpy(dtype=float)


def bb_width_population(close: np.ndarray, period: int = BB_PERIOD) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Population (ddof=0) Bollinger width. Returns (mid, width, std).

    Width is undefined (NaN) when mid <= 0 (fail-closed compression).
    """
    s = pd.Series(close)
    mid = s.rolling(period, min_periods=period).mean()
    std = s.rolling(period, min_periods=period).std(ddof=BB_DDOF)
    width = (4.0 * std) / mid
    width = width.mask(mid <= 0.0, np.nan)
    return mid.to_numpy(dtype=float), width.to_numpy(dtype=float), std.to_numpy(dtype=float)


def type7_percentile(sample: np.ndarray, q: float) -> float:
    """Linear interpolation percentile (numpy Type-7 / method='linear')."""
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, q, method="linear"))


def enumerate_grid_tuples() -> List[Tuple[int, int, float, float]]:
    """All 36 (L, q, m, a) tuples in lexicographic order."""
    return list(
        product(
            GRID_BREAKOUT_LOOKBACK,
            GRID_SQUEEZE_PERCENTILE,
            GRID_EXPANSION_MULTIPLIER,
            GRID_ATR_MULTIPLIER,
        )
    )


def enumerate_candidates() -> List[Dict[str, Any]]:
    """C00 then C01..C35 for the remaining lexicographic tuples."""
    c00 = (
        int(C00_PARAMS["breakout_lookback"]),
        int(C00_PARAMS["squeeze_percentile"]),
        float(C00_PARAMS["expansion_multiplier"]),
        float(C00_PARAMS["atr_multiplier"]),
    )
    rest = [t for t in enumerate_grid_tuples() if t != c00]
    out = [
        {
            "candidate_id": "C00",
            "breakout_lookback": c00[0],
            "squeeze_percentile": c00[1],
            "expansion_multiplier": c00[2],
            "atr_multiplier": c00[3],
        }
    ]
    for i, (L, q, m, a) in enumerate(rest, start=1):
        out.append(
            {
                "candidate_id": f"C{i:02d}",
                "breakout_lookback": L,
                "squeeze_percentile": q,
                "expansion_multiplier": m,
                "atr_multiplier": a,
            }
        )
    return out


class VolatilityCompressionBreakoutStrategy(BaseStrategy):
    name = "Volatility Compression Breakout"
    strategy_type = "volatility_compression_breakout"
    description = (
        "Bollinger-width squeeze + trailing-range breakout gated by ATR expansion "
        "and EMA20/EMA50 trend; fixed ATR stop and 2R target."
    )
    supports_spot = True
    supports_futures = True
    supports_short = True

    @classmethod
    def default_timeframes(cls) -> List[str]:
        return ["1h", "4h"]

    @classmethod
    def param_grid(cls) -> Dict[str, list]:
        return {
            "breakout_lookback": list(GRID_BREAKOUT_LOOKBACK),
            "squeeze_percentile": list(GRID_SQUEEZE_PERCENTILE),
            "expansion_multiplier": list(GRID_EXPANSION_MULTIPLIER),
            "atr_multiplier": list(GRID_ATR_MULTIPLIER),
        }

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        params = dict(params or {})
        params.setdefault("breakout_lookback", C00_PARAMS["breakout_lookback"])
        params.setdefault("squeeze_percentile", C00_PARAMS["squeeze_percentile"])
        params.setdefault("expansion_multiplier", C00_PARAMS["expansion_multiplier"])
        params.setdefault("atr_multiplier", C00_PARAMS["atr_multiplier"])
        params.setdefault("bb_period", BB_PERIOD)
        params.setdefault("squeeze_lookback", SQUEEZE_LOOKBACK)
        params.setdefault("expansion_lookback", EXPANSION_LOOKBACK)
        params.setdefault("fast_ema", FAST_EMA)
        params.setdefault("slow_ema", SLOW_EMA)
        params.setdefault("atr_period", ATR_PERIOD)
        params.setdefault("risk_reward_ratio", RISK_REWARD_RATIO)
        stop_spec = StopLossSpec(
            stop_type=StopType.ATR,
            atr_multiplier=float(params["atr_multiplier"]),
        )
        tp_spec = TakeProfitSpec(
            mode="rr",
            risk_reward_ratio=float(params["risk_reward_ratio"]),
        )
        super().__init__(params=params, stop_spec=stop_spec, tp_spec=tp_spec, **kwargs)

    def validate_params(self) -> None:
        p = self.params
        for k in (
            "breakout_lookback",
            "squeeze_percentile",
            "bb_period",
            "squeeze_lookback",
            "expansion_lookback",
            "fast_ema",
            "slow_ema",
            "atr_period",
        ):
            v = int(p[k])
            if v <= 0:
                raise ValueError(f"{k} must be > 0, got {v}")
        for k in ("expansion_multiplier", "atr_multiplier", "risk_reward_ratio"):
            v = float(p[k])
            if v <= 0:
                raise ValueError(f"{k} must be > 0, got {v}")
        q = float(p["squeeze_percentile"])
        if not (0.0 <= q <= 100.0):
            raise ValueError(f"squeeze_percentile out of range [0,100]: {q}")

    def setup(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "timestamp" in out.columns:
            out = out.sort_values("timestamp").reset_index(drop=True)
        else:
            out = out.reset_index(drop=True)

        close = out["close"].to_numpy(dtype=float)
        high = out["high"].to_numpy(dtype=float)
        low = out["low"].to_numpy(dtype=float)

        bb_p = int(self.params["bb_period"])
        atr_p = int(self.params["atr_period"])
        fast = int(self.params["fast_ema"])
        slow = int(self.params["slow_ema"])
        L = int(self.params["breakout_lookback"])
        S = int(self.params["squeeze_lookback"])
        X = int(self.params["expansion_lookback"])
        q = float(self.params["squeeze_percentile"])

        mid, width, std = bb_width_population(close, bb_p)
        out["bb_mid"] = mid
        out["bb_std"] = std
        out["bb_upper"] = mid + BB_STD_MULT * std
        out["bb_lower"] = mid - BB_STD_MULT * std
        out["bb_width"] = width

        atr = sma_atr(high, low, close, atr_p)
        out["atr_sma"] = atr
        # Engine helpers look for atr_14 for ATR-based stops; keep SMA-ATR here.
        out["atr_14"] = atr

        out["ema_fast"] = sma_seeded_ema(close, fast)
        out["ema_slow"] = sma_seeded_ema(close, slow)

        # Breakout reference: max high / min low of bars t-L … t-1 (current excluded)
        out["breakout_high"] = out["high"].shift(1).rolling(L, min_periods=L).max()
        out["breakout_low"] = out["low"].shift(1).rolling(L, min_periods=L).min()

        # Compression threshold: Type-7 percentile of width[t-S … t-1]
        out["squeeze_threshold"] = (
            out["bb_width"]
            .shift(1)
            .rolling(S, min_periods=S)
            .apply(lambda w: type7_percentile(np.asarray(w, dtype=float), q), raw=True)
        )

        # Expansion baseline: mean ATR[t-X … t-1]
        out["atr_baseline"] = out["atr_sma"].shift(1).rolling(X, min_periods=X).mean()
        return out

    # ------------------------------------------------------------ conditions
    def _finite(self, value: Any) -> bool:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False
        return np.isfinite(v)

    def compression_at(self, df: pd.DataFrame, i: int) -> bool:
        """BBWidth_t <= P_t(q); equality passes. Fail-closed on invalid baseline."""
        if i < WARMUP_MIN_INDEX:
            return False
        width = df["bb_width"].iloc[i]
        thresh = df["squeeze_threshold"].iloc[i]
        if not self._finite(width) or not self._finite(thresh):
            return False
        S = int(self.params["squeeze_lookback"])
        hist = df["bb_width"].iloc[i - S : i].to_numpy(dtype=float)
        if hist.size != S or not np.isfinite(hist).all():
            return False
        mid = df["bb_mid"].iloc[i]
        if not self._finite(mid) or float(mid) <= 0.0:
            return False
        return float(width) <= float(thresh)

    def long_break_at(self, df: pd.DataFrame, i: int) -> bool:
        L = int(self.params["breakout_lookback"])
        if i < L:
            return False
        ref = df["high"].iloc[i - L : i].to_numpy(dtype=float)
        if ref.size != L or not np.isfinite(ref).all():
            return False
        close_t = df["close"].iloc[i]
        if not self._finite(close_t):
            return False
        return float(close_t) > float(np.max(ref))

    def short_break_at(self, df: pd.DataFrame, i: int) -> bool:
        L = int(self.params["breakout_lookback"])
        if i < L:
            return False
        ref = df["low"].iloc[i - L : i].to_numpy(dtype=float)
        if ref.size != L or not np.isfinite(ref).all():
            return False
        close_t = df["close"].iloc[i]
        if not self._finite(close_t):
            return False
        return float(close_t) < float(np.min(ref))

    def expansion_at(self, df: pd.DataFrame, i: int) -> bool:
        """ATR_t >= m · avgATR_t; equality passes. Zero/invalid baseline fails."""
        X = int(self.params["expansion_lookback"])
        if i < X:
            return False
        atr_t = df["atr_sma"].iloc[i]
        base = df["atr_baseline"].iloc[i]
        if not self._finite(atr_t) or not self._finite(base):
            return False
        if float(base) <= 0.0:
            return False
        hist = df["atr_sma"].iloc[i - X : i].to_numpy(dtype=float)
        if hist.size != X or not np.isfinite(hist).all():
            return False
        m = float(self.params["expansion_multiplier"])
        return float(atr_t) >= m * float(base)

    def long_trend_at(self, df: pd.DataFrame, i: int) -> bool:
        fast = df["ema_fast"].iloc[i]
        slow = df["ema_slow"].iloc[i]
        if not self._finite(fast) or not self._finite(slow):
            return False
        return float(fast) > float(slow)

    def short_trend_at(self, df: pd.DataFrame, i: int) -> bool:
        fast = df["ema_fast"].iloc[i]
        slow = df["ema_slow"].iloc[i]
        if not self._finite(fast) or not self._finite(slow):
            return False
        return float(fast) < float(slow)

    def entry_signal(self, df: pd.DataFrame, i: int, ctx: Optional[Dict] = None) -> Tuple[Direction, str]:
        if i < WARMUP_MIN_INDEX:
            return Direction.NONE, ""
        if not self.compression_at(df, i):
            return Direction.NONE, ""
        if not self.expansion_at(df, i):
            return Direction.NONE, ""

        long_ok = self.long_break_at(df, i) and self.long_trend_at(df, i)
        short_ok = self.short_break_at(df, i) and self.short_trend_at(df, i)
        if long_ok and not short_ok:
            return Direction.LONG, "long_compression_breakout"
        if short_ok and not long_ok:
            return Direction.SHORT, "short_compression_breakout"
        return Direction.NONE, ""

    def generate_signal(self, df: pd.DataFrame, i: int, ctx: Optional[Dict] = None) -> Signal:
        sig = super().generate_signal(df, i, ctx)
        if sig.is_active:
            sig.meta["signal_bar"] = i
            atr_t = df["atr_sma"].iloc[i] if "atr_sma" in df.columns else None
            if self._finite(atr_t):
                sig.meta["atr_t"] = float(atr_t)
        return sig

    def prepare_fill(
        self,
        df: pd.DataFrame,
        signal_bar_index: int,
        direction: Direction,
        actual_fill_price: float,
        sig: Optional[Signal] = None,
    ) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
        """Fill-anchored stop/TP: R = atr_multiplier × ATR_t (signal bar)."""
        atr = float(df["atr_sma"].iloc[signal_bar_index])
        if not np.isfinite(atr) or atr <= 0:
            return None, None, {}
        a = float(self.params["atr_multiplier"])
        rr = float(self.params["risk_reward_ratio"])
        R = a * atr
        if direction == Direction.LONG:
            stop = actual_fill_price - R
            take = actual_fill_price + rr * R
        else:
            stop = actual_fill_price + R
            take = actual_fill_price - rr * R
        return stop, take, {
            "fill_price": actual_fill_price,
            "risk_R": R,
            "atr_t": atr,
            "signal_bar": signal_bar_index,
            "stop_direction": direction.value,
            "slippage_price_displacement": None,  # filled by engine as pos.slippage
        }

    def update_stop(self, df: pd.DataFrame, bar_index: int, position) -> None:
        """Fixed stop/target only — no breakeven or trailing."""
        return

    def compute_stop_loss(
        self,
        df: pd.DataFrame,
        i: int,
        direction: Direction,
        entry_price: float,
    ) -> Optional[float]:
        atr = self._atr_at(df, i)
        if atr is None or atr <= 0:
            return None
        R = float(self.params["atr_multiplier"]) * atr
        if direction == Direction.LONG:
            return entry_price - R
        return entry_price + R
