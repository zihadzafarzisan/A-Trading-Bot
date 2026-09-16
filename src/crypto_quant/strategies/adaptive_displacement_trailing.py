"""Adaptive Displacement & Structural Trailing strategy.

Implements PROTOCOL-STRATEGY-08-V1.3 exactly:
- Displacement candle at t-1 with body & volume qualification (Section 5)
- Signed Kaufman Efficiency Ratio regime filter (Section 7)
- Fill-price-anchored initial risk / initial stop / take profit (Section 2)
- Breakeven stop at FillPrice after +1.0R confirmation (Section 3)
- ATR trailing ratchet after breakeven (Section 4)

Signal at bar k -> execution at open[k+1] (engine enforces 1-bar delay).
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..backtesting.portfolio import Position
from ..indicators import trend as tn
from ..indicators import volatility as vt
from ..indicators.volume import volume_sma, volume_ratio
from ..logging_config import get_logger
from .base import BaseStrategy, Direction, Signal, StopLossSpec, TakeProfitSpec

logger = get_logger("strategies")

SLIPPAGE = 0.0005


class AdaptiveDisplacementTrailingStrategy(BaseStrategy):
    name = "Adaptive Displacement Trailing"
    strategy_type = "adaptive_displacement_trailing"
    description = (
        "Displacement/structural trailing: FVG + signed Kaufman ER entry, "
        "breakeven at FillPrice + 1.0R, then 14-period ATR trailing ratchet."
    )
    supports_spot = True
    supports_futures = True
    supports_short = True

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        params = dict(params or {})
        params.setdefault("body_mult", 1.6)
        params.setdefault("volume_mult", 1.7)
        params.setdefault("body_period", 20)
        params.setdefault("volume_period", 20)
        params.setdefault("ema_period", 50)
        params.setdefault("er_period", 10)
        params.setdefault("er_threshold", 0.30)
        params.setdefault("displacement_lookback", 5)
        params.setdefault("atr_period", 14)
        params.setdefault("atr_multiplier", 2.2)
        params.setdefault("risk_reward_ratio", 2.0)
        params.setdefault("breakeven_trigger_r", 1.0)
        params.setdefault("stop_type", "atr_breakeven")
        super().__init__(params, **kwargs)
        self.stop_spec = StopLossSpec(stop_type="atr_breakeven")
        self.tp_spec = TakeProfitSpec(mode="rr", risk_reward_ratio=self.params["risk_reward_ratio"])

    # ------------------------------------------------------------ contract
    def validate_params(self) -> None:
        p = self.params
        for k in ("body_mult", "volume_mult", "atr_multiplier", "risk_reward_ratio", "breakeven_trigger_r"):
            v = float(p[k])
            if v <= 0:
                raise ValueError(f"{k} must be > 0, got {v}")
        for k in ("body_period", "volume_period", "ema_period", "er_period", "atr_period", "displacement_lookback"):
            v = int(p[k])
            if v <= 0:
                raise ValueError(f"{k} must be > 0, got {v}")
        v = float(p["er_threshold"])
        if not (0 <= v <= 1):
            raise ValueError(f"er_threshold out of range [0,1]: {v}")

    # ------------------------------------------------------------ setup
    def setup(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        ema_p = int(self.params["ema_period"])
        atr_p = int(self.params["atr_period"])
        er_p = int(self.params["er_period"])
        body_p = int(self.params["body_period"])
        vol_p = int(self.params["volume_period"])
        out["ema_50"] = tn.ema(out["close"], ema_p, shift=1)
        out["atr_14"] = vt.atr(out["high"], out["low"], out["close"], atr_p, shift=0)
        out["vol_sma_20"] = volume_sma(out["volume"], vol_p, shift=0)
        out["body"] = (out["close"] - out["open"]).abs()
        out["avg_body_20"] = out["body"].rolling(body_p, min_periods=body_p).mean()
        out["avg_vol_20"] = out["volume"].rolling(vol_p, min_periods=vol_p).mean()
        # Signed Kaufman ER: directional change / total volatility (negative in downtrends).
        out["er"] = self._compute_er(out["close"], er_p)
        return out

    @staticmethod
    def _compute_er(close: pd.Series, n: int) -> pd.Series:
        # Signed directional Kaufman Efficiency Ratio, as required by the
        # frozen contract's directed regime filter (long: positive ER,
        # short: negative ER). Zero-volatility -> 0.0.
        directional = close - close.shift(n)
        total_vol = close.diff().abs().rolling(n).sum()
        er = directional / total_vol
        return er.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)

    # ------------------------------------------------------------ FVG
    def _fvg_bull_zone(self, df: pd.DataFrame, t: int) -> Optional[tuple[float, float]]:
        """Bullish FVG zone [High_(t-3), Low_(t-1)] if conditions met.

        Protocol V1.3 Section 5.1 & 5.3:
        - Evaluated on completed bar t-1 (displacement candle).
        - Requires 20 completed bars ending at t-2: bars t-21 through t-2.
        - Minimum reference bar index: t >= 21.
        - Geometric conditions:
            Close_(t-1) > Open_(t-1)
            Low_(t-1) > High_(t-3)
            Close_(t-1) > EMA50_(t-1)
        - Body qualification:
            Body_(t-1) >= body_mult * AvgBody_(t-1)
            where AvgBody_(t-1) = mean(Body[t-21 ... t-2]).
            Zero baseline -> FAIL.
            Equality -> PASS.
        - Volume qualification:
            Volume_(t-1) >= volume_mult * AvgVolume_(t-1)
            where AvgVolume_(t-1) = mean(Volume[t-21 ... t-2]).
            Zero baseline -> FAIL.
            Equality -> PASS.
        """
        if t < 21:
            return None
        t1 = t - 1
        t3 = t - 3

        ema_val = df["ema_50"].iloc[t1] if "ema_50" in df.columns else None
        if ema_val is None or pd.isna(ema_val):
            return None

        close_t1 = float(df["close"].iloc[t1])
        open_t1 = float(df["open"].iloc[t1])
        low_t1 = float(df["low"].iloc[t1])
        high_t3 = float(df["high"].iloc[t3])

        # Geometric conditions (V1.2 §5.1, preserved in V1.3 §5.3)
        if not (close_t1 > open_t1 and low_t1 > high_t3 and close_t1 > float(ema_val)):
            return None

        # Body baseline: 20 completed bars ending at t-2 (bars t-21 ... t-2)
        if "avg_body_20" in df.columns:
            avg_body = float(df["avg_body_20"].iloc[t - 2])
        else:
            avg_body = float((df["close"].iloc[t - 21 : t - 1] - df["open"].iloc[t - 21 : t - 1]).abs().mean())

        # Zero baseline or NaN fails
        if pd.isna(avg_body) or avg_body <= 0:
            return None

        body_t1 = abs(close_t1 - open_t1)
        body_mult = float(self.params["body_mult"])
        body_thresh = body_mult * avg_body
        # Body qualification: Body_(t-1) >= body_mult * AvgBody_(t-1) (equality allowed)
        if body_t1 < body_thresh and not np.isclose(body_t1, body_thresh, rtol=1e-9, atol=1e-12):
            return None

        # Volume baseline: 20 completed bars ending at t-2 (bars t-21 ... t-2)
        if "avg_vol_20" in df.columns:
            avg_vol = float(df["avg_vol_20"].iloc[t - 2])
        else:
            avg_vol = float(df["volume"].iloc[t - 21 : t - 1].mean())

        # Zero baseline or NaN fails
        if pd.isna(avg_vol) or avg_vol <= 0:
            return None

        vol_t1 = float(df["volume"].iloc[t1])
        vol_mult = float(self.params["volume_mult"])
        vol_thresh = vol_mult * avg_vol
        # Volume qualification: Volume_(t-1) >= volume_mult * AvgVolume_(t-1) (equality allowed)
        if vol_t1 < vol_thresh and not np.isclose(vol_t1, vol_thresh, rtol=1e-9, atol=1e-12):
            return None

        return high_t3, low_t1

    def _fvg_bear_zone(self, df: pd.DataFrame, t: int) -> Optional[tuple[float, float]]:
        """Bearish FVG zone [High_(t-1), Low_(t-3)] if conditions met.

        Protocol V1.3 Section 5.2 & 5.4:
        - Evaluated on completed bar t-1 (displacement candle).
        - Requires 20 completed bars ending at t-2: bars t-21 through t-2.
        - Minimum reference bar index: t >= 21.
        - Geometric conditions:
            Close_(t-1) < Open_(t-1)
            High_(t-1) < Low_(t-3)
            Close_(t-1) < EMA50_(t-1)
        - Body qualification:
            Body_(t-1) >= body_mult * AvgBody_(t-1)
            where AvgBody_(t-1) = mean(Body[t-21 ... t-2]).
            Zero baseline -> FAIL.
            Equality -> PASS.
        - Volume qualification:
            Volume_(t-1) >= volume_mult * AvgVolume_(t-1)
            where AvgVolume_(t-1) = mean(Volume[t-21 ... t-2]).
            Zero baseline -> FAIL.
            Equality -> PASS.
        """
        if t < 21:
            return None
        t1 = t - 1
        t3 = t - 3

        ema_val = df["ema_50"].iloc[t1] if "ema_50" in df.columns else None
        if ema_val is None or pd.isna(ema_val):
            return None

        close_t1 = float(df["close"].iloc[t1])
        open_t1 = float(df["open"].iloc[t1])
        high_t1 = float(df["high"].iloc[t1])
        low_t3 = float(df["low"].iloc[t3])

        # Geometric conditions (V1.2 §5.2, preserved in V1.3 §5.4)
        if not (close_t1 < open_t1 and high_t1 < low_t3 and close_t1 < float(ema_val)):
            return None

        # Body baseline: 20 completed bars ending at t-2 (bars t-21 ... t-2)
        if "avg_body_20" in df.columns:
            avg_body = float(df["avg_body_20"].iloc[t - 2])
        else:
            avg_body = float((df["close"].iloc[t - 21 : t - 1] - df["open"].iloc[t - 21 : t - 1]).abs().mean())

        # Zero baseline or NaN fails
        if pd.isna(avg_body) or avg_body <= 0:
            return None

        body_t1 = abs(close_t1 - open_t1)
        body_mult = float(self.params["body_mult"])
        body_thresh = body_mult * avg_body
        # Body qualification: Body_(t-1) >= body_mult * AvgBody_(t-1) (equality allowed)
        if body_t1 < body_thresh and not np.isclose(body_t1, body_thresh, rtol=1e-9, atol=1e-12):
            return None

        # Volume baseline: 20 completed bars ending at t-2 (bars t-21 ... t-2)
        if "avg_vol_20" in df.columns:
            avg_vol = float(df["avg_vol_20"].iloc[t - 2])
        else:
            avg_vol = float(df["volume"].iloc[t - 21 : t - 1].mean())

        # Zero baseline or NaN fails
        if pd.isna(avg_vol) or avg_vol <= 0:
            return None

        vol_t1 = float(df["volume"].iloc[t1])
        vol_mult = float(self.params["volume_mult"])
        vol_thresh = vol_mult * avg_vol
        # Volume qualification: Volume_(t-1) >= volume_mult * AvgVolume_(t-1) (equality allowed)
        if vol_t1 < vol_thresh and not np.isclose(vol_t1, vol_thresh, rtol=1e-9, atol=1e-12):
            return None

        return high_t1, low_t3

    def _active_fvg_zone(self, df: pd.DataFrame, i: int, direction: Direction):
        """Most recent active FVG zone valid at bar i (validity window incl.)."""
        lookback = int(self.params["displacement_lookback"])
        # FVG formed at reference t is valid while i in [t, t+lookback-1].
        # We scan possible formation bars t. Most recent = max t satisfying window.
        best = None
        for t in range(max(0, i - lookback + 1), i + 1):
            if direction == Direction.LONG:
                zone = self._fvg_bull_zone(df, t)
            else:
                zone = self._fvg_bear_zone(df, t)
            if zone is not None and t <= i <= t + lookback - 1:
                best = zone
        return best

    # ------------------------------------------------------------ entry
    def entry_signal(self, df: pd.DataFrame, i: int, ctx=None):
        if i < 3 or pd.isna(df["atr_14"].iloc[i]) or pd.isna(df["ema_50"].iloc[i]):
            return Direction.NONE, ""

        er = float(df["er"].iloc[i])
        er_thr = float(self.params["er_threshold"])

        direction = Direction.NONE
        reason = ""
        if er >= er_thr:
            zone = self._active_fvg_zone(df, i, Direction.LONG)
            if zone is not None:
                low_k = float(df["low"].iloc[i])
                if low_k <= zone[1] and low_k >= zone[0] and float(df["close"].iloc[i]) > float(df["ema_50"].iloc[i]):
                    direction = Direction.LONG
                    reason = "long_fvg_mitigation"
        elif er <= -er_thr:
            zone = self._active_fvg_zone(df, i, Direction.SHORT)
            if zone is not None:
                high_k = float(df["high"].iloc[i])
                if high_k >= zone[0] and high_k <= zone[1] and float(df["close"].iloc[i]) < float(df["ema_50"].iloc[i]):
                    direction = Direction.SHORT
                    reason = "short_fvg_mitigation"

        return direction, reason

    # ------------------------------------------------------------ fill (price-anchored)
    def prepare_fill(
        self,
        df: pd.DataFrame,
        signal_bar_index: int,
        direction: Direction,
        actual_fill_price: float,
        sig: Optional[Signal] = None,
    ) -> tuple[Optional[float], Optional[float], Dict[str, Any]]:
        """Return (stop, take, risk_state) based on EXACT fill price + signal-bar ATR."""
        p = self.params
        atr_p = int(p["atr_period"])
        atr = float(df["atr_14"].iloc[signal_bar_index])
        if pd.isna(atr) or atr <= 0:
            return None, None, {}
        R = float(p["atr_multiplier"]) * atr

        if direction == Direction.LONG:
            stop = actual_fill_price - R
            take = actual_fill_price + float(p["risk_reward_ratio"]) * R
        else:
            stop = actual_fill_price + R
            take = actual_fill_price - float(p["risk_reward_ratio"]) * R

        return stop, take, {
            "fill_price": actual_fill_price,
            "risk_R": R,
            "atr_k": atr,
            "signal_bar": signal_bar_index,
            "breakeven_triggered": False,
            "trailing_active": False,
            "stop_direction": direction.value,
        }

    # ------------------------------------------------------------ stop mgmt
    def update_stop(self, df: pd.DataFrame, bar_index: int, position: Position) -> None:
        state = position.extra_state
        if not state or "risk_R" not in state or state.get("risk_R") is None:
            return
        R = float(state["risk_R"])
        direction = state.get("stop_direction", position.direction)
        p = self.params
        # ---- Breakeven trigger (evaluated on completed bar m = bar_index - 1)
        if not state.get("breakeven_triggered", False):
            if bar_index < 1:
                return
            close_m = float(df["close"].iloc[bar_index - 1])
            fill = float(state["fill_price"])
            if direction == "long":
                if close_m >= fill + R * float(p["breakeven_trigger_r"]):
                    state["breakeven_triggered"] = True
                    state["trailing_active"] = True
                    position.active_stop = fill
                    position.breakeven_active = True
            else:
                if close_m <= fill - R * float(p["breakeven_trigger_r"]):
                    state["breakeven_triggered"] = True
                    state["trailing_active"] = True
                    position.active_stop = fill
                    position.breakeven_active = True
            return

        # ---- ATR trailing (completed bar j = bar_index - 1; effective next bar)
        if state.get("trailing_active", False) and bar_index >= 1:
            j = bar_index - 1
            if "atr_14" in df.columns:
                atr_j = float(df["atr_14"].iloc[j])
                close_j = float(df["close"].iloc[j])
                if not pd.isna(atr_j) and atr_j > 0:
                    atr_mult = float(p["atr_multiplier"])
                    if direction == "long":
                        candidate = close_j - atr_mult * atr_j
                        new_stop = max(float(position.active_stop or fill), candidate)
                    else:
                        candidate = close_j + atr_mult * atr_j
                        new_stop = min(float(position.active_stop or fill), candidate)
                    position.active_stop = new_stop

    # ------------------------------------------------------------ grid
    @classmethod
    def param_grid(cls) -> Dict[str, list]:
        return {
            "body_mult": [1.4, 1.7],
            "volume_mult": [1.5, 1.8],
            "er_threshold": [0.25, 0.35, 0.45],
            "atr_multiplier": [1.8, 2.2, 2.5],
        }


C00_PARAMS = {
    "candidate_id": "C00",
    "body_mult": 1.6,
    "volume_mult": 1.7,
    "body_period": 20,
    "volume_period": 20,
    "ema_period": 50,
    "er_period": 10,
    "er_threshold": 0.30,
    "displacement_lookback": 5,
    "atr_period": 14,
    "atr_multiplier": 2.2,
    "risk_reward_ratio": 2.0,
    "breakeven_trigger_r": 1.0,
    "stop_type": "atr_breakeven",
}
