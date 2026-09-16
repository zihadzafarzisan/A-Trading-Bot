"""Regime-Adaptive Trend / Mean-Reversion Strategy.

Quant Thesis
------------
Markets alternate between high-variance directional trends and mean-reverting ranges.
This strategy classifies market state causally per bar using ADX, Choppiness Index,
and Moving Average structure, dynamically deploying:
- Trend-Following execution during trending regimes (ADX >= threshold / Low Choppiness).
- Mean-Reversion execution during consolidating regimes (ADX < threshold / High Choppiness).
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..indicators import (
    adx as calc_adx,
    atr as calc_atr,
    bollinger_bands as calc_bb,
    choppiness_index as calc_chop,
    ema as calc_ema,
    rsi as calc_rsi,
    sma as calc_sma,
)
from ..logging_config import get_logger
from .base import BaseStrategy, Direction, StopLossSpec, StopType, TakeProfitSpec

logger = get_logger("strategies")


class RegimeAdaptiveStrategy(BaseStrategy):
    """Regime-Adaptive Trend / Mean-Reversion Strategy."""

    name = "Regime-Adaptive"
    strategy_type = "regime_adaptive"
    description = "Dynamically switches between trend-following and mean-reversion based on causal regime detection"
    supports_spot = True
    supports_futures = True
    supports_short = True

    @classmethod
    def default_timeframes(cls) -> List[str]:
        return ["1h", "4h", "1d"]

    @classmethod
    def param_grid(cls) -> Dict[str, list]:
        return {
            "adx_trend_thresh": [22.0, 25.0],
            "chop_range_thresh": [55.0, 60.0],
            "trend_ema_fast": [20],
            "trend_ema_slow": [50],
            "bb_period": [20],
            "bb_std": [2.0],
            "atr_multiplier": [2.0, 2.5],
            "risk_reward_ratio": [2.0, 2.5],
            "stop_type": ["atr", "swing"],
        }

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        params = dict(params or {})
        params.setdefault("adx_trend_thresh", 25.0)
        params.setdefault("chop_range_thresh", 55.0)
        params.setdefault("trend_ema_fast", 20)
        params.setdefault("trend_ema_slow", 50)
        params.setdefault("bb_period", 20)
        params.setdefault("bb_std", 2.0)
        params.setdefault("atr_multiplier", 2.0)
        params.setdefault("risk_reward_ratio", 2.0)
        params.setdefault("stop_type", "atr")
        params.setdefault("swing_lookback", 15)

        stop_type = StopType.SWING if params.get("stop_type") == "swing" else StopType.ATR
        stop_spec = StopLossSpec(
            stop_type=stop_type,
            atr_multiplier=float(params.get("atr_multiplier", 2.0)),
            swing_lookback=int(params.get("swing_lookback", 15)),
        )
        tp_spec = TakeProfitSpec(
            mode="rr",
            risk_reward_ratio=float(params.get("risk_reward_ratio", 2.0)),
        )
        super().__init__(params=params, stop_spec=stop_spec, tp_spec=tp_spec, **kwargs)

    def validate_params(self) -> None:
        p = self.params
        if float(p.get("adx_trend_thresh", 25.0)) <= 0:
            raise ValueError("adx_trend_thresh must be > 0")

    def setup(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy().sort_values("timestamp").reset_index(drop=True)
        p = self.params
        adx_th = float(p.get("adx_trend_thresh", 25.0))
        chop_th = float(p.get("chop_range_thresh", 55.0))
        t_fast = int(p.get("trend_ema_fast", 20))
        t_slow = int(p.get("trend_ema_slow", 50))
        bb_p = int(p.get("bb_period", 20))
        bb_s = float(p.get("bb_std", 2.0))

        # Causal shifted indicators
        out["adx"] = calc_adx(out["high"], out["low"], out["close"], 14, shift=1)
        out["chop"] = calc_chop(out["high"], out["low"], out["close"], 14, shift=1)
        out["atr_14"] = calc_atr(out["high"], out["low"], out["close"], 14, shift=1)
        out["ema_fast"] = calc_ema(out["close"], t_fast, shift=1)
        out["ema_slow"] = calc_ema(out["close"], t_slow, shift=1)
        out["rsi"] = calc_rsi(out["close"], 14, shift=1)

        bb = calc_bb(out["close"], bb_p, bb_s, shift=1)
        out["bb_mid"] = bb["bb_mid"]
        out["bb_upper"] = bb["bb_upper"]
        out["bb_lower"] = bb["bb_lower"]

        # Causal Regime Classification
        out["is_trend_regime"] = (out["adx"] >= adx_th) | (out["chop"] <= 42.0)
        out["is_range_regime"] = (out["adx"] < adx_th) & (out["chop"] >= chop_th)

        # 1. Trend Sub-Strategy Signals
        trend_long = out["is_trend_regime"] & (out["ema_fast"] > out["ema_slow"]) & (out["close"].shift(1) > out["ema_fast"]) & (out["rsi"] <= 60.0)
        trend_short = out["is_trend_regime"] & (out["ema_fast"] < out["ema_slow"]) & (out["close"].shift(1) < out["ema_fast"]) & (out["rsi"] >= 40.0)

        # 2. Mean-Reversion Sub-Strategy Signals
        mr_long = out["is_range_regime"] & (out["low"].shift(1) <= out["bb_lower"]) & (out["close"].shift(1) > out["open"].shift(1))
        mr_short = out["is_range_regime"] & (out["high"].shift(1) >= out["bb_upper"]) & (out["close"].shift(1) < out["open"].shift(1))

        out["_long_setup"] = trend_long | mr_long
        out["_short_setup"] = trend_short | mr_short

        return out

    def entry_signal(self, df: pd.DataFrame, i: int, ctx: Optional[Dict] = None) -> Tuple[Direction, str]:
        if i < 30:
            return Direction.NONE, ""

        is_trend = bool(df["is_trend_regime"].iloc[i])
        mode = "Trend" if is_trend else "MeanReversion"

        if bool(df["_long_setup"].iloc[i]):
            return Direction.LONG, f"Regime-Adaptive Long ({mode})"

        if bool(df["_short_setup"].iloc[i]):
            return Direction.SHORT, f"Regime-Adaptive Short ({mode})"

        return Direction.NONE, ""
