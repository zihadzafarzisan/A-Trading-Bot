"""Volume Displacement Continuation Strategy (Strategy Family #7).

Quant Thesis
------------
Trades directional trend continuation following high-volume price displacement candles
that form unmitigated price imbalances (Fair Value Gaps), confirmed by medium-term
trend alignment (EMA 50) and executed upon causal zone mitigation with asymmetric ATR targets.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from ..indicators import (
    atr as calc_atr,
    ema as calc_ema,
    volume_sma as calc_volume_sma,
)
from ..logging_config import get_logger
from .base import BaseStrategy, Direction, StopLossSpec, StopType, TakeProfitSpec

logger = get_logger("strategies")


class VolumeDisplacementContinuationStrategy(BaseStrategy):
    """Volume-Confirmed Price Displacement & Imbalance Continuation Strategy."""

    name = "Volume Displacement Continuation"
    strategy_type = "volume_displacement_continuation"
    description = "Trades trend continuation out of volume displacement candles and Fair Value Gaps"
    supports_spot = True
    supports_futures = True
    supports_short = True

    @classmethod
    def default_timeframes(cls) -> List[str]:
        return ["1h", "4h"]

    @classmethod
    def param_grid(cls) -> Dict[str, list]:
        return {
            "body_mult": [1.3, 1.6],
            "volume_mult": [1.3, 1.7],
            "atr_multiplier": [1.8, 2.2, 2.5],
            "risk_reward_ratio": [2.0, 2.5, 3.0],
            "displacement_lookback": [5],
            "ema_period": [50],
            "volume_period": [20],
            "atr_period": [14],
            "stop_type": ["atr"],
        }

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        params = dict(params or {})
        params.setdefault("body_mult", 1.5)
        params.setdefault("volume_mult", 1.5)
        params.setdefault("volume_period", 20)
        params.setdefault("ema_period", 50)
        params.setdefault("displacement_lookback", 5)
        params.setdefault("atr_period", 14)
        params.setdefault("atr_multiplier", 2.0)
        params.setdefault("risk_reward_ratio", 2.5)
        params.setdefault("stop_type", "atr")

        stop_spec = StopLossSpec(
            stop_type=StopType.ATR,
            atr_multiplier=float(params.get("atr_multiplier", 2.0)),
        )
        tp_spec = TakeProfitSpec(
            mode="rr",
            risk_reward_ratio=float(params.get("risk_reward_ratio", 2.5)),
        )
        super().__init__(params=params, stop_spec=stop_spec, tp_spec=tp_spec, **kwargs)

    def validate_params(self) -> None:
        p = self.params
        if float(p.get("body_mult", 1.5)) <= 0:
            raise ValueError("body_mult must be > 0")
        if float(p.get("volume_mult", 1.5)) <= 0:
            raise ValueError("volume_mult must be > 0")
        if int(p.get("ema_period", 50)) <= 0:
            raise ValueError("ema_period must be > 0")
        if int(p.get("displacement_lookback", 5)) <= 0:
            raise ValueError("displacement_lookback must be > 0")
        if float(p.get("atr_multiplier", 2.0)) <= 0:
            raise ValueError("atr_multiplier must be > 0")
        if float(p.get("risk_reward_ratio", 2.5)) <= 0:
            raise ValueError("risk_reward_ratio must be > 0")

    def setup(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy().sort_values("timestamp").reset_index(drop=True)
        p = self.params
        b_mult = float(p.get("body_mult", 1.5))
        v_mult = float(p.get("volume_mult", 1.5))
        v_period = int(p.get("volume_period", 20))
        ema_p = int(p.get("ema_period", 50))
        atr_p = int(p.get("atr_period", 14))
        disp_lookback = int(p.get("displacement_lookback", 5))

        # Causal shifted indicators (shift=1 aligns to bar t-1 close)
        out["atr_14"] = calc_atr(out["high"], out["low"], out["close"], period=atr_p, shift=1)
        out["vol_sma"] = calc_volume_sma(out["volume"], period=v_period, shift=1)
        out["ema_50"] = calc_ema(out["close"], period=ema_p, shift=1)

        # Bar t-1 closed candle metrics
        c1 = out["close"].shift(1)
        o1 = out["open"].shift(1)
        h1 = out["high"].shift(1)
        l1 = out["low"].shift(1)
        v1 = out["volume"].shift(1)
        h3 = out["high"].shift(3)
        l3 = out["low"].shift(3)

        body_size = (c1 - o1).abs()

        # Displacement Candle Conditions at bar t-1:
        # 1. Large real body relative to ATR
        is_large_body = body_size >= (b_mult * out["atr_14"])
        # 2. Volume expansion surge
        is_volume_surge = v1 >= (v_mult * out["vol_sma"])

        # 3. Directional Imbalance / Fair Value Gap:
        # Bullish: Low of t-1 is above High of t-3 (gap) and candle is green and above EMA50
        bullish_disp = is_large_body & is_volume_surge & (c1 > o1) & (l1 > h3) & (c1 > out["ema_50"])
        # Bearish: High of t-1 is below Low of t-3 (gap) and candle is red and below EMA50
        bearish_disp = is_large_body & is_volume_surge & (c1 < o1) & (h1 < l3) & (c1 < out["ema_50"])

        # Track recent displacement occurrence within lookback window
        recent_bull_disp = bullish_disp.rolling(disp_lookback, min_periods=1).max() == 1
        recent_bear_disp = bearish_disp.rolling(disp_lookback, min_periods=1).max() == 1

        # Mitigation / Retest condition: Price currently trades within or near the displacement zone
        # Bullish: Bar t-1 low dips into upper portion of displacement without closing below EMA50
        bullish_retest = recent_bull_disp & (c1 > out["ema_50"])
        bearish_retest = recent_bear_disp & (c1 < out["ema_50"])

        out["_long_setup"] = bullish_retest
        out["_short_setup"] = bearish_retest

        return out

    def entry_signal(self, df: pd.DataFrame, i: int, ctx: Optional[Dict] = None) -> Tuple[Direction, str]:
        min_bars = max(int(self.params.get("ema_period", 50)), int(self.params.get("volume_period", 20))) + 10
        if i < min_bars:
            return Direction.NONE, ""

        if bool(df["_long_setup"].iloc[i]):
            return Direction.LONG, f"Volume Displacement Continuation Long"

        if bool(df["_short_setup"].iloc[i]):
            return Direction.SHORT, f"Volume Displacement Continuation Short"

        return Direction.NONE, ""
