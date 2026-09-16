"""Trend Channel Breakout Strategy (Strategy Family #6).

Quant Thesis
------------
Captures structural momentum continuation and directional trend expansions.
Enters upon breakout of a N-period Donchian Channel on closed-bar price action,
filtered by medium-term EMA trend alignment and ADX trend strength to eliminate
range-bound chop and whipsaw stop-outs. Uses ATR-based asymmetric risk management.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from ..indicators import (
    atr as calc_atr,
    donchian_channel as calc_donchian,
    ema as calc_ema,
    adx as calc_adx,
)
from ..logging_config import get_logger
from .base import BaseStrategy, Direction, StopLossSpec, StopType, TakeProfitSpec

logger = get_logger("strategies")


class TrendChannelBreakoutStrategy(BaseStrategy):
    """Donchian Channel Breakout Strategy with EMA Trend Alignment and ADX Filtering."""

    name = "Trend Channel Breakout"
    strategy_type = "trend_channel_breakout"
    description = "Trades directional breakouts of Donchian channels confirmed by EMA trend and ADX strength"
    supports_spot = True
    supports_futures = True
    supports_short = True

    @classmethod
    def default_timeframes(cls) -> List[str]:
        return ["1h", "4h"]

    @classmethod
    def param_grid(cls) -> Dict[str, list]:
        return {
            "channel_period": [20, 30],
            "adx_threshold": [20.0, 25.0],
            "atr_multiplier": [1.8, 2.2, 2.5],
            "risk_reward_ratio": [2.0, 2.5, 3.0],
            "ema_period": [50],
            "adx_period": [14],
            "atr_period": [14],
            "stop_type": ["atr"],
        }

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        params = dict(params or {})
        params.setdefault("channel_period", 20)
        params.setdefault("ema_period", 50)
        params.setdefault("adx_period", 14)
        params.setdefault("adx_threshold", 20.0)
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
        if int(p.get("channel_period", 20)) <= 0:
            raise ValueError("channel_period must be > 0")
        if int(p.get("ema_period", 50)) <= 0:
            raise ValueError("ema_period must be > 0")
        if float(p.get("adx_threshold", 20.0)) < 0:
            raise ValueError("adx_threshold must be >= 0")
        if float(p.get("atr_multiplier", 2.0)) <= 0:
            raise ValueError("atr_multiplier must be > 0")
        if float(p.get("risk_reward_ratio", 2.5)) <= 0:
            raise ValueError("risk_reward_ratio must be > 0")

    def setup(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy().sort_values("timestamp").reset_index(drop=True)
        p = self.params
        ch_p = int(p.get("channel_period", 20))
        ema_p = int(p.get("ema_period", 50))
        adx_p = int(p.get("adx_period", 14))
        adx_thresh = float(p.get("adx_threshold", 20.0))
        atr_p = int(p.get("atr_period", 14))

        # Causal shifted indicators:
        # shift=2 for Donchian Channel ensures the channel is calculated over [t-1-period .. t-2],
        # so bar t-1's close can be compared against the prior channel boundary without look-ahead.
        dc = calc_donchian(out["high"], out["low"], period=ch_p, shift=2)
        out["dc_high"] = dc["dc_high"]
        out["dc_low"] = dc["dc_low"]
        out["dc_mid"] = dc["dc_mid"]

        # shift=1 aligns EMA, ADX, and ATR to bar t-1's close
        out["ema_trend"] = calc_ema(out["close"], period=ema_p, shift=1)
        out["adx_val"] = calc_adx(out["high"], out["low"], out["close"], period=adx_p, shift=1)
        out["atr_14"] = calc_atr(out["high"], out["low"], out["close"], period=atr_p, shift=1)

        # ADX trend strength filter at bar t-1
        adx_active = out["adx_val"] >= adx_thresh

        # Long Trigger: Bar t-1 Close breaches previous Donchian Upper, aligned above EMA trend, with strong ADX
        long_breakout = (out["close"].shift(1) > out["dc_high"]) & (out["close"].shift(1) > out["ema_trend"])
        # Short Trigger: Bar t-1 Close breaches previous Donchian Lower, aligned below EMA trend, with strong ADX
        short_breakout = (out["close"].shift(1) < out["dc_low"]) & (out["close"].shift(1) < out["ema_trend"])

        out["_long_setup"] = long_breakout & adx_active
        out["_short_setup"] = short_breakout & adx_active

        return out

    def entry_signal(self, df: pd.DataFrame, i: int, ctx: Optional[Dict] = None) -> Tuple[Direction, str]:
        # Safety warm-up guard
        min_bars = max(int(self.params.get("channel_period", 20)), int(self.params.get("ema_period", 50))) + 10
        if i < min_bars:
            return Direction.NONE, ""

        if bool(df["_long_setup"].iloc[i]):
            return Direction.LONG, f"Trend Channel Breakout Long (ADX={float(df['adx_val'].iloc[i]):.1f})"

        if bool(df["_short_setup"].iloc[i]):
            return Direction.SHORT, f"Trend Channel Breakout Short (ADX={float(df['adx_val'].iloc[i]):.1f})"

        return Direction.NONE, ""
