"""Breakout + Retest + Volume Confirmation Strategy.

Quant Thesis
------------
Breakout trading in crypto often suffers from false breakouts and liquidity sweeps.
This strategy filters false breakouts by requiring:
1. Breakout: A decisive close beyond recent swing channels / Donchian levels with volume.
2. Retest: A retracement back to the broken breakout level where volume dries up.
3. Confirmation: A reaction candle bouncing off the broken level confirming support/resistance flip.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..indicators import (
    atr as calc_atr,
    donchian_channel as calc_donchian,
    sma as calc_sma,
    volume_sma as calc_volume_sma,
)
from ..logging_config import get_logger
from .base import BaseStrategy, Direction, StopLossSpec, StopType, TakeProfitSpec

logger = get_logger("strategies")


class BreakoutRetestStrategy(BaseStrategy):
    """Breakout + Retest + Volume Confirmation Strategy."""

    name = "Breakout Retest"
    strategy_type = "breakout_retest"
    description = "Trades swing breakouts only after a successful retest with volume contraction"
    supports_spot = True
    supports_futures = True
    supports_short = True

    @classmethod
    def default_timeframes(cls) -> List[str]:
        return ["15m", "1h", "4h"]

    @classmethod
    def param_grid(cls) -> Dict[str, list]:
        return {
            "lookback": [20, 30, 50],
            "breakout_vol_mult": [1.1, 1.3, 1.5],
            "atr_exp_mult": [0.0, 1.0, 1.2],
            "retest_vol_max": [0.9, 1.0, 1.2],
            "retest_tol_atr": [0.4, 0.6, 0.8],
            "atr_multiplier": [1.8, 2.2, 2.5],
            "risk_reward_ratio": [1.8, 2.2, 2.8],
            "stop_type": ["atr", "swing"],
        }

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        params = dict(params or {})
        params.setdefault("lookback", 30)
        params.setdefault("breakout_vol_mult", 1.2)
        params.setdefault("atr_exp_mult", 1.0)
        params.setdefault("retest_vol_max", 1.0)
        params.setdefault("retest_tol_atr", 0.5)
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
        if int(p.get("lookback", 30)) < 5:
            raise ValueError("lookback must be >= 5")
        if float(p.get("breakout_vol_mult", 1.2)) <= 0:
            raise ValueError("breakout_vol_mult must be > 0")
        if float(p.get("atr_exp_mult", 1.0)) < 0:
            raise ValueError("atr_exp_mult must be >= 0")

    def setup(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy().sort_values("timestamp").reset_index(drop=True)
        p = self.params
        lb = int(p.get("lookback", 30))
        b_vol_m = float(p.get("breakout_vol_mult", 1.2))
        r_vol_m = float(p.get("retest_vol_max", 1.0))
        tol_atr = float(p.get("retest_tol_atr", 0.5))
        atr_exp = float(p.get("atr_exp_mult", 1.0))

        out["atr_14"] = calc_atr(out["high"], out["low"], out["close"], 14, shift=1)
        out["atr_sma50"] = calc_sma(out["atr_14"], 50, shift=1)
        out["vol_sma20"] = calc_volume_sma(out["volume"], 20, shift=1)

        # Prior channel levels (shifted by 1 so known at bar open)
        dc = calc_donchian(out["high"], out["low"], lb, shift=1)
        out["dc_high"] = dc["dc_high"]
        out["dc_low"] = dc["dc_low"]

        # Breakout signals in recent past (1 to 5 bars ago) with ATR expansion
        atr_expansion_ok = (atr_exp <= 0.0) | (out["atr_14"] >= atr_exp * out["atr_sma50"])
        high_break = (
            (out["close"].shift(1) > out["dc_high"].shift(1))
            & (out["volume"].shift(1) >= b_vol_m * out["vol_sma20"].shift(1))
            & atr_expansion_ok.shift(1)
        )
        low_break = (
            (out["close"].shift(1) < out["dc_low"].shift(1))
            & (out["volume"].shift(1) >= b_vol_m * out["vol_sma20"].shift(1))
            & atr_expansion_ok.shift(1)
        )

        out["recent_high_break"] = high_break.rolling(5, min_periods=1).max().astype(bool)
        out["recent_low_break"] = low_break.rolling(5, min_periods=1).max().astype(bool)

        # Retest condition at current closed bar:
        # Long: price tests dc_high within tolerance, volume is calm, and closed with bullish reaction
        dist_high = (out["low"] - out["dc_high"]).abs()
        retest_long_ok = (dist_high <= tol_atr * out["atr_14"]) & (out["volume"] <= r_vol_m * out["vol_sma20"]) & (out["close"] >= out["open"])

        # Short: price tests dc_low within tolerance, volume is calm, and closed with bearish reaction
        dist_low = (out["high"] - out["dc_low"]).abs()
        retest_short_ok = (dist_low <= tol_atr * out["atr_14"]) & (out["volume"] <= r_vol_m * out["vol_sma20"]) & (out["close"] <= out["open"])

        out["_long_setup"] = out["recent_high_break"] & retest_long_ok & (out["close"] >= out["dc_high"] * 0.998)
        out["_short_setup"] = out["recent_low_break"] & retest_short_ok & (out["close"] <= out["dc_low"] * 1.002)

        return out

    def entry_signal(self, df: pd.DataFrame, i: int, ctx: Optional[Dict] = None) -> Tuple[Direction, str]:
        if i < 30:
            return Direction.NONE, ""

        if bool(df["_long_setup"].iloc[i]):
            return Direction.LONG, "Breakout Retest Bounce"

        if bool(df["_short_setup"].iloc[i]):
            return Direction.SHORT, "Breakout Retest Rejection"

        return Direction.NONE, ""
