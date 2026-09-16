"""Volatility Squeeze Breakout Strategy (Strategy Family #5).

Quant Thesis
------------
In financial markets, extended periods of low volatility (consolidation/squeeze)
are systematically followed by explosive, fat-tailed volatility expansions.
This strategy detects when Bollinger Bands contract inside Keltner Channels
(volatility compression), and enters in the direction of the subsequent breakout
when confirmed by momentum and volume expansion.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from ..indicators import (
    atr as calc_atr,
    bollinger_bands as calc_bb,
    keltner_channel as calc_kc,
    momentum as calc_momentum,
    volume_sma as calc_volume_sma,
)
from ..logging_config import get_logger
from .base import BaseStrategy, Direction, StopLossSpec, StopType, TakeProfitSpec

logger = get_logger("strategies")


class VolatilitySqueezeBreakoutStrategy(BaseStrategy):
    """Keltner-Bollinger Volatility Squeeze Breakout Strategy."""

    name = "Volatility Squeeze Breakout"
    strategy_type = "volatility_squeeze_breakout"
    description = "Trades volatility expansions out of Keltner/Bollinger squeezes with momentum and volume confirmation"
    supports_spot = True
    supports_futures = True
    supports_short = True

    @classmethod
    def default_timeframes(cls) -> List[str]:
        return ["1h", "4h"]

    @classmethod
    def param_grid(cls) -> Dict[str, list]:
        return {
            "kc_mult": [1.2, 1.5, 1.8],
            "bb_std": [1.8, 2.0],
            "volume_mult": [1.0, 1.2],
            "atr_multiplier": [1.8, 2.2],
            "risk_reward_ratio": [2.0, 2.5],
            "bb_period": [20],
            "kc_period": [20],
            "mom_period": [12],
            "stop_type": ["atr"],
        }

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        params = dict(params or {})
        params.setdefault("bb_period", 20)
        params.setdefault("bb_std", 2.0)
        params.setdefault("kc_period", 20)
        params.setdefault("kc_atr_period", 14)
        params.setdefault("kc_mult", 1.5)
        params.setdefault("mom_period", 12)
        params.setdefault("volume_mult", 1.0)
        params.setdefault("volume_period", 20)
        params.setdefault("atr_period", 14)
        params.setdefault("atr_multiplier", 2.0)
        params.setdefault("risk_reward_ratio", 2.5)
        params.setdefault("squeeze_lookback", 10)
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
        if float(p.get("bb_std", 2.0)) <= 0:
            raise ValueError("bb_std must be > 0")
        if float(p.get("kc_mult", 1.5)) <= 0:
            raise ValueError("kc_mult must be > 0")
        if float(p.get("atr_multiplier", 2.0)) <= 0:
            raise ValueError("atr_multiplier must be > 0")

    def setup(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy().sort_values("timestamp").reset_index(drop=True)
        p = self.params
        bb_p = int(p.get("bb_period", 20))
        bb_s = float(p.get("bb_std", 2.0))
        kc_p = int(p.get("kc_period", 20))
        kc_atr_p = int(p.get("kc_atr_period", 14))
        kc_m = float(p.get("kc_mult", 1.5))
        mom_p = int(p.get("mom_period", 12))
        vol_m = float(p.get("volume_mult", 1.0))
        vol_p = int(p.get("volume_period", 20))
        atr_p = int(p.get("atr_period", 14))
        sqz_lookback = int(p.get("squeeze_lookback", 10))

        # Causal shifted indicators (shift=1 ensures values are point-in-time at bar close t-1)
        bb = calc_bb(out["close"], bb_p, bb_s, shift=1)
        out["bb_mid"] = bb["bb_mid"]
        out["bb_upper"] = bb["bb_upper"]
        out["bb_lower"] = bb["bb_lower"]

        kc = calc_kc(out["close"], out["high"], out["low"], kc_p, kc_atr_p, kc_m, shift=1)
        out["kc_mid"] = kc["kc_mid"]
        out["kc_upper"] = kc["kc_upper"]
        out["kc_lower"] = kc["kc_lower"]

        out["atr_14"] = calc_atr(out["high"], out["low"], out["close"], atr_p, shift=1)
        out["mom"] = calc_momentum(out["close"], mom_p, shift=1)
        out["vol_sma"] = calc_volume_sma(out["volume"], vol_p, shift=1)

        # Squeeze condition: Bollinger bands inside Keltner channel at t-1
        squeeze_on = (out["bb_upper"] <= out["kc_upper"]) & (out["bb_lower"] >= out["kc_lower"])
        # Recent squeeze: was in squeeze within the last `sqz_lookback` bars
        recent_squeeze = squeeze_on.rolling(sqz_lookback, min_periods=1).max() == 1

        # Volume confirmation
        vol_confirmed = out["volume"].shift(1) >= (vol_m * out["vol_sma"])

        # Long Trigger: Price breaks above KC Upper / BB Upper with positive momentum & volume expansion
        long_breakout = (out["close"].shift(1) > out["kc_upper"]) & (out["close"].shift(1) >= out["open"].shift(1)) & (out["mom"] > 0)
        # Short Trigger: Price breaks below KC Lower / BB Lower with negative momentum & volume expansion
        short_breakout = (out["close"].shift(1) < out["kc_lower"]) & (out["close"].shift(1) <= out["open"].shift(1)) & (out["mom"] < 0)

        out["_long_setup"] = recent_squeeze & long_breakout & vol_confirmed
        out["_short_setup"] = recent_squeeze & short_breakout & vol_confirmed

        return out

    def entry_signal(self, df: pd.DataFrame, i: int, ctx: Optional[Dict] = None) -> Tuple[Direction, str]:
        if i < 30:
            return Direction.NONE, ""

        if bool(df["_long_setup"].iloc[i]):
            return Direction.LONG, f"Squeeze Expansion Breakout Long (Mom={float(df['mom'].iloc[i]):.2f})"

        if bool(df["_short_setup"].iloc[i]):
            return Direction.SHORT, f"Squeeze Expansion Breakout Short (Mom={float(df['mom'].iloc[i]):.2f})"

        return Direction.NONE, ""