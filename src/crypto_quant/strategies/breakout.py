"""Breakout strategy.

Trades breakouts above the prior Donchian-channel high (long) or below the
prior low (short), requiring above-average volume confirmation. Stops use an
ATR-based mechanism by default.
"""

from typing import Any, Dict, Optional

import pandas as pd

from ..indicators import volatility as vt
from ..indicators import volume as vl
from ..logging_config import get_logger
from .base import BaseStrategy, Direction

logger = get_logger("strategies")


class BreakoutStrategy(BaseStrategy):
    """Donchian + volume-confirmed breakout."""

    name = "Breakout"
    strategy_type = "breakout"
    description = "Trades volume-confirmed breakouts from consolidation ranges"
    supports_spot = True
    supports_futures = True
    supports_short = True

    @classmethod
    def param_grid(cls) -> Dict[str, list]:
        return {
            "dc_period": [20, 30, 50],
            "volume_ratio": [1.0, 1.2, 1.5, 2.0],
            "atr_multiplier": [2.0, 2.5, 3.0],
            "risk_reward_ratio": [1.5, 2.0, 3.0],
        }

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        params = dict(params or {})
        params.setdefault("dc_period", 20)
        params.setdefault("volume_ratio", 1.5)
        super().__init__(params, **kwargs)

    def validate_params(self) -> None:
        p = self.params
        if int(p["dc_period"]) <= 0:
            raise ValueError("dc_period must be > 0")
        if float(p["volume_ratio"]) < 0:
            raise ValueError("volume_ratio must be >= 0")

    def setup(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        period = int(self.params["dc_period"])
        dc = vt.donchian_channel(out["high"], out["low"], period, shift=1)
        out["dc_high"] = dc["dc_high"]
        out["dc_low"] = dc["dc_low"]
        out["volume_ratio"] = vl.volume_ratio(out["volume"], 20, shift=1)
        return out

    def entry_signal(self, df: pd.DataFrame, i: int, ctx: Optional[Dict] = None):
        dc_high = df["dc_high"].iloc[i]
        dc_low = df["dc_low"].iloc[i]
        vol_ratio = df["volume_ratio"].iloc[i]
        close = df["close"].iloc[i]
        req_vol = float(self.params["volume_ratio"])

        if pd.isna(dc_high) or pd.isna(vol_ratio):
            return Direction.NONE, ""

        if close > dc_high and vol_ratio >= req_vol:
            return Direction.LONG, "Breakout above channel high + volume"
        if close < dc_low and vol_ratio >= req_vol:
            return Direction.SHORT, "Breakdown below channel low + volume"

        return Direction.NONE, ""