"""Trend-following strategy.

Goes long when a fast EMA is above a slow EMA (with optional ADX strength
confirmation); goes short when the reverse holds (futures only). Stops use an
ATR-based mechanism by default.
"""

from typing import Any, Dict, Optional

import pandas as pd

from ..indicators import trend as tn
from ..logging_config import get_logger
from .base import BaseStrategy, Direction

logger = get_logger("strategies")


class TrendStrategy(BaseStrategy):
    """EMA-crossover trend follower."""

    name = "Trend Following"
    strategy_type = "trend"
    description = "Follows market trends using EMA crossovers with optional ADX confirmation"
    supports_spot = True
    supports_futures = True
    supports_short = True

    # Default parameter grid for optimization
    @classmethod
    def param_grid(cls) -> Dict[str, list]:
        return {
            "ema_fast": [10, 20, 30],
            "ema_slow": [50, 100, 200],
            "adx_threshold": [0, 20, 25, 30],   # 0 disables ADX filter
            "atr_multiplier": [2.0, 2.5, 3.0],
            "risk_reward_ratio": [1.5, 2.0, 3.0],
        }

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        params = dict(params or {})
        params.setdefault("ema_fast", 20)
        params.setdefault("ema_slow", 50)
        params.setdefault("adx_threshold", 0)
        super().__init__(params, **kwargs)

    def validate_params(self) -> None:
        p = self.params
        for key in ("ema_fast", "ema_slow"):
            if int(p[key]) <= 0:
                raise ValueError(f"{key} must be > 0")
        if "adx_threshold" not in p:
            p["adx_threshold"] = 0
        if int(p["ema_fast"]) >= int(p["ema_slow"]):
            raise ValueError("ema_fast must be < ema_slow")
        if not 0 <= float(p.get("adx_threshold", 0)) <= 100:
            raise ValueError("adx_threshold must be 0..100")

    def setup(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        fast = int(self.params["ema_fast"])
        slow = int(self.params["ema_slow"])
        out["ema_fast"] = tn.ema(out["close"], fast, shift=1)
        out["ema_slow"] = tn.ema(out["close"], slow, shift=1)
        out["adx"] = tn.adx(out["high"], out["low"], out["close"], 14, shift=1)
        # long when fast > slow (shifted, so known at close of prior bar)
        out["_long_setup"] = out["ema_fast"] > out["ema_slow"]
        out["_short_setup"] = out["ema_fast"] < out["ema_slow"]
        return out

    def entry_signal(self, df: pd.DataFrame, i: int, ctx: Optional[Dict] = None):
        adx_thresh = float(self.params.get("adx_threshold", 0))
        adx_ok = adx_thresh <= 0 or df["adx"].iloc[i] > adx_thresh

        if not adx_ok:
            return Direction.NONE, ""

        if bool(df["_long_setup"].iloc[i]):
            return Direction.LONG, f"EMA{int(self.params['ema_fast'])}>EMA{int(self.params['ema_slow'])}"
        if bool(df["_short_setup"].iloc[i]):
            return Direction.SHORT, f"EMA{int(self.params['ema_fast'])}<EMA{int(self.params['ema_slow'])}"
        return Direction.NONE, ""