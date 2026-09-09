"""Mean-reversion strategy.

Buys when price is stretched below the lower Bollinger Band (oversold) and
sells at the mean; shorts the mirror setup (futures only). Uses RSI to confirm
the overextension.
"""

from typing import Any, Dict, Optional

import pandas as pd

from ..indicators.momentum import rsi as _rsi
from ..indicators import volatility as vt
from ..logging_config import get_logger
from .base import BaseStrategy, Direction, StopType, StopLossSpec

logger = get_logger("strategies")


class MeanReversionStrategy(BaseStrategy):
    """Bollinger + RSI mean reversion."""

    name = "Mean Reversion"
    strategy_type = "mean_reversion"
    description = "Reversion to the mean from stretched Bollinger Band extremes"
    supports_spot = True
    supports_futures = True
    supports_short = True

    @classmethod
    def param_grid(cls) -> Dict[str, list]:
        return {
            "bb_period": [20, 25, 30],
            "bb_std": [2.0, 2.5, 3.0],
            "rsi_period": [14],
            "rsi_extreme": [25, 30, 35],
            "atr_multiplier": [2.0, 2.5, 3.0],
            "risk_reward_ratio": [1.5, 2.0, 3.0],
        }

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        params = dict(params or {})
        params.setdefault("bb_period", 20)
        params.setdefault("bb_std", 2.5)
        params.setdefault("rsi_period", 14)
        params.setdefault("rsi_extreme", 30)
        # Mean reversion to target = band midpoint; stop outside the band.
        kwargs.setdefault("stop_spec", StopLossSpec(stop_type=StopType.PERCENT, percent=0.03))
        super().__init__(params, **kwargs)

    def validate_params(self) -> None:
        p = self.params
        if int(p["bb_period"]) <= 0:
            raise ValueError("bb_period must be > 0")
        if float(p["bb_std"]) <= 0:
            raise ValueError("bb_std must be > 0")
        if not 0 < float(p["rsi_extreme"]) < 50:
            raise ValueError("rsi_extreme must be in (0, 50)")

    def setup(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        period = int(self.params["bb_period"])
        std = float(self.params["bb_std"])
        rsi_p = int(self.params["rsi_period"])
        bb = vt.bollinger_bands(out["close"], period, std, shift=1)
        out["bb_lower"] = bb["bb_lower"]
        out["bb_upper"] = bb["bb_upper"]
        out["bb_mid"] = bb["bb_mid"]
        out["rsi"] = _rsi(out["close"], rsi_p, shift=1)
        return out

    def entry_signal(self, df: pd.DataFrame, i: int, ctx: Optional[Dict] = None):
        bb_lower = df["bb_lower"].iloc[i]
        bb_upper = df["bb_upper"].iloc[i]
        rsi_now = df["rsi"].iloc[i]
        close = df["close"].iloc[i]
        extreme = float(self.params["rsi_extreme"])

        if pd.isna(bb_lower) or pd.isna(rsi_now):
            return Direction.NONE, ""

        if close < bb_lower and rsi_now < extreme:
            return Direction.LONG, "Price<lower band + RSI oversold"
        if close > bb_upper and rsi_now > (100 - extreme):
            return Direction.SHORT, "Price>upper band + RSI overbought"

        return Direction.NONE, ""

    def compute_take_profit(self, df, i, direction, entry_price, stop_loss=None):
        # Revert to the band midpoint
        mid = df["bb_mid"].iloc[i] if "bb_mid" in df.columns else None
        if mid is not None and not pd.isna(mid):
            return float(mid)
        return super().compute_take_profit(df, i, direction, entry_price, stop_loss)