"""Momentum strategy.

Buys pullbacks-in-uptrend or momentum confirmations using RSI, MACD, and ROC.
Entry is triggered on RSI recovering from oversold with a positive MACD
histogram turn and positive momentum; the reverse for shorts (futures only).
"""

from typing import Any, Dict, Optional

import pandas as pd

from ..indicators.momentum import rsi as _rsi, macd as _macd, roc as _roc
from ..logging_config import get_logger
from .base import BaseStrategy, Direction

logger = get_logger("strategies")


class MomentumStrategy(BaseStrategy):
    """RSI + MACD momentum strategy."""

    name = "Momentum"
    strategy_type = "momentum"
    description = "Trades momentum confirmations using RSI and MACD"
    supports_spot = True
    supports_futures = True
    supports_short = True

    @classmethod
    def param_grid(cls) -> Dict[str, list]:
        return {
            "rsi_period": [7, 14, 21],
            "rsi_oversold": [30, 35, 40],
            "rsi_overbought": [60, 65, 70],
            "macd_fast": [12],
            "macd_slow": [26],
            "atr_multiplier": [2.0, 2.5, 3.0],
            "risk_reward_ratio": [1.5, 2.0, 3.0],
        }

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        params = dict(params or {})
        params.setdefault("rsi_period", 14)
        params.setdefault("rsi_oversold", 35)
        params.setdefault("rsi_overbought", 65)
        params.setdefault("macd_fast", 12)
        params.setdefault("macd_slow", 26)
        super().__init__(params, **kwargs)

    def validate_params(self) -> None:
        p = self.params
        if int(p["rsi_period"]) <= 0:
            raise ValueError("rsi_period must be > 0")
        if not 0 < float(p["rsi_oversold"]) < 50:
            raise ValueError("rsi_oversold must be in (0, 50)")
        if not 50 < float(p["rsi_overbought"]) < 100:
            raise ValueError("rsi_overbought must be in (50, 100)")
        if float(p["rsi_oversold"]) >= float(p["rsi_overbought"]):
            raise ValueError("rsi_oversold must be < rsi_overbought")

    def setup(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        rsi_p = int(self.params["rsi_period"])
        fast = int(self.params["macd_fast"])
        slow = int(self.params["macd_slow"])
        out["rsi"] = _rsi(out["close"], rsi_p, shift=1)
        macd_df = _macd(out["close"], fast, slow, 9, shift=1)
        out["macd_macd"] = macd_df["macd"]
        out["macd_hist"] = macd_df["histogram"]
        out["roc"] = _roc(out["close"], 12, shift=1)
        out["_prev_rsi"] = out["rsi"].shift(1)
        return out

    def entry_signal(self, df: pd.DataFrame, i: int, ctx: Optional[Dict] = None):
        oversold = float(self.params["rsi_oversold"])
        overbought = float(self.params["rsi_overbought"])

        rsi_now = df["rsi"].iloc[i]
        hist_now = df["macd_hist"].iloc[i]
        roc_now = df["roc"].iloc[i]
        prev_hist = df["macd_hist"].iloc[i - 1] if i > 0 else None

        if pd.isna(rsi_now) or pd.isna(hist_now):
            return Direction.NONE, ""

        # Long: oversold recovery + MACD histogram turning up + positive momentum
        long_hist_turn = prev_hist is not None and not pd.isna(prev_hist) and hist_now >= 0 > prev_hist
        if rsi_now > oversold and long_hist_turn and roc_now > 0:
            return Direction.LONG, "RSI recovery + MACD turn + ROC>0"
        if rsi_now > oversold and hist_now > 0 and roc_now > 0:
            return Direction.LONG, "RSI>oversold + MACD>0"

        # Short: overbought decline + MACD histogram turning down + negative momentum
        short_hist_turn = prev_hist is not None and not pd.isna(prev_hist) and hist_now <= 0 < prev_hist
        if rsi_now < overbought and short_hist_turn and roc_now < 0:
            return Direction.SHORT, "RSI<overbought + MACD turn -"
        if rsi_now < overbought and hist_now < 0 and roc_now < 0:
            return Direction.SHORT, "RSI<overbought + MACD<0"

        return Direction.NONE, ""