"""VWAP / Bollinger Mean Reversion Strategy.

Quant Thesis
------------
In non-trending or consolidation regimes, institutional market makers re-anchor prices
to volume-weighted fair value (VWAP) and Bollinger middle bands.
This strategy enters at outer statistical extremes (+/-2 std Bollinger breaches)
strictly when trend strength is low (ADX < threshold), targeting regression to
VWAP / Bollinger baseline.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..indicators import (
    adx as calc_adx,
    atr as calc_atr,
    bollinger_bands as calc_bb,
    rsi as calc_rsi,
    vwap as calc_vwap,
)
from ..logging_config import get_logger
from .base import BaseStrategy, Direction, StopLossSpec, StopType, TakeProfitSpec

logger = get_logger("strategies")


class VWAPBollingerMRStrategy(BaseStrategy):
    """VWAP / Bollinger Mean Reversion Strategy."""

    name = "VWAP Bollinger MR"
    strategy_type = "vwap_bollinger_mr"
    description = "Mean reversion fading outer Bollinger bands towards VWAP strictly during low-trend regimes"
    supports_spot = True
    supports_futures = True
    supports_short = True

    @classmethod
    def default_timeframes(cls) -> List[str]:
        return ["15m", "1h", "4h"]

    @classmethod
    def param_grid(cls) -> Dict[str, list]:
        return {
            "adx_max_thresh": [20.0, 22.0, 25.0],
            "bb_period": [20],
            "bb_std": [2.0, 2.2],
            "rsi_period": [14],
            "rsi_oversold": [30.0, 35.0],
            "rsi_overbought": [65.0, 70.0],
            "atr_multiplier": [1.5, 2.0, 2.5],
            "risk_reward_ratio": [1.5, 2.0],
            "stop_type": ["atr"],
        }

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        params = dict(params or {})
        params.setdefault("adx_max_thresh", 22.0)
        params.setdefault("bb_period", 20)
        params.setdefault("bb_std", 2.0)
        params.setdefault("vwap_period", 20)
        params.setdefault("rsi_period", 14)
        params.setdefault("rsi_oversold", 35.0)
        params.setdefault("rsi_overbought", 65.0)
        params.setdefault("atr_multiplier", 2.0)
        params.setdefault("risk_reward_ratio", 2.0)
        params.setdefault("stop_type", "atr")

        stop_spec = StopLossSpec(
            stop_type=StopType.ATR,
            atr_multiplier=float(params.get("atr_multiplier", 2.0)),
        )
        tp_spec = TakeProfitSpec(
            mode="rr",
            risk_reward_ratio=float(params.get("risk_reward_ratio", 2.0)),
        )
        super().__init__(params=params, stop_spec=stop_spec, tp_spec=tp_spec, **kwargs)

    def validate_params(self) -> None:
        p = self.params
        if float(p.get("adx_max_thresh", 22.0)) <= 0:
            raise ValueError("adx_max_thresh must be > 0")

    def setup(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy().sort_values("timestamp").reset_index(drop=True)
        p = self.params
        adx_max = float(p.get("adx_max_thresh", 22.0))
        bb_p = int(p.get("bb_period", 20))
        bb_s = float(p.get("bb_std", 2.0))
        vwap_p = int(p.get("vwap_period", 20))
        rsi_p = int(p.get("rsi_period", 14))
        rsi_os = float(p.get("rsi_oversold", 35.0))
        rsi_ob = float(p.get("rsi_overbought", 65.0))

        # Causal shifted indicators
        out["adx"] = calc_adx(out["high"], out["low"], out["close"], 14, shift=1)
        out["rsi"] = calc_rsi(out["close"], rsi_p, shift=1)
        out["atr_14"] = calc_atr(out["high"], out["low"], out["close"], 14, shift=1)
        out["vwap"] = calc_vwap(out["high"], out["low"], out["close"], out["volume"], vwap_p, shift=1)

        bb = calc_bb(out["close"], bb_p, bb_s, shift=1)
        out["bb_mid"] = bb["bb_mid"]
        out["bb_upper"] = bb["bb_upper"]
        out["bb_lower"] = bb["bb_lower"]
        out["bb_width"] = bb["bb_width"]

        # Range filter: strictly low directional momentum
        range_regime = out["adx"] <= adx_max

        # Long Trigger: Low poked below lower band, closed with green reaction, RSI oversold
        long_breach = (out["low"].shift(1) <= out["bb_lower"]) & (out["rsi"] <= rsi_os) & (out["close"].shift(1) >= out["open"].shift(1))

        # Short Trigger: High poked above upper band, closed with red reaction, RSI overbought
        short_breach = (out["high"].shift(1) >= out["bb_upper"]) & (out["rsi"] >= rsi_ob) & (out["close"].shift(1) <= out["open"].shift(1))

        out["_long_setup"] = range_regime & long_breach
        out["_short_setup"] = range_regime & short_breach

        return out

    def entry_signal(self, df: pd.DataFrame, i: int, ctx: Optional[Dict] = None) -> Tuple[Direction, str]:
        if i < 30:
            return Direction.NONE, ""

        if bool(df["_long_setup"].iloc[i]):
            return Direction.LONG, f"VWAP/BB MR Long (RSI={float(df['rsi'].iloc[i]):.1f})"

        if bool(df["_short_setup"].iloc[i]):
            return Direction.SHORT, f"VWAP/BB MR Short (RSI={float(df['rsi'].iloc[i]):.1f})"

        return Direction.NONE, ""
