"""Trend-Filtered RSI Pullback Strategy.

Quant Thesis
------------
Standard RSI mean reversion fails in strong directional trends because oscillators
remain pinned in extreme zones. By combining a strict structural moving average
trend filter (SMA50 > SMA200, Close > SMA200) with local RSI oversold pullbacks,
we systematically buy the dip in bull markets and short the rally in bear markets.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..indicators import (
    adx as calc_adx,
    atr as calc_atr,
    rsi as calc_rsi,
    sma as calc_sma,
)
from ..logging_config import get_logger
from .base import BaseStrategy, Direction, StopLossSpec, StopType, TakeProfitSpec

logger = get_logger("strategies")


class TrendFilteredRSIStrategy(BaseStrategy):
    """Trend-Filtered RSI Pullback Strategy."""

    name = "Trend-Filtered RSI"
    strategy_type = "trend_filtered_rsi"
    description = "Trades RSI oversold dips in structural uptrends and overbought rallies in downtrends"
    supports_spot = True
    supports_futures = True
    supports_short = True

    @classmethod
    def default_timeframes(cls) -> List[str]:
        return ["15m", "1h", "4h", "1d"]

    @classmethod
    def param_grid(cls) -> Dict[str, list]:
        return {
            "sma_fast": [50],
            "sma_slow": [200],
            "adx_threshold": [0.0, 20.0, 25.0],
            "rsi_period": [2, 14],
            "rsi_oversold": [10.0, 30.0, 35.0],
            "rsi_overbought": [70.0, 65.0, 90.0],
            "atr_multiplier": [1.8, 2.2, 2.5],
            "risk_reward_ratio": [1.8, 2.2, 2.8],
            "stop_type": ["atr", "swing"],
        }

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        params = dict(params or {})
        params.setdefault("sma_fast", 50)
        params.setdefault("sma_slow", 200)
        params.setdefault("adx_threshold", 20.0)
        params.setdefault("rsi_period", 14)
        params.setdefault("rsi_oversold", 35.0)
        params.setdefault("rsi_overbought", 65.0)
        params.setdefault("atr_multiplier", 2.0)
        params.setdefault("risk_reward_ratio", 2.2)
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
            risk_reward_ratio=float(params.get("risk_reward_ratio", 2.2)),
        )
        super().__init__(params=params, stop_spec=stop_spec, tp_spec=tp_spec, **kwargs)

    def validate_params(self) -> None:
        p = self.params
        if int(p.get("sma_fast", 50)) >= int(p.get("sma_slow", 200)):
            raise ValueError("sma_fast must be < sma_slow")
        if float(p.get("rsi_oversold", 35)) >= float(p.get("rsi_overbought", 65)):
            raise ValueError("rsi_oversold must be < rsi_overbought")

    def setup(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy().sort_values("timestamp").reset_index(drop=True)
        p = self.params
        s_fast = int(p.get("sma_fast", 50))
        s_slow = int(p.get("sma_slow", 200))
        rsi_p = int(p.get("rsi_period", 14))
        adx_t = float(p.get("adx_threshold", 20.0))
        rsi_os = float(p.get("rsi_oversold", 35.0))
        rsi_ob = float(p.get("rsi_overbought", 65.0))

        # Shift=1 for causal signals
        out["sma_fast"] = calc_sma(out["close"], s_fast, shift=1)
        out["sma_slow"] = calc_sma(out["close"], s_slow, shift=1)
        out["adx"] = calc_adx(out["high"], out["low"], out["close"], 14, shift=1)
        out["rsi"] = calc_rsi(out["close"], rsi_p, shift=1)
        out["atr_14"] = calc_atr(out["high"], out["low"], out["close"], 14, shift=1)

        # Macro trend regime
        adx_ok = (adx_t <= 0) | (out["adx"] >= adx_t)
        bull_trend = (out["sma_fast"] > out["sma_slow"]) & (out["close"].shift(1) > out["sma_slow"]) & adx_ok
        bear_trend = (out["sma_fast"] < out["sma_slow"]) & (out["close"].shift(1) < out["sma_slow"]) & adx_ok

        # RSI inflection: oversold dip followed by turnaround
        rsi_prev = out["rsi"].shift(1)
        long_pullback = (out["rsi"] <= rsi_os) | ((rsi_prev <= rsi_os) & (out["rsi"] > rsi_prev))
        short_pullback = (out["rsi"] >= rsi_ob) | ((rsi_prev >= rsi_ob) & (out["rsi"] < rsi_prev))

        out["_long_setup"] = bull_trend & long_pullback & (out["close"].shift(1) >= out["open"].shift(1))
        out["_short_setup"] = bear_trend & short_pullback & (out["close"].shift(1) <= out["open"].shift(1))

        return out

    def entry_signal(self, df: pd.DataFrame, i: int, ctx: Optional[Dict] = None) -> Tuple[Direction, str]:
        if i < 30:
            return Direction.NONE, ""

        if bool(df["_long_setup"].iloc[i]):
            return Direction.LONG, f"Trend-Filtered RSI Dip ({float(df['rsi'].iloc[i]):.1f})"

        if bool(df["_short_setup"].iloc[i]):
            return Direction.SHORT, f"Trend-Filtered RSI Rally ({float(df['rsi'].iloc[i]):.1f})"

        return Direction.NONE, ""
