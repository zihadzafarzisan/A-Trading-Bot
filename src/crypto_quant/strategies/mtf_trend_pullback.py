"""Multi-Timeframe Trend + Pullback Strategy (MTFTrendPullbackStrategy).

Quant Thesis
------------
Single-timeframe trend following suffers from poor risk-to-reward because entries
occur after momentum is extended, or suffers whipsaws in consolidation.
This strategy decouples directional bias from execution across 3 timeframes:
1. Higher Timeframe (HTF): Confirms macro trend regime (EMA alignment + ADX strength).
2. Intermediate Timeframe (ITF): Detects healthy pullback into value area (EMA ribbon / RSI dip).
3. Lower Timeframe (LTF): Executes on local exhaustion & micro-structure reclaim with volume.

Guarantees & Causal Alignment
-----------------------------
- Strict causal alignment: HTF/ITF indicators are computed exclusively on closed bars
  and mapped forward by exact bar-close timestamps (open_ts + duration_ms).
- Lower-timeframe bars at timestamp T can only observe HTF/ITF bars that closed at or
  before T. No future data leaks through forward fill or incomplete candles.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..indicators import (
    adx as calc_adx,
    dmi as calc_dmi,
    atr as calc_atr,
    ema as calc_ema,
    rsi as calc_rsi,
    sma as calc_sma,
)
from ..logging_config import get_logger
from .base import BaseStrategy, Direction, StopLossSpec, StopType, TakeProfitSpec

logger = get_logger("strategies")

# Timeframe duration in milliseconds
TIMEFRAME_MS: Dict[str, int] = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

# Supported MTF configuration triples: (HTF, ITF, LTF)
MTF_COMBOS: Dict[str, Tuple[str, str, str]] = {
    "4h_1h_15m": ("4h", "1h", "15m"),
    "4h_1h_5m": ("4h", "1h", "5m"),
    "1d_4h_1h": ("1d", "4h", "1h"),
}


def resample_ohlcv_causal(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """Resample a base OHLCV DataFrame to a higher timeframe with causal timestamps.

    Each resampled bar is timestamped with:
    - ``open_ts``: UNIX ms at bar open (epoch-aligned).
    - ``available_ts``: UNIX ms when the bar is fully closed (open_ts + duration_ms).

    Returns a DataFrame with columns:
        open_ts, available_ts, open, high, low, close, volume.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["open_ts", "available_ts", "open", "high", "low", "close", "volume"])

    duration_ms = TIMEFRAME_MS.get(target_tf)
    if duration_ms is None:
        raise ValueError(f"Unsupported target timeframe: '{target_tf}'")

    ts = df["timestamp"].to_numpy(dtype=np.int64)
    bucket = (ts // duration_ms) * duration_ms

    tmp = df[["open", "high", "low", "close", "volume"]].copy()
    tmp["open_ts"] = bucket

    resampled = tmp.groupby("open_ts", as_index=False, sort=True).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    resampled["available_ts"] = resampled["open_ts"] + duration_ms
    resampled["timestamp"] = resampled["open_ts"]
    return resampled


class MTFTrendPullbackStrategy(BaseStrategy):
    """Multi-Timeframe Trend + Pullback + Confluence Strategy."""

    name = "MTF Trend Pullback"
    strategy_type = "mtf_trend_pullback"
    description = (
        "Multi-timeframe trend following with intermediate pullback and "
        "micro-structure reclaim trigger with volume confirmation"
    )
    supports_spot = True
    supports_futures = True
    supports_short = True

    @classmethod
    def default_timeframes(cls) -> List[str]:
        return ["15m", "5m", "1h"]

    @classmethod
    def param_grid(cls) -> Dict[str, list]:
        """Parameter grid for research discovery and optimization."""
        return {
            "mtf_combo": ["4h_1h_15m", "4h_1h_5m", "1d_4h_1h"],
            "htf_ema_fast": [20, 50],
            "htf_ema_slow": [50, 100, 200],
            "htf_adx_threshold": [0, 20, 25],
            "itf_ema_fast": [20, 30],
            "itf_rsi_pullback": [45, 50, 55],
            "ltf_rsi_oversold": [32, 38, 42],
            "ltf_rsi_overbought": [58, 62, 68],
            "volume_mult": [0.0, 1.0, 1.2],
            "atr_multiplier": [1.5, 2.0, 2.5, 3.0],
            "risk_reward_ratio": [1.5, 2.0, 2.5, 3.0],
            "stop_type": ["atr", "swing"],
        }

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        params = dict(params or {})
        params.setdefault("mtf_combo", "4h_1h_15m")
        params.setdefault("htf_ema_fast", 20)
        params.setdefault("htf_ema_slow", 50)
        params.setdefault("htf_adx_threshold", 20.0)
        params.setdefault("itf_ema_fast", 20)
        params.setdefault("itf_ema_slow", 50)
        params.setdefault("itf_rsi_period", 14)
        params.setdefault("itf_rsi_pullback", 50.0)
        params.setdefault("ltf_ema_fast", 20)
        params.setdefault("ltf_rsi_period", 14)
        params.setdefault("ltf_rsi_oversold", 38.0)
        params.setdefault("ltf_rsi_overbought", 62.0)
        params.setdefault("volume_mult", 1.0)
        params.setdefault("stop_type", "atr")
        params.setdefault("atr_multiplier", 2.0)
        params.setdefault("risk_reward_ratio", 2.0)
        params.setdefault("swing_lookback", 15)

        # Sync stop/tp specs from params
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
        combo = p.get("mtf_combo", "4h_1h_15m")
        if combo not in MTF_COMBOS:
            if not ("htf" in p and "itf" in p and "ltf" in p):
                raise ValueError(
                    f"Invalid mtf_combo '{combo}'. Supported: {list(MTF_COMBOS.keys())}"
                )

        if int(p.get("htf_ema_fast", 20)) <= 0 or int(p.get("htf_ema_slow", 50)) <= 0:
            raise ValueError("htf_ema_fast and htf_ema_slow must be > 0")
        if int(p.get("htf_ema_fast", 20)) >= int(p.get("htf_ema_slow", 50)):
            raise ValueError("htf_ema_fast must be < htf_ema_slow")
        if float(p.get("ltf_rsi_oversold", 38)) >= float(p.get("ltf_rsi_overbought", 62)):
            raise ValueError("ltf_rsi_oversold must be < ltf_rsi_overbought")

    def _resolve_timeframes(self) -> Tuple[str, str, str]:
        """Return (htf, itf, ltf) strings."""
        p = self.params
        combo = p.get("mtf_combo", "4h_1h_15m")
        if combo in MTF_COMBOS:
            return MTF_COMBOS[combo]
        return (p["htf"], p["itf"], p["ltf"])

    def setup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Precompute indicators across HTF, ITF, and LTF with causal alignment."""
        if df is None or df.empty:
            return df

        out = df.copy().sort_values("timestamp").reset_index(drop=True)
        htf, itf, ltf = self._resolve_timeframes()
        p = self.params

        # -------------------------------------------------------------
        # 1. Base / Execution (LTF) Indicators (shifted to avoid lookahead)
        # -------------------------------------------------------------
        ltf_fast = int(p.get("ltf_ema_fast", 20))
        ltf_rsi_p = int(p.get("ltf_rsi_period", 14))

        out["ltf_ema_fast"] = calc_ema(out["close"], ltf_fast, shift=1)
        out["ltf_rsi"] = calc_rsi(out["close"], ltf_rsi_p, shift=1)
        out["ltf_atr"] = calc_atr(out["high"], out["low"], out["close"], 14, shift=1)
        out["atr_14"] = out["ltf_atr"]  # for BaseStrategy ATR stop sizing
        out["ltf_vol_sma"] = calc_sma(out["volume"], 20, shift=1)
        out["ltf_prev_high"] = out["high"].shift(1)
        out["ltf_prev_low"] = out["low"].shift(1)
        out["ltf_prev_close"] = out["close"].shift(1)

        # -------------------------------------------------------------
        # 2. Intermediate Timeframe (ITF) Resampling & Indicators
        # -------------------------------------------------------------
        itf_df = resample_ohlcv_causal(out, itf)
        itf_fast = int(p.get("itf_ema_fast", 20))
        itf_slow = int(p.get("itf_ema_slow", 50))
        itf_rsi_p = int(p.get("itf_rsi_period", 14))

        # Computed on closed bars (shift=0 because available_ts = open_ts + duration_ms)
        itf_df["itf_ema_fast"] = calc_ema(itf_df["close"], itf_fast, shift=0)
        itf_df["itf_ema_slow"] = calc_ema(itf_df["close"], itf_slow, shift=0)
        itf_df["itf_rsi"] = calc_rsi(itf_df["close"], itf_rsi_p, shift=0)
        itf_df["itf_close"] = itf_df["close"]
        itf_df["itf_low"] = itf_df["low"]
        itf_df["itf_high"] = itf_df["high"]

        itf_cols = [
            "available_ts", "itf_ema_fast", "itf_ema_slow", "itf_rsi",
            "itf_close", "itf_low", "itf_high"
        ]
        out = pd.merge_asof(
            out,
            itf_df[itf_cols].sort_values("available_ts"),
            left_on="timestamp",
            right_on="available_ts",
            direction="backward",
        ).drop(columns=["available_ts"], errors="ignore")

        # -------------------------------------------------------------
        # 3. Higher Timeframe (HTF) Resampling & Indicators
        # -------------------------------------------------------------
        htf_df = resample_ohlcv_causal(out, htf)
        htf_fast = int(p.get("htf_ema_fast", 20))
        htf_slow = int(p.get("htf_ema_slow", 50))
        adx_thresh = float(p.get("htf_adx_threshold", 20.0))

        htf_df["htf_ema_fast"] = calc_ema(htf_df["close"], htf_fast, shift=0)
        htf_df["htf_ema_slow"] = calc_ema(htf_df["close"], htf_slow, shift=0)
        htf_df["htf_adx"] = calc_adx(htf_df["high"], htf_df["low"], htf_df["close"], 14, shift=0)
        p_di, m_di = calc_dmi(htf_df["high"], htf_df["low"], htf_df["close"], 14, shift=0)
        htf_df["htf_plus_di"] = p_di
        htf_df["htf_minus_di"] = m_di
        htf_df["htf_close"] = htf_df["close"]

        htf_cols = [
            "available_ts", "htf_ema_fast", "htf_ema_slow", "htf_adx",
            "htf_plus_di", "htf_minus_di", "htf_close"
        ]
        out = pd.merge_asof(
            out,
            htf_df[htf_cols].sort_values("available_ts"),
            left_on="timestamp",
            right_on="available_ts",
            direction="backward",
        ).drop(columns=["available_ts"], errors="ignore")

        # -------------------------------------------------------------
        # 4. Precompute Setup Booleans for Fast Signal Generation
        # -------------------------------------------------------------
        # HTF Trend
        htf_adx_ok = (adx_thresh <= 0) | (out["htf_adx"] >= adx_thresh)
        htf_bull = (
            (out["htf_ema_fast"] > out["htf_ema_slow"])
            & htf_adx_ok
            & (out["htf_plus_di"] >= out["htf_minus_di"])
            & (out["htf_close"] >= out["htf_ema_slow"] * 0.99)
        )
        htf_bear = (
            (out["htf_ema_fast"] < out["htf_ema_slow"])
            & htf_adx_ok
            & (out["htf_minus_di"] >= out["htf_plus_di"])
            & (out["htf_close"] <= out["htf_ema_slow"] * 1.01)
        )

        # ITF Pullback
        itf_rsi_limit = float(p.get("itf_rsi_pullback", 50.0))
        itf_bull_pullback = (out["itf_low"] <= out["itf_ema_fast"] * 1.015) | (out["itf_rsi"] <= itf_rsi_limit)
        itf_bull_structure = out["itf_close"] >= out["itf_ema_slow"] * 0.985
        itf_long_ok = itf_bull_pullback & itf_bull_structure

        itf_bear_pullback = (out["itf_high"] >= out["itf_ema_fast"] * 0.985) | (out["itf_rsi"] >= 100 - itf_rsi_limit)
        itf_bear_structure = out["itf_close"] <= out["itf_ema_slow"] * 1.015
        itf_short_ok = itf_bear_pullback & itf_bear_structure

        # LTF Exhaustion & Reclaim
        ltf_rsi_os = float(p.get("ltf_rsi_oversold", 38.0))
        ltf_rsi_ob = float(p.get("ltf_rsi_overbought", 62.0))
        vol_m = float(p.get("volume_mult", 1.0))

        vol_ok = (vol_m <= 0) | (out["volume"].shift(1) >= vol_m * out["ltf_vol_sma"])

        ltf_long_trigger = (
            ((out["ltf_rsi"] <= ltf_rsi_os) | (out["ltf_rsi"].shift(1) <= ltf_rsi_os))
            & (out["ltf_prev_close"] >= out["ltf_ema_fast"] * 0.995)
            & vol_ok
        )

        ltf_short_trigger = (
            ((out["ltf_rsi"] >= ltf_rsi_ob) | (out["ltf_rsi"].shift(1) >= ltf_rsi_ob))
            & (out["ltf_prev_close"] <= out["ltf_ema_fast"] * 1.005)
            & vol_ok
        )

        out["_long_setup"] = htf_bull & itf_long_ok & ltf_long_trigger
        out["_short_setup"] = htf_bear & itf_short_ok & ltf_short_trigger

        return out

    def entry_signal(
        self, df: pd.DataFrame, i: int, ctx: Optional[Dict] = None
    ) -> Tuple[Direction, str]:
        """Return trade entry direction and reason for bar i."""
        if i < 30:
            return Direction.NONE, ""

        if bool(df["_long_setup"].iloc[i]):
            return Direction.LONG, f"MTF Bullish Reclaim ({self.params.get('mtf_combo')})"

        if bool(df["_short_setup"].iloc[i]):
            return Direction.SHORT, f"MTF Bearish Breakdown ({self.params.get('mtf_combo')})"

        return Direction.NONE, ""
