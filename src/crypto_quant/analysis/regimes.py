"""Market regime detection.

Classifies every bar into a trend regime (bull / bear / sideways) and a
volatility regime (high / normal / low). All signals use shifted indicators so
the classification of a bar is known only at its close — no look-ahead.

Regime definitions (configurable thresholds, documented in the docstrings):
- bull:      SMA50 above SMA200 and close above SMA50
- bear:      SMA50 below SMA200 and close below SMA50
- sideways:  neither (or a narrow gap between the SMAs)
- high/low volatility: ATR percentile within its historical window
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from ..indicators import trend as tn
from ..indicators import volatility as vt
from ..logging_config import get_logger

logger = get_logger("analysis")

TREND_REGIMES = ("bull", "bear", "sideways")
VOL_REGIMES = ("high_volatility", "normal_volatility", "low_volatility")


class RegimeClassifier:
    """Classifies bars into trend and volatility regimes."""

    def __init__(
        self,
        sma_fast: int = 50,
        sma_slow: int = 200,
        adx_period: int = 14,
        adx_sideways_threshold: float = 25.0,
        atr_period: int = 14,
        atr_window: int = 252,
        vol_high_percentile: float = 0.75,
        vol_low_percentile: float = 0.25,
    ):
        """Initialize with configurable thresholds."""
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.adx_period = adx_period
        self.adx_sideways_threshold = adx_sideways_threshold
        self.atr_period = atr_period
        self.atr_window = atr_window
        self.vol_high_percentile = vol_high_percentile
        self.vol_low_percentile = vol_low_percentile

    # ------------------------------------------------------------ main API
    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with 'regime' and 'vol_regime' columns per bar.

        Rows are aligned to df.index. Regimes are computed from shifted
        indicators (known at bar close).
        """
        out = pd.DataFrame(index=df.index)
        out["regime"] = self._trend_regime(df)
        out["vol_regime"] = self._volatility_regime(df)
        return out

    def classify_series(self, df: pd.DataFrame) -> pd.Series:
        """Return a single combined regime label per bar."""
        reg = self.classify(df)
        return reg["regime"] + "_" + reg["vol_regime"]

    # ------------------------------------------------------------ internal
    def _trend_regime(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        sma_f = tn.sma(close, self.sma_fast, shift=1)
        sma_s = tn.sma(close, self.sma_slow, shift=1)
        adx = tn.adx(df["high"], df["low"], close, self.adx_period, shift=1)

        regime = pd.Series("sideways", index=df.index)
        bull = (sma_f > sma_s) & (close > sma_f)
        bear = (sma_f < sma_s) & (close < sma_f)

        # Sideways confirmation: low ADX OR overlapping SMAs
        adx_low = adx < self.adx_sideways_threshold
        bull = bull & ~adx_low
        bear = bear & ~adx_low

        regime[bull.fillna(False)] = "bull"
        regime[bear.fillna(False)] = "bear"
        return regime

    def _volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        atr_ratio = vt.atr(df["high"], df["low"], df["close"], self.atr_period, shift=1) / df["close"]
        pct = atr_ratio.rolling(self.atr_window, min_periods=20).rank(pct=True)

        regime = pd.Series("normal_volatility", index=df.index)
        regime[pct > self.vol_high_percentile] = "high_volatility"
        regime[pct < self.vol_low_percentile] = "low_volatility"
        return regime

    # ------------------------------------------------------------ helpers
    def describe(self) -> dict:
        """Return the classifier configuration (for reproducibility)."""
        return {
            "sma_fast": self.sma_fast,
            "sma_slow": self.sma_slow,
            "adx_period": self.adx_period,
            "adx_sideways_threshold": self.adx_sideways_threshold,
            "atr_period": self.atr_period,
            "atr_window": self.atr_window,
            "vol_high_percentile": self.vol_high_percentile,
            "vol_low_percentile": self.vol_low_percentile,
        }

    def regime_distribution(self, classified: pd.DataFrame) -> dict:
        """Distribution of bars across regimes (share of time)."""
        n = len(classified)
        if n == 0:
            return {}
        dist = {}
        for col in ("regime", "vol_regime"):
            counts = classified[col].value_counts(normalize=True).to_dict()
            dist[col] = {k: float(v) for k, v in counts.items()}
        return dist