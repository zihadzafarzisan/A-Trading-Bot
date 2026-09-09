"""Feature engineering engine.

Combines all technical indicators with price-structure and time features into a
single feature matrix. Every feature is computed with a built-in shift so that
the value on a given bar uses ONLY data available at the close of that bar —
signals/features are usable for the NEXT bar's decision, preventing look-ahead
bias. See: `FeatureEngine.DEFAULT_SHIFT`.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..indicators import trend, volatility, volume
from ..indicators.momentum import (
    rsi as _rsi,
    macd as _macd,
    stochastic as _stochastic,
    roc as _roc,
    momentum as _momentum,
    williams_r as _williams_r,
    cci as _cci,
)
from ..logging_config import get_logger

logger = get_logger("features")


class FeatureEngine:
    """Builds a feature matrix from an OHLCV DataFrame."""

    # Every feature is shifted by this many bars so decision at bar t uses
    # only data through bar t-1 (no look-ahead).
    DEFAULT_SHIFT = 1

    # Column name for the (time-shifted) logged close used as basis.
    def __init__(self, shift: int = DEFAULT_SHIFT):
        """Initialize feature engine.

        Args:
            shift: Number of bars to shift all features (default 1).
        """
        self.shift = shift

    # ------------------------------------------------------------ main API
    def compute(
        self,
        df: pd.DataFrame,
        time_bucket_minutes: int = 0,
    ) -> pd.DataFrame:
        """Compute the full feature matrix for an OHLCV DataFrame.

        Args:
            df: DataFrame with columns timestamp, open, high, low, close, volume
                (sorted ascending by timestamp).
            time_bucket_minutes: If > 0, adds a session/time bucket feature
                bucketing the bar's UTC time into chunks of this many minutes.
                If 0, uses the configurable default hour bucket.

        Returns:
            DataFrame indexed like df with feature columns plus original OHLCV.
        """
        if df is None or df.empty:
            logger.warning("compute() called on empty DataFrame")
            return df.copy()

        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume_s = df["volume"]

        out = pd.DataFrame(index=df.index)
        if "timestamp" in df.columns:
            out["timestamp"] = df["timestamp"].values
        out["close"] = close  # keep reference (NOTE: caller must not trade on it)

        features: Dict[str, pd.Series] = {}

        # ---- Trend ----
        for p in (10, 20, 50, 100, 200):
            features[f"sma_{p}"] = trend.sma(close, p, shift=self.shift)
            features[f"ema_{p}"] = trend.ema(close, p, shift=self.shift)
        features["ema_cross_20_50"] = trend.ema_crossover(close, 20, 50, shift=self.shift)
        features["ema_cross_50_200"] = trend.ema_crossover(close, 50, 200, shift=self.shift)
        features["adx_14"] = trend.adx(high, low, close, 14, shift=self.shift)
        features["adx_20"] = trend.adx(high, low, close, 20, shift=self.shift)
        features["slope_20"] = trend.trend_slope(close, 20, shift=self.shift)

        # Distance from moving averages (normalized as % of price)
        for p in (20, 50, 200):
            ma = trend.sma(close, p, shift=self.shift)
            features[f"dist_sma_{p}"] = (close - ma) / close * 100.0

        # ---- Momentum ----
        features["rsi_14"] = _rsi(close, 14, shift=self.shift)
        features["rsi_7"] = _rsi(close, 7, shift=self.shift)
        macd_df = _macd(close, 12, 26, 9, shift=self.shift)
        for col in ("macd", "signal", "histogram"):
            features[f"macd_{col}"] = macd_df[col]
        stoch_df = _stochastic(high, low, close, 14, 3, shift=self.shift)
        for col in ("stoch_k", "stoch_d"):
            features[col] = stoch_df[col]
        features["roc_12"] = _roc(close, 12, shift=self.shift)
        features["mom_10"] = _momentum(close, 10, shift=self.shift)
        features["williams_r"] = _williams_r(high, low, close, 14, shift=self.shift)
        features["cci_20"] = _cci(high, low, close, 20, shift=self.shift)

        # ---- Volatility ----
        features["atr_14"] = volatility.atr(high, low, close, 14, shift=self.shift)
        features["atr_ratio"] = (
            volatility.atr(high, low, close, 14, shift=self.shift) / close
        )
        bb = volatility.bollinger_bands(close, 20, 2.0, shift=self.shift)
        for col in ("bb_mid", "bb_upper", "bb_lower", "bb_width"):
            features[col] = bb[col]
        # %B position: (close - lower) / (upper - lower)
        rng = (bb["bb_upper"] - bb["bb_lower"]).replace(0, np.nan)
        features["bb_pct_b"] = (close - bb["bb_lower"]) / rng
        features["hist_vol_20"] = volatility.historical_volatility(
            close, 20, annualization=365.0, shift=self.shift
        )
        features["vol_percentile"] = volatility.volatility_percentile(
            close, 20, 252, shift=self.shift
        )
        dc = volatility.donchian_channel(high, low, 20, shift=self.shift)
        for col in ("dc_high", "dc_low", "dc_mid"):
            features[col] = dc[col]

        # ---- Volume ----
        features["volume_sma_20"] = volume.volume_sma(volume_s, 20, shift=self.shift)
        features["volume_ratio"] = volume.volume_ratio(volume_s, 20, shift=self.shift)
        features["obv"] = volume.obv(close, volume_s, shift=self.shift)
        features["vol_mom_10"] = volume.volume_momentum(volume_s, 10, shift=self.shift)
        features["mfi_14"] = volume.mfi(high, low, close, volume_s, 14, shift=self.shift)
        features["vwap_20"] = volume.vwap(high, low, close, volume_s, 20, shift=self.shift)

        # ---- Price structure ----
        features["higher_high"] = self._higher_high(high, 5, shift=self.shift)
        features["lower_low"] = self._lower_low(low, 5, shift=self.shift)
        features["hh_breakout"] = self._hh_breakout(high, 20, shift=self.shift)
        features["ll_breakdown"] = self._ll_breakdown(low, 20, shift=self.shift)
        features["above_sma_50"] = (close > trend.sma(close, 50, shift=self.shift)).astype(float)
        features["near_resistance"] = self._near_level(
            close, self._rolling_max(high, 50, shift=self.shift), tol_pct=0.5
        )
        features["near_support"] = self._near_level(
            close, self._rolling_min(low, 50, shift=self.shift), tol_pct=0.5
        )

        # ---- Time features ----
        time_df = self._time_features(df, time_bucket_minutes)
        features.update(time_df)

        # ---- Assemble ----
        for name, s in features.items():
            out[name] = s.values if isinstance(s, pd.Series) else s

        return out

    # ------------------------------------------------------- price structure
    def _higher_high(self, high: pd.Series, lookback: int, shift: int = 1) -> pd.Series:
        """1.0 if current high > max of previous `lookback` highs."""
        prev_max = high.rolling(lookback).max().shift()
        result = (high > prev_max).astype(float)
        return result.shift(shift) if shift else result

    def _lower_low(self, low: pd.Series, lookback: int, shift: int = 1) -> pd.Series:
        """1.0 if current low < min of previous `lookback` lows."""
        prev_min = low.rolling(lookback).min().shift()
        result = (low < prev_min).astype(float)
        return result.shift(shift) if shift else result

    def _hh_breakout(self, high: pd.Series, period: int, shift: int = 1) -> pd.Series:
        """1.0 if current high breaks above the previous `period` high."""
        prev_high = high.rolling(period).max().shift()
        result = (high > prev_high).astype(float)
        return result.shift(shift) if shift else result

    def _ll_breakdown(self, low: pd.Series, period: int, shift: int = 1) -> pd.Series:
        """1.0 if current low breaks below the previous `period` low."""
        prev_low = low.rolling(period).min().shift()
        result = (low < prev_low).astype(float)
        return result.shift(shift) if shift else result

    def _rolling_max(self, high: pd.Series, period: int, shift: int = 1) -> pd.Series:
        result = high.rolling(period, min_periods=1).max()
        return result.shift(shift) if shift else result

    def _rolling_min(self, low: pd.Series, period: int, shift: int = 1) -> pd.Series:
        result = low.rolling(period, min_periods=1).min()
        return result.shift(shift) if shift else result

    def _near_level(self, close: pd.Series, level: pd.Series, tol_pct: float) -> pd.Series:
        """1.0 if close is within tol_pct% of a reference level."""
        if level is None:
            return pd.Series(0.0, index=close.index)
        dist = (close - level).abs() / close
        return (dist <= tol_pct / 100.0).astype(float)

    # ------------------------------------------------------------ time features
    def _time_features(self, df: pd.DataFrame, bucket_minutes: int) -> Dict[str, pd.Series]:
        """Build time-based features from the timestamp column (UTC)."""
        if "timestamp" not in df.columns:
            return {}

        ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        features = {
            "hour": ts.dt.hour.astype(float),
            "day_of_week": ts.dt.dayofweek.astype(float),
            "month": ts.dt.month.astype(float),
            "is_weekend": (ts.dt.dayofweek >= 5).astype(float),
            "day": ts.dt.day.astype(float),
        }

        if bucket_minutes and bucket_minutes > 0:
            # Session/time bucket: minutes elapsed since midnight // bucket size
            minutes_of_day = ts.dt.hour * 60 + ts.dt.minute
            features["session_bucket"] = (minutes_of_day // bucket_minutes).astype(float)
        else:
            features["session_bucket"] = (ts.dt.hour // 4).astype(float)  # 4-hr buckets

        return features

    # ------------------------------------------------------------- helper API
    @staticmethod
    def feature_names() -> List[str]:
        """Return the canonical ordered list of feature column names."""
        return [
            # trend
            *[f"sma_{p}" for p in (10, 20, 50, 100, 200)],
            *[f"ema_{p}" for p in (10, 20, 50, 100, 200)],
            "ema_cross_20_50", "ema_cross_50_200", "adx_14", "adx_20", "slope_20",
            "dist_sma_20", "dist_sma_50", "dist_sma_200",
            # momentum
            "rsi_14", "rsi_7", "macd_macd", "macd_signal", "macd_histogram",
            "stoch_k", "stoch_d", "roc_12", "mom_10", "williams_r", "cci_20",
            # volatility
            "atr_14", "atr_ratio", "bb_mid", "bb_upper", "bb_lower", "bb_width",
            "bb_pct_b", "hist_vol_20", "vol_percentile", "dc_high", "dc_low", "dc_mid",
            # volume
            "volume_sma_20", "volume_ratio", "obv", "vol_mom_10", "mfi_14", "vwap_20",
            # price structure
            "higher_high", "lower_low", "hh_breakout", "ll_breakdown",
            "above_sma_50", "near_resistance", "near_support",
            # time
            "hour", "day_of_week", "month", "is_weekend", "day", "session_bucket",
        ]