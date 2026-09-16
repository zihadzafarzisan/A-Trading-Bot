"""Trend indicators.

All functions return the indicator value using ONLY data available up to and
including the current row (no future data). Functions accept an optional
`shift` parameter (default 1) so callers can align signals to the NEXT bar's
decision point, preventing look-ahead bias when generating trade signals.

Every function operates on pandas Series/DataFrames and is fully vectorized.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from ..logging_config import get_logger

logger = get_logger("indicators")


def sma(series: pd.Series, period: int, shift: int = 1) -> pd.Series:
    """Simple moving average.

    Args:
        series: Price/close series.
        period: Lookback window.
        shift: Bars to shift (default 1 to avoid look-ahead).

    Returns:
        Series of SMA values.
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    result = series.rolling(window=period, min_periods=period).mean()
    if shift:
        result = result.shift(shift)
    return result


def ema(series: pd.Series, period: int, shift: int = 1) -> pd.Series:
    """Exponential moving average.

    Args:
        series: Price/close series.
        period: Lookback window.
        shift: Bars to shift.

    Returns:
        Series of EMA values.
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    result = series.ewm(span=period, adjust=False, min_periods=period).mean()
    if shift:
        result = result.shift(shift)
    return result


def ema_crossover(
    series: pd.Series,
    fast_period: int,
    slow_period: int,
    shift: int = 1,
) -> pd.Series:
    """EMA crossover signal.

    Computes an integer series: +1 following a golden cross (fast crosses above
    slow), -1 following a death cross, and holding signal otherwise.

    Args:
        series: Price/close series.
        fast_period: Fast EMA period.
        slow_period: Slow EMA period.
        shift: Bars to shift.

    Returns:
        Series encoded as -1/+1/0.
    """
    ema_fast = ema(series, fast_period, shift=0)
    ema_slow = ema(series, slow_period, shift=0)

    diff = ema_fast - ema_slow
    prev_diff = diff.shift(1)
    prev_diff = prev_diff.fillna(diff)  # avoid bool on NaN

    signal = pd.Series(0, index=series.index)
    signal[(diff > 0) & (prev_diff <= 0)] = 1   # golden cross
    signal[(diff < 0) & (prev_diff >= 0)] = -1  # death cross

    if shift:
        signal = signal.shift(shift)
    return signal


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    shift: int = 1,
) -> pd.Series:
    """Average Directional Index (Wilder).

    Measures trend strength from 0 (no trend) to 100 (strong trend).

    Returns:
        Series of ADX values in [0, 100].
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    tr = _true_range(high, low)

    # Wilder smoothing (RMA)
    atr = _wilder_rma(tr, period)
    plus_di = 100 * _wilder_rma(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100 * _wilder_rma(minus_dm, period) / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_series = _wilder_rma(dx, period)
    adx_series = adx_series.replace([np.inf, -np.inf], np.nan)

    if shift:
        adx_series = adx_series.shift(shift)
    return adx_series.fillna(0.0)


def dmi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    shift: int = 1,
) -> Tuple[pd.Series, pd.Series]:
    """Directional Movement Indicators (+DI, -DI).

    Returns:
        Tuple of (plus_di, minus_di) Series.
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    tr = _true_range(high, low)
    atr = _wilder_rma(tr, period)

    plus_di = 100 * _wilder_rma(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100 * _wilder_rma(minus_dm, period) / atr.replace(0, np.nan)

    plus_di = plus_di.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    minus_di = minus_di.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if shift:
        plus_di = plus_di.shift(shift)
        minus_di = minus_di.shift(shift)

    return plus_di, minus_di


def trend_slope(
    series: pd.Series,
    period: int = 20,
    shift: int = 1,
) -> pd.Series:
    """Trend slope: slope of a linear regression over `period` bars.

    Positive = upward trend; negative = downward.

    Returns:
        Series of slope values.
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    x = np.arange(period)
    slope = series.rolling(period).apply(
        lambda w: np.polyfit(x, w, 1)[0] if (~np.isnan(w)).all() else np.nan,
        raw=True,
    )
    if shift:
        slope = slope.shift(shift)
    return slope


def macd_line(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    shift: int = 1,
) -> pd.Series:
    """MACD line (fast EMA - slow EMA)."""
    result = ema(series, fast_period, shift=0) - ema(series, slow_period, shift=0)
    if shift:
        result = result.shift(shift)
    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _true_range(high: pd.Series, low: pd.Series) -> pd.Series:
    """Wilder true range."""
    prev_close = high.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's running moving average (RMA)."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()