"""Volume indicators.

All functions vectorized, using only data up to the current row, with optional
`shift` (default 1) to prevent look-ahead bias.
"""

import numpy as np
import pandas as pd


def volume_sma(volume: pd.Series, period: int = 20, shift: int = 1) -> pd.Series:
    """Simple moving average of volume."""
    if period <= 0:
        raise ValueError("period must be > 0")
    result = volume.rolling(period, min_periods=period).mean()
    if shift:
        result = result.shift(shift)
    return result


def volume_ratio(volume: pd.Series, period: int = 20, shift: int = 1) -> pd.Series:
    """Current volume divided by its `period` moving average.

    Value > 1 means above-average volume.
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    avg = volume.rolling(period, min_periods=period).mean().replace(0, np.nan)
    result = volume / avg
    if shift:
        result = result.shift(shift)
    return result


def obv(close: pd.Series, volume: pd.Series, shift: int = 1) -> pd.Series:
    """On-Balance Volume.

    Cumulative volume signed by price direction.
    """
    direction = np.sign(close.diff()).fillna(0)
    result = (direction * volume).cumsum()
    if shift:
        result = result.shift(shift)
    return result


def volume_momentum(volume: pd.Series, period: int = 10, shift: int = 1) -> pd.Series:
    """Rate of change of volume (percent)."""
    if period <= 0:
        raise ValueError("period must be > 0")
    result = volume.pct_change(period) * 100.0
    if shift:
        result = result.shift(shift)
    return result


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
    shift: int = 1,
) -> pd.Series:
    """Money Flow Index in [0, 100].

    Volume-weighted RSI-like oscillator.
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    typical = (high + low + close) / 3.0
    flow = typical * volume
    positive_flow = flow.where(typical > typical.shift(1), 0.0)
    negative_flow = flow.where(typical < typical.shift(1), 0.0)

    pos_sum = positive_flow.rolling(period, min_periods=period).sum()
    neg_sum = negative_flow.rolling(period, min_periods=period).sum()

    ratio = pos_sum / neg_sum.replace(0, np.nan)
    result = 100 - (100 / (1 + ratio))
    result = result.fillna(50.0)

    if shift:
        result = result.shift(shift)
    return result


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
    shift: int = 1,
) -> pd.Series:
    """Volume-Weighted Average Price over `period` bars.

    Represents the average price weighted by volume; useful as a reference
    level (not a short-term daily session VWAP).
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    typical = (high + low + close) / 3.0
    tpv = typical * volume
    result = tpv.rolling(period, min_periods=period).sum() / \
        volume.rolling(period, min_periods=period).sum().replace(0, np.nan)
    if shift:
        result = result.shift(shift)
    return result