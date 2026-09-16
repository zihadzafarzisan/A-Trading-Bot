"""Volatility indicators.

All functions vectorized, using only data up to the current row, with optional
`shift` (default 1) to prevent look-ahead bias.
"""

import numpy as np
import pandas as pd


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    shift: int = 1,
) -> pd.Series:
    """Average True Range (Wilder)."""
    if period <= 0:
        raise ValueError("period must be > 0")

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    result = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    if shift:
        result = result.shift(shift)
    return result


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
    shift: int = 1,
) -> pd.DataFrame:
    """Bollinger Bands.

    Returns:
        DataFrame with columns: bb_mid, bb_upper, bb_lower, bb_width.
    """
    if period <= 0 or num_std <= 0:
        raise ValueError("period and num_std must be > 0")

    mid = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std()

    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid.replace(0, np.nan)

    df = pd.DataFrame({
        "bb_mid": mid,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_width": width,
    })
    if shift:
        df = df.shift(shift)
    return df


def historical_volatility(
    series: pd.Series,
    period: int = 20,
    annualization: float = 365.0,
    shift: int = 1,
) -> pd.Series:
    """Annualized historical volatility of a price series.

    Computed as annualized std of log returns.
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    log_ret = np.log(series / series.shift(1))
    result = log_ret.rolling(period, min_periods=period).std(ddof=1) * np.sqrt(annualization)

    if shift:
        result = result.shift(shift)
    return result


def volatility_percentile(
    series: pd.Series,
    vol_period: int = 20,
    percentile_period: int = 252,
    shift: int = 1,
) -> pd.Series:
    """Percentile rank of current volatility within a historical window.

    Value in [0, 1]; high = currently more volatile than usual.
    """
    vol = historical_volatility(series, vol_period, shift=0)
    result = vol.rolling(percentile_period, min_periods=1).apply(
        lambda w: (w[:-1] <= w[-1]).mean() if len(w) > 1 else np.nan,
        raw=True,
    )
    if shift:
        result = result.shift(shift)
    return result


def donchian_channel(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
    shift: int = 1,
) -> pd.DataFrame:
    """Donchian Channel (breakout support/resistance).

    Returns:
        DataFrame with columns: dc_high, dc_low, dc_mid.
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    dc_high = high.rolling(period, min_periods=period).max()
    dc_low = low.rolling(period, min_periods=period).min()
    dc_mid = (dc_high + dc_low) / 2.0

    df = pd.DataFrame({"dc_high": dc_high, "dc_low": dc_low, "dc_mid": dc_mid})
    if shift:
        df = df.shift(shift)
    return df


def keltner_channel(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
    shift: int = 1,
) -> pd.DataFrame:
    """Keltner Channel.

    Returns:
        DataFrame with columns: kc_mid, kc_upper, kc_lower.
    """
    if ema_period <= 0 or atr_period <= 0:
        raise ValueError("periods must be > 0")

    mid = close.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    atr_v = atr(high, low, close, atr_period, shift=0)

    df = pd.DataFrame({
        "kc_mid": mid,
        "kc_upper": mid + multiplier * atr_v,
        "kc_lower": mid - multiplier * atr_v,
    })
    if shift:
        df = df.shift(shift)
    return df


def atr_percentile(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 14,
    window: int = 252,
    shift: int = 1,
) -> pd.Series:
    """ATR percentile rank in [0, 1] over a historical window."""
    atr_v = atr(high, low, close, atr_period, shift=0)
    result = atr_v.rolling(window, min_periods=1).apply(
        lambda w: (w[:-1] <= w[-1]).mean() if len(w) > 1 else np.nan,
        raw=True,
    )
    if shift:
        result = result.shift(shift)
    return result


def choppiness_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    shift: int = 1,
) -> pd.Series:
    """Choppiness Index in [0, 100].

    Values > 61.8 indicate consolidation / choppy market.
    Values < 38.2 indicate strong directional trend.
    """
    if period <= 1:
        raise ValueError("period must be > 1")

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    tr_sum = tr.rolling(period, min_periods=period).sum()
    highest_high = high.rolling(period, min_periods=period).max()
    lowest_low = low.rolling(period, min_periods=period).min()
    price_range = (highest_high - lowest_low).replace(0, np.nan)

    ratio = tr_sum / price_range
    chop = 100.0 * np.log10(ratio.replace(0, np.nan)) / np.log10(period)
    chop = chop.replace([np.inf, -np.inf], np.nan).fillna(50.0)

    if shift:
        chop = chop.shift(shift)
    return chop