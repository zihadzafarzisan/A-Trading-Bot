"""Momentum indicators.

All functions vectorized, using only data up to the current row, with optional
`shift` (default 1) to prevent look-ahead bias in signal generation.
"""

import numpy as np
import pandas as pd


def rsi(
    series: pd.Series,
    period: int = 14,
    shift: int = 1,
) -> pd.Series:
    """Relative Strength Index (Wilder).

    Oscillator in [0, 100]. Values >70 typically overbought, <30 oversold.

    Returns:
        Series of RSI values.
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = _wilder_rma(gain, period)
    avg_loss = _wilder_rma(loss, period)

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.where(avg_loss != 0, 100.0)
    result = result.fillna(50.0)  # neutral when no movement yet

    if shift:
        result = result.shift(shift)
    return result


def macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    shift: int = 1,
) -> pd.DataFrame:
    """MACD indicator returning (macd, signal, histogram).

    Returns:
        DataFrame with columns: macd, signal, histogram.
    """
    if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
        raise ValueError("periods must be > 0")

    macd_line = _ema(series, fast_period) - _ema(series, slow_period)
    signal_line = _ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    df = pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    })
    if shift:
        df = df.shift(shift)
    return df


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
    shift: int = 1,
) -> pd.DataFrame:
    """Stochastic oscillator %K and %D.

    Returns:
        DataFrame with columns: stoch_k, stoch_d.
    """
    if k_period <= 0 or d_period <= 0:
        raise ValueError("periods must be > 0")

    lowest_low = low.rolling(k_period, min_periods=k_period).min()
    highest_high = high.rolling(k_period, min_periods=k_period).max()

    rng = highest_high - lowest_low
    pct_k = 100 * (close - lowest_low) / rng.replace(0, np.nan)
    pct_k = pct_k.fillna(50.0)

    pct_d = pct_k.rolling(d_period, min_periods=d_period).mean()

    df = pd.DataFrame({"stoch_k": pct_k, "stoch_d": pct_d})
    if shift:
        df = df.shift(shift)
    return df


def roc(series: pd.Series, period: int = 12, shift: int = 1) -> pd.Series:
    """Rate of change (percentage, 0-centered).

    Returns:
        Series of ROC values (percent).
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    result = series.pct_change(period) * 100.0
    if shift:
        result = result.shift(shift)
    return result


def momentum(series: pd.Series, period: int = 10, shift: int = 1) -> pd.Series:
    """Momentum = current price - price `period` bars ago."""
    if period <= 0:
        raise ValueError("period must be > 0")
    result = series - series.shift(period)
    if shift:
        result = result.shift(shift)
    return result


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    shift: int = 1,
) -> pd.Series:
    """Williams %R oscillator in [-100, 0]."""
    if period <= 0:
        raise ValueError("period must be > 0")

    highest = high.rolling(period, min_periods=period).max()
    lowest = low.rolling(period, min_periods=period).min()
    rng = highest - lowest
    result = -100 * (highest - close) / rng.replace(0, np.nan)
    result = result.fillna(-50.0)

    if shift:
        result = result.shift(shift)
    return result


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    shift: int = 1,
) -> pd.Series:
    """Commodity Channel Index."""
    if period <= 0:
        raise ValueError("period must be > 0")

    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(period, min_periods=period).mean()
    mad = tp.rolling(period, min_periods=period).apply(
        lambda w: np.abs(w - w.mean()).mean(), raw=True
    )
    result = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
    result = result.replace([np.inf, -np.inf], np.nan)

    if shift:
        result = result.shift(shift)
    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()