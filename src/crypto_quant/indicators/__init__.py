"""Technical indicators package.

Every indicator is vectorized and uses only data up to the current row.
By default each indicator shifts its output by one bar so a decision at bar `t`
only sees information available at the close of bar `t-1` — this is the core
anti-look-ahead guarantee. Callers may pass `shift=0` to obtain the raw
(point-in-time) value, e.g. for analytics rather than trade signals.
"""

from .trend import (
    sma,
    ema,
    ema_crossover,
    adx,
    trend_slope,
    macd_line,
)
from .momentum import (
    rsi,
    macd,
    stochastic,
    roc,
    momentum,
    williams_r,
    cci,
)
from .volatility import (
    atr,
    bollinger_bands,
    historical_volatility,
    volatility_percentile,
    donchian_channel,
    keltner_channel,
    atr_percentile,
)
from .volume import (
    volume_sma,
    volume_ratio,
    obv,
    volume_momentum,
    mfi,
    vwap,
)

# Central registry of all indicator functions (name -> callable). Used by the
# research/discovery pipeline to enumerate candidate features.
INDICATOR_REGISTRY = {
    # trend
    "sma": sma,
    "ema": ema,
    "ema_crossover": ema_crossover,
    "adx": adx,
    "trend_slope": trend_slope,
    "macd_line": macd_line,
    # momentum
    "rsi": rsi,
    "macd": macd,
    "stochastic": stochastic,
    "roc": roc,
    "momentum": momentum,
    "williams_r": williams_r,
    "cci": cci,
    # volatility
    "atr": atr,
    "bollinger_bands": bollinger_bands,
    "historical_volatility": historical_volatility,
    "volatility_percentile": volatility_percentile,
    "donchian_channel": donchian_channel,
    "keltner_channel": keltner_channel,
    "atr_percentile": atr_percentile,
    # volume
    "volume_sma": volume_sma,
    "volume_ratio": volume_ratio,
    "obv": obv,
    "volume_momentum": volume_momentum,
    "mfi": mfi,
    "vwap": vwap,
}

__all__ = [
    "sma", "ema", "ema_crossover", "adx", "trend_slope", "macd_line",
    "rsi", "macd", "stochastic", "roc", "momentum", "williams_r", "cci",
    "atr", "bollinger_bands", "historical_volatility", "volatility_percentile",
    "donchian_channel", "keltner_channel", "atr_percentile",
    "volume_sma", "volume_ratio", "obv", "volume_momentum", "mfi", "vwap",
    "INDICATOR_REGISTRY",
]