"""Strategy framework package.

Base abstractions plus concrete strategy families and the registry.
"""

from .base import (
    BaseStrategy,
    Signal,
    Direction,
    StopType,
    StopLossSpec,
    TakeProfitSpec,
    ExitReason,
)
from .trend import TrendStrategy
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .breakout import BreakoutStrategy
from .mtf_trend_pullback import MTFTrendPullbackStrategy
from .breakout_retest import BreakoutRetestStrategy
from .trend_filtered_rsi import TrendFilteredRSIStrategy
from .regime_adaptive import RegimeAdaptiveStrategy
from .vwap_bollinger_mr import VWAPBollingerMRStrategy
from .volatility_squeeze_breakout import VolatilitySqueezeBreakoutStrategy
from .trend_channel_breakout import TrendChannelBreakoutStrategy
from .volume_displacement_continuation import VolumeDisplacementContinuationStrategy
from .adaptive_displacement_trailing import AdaptiveDisplacementTrailingStrategy
from .volatility_compression_breakout import VolatilityCompressionBreakoutStrategy
from .registry import (
    STRATEGY_REGISTRY,
    get_strategy_class,
    create_strategy,
    available_strategy_types,
    strategy_param_grid,
    register_strategy,
)

STRATEGY_TYPES = available_strategy_types()

__all__ = [
    "BaseStrategy", "Signal", "Direction", "StopType", "StopLossSpec",
    "TakeProfitSpec", "ExitReason",
    "TrendStrategy", "MomentumStrategy", "MeanReversionStrategy", "BreakoutStrategy",
    "MTFTrendPullbackStrategy", "BreakoutRetestStrategy", "TrendFilteredRSIStrategy",
    "RegimeAdaptiveStrategy", "VWAPBollingerMRStrategy", "VolatilitySqueezeBreakoutStrategy",
    "TrendChannelBreakoutStrategy", "VolumeDisplacementContinuationStrategy",
    "AdaptiveDisplacementTrailingStrategy",
    "VolatilityCompressionBreakoutStrategy",
    "STRATEGY_REGISTRY", "get_strategy_class", "create_strategy",
    "available_strategy_types", "strategy_param_grid", "register_strategy",
    "STRATEGY_TYPES",
]