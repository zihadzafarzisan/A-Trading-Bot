"""Strategy registry.

Maps strategy_type -> strategy class and provides factory helpers used by the
research/discovery pipeline. Registering a new strategy family is a one-line
addition to STRATEGY_REGISTRY.
"""

from typing import Any, Dict, List, Optional, Type

from .base import BaseStrategy
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

# Central registry: strategy_type -> strategy class.
STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    "trend": TrendStrategy,
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "breakout": BreakoutStrategy,
    "mtf_trend_pullback": MTFTrendPullbackStrategy,
    "breakout_retest": BreakoutRetestStrategy,
    "trend_filtered_rsi": TrendFilteredRSIStrategy,
    "regime_adaptive": RegimeAdaptiveStrategy,
    "vwap_bollinger_mr": VWAPBollingerMRStrategy,
    "volatility_squeeze_breakout": VolatilitySqueezeBreakoutStrategy,
    "trend_channel_breakout": TrendChannelBreakoutStrategy,
    "volume_displacement_continuation": VolumeDisplacementContinuationStrategy,
    "adaptive_displacement_trailing": AdaptiveDisplacementTrailingStrategy,
    "volatility_compression_breakout": VolatilityCompressionBreakoutStrategy,
}


def get_strategy_class(strategy_type: str) -> Type[BaseStrategy]:
    """Return the strategy class for a strategy type."""
    key = strategy_type.lower()
    if key not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Unknown strategy type '{strategy_type}'. Available: {list(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[key]


def create_strategy(
    strategy_type: str,
    params: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> BaseStrategy:
    """Instantiate a strategy by type with optional parameters."""
    cls = get_strategy_class(strategy_type)
    return cls(params=params, **kwargs)


def available_strategy_types() -> List[str]:
    """Return the list of registered strategy types."""
    return list(STRATEGY_REGISTRY.keys())


def strategy_param_grid(strategy_type: str) -> Dict[str, list]:
    """Return the parameter grid for a strategy type."""
    return get_strategy_class(strategy_type).param_grid()


def register_strategy(strategy_type: str, cls: Type[BaseStrategy]) -> None:
    """Register a new strategy class (for plugins/extensions)."""
    if not (isinstance(cls, type) and issubclass(cls, BaseStrategy)):
        raise TypeError("strategy must subclass BaseStrategy")
    STRATEGY_REGISTRY[strategy_type.lower()] = cls