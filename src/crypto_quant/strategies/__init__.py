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
    "STRATEGY_REGISTRY", "get_strategy_class", "create_strategy",
    "available_strategy_types", "strategy_param_grid", "register_strategy",
    "STRATEGY_TYPES",
]