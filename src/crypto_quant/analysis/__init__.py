"""Trade analysis package."""

from .regimes import RegimeClassifier, TREND_REGIMES, VOL_REGIMES
from .statistics import (
    cohens_d, welch_t_pvalue, compare_effect, percentile_range,
    category_distribution, EffectSize,
)
from .trades import TradeAnalyzer, TradeAnalysisReport, FeatureFinding, CategoryFinding

__all__ = [
    "RegimeClassifier", "TREND_REGIMES", "VOL_REGIMES",
    "cohens_d", "welch_t_pvalue", "compare_effect", "percentile_range",
    "category_distribution", "EffectSize",
    "TradeAnalyzer", "TradeAnalysisReport", "FeatureFinding", "CategoryFinding",
]