"""Research & optimization package."""

from .experiments import ExperimentTracker, CandidateRecord
from .optimizer import ParameterSearcher, SearchConfig
from .ranking import RankingEngine, QualityFilters, RankedCandidate, OBJECTIVES
from .discovery import DiscoveryEngine, DiscoveryConfig, DiscoveryResult
from .funding_scanner import CarryOpportunity, FundingScanner

__all__ = [
    "ExperimentTracker", "CandidateRecord",
    "ParameterSearcher", "SearchConfig",
    "RankingEngine", "QualityFilters", "RankedCandidate", "OBJECTIVES",
    "DiscoveryEngine", "DiscoveryConfig", "DiscoveryResult",
    "CarryOpportunity", "FundingScanner",
]