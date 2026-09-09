"""Parameter search / optimization.

Supports:
- Grid search: exhaustive product of parameter grids (honoring max_combinations).
- Random search: uniform sampling of parameter combinations.
- Adaptive search (lightweight Bayesian-style): starts with random exploration,
  then concentrates sampling near the best observed region with Gaussian
  perturbation, retaining some exploration.

All searches respect a configurable combination limit so uncontrolled
brute-force optimization is never possible.
"""

import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np

from ..logging_config import get_logger

logger = get_logger("research")


@dataclass
class SearchConfig:
    """Search strategy configuration."""

    search_type: str = "grid"            # grid | random | adaptive
    max_combinations: int = 500          # hard cap on evaluations
    random_seed: Optional[int] = 42
    adaptive_initial: int = 25           # initial random phase for adaptive
    adaptive_perturbation: float = 0.2   # mutation scale (fraction of range)


class ParameterSearcher:
    """Generates parameter combinations for a strategy family."""

    SEARCH_TYPES = ("grid", "random", "adaptive")

    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        if self.config.search_type not in self.SEARCH_TYPES:
            raise ValueError(
                f"Unknown search type '{self.config.search_type}'. "
                f"Options: {self.SEARCH_TYPES}"
            )
        self._rng = random.Random(self.config.random_seed)

    # ------------------------------------------------------------ main API
    def combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Return up to max_combinations parameter combinations."""
        if not param_grid:
            return [{}]

        if self.config.search_type == "grid":
            combos = self._grid(param_grid)
        elif self.config.search_type == "random":
            combos = self._random(param_grid)
        else:
            combos = self._adaptive(param_grid)

        # Cap and dedupe
        seen = set()
        capped = []
        for c in combos:
            key = _canonical_key(c)
            if key in seen:
                continue
            seen.add(key)
            capped.append(c)
            if len(capped) >= self.config.max_combinations:
                break
        return capped

    # ------------------------------------------------------------ strategies
    def _grid(self, grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        keys = list(grid.keys())
        values = [grid[k] for k in keys]
        combos = []
        for combo in itertools.product(*values):
            combos.append(dict(zip(keys, combo)))
        # Keep deterministic order
        if len(combos) > self.config.max_combinations:
            logger.warning(
                "Grid has %d combos, capping at %d (consider random search)",
                len(combos), self.config.max_combinations,
            )
        return combos

    def _random(self, grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        keys = list(grid.keys())
        combos = []
        # Bound the random pool by grid product size (avoid huge products)
        pool_size = min(self.config.max_combinations * 2, _product_size(grid))
        for _ in range(pool_size):
            combos.append({k: self._rng.choice(grid[k]) for k in keys})
        return combos

    def _adaptive(self, grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Adaptive search.

        Phase 1 (initial random combos) — evaluated externally; phase 2 builds
        mutations around the best-so-far using a score callback if provided,
        otherwise returns initial + random perturbations.
        """
        keys = list(grid.keys())
        combos = []
        n_init = min(self.config.adaptive_initial, self.config.max_combinations)
        for _ in range(n_init):
            combos.append({k: self._rng.choice(grid[k]) for k in keys})
        # Add perturbed random combos to reach the cap (refinement happens in
        # the DiscoveryEngine which can re-perturb around the best observed).
        budget = self.config.max_combinations - len(combos)
        for _ in range(max(0, budget)):
            base = {k: self._rng.choice(grid[k]) for k in keys}
            combo = self._perturb(base, grid)
            combos.append(combo)
        return combos

    def refine_around(self, best_params: Dict[str, Any], grid: Dict[str, List[Any]], n: int) -> List[Dict[str, Any]]:
        """Generate n mutations around a best parameter set (adaptive step)."""
        combos = []
        for _ in range(n):
            combos.append(self._perturb(best_params, grid))
        return combos

    def _perturb(self, base: Dict[str, Any], grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Gaussian-style mutation within each parameter's allowed values."""
        combo = {}
        scale = self.config.adaptive_perturbation
        for k, v in base.items():
            values = grid[k]
            if len(values) == 1:
                combo[k] = values[0]
                continue
            # Treat values as discrete positions; perturb by index
            idx = values.index(v) if v in values else 0
            sigma = max(1.0, scale * len(values))
            new_idx = int(round(self._rng.gauss(idx, sigma)))
            new_idx = max(0, min(len(values) - 1, new_idx))
            combo[k] = values[new_idx]
        return combo


def _product_size(grid: Dict[str, List[Any]]) -> int:
    size = 1
    for v in grid.values():
        size *= len(v)
    return size


def grid_neighbors(params: Dict[str, Any], grid: Dict[str, List[Any]], radius: int = 1) -> List[Dict[str, Any]]:
    """Return one-step grid neighbors of a parameter set (for stability checks).

    Each neighbor varies exactly one parameter by ±radius positions within the
    grid. Used by parameter-stability analysis (spec #49).
    """
    neighbors = []
    for k, v in params.items():
        values = grid.get(k)
        if not values or len(values) <= 1 or v not in values:
            continue
        idx = values.index(v)
        for delta in (-radius, radius):
            ni = idx + delta
            if 0 <= ni < len(values):
                nparams = dict(params)
                nparams[k] = values[ni]
                neighbors.append(nparams)
    return neighbors


def _canonical_key(combo: Dict[str, Any]) -> str:
    return "|".join(f"{k}={v}" for k, v in sorted(combo.items()))