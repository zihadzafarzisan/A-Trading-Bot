"""Statistical helpers for trade analysis.

Provides effect sizes and significance tests using only numpy (no scipy
dependency). Implements:
- Cohen's d effect size
- Welch's t-test p-value (approximated via the normal for large n)
- percentile ranges for describing distributions
- categorical chi-square-like comparison via proportional difference

These tools are used to support trade-pattern findings statistically rather
than by eyeballing means.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class EffectSize:
    """Effect size and significance between two distributions."""

    cohens_d: float
    p_value: Optional[float]
    winners_higher: bool
    meaningful: bool  # |d| >= 0.3 and (p < 0.10 if p is not None)
    n_a: int
    n_b: int

    def to_dict(self) -> dict:
        return {
            "cohens_d": round(self.cohens_d, 3),
            "p_value": round(self.p_value, 4) if self.p_value is not None else None,
            "winners_higher": self.winners_higher,
            "meaningful": self.meaningful,
            "n_winners": self.n_a,
            "n_losers": self.n_b,
        }


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d effect size between two independent samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    na, nb = len(a), len(b)
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled = math.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def welch_t_pvalue(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Two-sided p-value for Welch's t-test, normal approximation.

    Returns None when the test is not applicable (too few samples).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 3 or nb < 3:
        return None

    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    denom = var_a / na + var_b / nb
    if denom == 0:
        return 0.0  # identical variance, mean difference exact

    t = (a.mean() - b.mean()) / math.sqrt(denom)
    # Welch–Satterthwaite degrees of freedom
    df = denom ** 2 / ((var_a / na) ** 2 / (na - 1) + (var_b / nb) ** 2 / (nb - 1))
    if df <= 0:
        return None
    # Normal approximation for the t CDF
    z = t / math.sqrt(1 + (df - 2) / (df - 1)) if df > 1 else t
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return float(min(p, 1.0))


def compare_effect(a: Sequence[float], b: Sequence[float]) -> EffectSize:
    """Compare two distributions and report effect + significance."""
    d = cohens_d(a, b)
    p = welch_t_pvalue(a, b)
    meaningful = abs(d) >= 0.3 and (p is None or p < 0.10)
    return EffectSize(
        cohens_d=d, p_value=p, winners_higher=(d > 0),
        meaningful=meaningful, n_a=len(a), n_b=len(b),
    )


def percentile_range(values: Sequence[float], lo: float = 25.0, hi: float = 75.0) -> Tuple[float, float]:
    """Return (lo, hi) percentile bounds of a distribution."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (0.0, 0.0)
    return (float(np.percentile(v, lo)), float(np.percentile(v, hi)))


def category_distribution(
    values: Sequence[str],
    categories: Optional[List[str]] = None,
) -> dict:
    """Distribution (count and share) of categorical values."""
    vals = list(values)
    n = len(vals) or 1
    counts: dict = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    if categories:
        for c in categories:
            counts.setdefault(c, 0)
    return {k: {"count": c, "share": c / n} for k, c in sorted(counts.items())}


def _normal_cdf(x: float) -> float:
    """Standard normal CDF (via math.erf)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))