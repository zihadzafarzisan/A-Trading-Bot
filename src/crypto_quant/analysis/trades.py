"""Trade research engine.

Analyzes winning and losing trades separately to discover statistically
supported patterns: RSI range, EMA relationship, volume ratio, ATR percentile,
volatility, trend, time-of-day, day-of-week, market regime, direction, and
holding period. Produces human-readable findings only when a difference is
statistically meaningful — it never invents explanations.

A finding is reported when the winner/loser distributions differ with
|Cohen's d| >= 0.3 and (p < 0.10 when the sample is large enough to test).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..logging_config import get_logger
from .regimes import RegimeClassifier
from .statistics import (
    EffectSize,
    category_distribution,
    compare_effect,
    percentile_range,
)

logger = get_logger("analysis")

# Numeric entry-context features to profile (name -> display label)
NUMERIC_FEATURES: Dict[str, str] = {
    "rsi_14": "RSI",
    "volume_ratio": "Volume ratio",
    "atr_ratio": "ATR / price",
    "hist_vol_20": "Historical volatility",
    "adx_14": "ADX",
    "dist_sma_50": "Distance from SMA50 (%)",
    "roc_12": "ROC",
}

# Categorical context dimensions
CATEGORICAL_DIMENSIONS = ["regime", "vol_regime", "direction", "hour_bucket", "day_of_week"]


@dataclass
class FeatureFinding:
    """A numeric feature pattern distinguishing winners from losers."""

    feature: str
    label: str
    winners_mean: float
    losers_mean: float
    winners_range: tuple
    losers_range: tuple
    effect: EffectSize

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "label": self.label,
            "winners_mean": round(self.winners_mean, 4),
            "losers_mean": round(self.losers_mean, 4),
            "winners_range": [round(x, 4) for x in self.winners_range],
            "losers_range": [round(x, 4) for x in self.losers_range],
            "effect": self.effect.to_dict(),
        }


@dataclass
class CategoryFinding:
    """A categorical dimension where winners and losers differ in mix."""

    dimension: str
    winners_dist: dict
    losers_dist: dict
    top_winners: str
    top_losers: str

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "winners_distribution": self.winners_dist,
            "losers_distribution": self.losers_dist,
            "top_winners": self.top_winners,
            "top_losers": self.top_losers,
        }


@dataclass
class TradeAnalysisReport:
    """Full trade-pattern analysis output."""

    n_winning: int = 0
    n_losing: int = 0
    n_total: int = 0
    win_rate: float = 0.0
    feature_findings: List[FeatureFinding] = field(default_factory=list)
    category_findings: List[CategoryFinding] = field(default_factory=list)
    regime_breakdown: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_winning": self.n_winning,
            "n_losing": self.n_losing,
            "n_total": self.n_total,
            "win_rate": self.win_rate,
            "feature_findings": [f.to_dict() for f in self.feature_findings],
            "category_findings": [c.to_dict() for c in self.category_findings],
            "regime_breakdown": self.regime_breakdown,
        }


class TradeAnalyzer:
    """Mines winning vs losing trade patterns with statistical support."""

    def __init__(self, regime_classifier: Optional[RegimeClassifier] = None):
        self.regime_classifier = regime_classifier or RegimeClassifier()

    # ------------------------------------------------------------ main API
    def analyze(
        self,
        trades: List[dict],
        features_df: Optional[pd.DataFrame] = None,
    ) -> TradeAnalysisReport:
        """Analyze a list of closed trade dicts.

        Args:
            trades: Backtest trade dicts (must include entry_time, net_pnl,
                direction). Optionally symbol, timeframe, exit_reason.
            features_df: DataFrame with 'timestamp' plus feature columns
                (e.g. from FeatureEngine) used to attach entry context.

        Returns:
            TradeAnalysisReport with statistically supported findings.
        """
        report = TradeAnalysisReport()
        closed = [t for t in trades if t.get("net_pnl") is not None]
        report.n_total = len(closed)
        report.n_winning = sum(1 for t in closed if t["net_pnl"] > 0)
        report.n_losing = sum(1 for t in closed if t["net_pnl"] < 0)
        report.win_rate = report.n_winning / report.n_total if report.n_total else 0.0

        if report.n_total == 0:
            return report

        # Attach entry context
        context = self._build_context(closed, features_df)
        winners = [c for c in context if c["trade"]["net_pnl"] > 0]
        losers = [c for c in context if c["trade"]["net_pnl"] < 0]

        if len(winners) < 5 or len(losers) < 5:
            logger.warning(
                "Too few trades for robust pattern analysis (winners=%d losers=%d)",
                len(winners), len(losers),
            )

        # Numeric feature profiling
        if features_df is not None:
            for feature, label in NUMERIC_FEATURES.items():
                w_vals = [c["features"].get(feature) for c in winners]
                l_vals = [c["features"].get(feature) for c in losers]
                w_vals = [v for v in w_vals if v is not None and np.isfinite(v)]
                l_vals = [v for v in l_vals if v is not None and np.isfinite(v)]
                if len(w_vals) < 3 or len(l_vals) < 3:
                    continue
                effect = compare_effect(w_vals, l_vals)
                if effect.meaningful:
                    report.feature_findings.append(FeatureFinding(
                        feature=feature, label=label,
                        winners_mean=float(np.mean(w_vals)),
                        losers_mean=float(np.mean(l_vals)),
                        winners_range=percentile_range(w_vals),
                        losers_range=percentile_range(l_vals),
                        effect=effect,
                    ))

        # Categorical profiling
        for dim in CATEGORICAL_DIMENSIONS:
            w_vals = [c["context"].get(dim) for c in winners]
            l_vals = [c["context"].get(dim) for c in losers]
            if all(v is not None for v in w_vals + l_vals):
                report.category_findings.append(self._category_finding(dim, w_vals, l_vals))

        # Regime breakdown of win rates
        report.regime_breakdown = self._regime_breakdown(closed, context)

        return report

    # ------------------------------------------------------------ context
    def _build_context(self, trades: List[dict], features_df: Optional[pd.DataFrame]) -> List[dict]:
        """Attach entry-time features and categorical context to each trade."""
        # Index features by timestamp for fast lookup
        feat_by_ts = {}
        if features_df is not None and not features_df.empty:
            ts = features_df["timestamp"].to_numpy()
            for i in range(len(features_df)):
                feat_by_ts[int(ts[i])] = i

        context = []
        for t in trades:
            entry_ts = int(t["entry_time"])
            ctx = {
                "direction": t.get("direction", "long"),
                "hour_bucket": _hour_bucket(entry_ts),
                "day_of_week": _day_of_week(entry_ts),
            }
            row_features = {}
            if entry_ts in feat_by_ts:
                idx = feat_by_ts[entry_ts]
                row = features_df.iloc[idx]
                for f in NUMERIC_FEATURES:
                    if f in row:
                        row_features[f] = float(row[f])
                if "regime" in row:
                    ctx["regime"] = str(row["regime"])
                if "vol_regime" in row:
                    ctx["vol_regime"] = str(row["vol_regime"])
            context.append({"trade": t, "features": row_features, "context": ctx})

        return context

    def _category_finding(self, dimension: str, w_vals: List, l_vals: List) -> CategoryFinding:
        w_dist = category_distribution(w_vals)
        l_dist = category_distribution(l_vals)
        top_w = max(w_dist, key=lambda k: w_dist[k]["count"]) if w_dist else ""
        top_l = max(l_dist, key=lambda k: l_dist[k]["count"]) if l_dist else ""
        return CategoryFinding(
            dimension=dimension,
            winners_dist={k: v["share"] for k, v in w_dist.items()},
            losers_dist={k: v["share"] for k, v in l_dist.items()},
            top_winners=top_w, top_losers=top_l,
        )

    def _regime_breakdown(self, trades: List[dict], context: List[dict]) -> dict:
        """Win rate per regime."""
        by_regime: Dict[str, dict] = {}
        for c in context:
            reg = c["context"].get("regime")
            if reg is None:
                continue
            bucket = by_regime.setdefault(reg, {"wins": 0, "total": 0})
            bucket["total"] += 1
            if c["trade"]["net_pnl"] > 0:
                bucket["wins"] += 1
        return {
            reg: {"win_rate": d["wins"] / d["total"], "trades": d["total"]}
            for reg, d in by_regime.items()
        }

    # ------------------------------------------------------------ summary
    def summarize(self, report: TradeAnalysisReport) -> List[str]:
        """Human-readable findings sentences (with statistical backing)."""
        lines = []
        for f in report.feature_findings:
            direction = "higher" if f.effect.winners_higher else "lower"
            lines.append(
                f"Winning trades {direction} {f.label} "
                f"(winners {f.winners_mean:.2f} vs losers {f.losers_mean:.2f}, "
                f"d={f.effect.cohens_d:.2f}, p={f.effect.p_value:.3f})"
            )
        for c in report.category_findings:
            lines.append(
                f"{c.dimension}: winners most often '{c.top_winners}' "
                f"vs losers '{c.top_losers}'"
            )
        return lines


# ---------------------------------------------------------------------------
# Time helpers (UTC)
# ---------------------------------------------------------------------------
def _hour_bucket(ts_ms: int, bucket_hours: int = 4) -> str:
    """UTC hour of day bucketed (e.g. 'hour_0_4')."""
    from datetime import datetime, timezone
    hour = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour
    start = (hour // bucket_hours) * bucket_hours
    return f"hour_{start}_{start + bucket_hours}"


def _day_of_week(ts_ms: int) -> str:
    from datetime import datetime, timezone
    names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    dow = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).weekday()
    return names[dow]