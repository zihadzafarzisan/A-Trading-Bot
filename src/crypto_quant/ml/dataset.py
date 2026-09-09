"""ML dataset construction.

Builds a supervised dataset from backtest trades + entry-time features. Every
feature is computed from the FeatureEngine (already shifted, so no future-candle
leakage). The dataset retains each sample's entry timestamp so the pipeline can
perform STRICT CHRONOLOGICAL splits — never random shuffles.

Target: y = 1 if the trade was profitable (net_pnl > 0), else 0.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..logging_config import get_logger

logger = get_logger("ml")


@dataclass
class MLDataset:
    """A ready-to-model dataset with preserved time ordering."""

    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    timestamps: np.ndarray
    trade_ids: List[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.y)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def baseline_win_rate(self) -> float:
        return float(self.y.mean()) if len(self.y) else 0.0


# Categorical -> one-hot style numeric columns added to the feature matrix
CATEGORICAL_FEATURES = ["direction", "regime", "vol_regime", "hour_bucket", "day_of_week"]


class MLDatasetBuilder:
    """Builds an MLDataset from trades and a feature DataFrame."""

    def __init__(self, feature_cols: Optional[List[str]] = None):
        """Initialize.

        Args:
            feature_cols: Numeric feature columns to use. Defaults to the
                canonical FeatureEngine feature set.
        """
        if feature_cols is None:
            from ..features.engine import FeatureEngine
            feature_cols = FeatureEngine.feature_names()
        self.feature_cols = [c for c in feature_cols if c != "timestamp"]

    # ------------------------------------------------------------ main API
    def build(
        self,
        trades: List[Dict[str, Any]],
        features_df: pd.DataFrame,
        symbol: str = "",
        timeframe: str = "",
        strategy_type: str = "",
    ) -> MLDataset:
        """Build the dataset.

        Args:
            trades: Backtest trade dicts (entry_time, net_pnl, direction).
            features_df: FeatureEngine output (timestamp + feature columns).
            symbol/timeframe/strategy_type: Optional metadata columns added
                as numeric indicators.

        Returns:
            MLDataset (samples sorted chronologically by entry time).
        """
        # Index features by timestamp
        feat_by_ts = {}
        if features_df is not None and not features_df.empty:
            ts = features_df["timestamp"].to_numpy()
            for i in range(len(features_df)):
                feat_by_ts[int(ts[i])] = i

        rows = []
        ys = []
        timestamps = []
        ids = []
        meta_cols: Dict[str, List] = {c: [] for c in CATEGORICAL_FEATURES}

        for idx, t in enumerate(trades):
            entry_ts = int(t["entry_time"])
            net_pnl = t.get("net_pnl")
            if net_pnl is None:
                continue
            timestamps.append(entry_ts)
            ys.append(1 if net_pnl > 0 else 0)
            ids.append(t.get("trade_id") or f"trade_{idx}")

            feat_row = None
            if entry_ts in feat_by_ts:
                fr = features_df.iloc[feat_by_ts[entry_ts]]
                feat_row = {c: fr[c] for c in self.feature_cols if c in fr.index}

            # Add meta/categorical context
            row = {}
            if feat_row:
                row.update(feat_row)
            row["symbol"] = 1.0 if symbol else 0.0
            row["timeframe"] = 1.0 if timeframe else 0.0
            row["strategy_type"] = 1.0 if strategy_type else 0.0
            row["direction"] = 1.0 if t.get("direction") == "long" else 0.0

            rows.append(row)

        if not rows:
            logger.warning("MLDatasetBuilder: no usable trades")
            return MLDataset(
                X=np.empty((0, len(self.feature_cols) + 4)),
                y=np.array([]), feature_names=[], timestamps=np.array([]),
            )

        X = np.array([self._vectorize(r) for r in rows], dtype=float)
        y = np.array(ys, dtype=int)
        timestamps = np.array(timestamps)
        feature_names = self.feature_cols + ["symbol", "timeframe", "strategy_type", "direction"]

        # Chronological sort
        order = np.argsort(timestamps, kind="stable")
        return MLDataset(
            X=X[order], y=y[order], feature_names=feature_names,
            timestamps=timestamps[order],
            trade_ids=[ids[i] for i in order],
        )

    def _vectorize(self, row: Dict[str, float]) -> np.ndarray:
        vec = []
        for c in self.feature_cols + ["symbol", "timeframe", "strategy_type", "direction"]:
            v = row.get(c)
            vec.append(0.0 if v is None else (float(v) if np.isfinite(v) else 0.0))
        return np.array(vec, dtype=float)


def chronological_split(
    dataset: MLDataset,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[MLDataset, MLDataset, MLDataset]:
    """Split a dataset chronologically by entry timestamp (never random).

    Returns (train, validation, test).
    """
    n = dataset.n_samples
    if n == 0:
        raise ValueError("Cannot split an empty dataset")
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    def _slice(a, lo, hi):
        return a[lo:hi]

    train = MLDataset(
        X=_slice(dataset.X, 0, train_end),
        y=_slice(dataset.y, 0, train_end),
        feature_names=dataset.feature_names,
        timestamps=_slice(dataset.timestamps, 0, train_end),
        trade_ids=dataset.trade_ids[:train_end],
    )
    validation = MLDataset(
        X=_slice(dataset.X, train_end, val_end),
        y=_slice(dataset.y, train_end, val_end),
        feature_names=dataset.feature_names,
        timestamps=_slice(dataset.timestamps, train_end, val_end),
        trade_ids=dataset.trade_ids[train_end:val_end],
    )
    test = MLDataset(
        X=_slice(dataset.X, val_end, n),
        y=_slice(dataset.y, val_end, n),
        feature_names=dataset.feature_names,
        timestamps=_slice(dataset.timestamps, val_end, n),
        trade_ids=dataset.trade_ids[val_end:],
    )
    return train, validation, test