"""Prediction integration.

Loads a trained model and produces the ML decision for a candidate trade:
    Trade Signal: LONG
    ML Probability: 0.73
    Decision: TAKE / SKIP

The decision threshold is configurable. The predictor never re-fits anything;
it only applies an already-trained (leakage-safe) model.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .train import TrainedModel


@dataclass
class Prediction:
    """ML decision for a single candidate trade."""

    direction: str
    probability: float
    decision: str            # 'TAKE' | 'SKIP'
    threshold: float
    confidence: float        # distance from threshold

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "ml_probability": round(self.probability, 3),
            "decision": self.decision,
            "threshold": self.threshold,
            "confidence": round(self.confidence, 3),
        }


class TradePredictor:
    """Applies a trained model to candidate trade features."""

    def __init__(self, trained: TrainedModel, default_threshold: float = 0.60):
        """Initialize with a trained model."""
        self.trained = trained
        self.default_threshold = default_threshold

    # ------------------------------------------------------------ main API
    def predict(
        self,
        features: Dict[str, float],
        direction: str = "long",
        threshold: Optional[float] = None,
    ) -> Prediction:
        """Predict the probability a trade is profitable.

        Args:
            features: Feature dict keyed by the model's feature names.
            direction: Trade direction ('long'/'short').
            threshold: TAKE threshold (defaults to the configured value).

        Returns:
            Prediction with TAKE/SKIP decision.
        """
        thr = threshold if threshold is not None else self.default_threshold
        x = self._vectorize(features)
        x_scaled = self.trained.scaler.transform(x.reshape(1, -1))
        proba = float(self.trained.model.predict_proba(x_scaled)[0, 1])
        decision = "TAKE" if proba >= thr else "SKIP"
        return Prediction(
            direction=direction,
            probability=proba,
            decision=decision,
            threshold=thr,
            confidence=proba - thr,
        )

    def predict_batch(
        self,
        rows: List[Dict[str, float]],
        directions: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ) -> List[Prediction]:
        """Predict for many candidate trades at once."""
        return [
            self.predict(f, directions[i] if directions else "long", threshold)
            for i, f in enumerate(rows)
        ]

    # ------------------------------------------------------------ helpers
    def _vectorize(self, features: Dict[str, float]) -> np.ndarray:
        vec = []
        for name in self.trained.feature_names:
            v = features.get(name)
            vec.append(0.0 if v is None else (float(v) if np.isfinite(v) else 0.0))
        return np.array(vec, dtype=float)