"""Model evaluation.

Evaluates trained models on the untouched TEST set (strictly after
validation/model selection — no test-set leakage). Also answers the honest
question: does ML filtering actually improve the strategy's win rate?
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..logging_config import get_logger
from .dataset import MLDataset, chronological_split
from .train import TrainedModel

logger = get_logger("ml")


@dataclass
class EvaluationResult:
    """Evaluation of a model on the test set + ML-vs-baseline comparison."""

    model_name: str
    test_metrics: Dict[str, float]
    baseline_win_rate: float
    ml_filtered_win_rate: Optional[float]
    ml_improvement: Optional[float]        # filtered - baseline (pp)
    ml_improves: Optional[bool]            # honest verdict
    n_tested: int
    n_taken: int
    threshold: float
    feature_importance: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "test_metrics": self.test_metrics,
            "baseline_win_rate": self.baseline_win_rate,
            "ml_filtered_win_rate": self.ml_filtered_win_rate,
            "ml_improvement_pp": self.ml_improvement,
            "ml_improves": self.ml_improves,
            "n_tested": self.n_tested,
            "n_taken": self.n_taken,
            "threshold": self.threshold,
            "feature_importance": self.feature_importance,
            "warnings": self.warnings,
        }


class ModelEvaluator:
    """Evaluates a trained model on the out-of-sample test split."""

    def __init__(self, train_frac: float = 0.7, val_frac: float = 0.15):
        """Initialize with the same split fractions used in training."""
        self.train_frac = train_frac
        self.val_frac = val_frac

    # ------------------------------------------------------------ main API
    def evaluate(
        self,
        trained: TrainedModel,
        dataset: MLDataset,
        threshold: float = 0.60,
    ) -> EvaluationResult:
        """Evaluate a trained model on the test split.

        Args:
            trained: A TrainedModel (scaler fit on train only).
            dataset: Full dataset (will be chronologically split).
            threshold: Probability above which a trade is 'TAKE'.

        Returns:
            EvaluationResult including the honest ML-vs-baseline verdict.
        """
        _, _, test = chronological_split(dataset, self.train_frac, self.val_frac)
        if test.n_samples < 10:
            warnings = ["LOW TEST SAMPLE SIZE"]

        X_test = trained.scaler.transform(test.X)
        proba = trained.model.predict_proba(X_test)[:, 1]

        test_metrics = _classification_metrics(test.y, proba)
        baseline = test.baseline_win_rate

        # ML-filtered subset: trades the model says TAKE
        take_mask = proba >= threshold
        n_taken = int(take_mask.sum())
        if n_taken >= 5:
            ml_wr = float(test.y[take_mask].mean())
            improvement = ml_wr - baseline
            ml_improves = improvement > 0.01   # > 1pp improvement counts
            warnings = _ml_warnings(ml_improves, n_taken, baseline, ml_wr)
        else:
            ml_wr = None
            improvement = None
            ml_improves = None
            warnings = ["TOO FEW TAKES AT THIS THRESHOLD"]

        return EvaluationResult(
            model_name=trained.name,
            test_metrics=test_metrics,
            baseline_win_rate=baseline,
            ml_filtered_win_rate=ml_wr,
            ml_improvement=improvement,
            ml_improves=ml_improves,
            n_tested=test.n_samples,
            n_taken=n_taken,
            threshold=threshold,
            feature_importance=self._feature_importance(trained),
            warnings=warnings,
        )

    # ------------------------------------------------------------ helpers
    def _feature_importance(self, trained: TrainedModel) -> Dict[str, float]:
        model = trained.model
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_[0])
            else:
                return {}
            names = trained.feature_names
            return {n: float(v) for n, v in sorted(zip(names, importances), key=lambda p: -p[1])}
        except Exception:
            return {}


def _classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    )
    y_pred = (proba >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y_true, proba))
    except ValueError:
        auc = 0.5
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": auc,
    }


def _ml_warnings(ml_improves: bool, n_taken: int, baseline: float, ml_wr: float) -> List[str]:
    out = []
    if not ml_improves:
        out.append("ML DOES NOT IMPROVE WIN RATE — take result honestly")
    if n_taken < 20:
        out.append("LOW NUMBER OF TAKES")
    return out