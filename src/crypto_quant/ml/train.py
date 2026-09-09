"""Model training with strict anti-leakage.

Preprocessing (standardization) is FIT ONLY ON THE TRAIN SET, then applied to
validation and test. No dataset-wide normalization. Trains Logistic Regression,
Random Forest, and XGBoost, and compares them on validation.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import joblib

from ..logging_config import get_logger
from .dataset import MLDataset, chronological_split

logger = get_logger("ml")


@dataclass
class TrainedModel:
    """A trained model with its scaler and metadata."""

    name: str
    model: Any
    scaler: Any
    feature_names: List[str]
    metrics: Dict[str, Any]
    model_path: Optional[str] = None

    def save(self, directory: str) -> str:
        """Persist model + scaler + metadata to a directory."""
        os.makedirs(directory, exist_ok=True)
        base = os.path.join(directory, f"{self.name}.joblib")
        joblib.dump(
            {"model": self.model, "scaler": self.scaler,
             "feature_names": self.feature_names, "metrics": self.metrics},
            base,
        )
        self.model_path = base
        return base


# Model factories (kept small and deterministic)
def _make_logistic(**kwargs):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, C=1.0, **kwargs)


def _make_random_forest(**kwargs):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=200, max_depth=6,
                                  min_samples_leaf=5, random_state=42, **kwargs)


def _make_xgboost(**kwargs):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         eval_metric="logloss", random_state=42, **kwargs)


MODEL_REGISTRY = {
    "logistic_regression": _make_logistic,
    "random_forest": _make_random_forest,
    "xgboost": _make_xgboost,
}


@dataclass
class TrainingResult:
    """Comparison of models trained on the same splits."""

    trained: Dict[str, TrainedModel]
    val_metrics: Dict[str, Dict[str, float]]
    best_model_name: str
    baseline_win_rate: float

    def to_dict(self) -> dict:
        return {
            "models": {name: m.metrics for name, m in self.trained.items()},
            "best_model": self.best_model_name,
            "baseline_win_rate": self.baseline_win_rate,
        }


class ModelTrainer:
    """Trains and compares models with chronological, leakage-safe splits."""

    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        random_state: int = 42,
    ):
        """Initialize trainer."""
        self.model_names = model_names or list(MODEL_REGISTRY.keys())
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.random_state = random_state
        for name in self.model_names:
            if name not in MODEL_REGISTRY:
                raise ValueError(f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY)}")

    # ------------------------------------------------------------ main API
    def train(
        self,
        dataset: MLDataset,
        save_dir: Optional[str] = None,
    ) -> TrainingResult:
        """Train all models on chronological splits.

        The scaler is fit on TRAIN ONLY and reused for validation (and later
        test) — the critical anti-leakage step.
        """
        train, val, _ = chronological_split(dataset, self.train_frac, self.val_frac)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(train.X)
        X_train = scaler.transform(train.X)
        X_val = scaler.transform(val.X)

        trained: Dict[str, TrainedModel] = {}
        val_metrics: Dict[str, Dict[str, float]] = {}

        for name in self.model_names:
            factory = MODEL_REGISTRY[name]
            model = factory()
            model.fit(X_train, train.y)
            proba = model.predict_proba(X_val)[:, 1]
            metrics = _classification_metrics(val.y, proba)
            val_metrics[name] = metrics
            tm = TrainedModel(
                name=name, model=model, scaler=scaler,
                feature_names=dataset.feature_names, metrics=metrics,
            )
            if save_dir:
                tm.save(save_dir)
            trained[name] = tm
            logger.info("Trained %s: val_auc=%.3f val_acc=%.3f", name, metrics["auc"], metrics["accuracy"])

        best_name = max(val_metrics, key=lambda n: val_metrics[n]["auc"])
        return TrainingResult(
            trained=trained,
            val_metrics=val_metrics,
            best_model_name=best_name,
            baseline_win_rate=dataset.baseline_win_rate,
        )

    # ------------------------------------------------------------ helpers
    @staticmethod
    def load(path: str) -> TrainedModel:
        """Load a TrainedModel from a joblib file."""
        data = joblib.load(path)
        return TrainedModel(
            name=os.path.basename(path).replace(".joblib", ""),
            model=data["model"], scaler=data["scaler"],
            feature_names=data["feature_names"], metrics=data["metrics"],
            model_path=path,
        )


def _classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    """Accuracy, precision, recall, f1, AUC at the 0.5 threshold."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    )
    y_pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, proba)),
    }
    return metrics