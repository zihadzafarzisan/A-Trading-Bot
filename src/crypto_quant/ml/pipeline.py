"""End-to-end ML pipeline orchestrator.

dataset -> train -> evaluate  with strict chronological splits and
no test-set leakage. Produces an honest report of whether ML improves the
underlying strategy.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..logging_config import get_logger
from .dataset import MLDataset, MLDatasetBuilder
from .evaluate import EvaluationResult, ModelEvaluator
from .predict import TradePredictor
from .train import MODEL_REGISTRY, ModelTrainer, TrainingResult

logger = get_logger("ml")


@dataclass
class MLPipelineReport:
    """Full ML pipeline output."""

    n_samples: int
    n_features: int
    baseline_win_rate: float
    training: TrainingResult
    evaluation: EvaluationResult
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "baseline_win_rate": self.baseline_win_rate,
            "training": self.training.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "warnings": self.warnings,
        }


class MLPipeline:
    """Runs the complete ML workflow for a strategy's trades."""

    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        threshold: float = 0.60,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        save_dir: Optional[str] = None,
    ):
        """Initialize pipeline."""
        self.model_names = model_names or list(MODEL_REGISTRY.keys())
        self.threshold = threshold
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.save_dir = save_dir

    # ------------------------------------------------------------ main API
    def run(
        self,
        trades: List[Dict[str, Any]],
        features_df,
        symbol: str = "",
        timeframe: str = "",
        strategy_type: str = "",
    ) -> MLPipelineReport:
        """Run the full pipeline.

        Args:
            trades: Backtest trade dicts.
            features_df: FeatureEngine output aligned to entry times.
            symbol/timeframe/strategy_type: Optional metadata.

        Returns:
            MLPipelineReport.
        """
        builder = MLDatasetBuilder()
        dataset = builder.build(trades, features_df, symbol, timeframe, strategy_type)

        if dataset.n_samples < 30:
            warnings = ["LOW SAMPLE SIZE — ML results unreliable"]
            logger.warning("ML pipeline got only %d samples", dataset.n_samples)
        else:
            warnings = []

        trainer = ModelTrainer(
            model_names=self.model_names, train_frac=self.train_frac, val_frac=self.val_frac,
        )
        training = trainer.train(dataset, save_dir=self.save_dir)

        best_trained = training.trained[training.best_model_name]
        evaluator = ModelEvaluator(train_frac=self.train_frac, val_frac=self.val_frac)
        evaluation = evaluator.evaluate(best_trained, dataset, threshold=self.threshold)

        if evaluation.ml_improves is False:
            warnings.append("ML DID NOT IMPROVE THE STRATEGY — reported honestly")

        return MLPipelineReport(
            n_samples=dataset.n_samples,
            n_features=dataset.n_features,
            baseline_win_rate=dataset.baseline_win_rate,
            training=training,
            evaluation=evaluation,
            warnings=warnings,
        )

    # ------------------------------------------------------------ helpers
    @staticmethod
    def make_predictor(trained_model) -> TradePredictor:
        """Build a TradePredictor from a TrainedModel."""
        return TradePredictor(trained_model)