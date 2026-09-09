"""Machine learning package."""

from .dataset import MLDataset, MLDatasetBuilder, chronological_split
from .train import MODEL_REGISTRY, ModelTrainer, TrainingResult, TrainedModel
from .evaluate import ModelEvaluator, EvaluationResult
from .predict import TradePredictor, Prediction
from .pipeline import MLPipeline, MLPipelineReport

__all__ = [
    "MLDataset", "MLDatasetBuilder", "chronological_split",
    "MODEL_REGISTRY", "ModelTrainer", "TrainingResult", "TrainedModel",
    "ModelEvaluator", "EvaluationResult",
    "TradePredictor", "Prediction",
    "MLPipeline", "MLPipelineReport",
]