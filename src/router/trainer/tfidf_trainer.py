"""Trainer for the multiclass TF-IDF baseline."""

from pathlib import Path

from src.router.models import tfidf_model
from src.router.trainer.base import BaseTrainer


class TfidfTrainer(BaseTrainer):
    """Fits and persists the TF-IDF logistic-regression router."""

    def __init__(self, artifact_path="artifacts/tfidf/tfidf.joblib", **kwargs):
        super().__init__(**kwargs)
        self.artifact_path = Path(artifact_path)

    def train(self):
        train_df = self.load_dataset("training.csv")
        model = tfidf_model.TfIdfLogisticRegression()
        trainer = tfidf_model.TfIdfLogisticRegressionTrain(model=model)
        trainer.train(train_df["query"], train_df["path"])
        trainer.save(self.artifact_path)
        return model
