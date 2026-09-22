"""Adapters that give every router model the same prediction interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.router.models import frozen, full_fine_tuning, partial_fine_tuning
from src.router.models.tfidf_model import TfIdfLogisticRegressionTrain


ROUTER_CLASSES = np.array([0, 1, 2])


class TfidfAdapter:
    """Adapts the persisted TF-IDF wrapper to the tester interface."""

    def __init__(self, artifact_path):
        self.model = TfIdfLogisticRegressionTrain.load(artifact_path)
        self.classes_ = np.asarray(self.model.model.classes_)

    def predict(self, queries):
        return self.model.predict(queries)

    def predict_proba(self, queries):
        return self.model.predict_proba(queries)


class FrozenAdapter:
    """Loads the frozen encoder and head for test-time predictions."""

    classes_ = ROUTER_CLASSES

    def __init__(self, artifact_path):
        self.encoder = frozen.FrozenSentenceTransformer()
        self.classifier = frozen.ClassificationFrozenLinearHead()
        self.classifier.load_state_dict(
            torch.load(artifact_path, map_location="cpu", weights_only=True)
        )
        self.classifier.eval()

    def predict_proba(self, queries, batch_size=32):
        embeddings = self.encoder.generate_embeddings(queries, batch_size=batch_size)
        with torch.no_grad():
            return torch.softmax(self.classifier(embeddings), dim=1).numpy()

    def predict(self, queries):
        return self.predict_proba(queries).argmax(axis=1)


class FineTunedAdapter:
    """Adapts partial and full fine-tuned classifiers for batched inference."""

    classes_ = ROUTER_CLASSES

    def __init__(self, model, artifact_path):
        self.model = model
        self.model.load_state_dict(
            torch.load(artifact_path, map_location=self.model.device, weights_only=True)
        )
        self.model.eval()

    def predict_proba(self, queries, batch_size=32):
        batches = []
        queries = list(queries)
        with torch.no_grad():
            for start in range(0, len(queries), batch_size):
                features = self.model.preprocess(queries[start : start + batch_size])
                features = {
                    name: value.to(self.model.device)
                    if isinstance(value, torch.Tensor)
                    else value
                    for name, value in features.items()
                }
                probabilities = torch.softmax(self.model(features), dim=1)
                batches.append(probabilities.cpu().numpy())
        return np.concatenate(batches, axis=0)

    def predict(self, queries):
        return self.predict_proba(queries).argmax(axis=1)


def load_model_adapter(model_name, artifacts_dir="artifacts"):
    """Loads one persisted router model by its canonical experiment name."""

    artifacts_dir = Path(artifacts_dir)
    normalized_name = model_name.lower().replace("-", "_")

    if normalized_name == "tfidf":
        return TfidfAdapter(artifacts_dir / "tfidf" / "tfidf.joblib")
    if normalized_name == "frozen":
        return FrozenAdapter(artifacts_dir / "frozen" / "classifier.pt")
    if normalized_name == "partial_fine_tuning":
        return FineTunedAdapter(
            partial_fine_tuning.PartialFineTuningModel(),
            artifacts_dir / "partial_fine_tuning" / "model.pt",
        )
    if normalized_name == "full_fine_tuning":
        return FineTunedAdapter(
            full_fine_tuning.FullFineTuningModel(),
            artifacts_dir / "full_fine_tuning" / "model.pt",
        )

    supported = "tfidf, frozen, partial_fine_tuning, full_fine_tuning"
    raise ValueError(f"Unknown model '{model_name}'. Supported models: {supported}")
