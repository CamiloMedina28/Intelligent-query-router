"""Generic multiclass test evaluator for router-model adapters."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


ROUTER_CLASSES = np.array([0, 1, 2])
ROUTE_LABELS = {0: "LLM", 1: "API", 2: "UNDETERMINED"}


class ModelTester:
    """Tests one model and writes comparable multiclass evaluation artifacts."""

    def __init__(self, model, model_name, dataset="data/processed/test.csv", output_dir="artifacts/testing"):
        self.model = model
        self.model_name = model_name
        self.dataset = Path(dataset)
        self.output_dir = Path(output_dir) / model_name

    def _load_dataset(self):
        dataset = pd.read_csv(self.dataset)
        required_columns = {"query", "path"}
        missing_columns = required_columns - set(dataset.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
        if dataset.empty:
            raise ValueError("The test dataset must not be empty")

        labels = set(dataset["path"].astype(int).unique())
        if not labels.issubset(set(ROUTER_CLASSES)):
            raise ValueError(f"Unsupported labels in test dataset: {sorted(labels)}")
        return dataset

    def _validate_model_classes(self, probabilities):
        classes = np.asarray(getattr(self.model, "classes_", []), dtype=int)
        if not np.array_equal(classes, ROUTER_CLASSES):
            raise ValueError(
                f"{self.model_name} exposes classes {classes.tolist()}, but the router "
                "requires [0, 1, 2]. Retrain this model with all three classes."
            )
        if probabilities.shape[1] != len(ROUTER_CLASSES):
            raise ValueError(
                f"{self.model_name} returned {probabilities.shape[1]} probability "
                "columns; expected 3."
            )

    def run(self):
        """Runs test inference, metrics, confusion matrix, and artifact persistence."""

        dataset = self._load_dataset()
        queries = dataset["query"].tolist()
        y_true = dataset["path"].astype(int).to_numpy()
        probabilities = np.asarray(self.model.predict_proba(queries))
        self._validate_model_classes(probabilities)
        y_pred = ROUTER_CLASSES[probabilities.argmax(axis=1)]

        matrix = confusion_matrix(y_true, y_pred, labels=ROUTER_CLASSES)
        report = classification_report(
            y_true,
            y_pred,
            labels=ROUTER_CLASSES,
            target_names=[ROUTE_LABELS[label] for label in ROUTER_CLASSES],
            output_dict=True,
            zero_division=0,
        )
        results = {
            "model": self.model_name,
            "samples": int(len(dataset)),
            "class_distribution": {
                str(label): int((y_true == label).sum()) for label in ROUTER_CLASSES
            },
            "metrics": {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "precision_macro": float(
                    precision_score(y_true, y_pred, average="macro", zero_division=0)
                ),
                "recall_macro": float(
                    recall_score(y_true, y_pred, average="macro", zero_division=0)
                ),
                "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
                "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
                "log_loss": float(log_loss(y_true, probabilities, labels=ROUTER_CLASSES)),
            },
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
        }

        self._save_artifacts(dataset, y_pred, probabilities, results, matrix)
        return results

    def _save_artifacts(self, dataset, y_pred, probabilities, results, matrix):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        predictions = dataset.copy()
        predictions["prediction"] = y_pred
        predictions["prediction_label"] = [ROUTE_LABELS[label] for label in y_pred]
        predictions["probability_llm"] = probabilities[:, 0]
        predictions["probability_api"] = probabilities[:, 1]
        predictions["probability_undetermined"] = probabilities[:, 2]
        predictions.to_csv(self.output_dir / "predictions.csv", index=False, encoding="utf-8")

        with (self.output_dir / "results.json").open("w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=2)
        self._save_confusion_matrix(matrix)

    def _save_confusion_matrix(self, matrix):
        figure, axis = plt.subplots(figsize=(7, 5))
        image = axis.imshow(matrix, cmap="Blues")
        figure.colorbar(image, ax=axis)
        axis.set(
            xticks=range(len(ROUTER_CLASSES)),
            yticks=range(len(ROUTER_CLASSES)),
            xticklabels=[ROUTE_LABELS[label] for label in ROUTER_CLASSES],
            yticklabels=[ROUTE_LABELS[label] for label in ROUTER_CLASSES],
            xlabel="Predicted route",
            ylabel="Actual route",
            title=f"Confusion Matrix — {self.model_name}",
        )
        for row in range(len(ROUTER_CLASSES)):
            for column in range(len(ROUTER_CLASSES)):
                axis.text(column, row, matrix[row, column], ha="center", va="center")
        figure.tight_layout()
        figure.savefig(self.output_dir / "confusion_matrix.png", dpi=150)
        plt.close(figure)
