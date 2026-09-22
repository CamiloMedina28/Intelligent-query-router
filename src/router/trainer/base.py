"""Shared utilities and contracts for router model trainers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATA_DIR = Path("data/processed")
REQUIRED_COLUMNS = {"query", "path"}


class TextClassificationDataset(Dataset):
    """Raw text queries paired with their integer route labels."""

    def __init__(self, queries, labels):
        self.queries = [str(query) for query in queries]
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        if len(self.queries) != len(self.labels):
            raise ValueError("queries and labels must have the same length")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.queries[index], self.labels[index]


class BaseTrainer(ABC):
    """Base class for shared dataset, reproducibility, and artifact behavior."""

    def __init__(self, data_dir=DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)

    def load_dataset(self, filename):
        """Loads and validates a processed router dataset."""

        dataset = pd.read_csv(self.data_dir / filename)
        missing_columns = REQUIRED_COLUMNS - set(dataset.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
        if dataset.empty:
            raise ValueError(f"{filename} must not be empty")
        if dataset["query"].isna().any() or dataset["path"].isna().any():
            raise ValueError(f"{filename} must not contain null queries or labels")
        return dataset

    def load_train_validation_datasets(self):
        """Loads the standard train and validation splits."""

        return self.load_dataset("training.csv"), self.load_dataset("validation.csv")

    @staticmethod
    def set_seed(seed=42):
        """Sets PyTorch seeds used by all neural-model training runs."""

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def class_weights(labels, num_classes):
        """Returns inverse-frequency weights and requires every class to exist."""

        counts = pd.Series(labels).value_counts().reindex(
            range(num_classes), fill_value=0
        )
        if (counts == 0).any():
            raise ValueError(
                "training.csv must contain every class from 0 to "
                f"{num_classes - 1}"
            )
        return (len(labels) / (num_classes * counts.to_numpy())).tolist()

    @staticmethod
    def create_text_dataloader(dataset, model, batch_size, shuffle):
        """Tokenizes raw queries per batch without breaking encoder gradients."""

        def collate(batch):
            queries, labels = zip(*batch)
            return model.preprocess(queries), torch.stack(list(labels))

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate,
        )

    @staticmethod
    def move_features_to_device(features, device):
        """Moves token tensors while preserving any non-tensor metadata."""

        return {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in features.items()
        }

    def evaluate_loss(self, model, data_loader, criterion):
        """Calculates validation loss without changing model weights."""

        model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for features, labels in data_loader:
                features = self.move_features_to_device(features, model.device)
                labels = labels.to(model.device)
                loss = criterion(model(features), labels)
                total_loss += loss.item() * len(labels)
                total_samples += len(labels)

        return total_loss / total_samples

    def train_text_classifier(
        self,
        model,
        train_df,
        validation_df,
        num_classes,
        epochs,
        batch_size,
        encoder_learning_rate,
        head_learning_rate,
        weight_decay,
    ):
        """Trains any router model exposing ``encoder``, ``classifier``, and ``device``."""

        train_loader = self.create_text_dataloader(
            TextClassificationDataset(train_df["query"], train_df["path"].values),
            model,
            batch_size,
            shuffle=True,
        )
        validation_loader = self.create_text_dataloader(
            TextClassificationDataset(
                validation_df["query"], validation_df["path"].values
            ),
            model,
            batch_size,
            shuffle=False,
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(
                self.class_weights(train_df["path"].to_numpy(), num_classes),
                dtype=torch.float32,
                device=model.device,
            )
        )
        encoder_parameters = [
            parameter for parameter in model.encoder.parameters() if parameter.requires_grad
        ]
        if not encoder_parameters:
            raise RuntimeError("The model does not expose trainable encoder parameters")

        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_parameters, "lr": encoder_learning_rate},
                {"params": model.classifier.parameters(), "lr": head_learning_rate},
            ],
            weight_decay=weight_decay,
        )

        history = {"train_loss": [], "validation_loss": []}
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            total_samples = 0
            for features, labels in train_loader:
                features = self.move_features_to_device(features, model.device)
                labels = labels.to(model.device)
                optimizer.zero_grad()
                loss = criterion(model(features), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * len(labels)
                total_samples += len(labels)

            train_loss = total_loss / total_samples
            validation_loss = self.evaluate_loss(model, validation_loader, criterion)
            history["train_loss"].append(train_loss)
            history["validation_loss"].append(validation_loss)
            print(
                f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f} "
                f"- validation_loss: {validation_loss:.4f}"
            )

        return history

    def save_text_classifier_artifacts(self, model, artifact_dir, config, history):
        """Persists a fine-tuned model checkpoint, its configuration, and history."""

        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), artifact_dir / "model.pt")
        self.save_json(config, artifact_dir / "config.json")
        self.save_json(history, artifact_dir / "history.json")

    @staticmethod
    def save_json(payload, path):
        """Writes JSON artifacts and creates their parent directory if needed."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    @abstractmethod
    def train(self, *args, **kwargs):
        """Runs the trainer-specific training workflow."""
