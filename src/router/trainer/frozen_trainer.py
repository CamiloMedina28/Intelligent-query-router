"""Trainer for the frozen-encoder linear-probing baseline."""

from pathlib import Path

import torch

from src.router.models import frozen
from src.router.trainer.base import BaseTrainer


class FrozenTrainer(BaseTrainer):
    """Trains only the classification head on precomputed frozen embeddings."""

    def __init__(self, artifact_dir="artifacts/frozen", **kwargs):
        super().__init__(**kwargs)
        self.artifact_dir = Path(artifact_dir)

    def train(self, epochs=12, batch_size=32, learning_rate=1e-3, seed=42):
        self.set_seed(seed)
        train_df, validation_df = self.load_train_validation_datasets()

        encoder = frozen.FrozenSentenceTransformer()
        train_embeddings = encoder.generate_embeddings(train_df["query"], batch_size)
        validation_embeddings = encoder.generate_embeddings(
            validation_df["query"], batch_size
        )
        train_loader = frozen.create_dataloader(
            train_embeddings, train_df["path"].values, batch_size, shuffle=True
        )
        validation_loader = frozen.create_dataloader(
            validation_embeddings,
            validation_df["path"].values,
            batch_size,
            shuffle=False,
        )

        classifier = frozen.ClassificationFrozenLinearHead()
        trainer = frozen.ClassificationFrozenLinearHeadTrain(
            model=classifier,
            learning_rate=learning_rate,
            class_weights=self.class_weights(
                train_df["path"].to_numpy(), frozen.NUM_CLASSES
            ),
        )
        history = trainer.train(train_loader, validation_loader, epochs)

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(classifier.state_dict(), self.artifact_dir / "classifier.pt")
        self.save_json(
            {
                "training_mode": "linear_probing",
                "encoder": frozen.EMBEDDINGS_MODEL,
                "embedding_dimension": frozen.DIMENSION,
                "hidden_dimension": frozen.HIDDEN_DIMENSION,
                "num_classes": frozen.NUM_CLASSES,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
            },
            self.artifact_dir / "config.json",
        )
        self.save_json(history, self.artifact_dir / "history.json")
        return history
