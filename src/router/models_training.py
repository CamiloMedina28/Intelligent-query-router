import json
from pathlib import Path

import pandas as pd
import torch

from src.router.models import frozen
from src.router.models import tfidf_model


DATA_DIR = Path("data/processed")
TFIDF_ARTIFACT = Path("artifacts/tfidf/tfidf.joblib")
FROZEN_ARTIFACT_DIR = Path("artifacts/frozen")


def load_dataset(filename):
    """Carga un dataset de tres clases desde data/processed."""

    dataset = pd.read_csv(DATA_DIR / filename)
    return dataset


def train_tfidf():
    """Entrena y guarda el modelo TF-IDF."""
    train_df = load_dataset("training.csv")
    modelo = tfidf_model.TfIdfLogisticRegression()
    entrenador = tfidf_model.TfIdfLogisticRegressionTrain(model=modelo)
    entrenador.train(train_df["query"], train_df["path"])
    entrenador.save(TFIDF_ARTIFACT)


def train_frozen(epochs=12, batch_size=32, learning_rate=1e-3):
    """Entrena la cabeza sobre embeddings de un encoder congelado.

    ``select_datasets.py`` debe ejecutarse previamente para incluir ejemplos
    de las tres clases en training.csv y validation.csv.
    """
    torch.manual_seed(42)

    train_df = load_dataset("training.csv")
    validation_df = load_dataset("validation.csv")

    encoder = frozen.FrozenSentenceTransformer()

    train_embeddings = encoder.generate_embeddings(train_df["query"], batch_size)
    validation_embeddings = encoder.generate_embeddings(validation_df["query"], batch_size)

    train_loader = frozen.create_dataloader(
        train_embeddings,
        train_df["path"].values,
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = frozen.create_dataloader(
        validation_embeddings,
        validation_df["path"].values,
        batch_size=batch_size,
        shuffle=False,
    )

    classifier = frozen.ClassificationFrozenLinearHead()
    trainer = frozen.ClassificationFrozenLinearHeadTrain(
        model=classifier,
        learning_rate=learning_rate,
        class_weights=_class_weights(train_df["path"].to_numpy()),
    )

    history = trainer.train(
        train_loader=train_loader,
        validation_loader=validation_loader,
        epochs=epochs,
    )

    FROZEN_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        classifier.state_dict(),
        FROZEN_ARTIFACT_DIR / "classifier.pt",
    )

    with (FROZEN_ARTIFACT_DIR / "config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "encoder": frozen.EMBEDDINGS_MODEL,
                "embedding_dimension": frozen.DIMENSION,
                "hidden_dimension": frozen.DIMENSION_OCULTA,
                "num_classes": frozen.NUM_CLASSES,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": 42,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    with (FROZEN_ARTIFACT_DIR / "history.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(history, file, ensure_ascii=False, indent=2)

    return history


def _class_weights(labels):
    """Da mayor peso a clases menos frecuentes durante el entrenamiento."""
    counts = pd.Series(labels).value_counts().reindex(range(frozen.NUM_CLASSES), fill_value=0)
    if (counts == 0).any():
        raise ValueError("training.csv debe contener las tres clases: 0, 1 y 2")
    weights = len(labels) / (frozen.NUM_CLASSES * counts.to_numpy())
    return weights.tolist()


if __name__ == "__main__":
    train_frozen()
