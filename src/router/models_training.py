"""Convenience entry points for the router model trainers.

Concrete training workflows live in ``src.router.trainer``. These functions
preserve a compact public API for scripts and command-line use.
"""

from src.router.trainer import (
    FrozenTrainer,
    FullFineTuningTrainer,
    PartialFineTuningTrainer,
    TfidfTrainer,
)


def load_dataset(filename):
    """Loads a processed dataset using the shared trainer validation rules."""

    return TfidfTrainer().load_dataset(filename)


def train_tfidf():
    """Trains and saves the multiclass TF-IDF baseline."""

    return TfidfTrainer().train()


def train_frozen(epochs=12, batch_size=32, learning_rate=1e-3):
    """Trains the frozen-encoder linear-probing baseline."""

    return FrozenTrainer().train(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )


def train_partial_fine_tuning(
    epochs=4,
    batch_size=16,
    last_unfrozen_layers=2,
    encoder_learning_rate=2e-5,
    head_learning_rate=1e-3,
    weight_decay=0.01,
):
    """Trains the partial fine-tuning model with the selected Transformer depth."""

    return PartialFineTuningTrainer().train(
        epochs=epochs,
        batch_size=batch_size,
        last_unfrozen_layers=last_unfrozen_layers,
        encoder_learning_rate=encoder_learning_rate,
        head_learning_rate=head_learning_rate,
        weight_decay=weight_decay,
    )


def train_full_fine_tuning(
    epochs=2,
    batch_size=8,
    encoder_learning_rate=1e-5,
    head_learning_rate=3e-4,
    weight_decay=0.01,
):
    """Trains the full fine-tuning model across the entire encoder."""

    return FullFineTuningTrainer().train(
        epochs=epochs,
        batch_size=batch_size,
        encoder_learning_rate=encoder_learning_rate,
        head_learning_rate=head_learning_rate,
        weight_decay=weight_decay,
    )


if __name__ == "__main__":
    train_full_fine_tuning()
