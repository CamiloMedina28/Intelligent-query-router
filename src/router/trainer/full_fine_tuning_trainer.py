"""Trainer for the full fine-tuning router model."""

from pathlib import Path

from src.router.models import full_fine_tuning
from src.router.trainer.base import BaseTrainer


class FullFineTuningTrainer(BaseTrainer):
    """Trains every SentenceTransformer parameter and the classification head."""

    def __init__(self, artifact_dir="artifacts/full_fine_tuning", **kwargs):
        super().__init__(**kwargs)
        self.artifact_dir = Path(artifact_dir)

    def train(
        self,
        epochs=5,
        batch_size=8,
        encoder_learning_rate=1e-5,
        head_learning_rate=3e-4,
        weight_decay=0.01,
        seed=42,
    ):
        self.set_seed(seed)
        train_df, validation_df = self.load_train_validation_datasets()
        model = full_fine_tuning.FullFineTuningModel()

        history = self.train_text_classifier(
            model=model,
            train_df=train_df,
            validation_df=validation_df,
            num_classes=full_fine_tuning.NUM_CLASSES,
            epochs=epochs,
            batch_size=batch_size,
            encoder_learning_rate=encoder_learning_rate,
            head_learning_rate=head_learning_rate,
            weight_decay=weight_decay,
        )
        self.save_text_classifier_artifacts(
            model,
            self.artifact_dir,
            config={
                "training_mode": "full_fine_tuning",
                "encoder": full_fine_tuning.EMBEDDINGS_MODEL,
                "embedding_dimension": full_fine_tuning.EMBEDDINGS_DIMENSION,
                "num_classes": full_fine_tuning.NUM_CLASSES,
                "epochs": epochs,
                "batch_size": batch_size,
                "encoder_learning_rate": encoder_learning_rate,
                "head_learning_rate": head_learning_rate,
                "weight_decay": weight_decay,
                "trainable_parameter_count": model.trainable_parameter_count,
                "seed": seed,
            },
            history=history,
        )
        return history
