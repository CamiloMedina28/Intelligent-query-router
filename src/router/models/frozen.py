"""Sentence Transformer congelado y cabeza de clasificación binaria."""

import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


EMBEDDINGS_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIMENSION = 384
HIDDEN_DIMENSION = 128
NUM_CLASSES = 3


class FrozenSentenceTransformer:
    """Encoder only used to generate embeddings."""

    def __init__(self, model_name=EMBEDDINGS_MODEL, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = SentenceTransformer(model_name, device=self.device)
        self.encoder.eval()

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def generate_embeddings(self, queries, batch_size=32):
        """Generates embeddings without calculating gradients."""

        queries = list(queries)

        with torch.no_grad():
            embeddings = self.encoder.encode(
                queries,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
            )

        return embeddings.detach().cpu().float()


def create_dataloader(embeddings, labels, batch_size=32, shuffle=True):
    """Crea batches a partir de embeddings y etiquetas de tres clases."""

    embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)

    if len(embeddings) != len(labels):
        raise ValueError("embeddings y labels deben tener la misma longitud")

    dataset = TensorDataset(embeddings, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class ClassificationFrozenLinearHead(nn.Module):
    def __init__(
        self,
        embedding_dim=DIMENSION,
        hidden_dim=HIDDEN_DIMENSION,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, NUM_CLASSES),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)


class ClassificationFrozenLinearHeadTrain:
    def __init__(
        self,
        model: ClassificationFrozenLinearHead,
        learning_rate=1e-3,
        class_weights=None,
    ):
        self.model = model
        weights = (
            torch.as_tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

    def _evaluate_loader(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for embeddings, labels in data_loader:
                outputs = self.model(embeddings)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * len(labels)
                total_samples += len(labels)

        return total_loss / total_samples

    def train(self, train_loader, validation_loader=None, epochs=10):
        """Entrena la cabeza y devuelve las pérdidas por época."""

        history = {
            "train_loss": [],
            "validation_loss": [],
        }

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_samples = 0

            for embeddings, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(embeddings)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(labels)
                total_samples += len(labels)

            train_loss = total_loss / total_samples
            history["train_loss"].append(train_loss)

            if validation_loader is not None:
                validation_loss = self._evaluate_loader(validation_loader)
                history["validation_loss"].append(validation_loss)

            print(f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f}")

        return history
