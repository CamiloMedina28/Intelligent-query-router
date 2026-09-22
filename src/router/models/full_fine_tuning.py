"""Full fine-tuning model for the three-class router."""

import torch
from sentence_transformers import SentenceTransformer
from torch import nn


EMBEDDINGS_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDINGS_DIMENSION = 384
HIDDEN_DIMENSION = 128
NUM_CLASSES = 3


class FullFineTuningModel(nn.Module):
    """Fine-tunes every encoder parameter and a three-class classification head."""

    def __init__(
        self,
        model_name: str = EMBEDDINGS_MODEL,
        embedding_dimension: int = EMBEDDINGS_DIMENSION,
        hidden_dimension: int = HIDDEN_DIMENSION,
        num_classes: int = NUM_CLASSES,
        device: str | None = None,
    ):
        super().__init__()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.encoder = SentenceTransformer(model_name, device=str(self.device))

        # Full fine-tuning updates the embedding layer and every Transformer block.
        for parameter in self.encoder.parameters():
            parameter.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dimension, num_classes),
        )
        self.to(self.device)

    def preprocess(self, queries):
        """Tokenizes a batch of text queries for the encoder forward pass."""

        return self.encoder.preprocess(list(queries))

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns three-class logits for every tokenized query."""

        sentence_embedding = self.encoder(features)["sentence_embedding"]
        return self.classifier(sentence_embedding)

    @property
    def trainable_parameter_count(self) -> int:
        """Number of parameters that the optimizer can update."""

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
