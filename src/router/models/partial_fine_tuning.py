"""Partial fine-tuning model for the three-class router."""

import torch
from torch import nn
from sentence_transformers import SentenceTransformer

EMBEDDINGS_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDINGS_DIMENSION = 384
HIDDEN_DIMENSION = 128
NUM_CLASSES = 3


class PartialFineTuningModel(nn.Module):
    """Fine-tunes only the final Transformer blocks and a classification head."""

    def __init__(
        self,
        model_name: str = EMBEDDINGS_MODEL,
        embedding_dimension: int = EMBEDDINGS_DIMENSION,
        hidden_dimension: int = HIDDEN_DIMENSION,
        num_classes: int = NUM_CLASSES,
        last_unfrozen_layers: int = 2,
        device: str | None = None,
    ):
        super().__init__()

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.encoder = SentenceTransformer(model_name, device=str(self.device))
        self.last_unfrozen_layers = last_unfrozen_layers

        self._freeze_encoder()
        self._unfreeze_last_transformer_layers(last_unfrozen_layers)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dimension, num_classes),
        )
        self.to(self.device)

    def _freeze_encoder(self) -> None:
        """Disables gradients for every parameter in the pre-trained encoder."""

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def _transformer_layers(self):
        """Returns the ordered Transformer blocks of the wrapped MiniLM model."""

        transformer_module = self.encoder._first_module()
        auto_model = transformer_module.auto_model
        layers = getattr(getattr(auto_model, "encoder", None), "layer", None)

        if layers is None:
            raise TypeError(
                "The selected encoder does not expose BERT-style encoder.layer blocks. "
                "Update _transformer_layers for this architecture."
            )

        return layers

    def _unfreeze_last_transformer_layers(self, count: int) -> None:
        """Enables gradients only for the last ``count`` Transformer blocks."""

        layers = self._transformer_layers()
        if not 1 <= count <= len(layers):
            raise ValueError(
                f"last_unfrozen_layers must be between 1 and {len(layers)}; "
                f"received {count}."
            )

        for layer in layers[-count:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    def preprocess(self, queries):
        """Tokenizes a batch of text queries for the encoder forward pass."""

        return self.encoder.preprocess(list(queries))

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns one logit per router class for every tokenized query."""

        sentence_embedding = self.encoder(features)["sentence_embedding"]
        return self.classifier(sentence_embedding)

    @property
    def trainable_parameter_count(self) -> int:
        """Number of parameters that the optimizer is allowed to update."""

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


