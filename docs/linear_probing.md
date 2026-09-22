# Linear Probing: Feature Extraction with a Frozen Encoder

## Objective

This approach classifies each router query into one of three routes:

| Label | Route |
| --- | --- |
| `0` | LLM |
| `1` | API |
| `2` | UNDETERMINED |

Rather than updating the entire language model, a pre-trained encoder converts
queries into embeddings, while only the final classification layers are
trained. This lowers training cost and preserves the base encoder's linguistic
knowledge.

Strictly speaking, *linear probing* uses a single linear layer as its head.
This project's implementation includes a hidden layer, so the more precise
name is **frozen encoder with a trainable classification head**. The term
*linear probing* is used here to refer to this feature-extraction strategy
with a frozen encoder.

## Architecture

![Models architecture](./media/Feature%20Extraction%20with%20a%20Frozen%20Encoder%20architecture.jpeg)

The encoder is defined in `src/router/models/frozen.py`. Its parameters use
`requires_grad = False`, and embeddings are generated inside `torch.no_grad()`.
Consequently, no gradients are computed or updated for the
`SentenceTransformer` during training; only the
`ClassificationFrozenLinearHead` weights change.

## Data

Processed datasets are stored in `data/processed/`:

| File | Purpose |
| --- | --- |
| `training.csv` | Fits the classification head |
| `validation.csv` | Tracks loss and evaluates validation performance |
| `test.csv` | Final evaluation across all three classes |

Each CSV must include `query` and `path` columns. Training requires all three
labels, `0`, `1`, and `2`; class-weight calculation fails if any class is
missing. However, data is not included in the public repository.

## Training

From the repository root, run:

```powershell
python -m src.router.models_training
```

The current configuration is reproducible with seed `42` and uses:

| Parameter | Value |
| --- | ---: |
| Epochs | 12 |
| Batch size | 32 |
| Learning rate | 0.001 |
| Embedding dimension | 384 |
| Hidden dimension | 128 |
| Number of classes | 3 |

`CrossEntropyLoss` uses weights inversely proportional to each class frequency.
Training produces:

| Artifact | Contents |
| --- | --- |
| `artifacts/frozen/classifier.pt` | Trained classification-head weights |
| `artifacts/frozen/config.json` | Hyperparameters and encoder identifier |
| `artifacts/frozen/history.json` | Training and validation loss per epoch |

The encoder is not stored in the checkpoint because it is reloaded by its
Hugging Face identifier during inference.

## Inference

For one or more queries, the model generates embeddings, obtains the head's
*logits*, and applies `softmax`. The predicted class is the index with the
highest probability:

```python
probabilities = torch.softmax(logits, dim=1)
route = probabilities.argmax(axis=1)
```

Probability columns always follow the `LLM`, `API`, `UNDETERMINED` order
(indices `0`, `1`, and `2`, respectively).

## Validation and Testing

To regenerate validation metrics and plots:

```powershell
python -m src.router.models_validation
```

Results are stored in `artifacts/validation/frozen/`, including `results.json`,
`predictions.csv`, and the confusion matrix.

To evaluate the test set and save predictions for the ambiguous holdout:

```powershell
python -m src.router.frozen_testing
```

This produces `artifacts/testing/frozen/confusion_matrix.png` and
`artifacts/testing/frozen/ambiguous_predictions.csv`.

## Current Results

The current validation set has 1,378 samples, with a balanced distribution of
about 459 examples per class. The model achieved:

| Metric | Result |
| --- | ---: |
| Accuracy | 0.9528 |
| Balanced accuracy | 0.9528 |
| Macro F1 | 0.9529 |
| Log loss | 0.1352 |

These values come from `artifacts/validation/frozen/results.json` and must be
recomputed after changing the data, encoder, or hyperparameters.

## When to Use This Approach

This is a strong first option when labeled data is limited, training must be
fast, or a robust multilingual baseline is needed. If the model consistently
confuses domain-specific cases despite good data, the next step is
*fine-tuning*: unfreeze part or all of the encoder and train it with a lower
learning rate. That brings a higher cost, greater overfitting risk, and a need
for more careful evaluation.
