# Full Fine-Tuning

## Overview

Full fine-tuning routes each query to one of three destinations by updating
every parameter in a pre-trained sentence-embedding model, as well as a new
classification head:

| Label | Route |
| --- | --- |
| `0` | LLM |
| `1` | API |
| `2` | UNDETERMINED |

The implementation uses
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, a multilingual
encoder that produces 384-dimensional sentence embeddings. Unlike partial
fine-tuning, full fine-tuning does not freeze any encoder components: the
embedding layer, every Transformer block, and the pooling-related encoder
parameters can all be updated from the routing data.

This gives the model the greatest capacity to adapt to domain-specific routing
language, but it also requires more GPU memory and compute and presents a
higher risk of overfitting than a frozen encoder or partial fine-tuning.

## Architecture

![Full fine-tuning architecture](./media/full%20fine%20tuning.jpeg)

The model is implemented in `src/router/models/full_fine_tuning.py`.

```text
Query text
   |
   v
MiniLM tokenizer
   |
   v
Pre-trained multilingual MiniLM encoder
   |-- embedding layer: trainable
   |-- all Transformer blocks: trainable
   |-- sentence-embedding output: trainable
   |
   v
384-dimensional sentence embedding
   |
   v
Linear(384, 128) -> ReLU -> Dropout(0.2) -> Linear(128, 3)
   |
   v
Logits for LLM, API, UNDETERMINED
```

The classification head is a small multi-layer perceptron: two linear layers
separated by a 128-unit hidden layer, ReLU activation, and dropout with a
probability of `0.2`. It is not a single linear layer.

## What Is Trainable

| Component | Updated during training? |
| --- | --- |
| Tokenizer and input preparation | No |
| MiniLM embedding layer | Yes |
| Every MiniLM Transformer block | Yes |
| Other encoder parameters | Yes |
| Classification head | Yes |

`FullFineTuningModel` explicitly sets `requires_grad = True` for every
parameter exposed by the `SentenceTransformer` encoder. As a result, gradients
flow from the routing loss through the classification head and across the
entire encoder during every training step.

## Data

Processed datasets live in `data/processed/`:

| File | Purpose |
| --- | --- |
| `training.csv` | Fits all encoder parameters and the classification head. |
| `validation.csv` | Measures validation loss after every epoch. |
| `test.csv` | Provides final, held-out evaluation. |

Each CSV must include `query` and `path` columns. `path` must use the integer
labels `0`, `1`, and `2`. The training split must contain all three classes:
the training loss uses inverse-frequency class weights and raises an error when
any class is missing. The public repository does not include the datasets.

## Training

Run the full fine-tuning workflow from the repository root:

```powershell
python -m src.router.models_training
```

This works because the module's current `__main__` entry point calls
`train_full_fine_tuning()`. The equivalent explicit command is:

```powershell
python -c "from src.router.models_training import train_full_fine_tuning; train_full_fine_tuning()"
```

The public entry point uses these defaults:

| Parameter | Default | Purpose |
| --- | ---: | --- |
| Epochs | 2 | Number of passes through `training.csv`. |
| Batch size | 8 | Queries per optimization step. |
| Encoder learning rate | `1e-5` | Conservative rate for all pre-trained encoder weights. |
| Head learning rate | `3e-4` | Higher rate for the new classification head. |
| Weight decay | `0.01` | AdamW regularization. |
| Seed | 42 | PyTorch random seed. |

For example, to train for three epochs with a smaller batch size:

```powershell
python -c "from src.router.models_training import train_full_fine_tuning; train_full_fine_tuning(epochs=3, batch_size=4)"
```

The trainer uses `AdamW` with two parameter groups: every encoder parameter
uses `encoder_learning_rate`, and the classification head uses
`head_learning_rate`. `CrossEntropyLoss` applies inverse-frequency class
weights. Gradients are clipped to a maximum norm of `1.0` before every
optimizer step. Validation runs under `torch.no_grad()` and records loss
without changing model weights.

## Training Artifacts

Each run writes to `artifacts/full_fine_tuning/`:

| Artifact | Contents |
| --- | --- |
| `model.pt` | Model state dictionary containing the fine-tuned encoder and classification-head weights. |
| `config.json` | Encoder identifier, hyperparameters, seed, and trainable-parameter count. |
| `history.json` | Training and validation loss for each epoch. |

Keep `config.json` with its checkpoint so the training setup remains
traceable. Full fine-tuning checkpoints are substantially larger than frozen
head-only checkpoints because they include the complete encoder state.

## Inference

For inference, the model tokenizes each query, runs it through the fine-tuned
encoder, and passes the resulting sentence embedding through the classification
head. `softmax` converts the three logits into probabilities; the largest
probability selects the route:

```python
probabilities = torch.softmax(logits, dim=1)
route = probabilities.argmax(dim=1)
```

Probability columns always follow this order: `LLM` (`0`), `API` (`1`), and
`UNDETERMINED` (`2`).

## Test Evaluation

After choosing the configuration using validation results, evaluate the held-
out test split with:

```powershell
python -c "from src.router.models_testing import test_model; print(test_model('full_fine_tuning'))"
```

This produces `results.json`, `predictions.csv`, and `confusion_matrix.png` in
`artifacts/testing/full_fine_tuning/`. Do not use the test split to choose
epochs, batch size, or learning rates; reserve it for the final selected
configuration.

## When to Use It

Use full fine-tuning when you have enough representative labeled data and the
frozen and partially fine-tuned models still miss domain-specific distinctions.
It is the most adaptable approach in this repository, but should be compared
against the lighter alternatives on the validation split to confirm that its
additional complexity improves generalization.
