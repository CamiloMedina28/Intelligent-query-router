# Partial Fine-Tuning

## Overview

Partial fine-tuning routes each query to one of three destinations while
updating only a small part of a pre-trained sentence-embedding model. It is a
middle ground between a frozen encoder (feature extraction) and full
fine-tuning:

| Label | Route |
| --- | --- |
| `0` | LLM |
| `1` | API |
| `2` | UNDETERMINED |

The implementation starts with
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, which produces
384-dimensional sentence embeddings. All encoder parameters are frozen first.
It then unfreezes only the final Transformer blocks and trains those blocks
together with a classification head.

By default, the last **two** Transformer blocks are fine-tuned. This is not a
model that fine-tunes only its final classification layer: the classification
head is trained, and so are the encoder parameters in the selected final
Transformer blocks.

## Architecture

![Partial fine-tuning architecture](./media/partial%20fine%20tuning.jpeg)

The model is implemented in `src/router/models/partial_fine_tuning.py`.

```text
Query text
   |
   v
MiniLM tokenizer
   |
   v
Pre-trained multilingual MiniLM encoder
   |-- earlier Transformer blocks: frozen
   |-- final 2 Transformer blocks: trainable by default
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

The classification head contains two linear layers with a hidden layer, ReLU,
and dropout. Therefore, it is more precise to call it a **trainable
classification head** than a single linear head.

During construction, `PartialFineTuningModel` sets every encoder parameter's
`requires_grad` flag to `False`. It then sets the flag to `True` only for the
parameters in the final `last_unfrozen_layers` blocks. The selected encoder
must expose BERT-style `encoder.layer` blocks; otherwise the model raises an
error rather than silently fine-tuning an unintended architecture.

## What Is Trainable

| Component | Default state | Updated during training? |
| --- | --- | --- |
| Tokenizer and input preparation | Fixed | No |
| Earlier MiniLM Transformer blocks | Frozen | No |
| Final two MiniLM Transformer blocks | Unfrozen | Yes |
| Classification head | Trainable | Yes |

Set `last_unfrozen_layers` to select a different number of final Transformer
blocks. It must be at least `1` and no greater than the number of blocks in
the loaded encoder. A smaller value reduces compute and limits adaptation; a
larger value allows more adaptation but increases memory use and overfitting
risk.

## Data

Processed datasets live in `data/processed/`:

| File | Purpose |
| --- | --- |
| `training.csv` | Fits the trainable encoder blocks and classification head. |
| `validation.csv` | Measures validation loss after every epoch. |
| `test.csv` | Provides final, held-out evaluation. |

Each CSV must contain `query` and `path` columns. `path` must use the integer
labels `0`, `1`, and `2`. The training split must include every class because
the loss uses inverse-frequency class weights; training stops with an error if
any class is absent. The public repository does not include the datasets.

## Training

Run the partial fine-tuning entry point from the repository root:

```powershell
python -c "from src.router.models_training import train_partial_fine_tuning; train_partial_fine_tuning()"
```

`python -m src.router.models_training` is not the partial fine-tuning command:
the module's current `__main__` entry point trains the full fine-tuning model.

The default partial fine-tuning configuration is:

| Parameter | Default | Purpose |
| --- | ---: | --- |
| Epochs | 4 | Number of passes through `training.csv`. |
| Batch size | 16 | Queries per optimization step. |
| Unfrozen final blocks | 2 | Encoder Transformer blocks allowed to adapt. |
| Encoder learning rate | `2e-5` | Lower rate for the pre-trained blocks. |
| Head learning rate | `1e-3` | Higher rate for the new classification head. |
| Weight decay | `0.01` | AdamW regularization. |
| Seed | 42 | PyTorch random seed. |

For example, to fine-tune only the final Transformer block for six epochs:

```powershell
python -c "from src.router.models_training import train_partial_fine_tuning; train_partial_fine_tuning(epochs=6, last_unfrozen_layers=1)"
```

The trainer uses `AdamW` with two parameter groups: the selected encoder blocks
use `encoder_learning_rate`, while the classification head uses
`head_learning_rate`. `CrossEntropyLoss` applies weights inversely proportional
to class frequency. Gradients are clipped to a maximum norm of `1.0` before
each optimizer step. Validation runs with gradients disabled and records loss;
it does not update any parameters.

## Training Artifacts

Each run writes to `artifacts/partial_fine_tuning/`:

| Artifact | Contents |
| --- | --- |
| `model.pt` | The model state dictionary, including encoder and classification-head weights. |
| `config.json` | Encoder identifier, trainable depth, hyperparameters, seed, and trainable-parameter count. |
| `history.json` | Training and validation loss for each epoch. |

The saved checkpoint is loaded for inference by the shared model tester. Keep
`config.json` with the checkpoint so the experiment settings remain traceable.

## Inference

At inference time, the model tokenizes the queries, generates sentence
embeddings through the encoder, and produces three logits through the
classification head. Applying `softmax` converts logits to probabilities. The
highest-probability index is the predicted route:

```python
probabilities = torch.softmax(logits, dim=1)
route = probabilities.argmax(dim=1)
```

Probability columns use this fixed order: `LLM` (`0`), `API` (`1`), and
`UNDETERMINED` (`2`).

## Test Evaluation

After selecting a configuration using validation results, evaluate the held-out
test set with:

```powershell
python -c "from src.router.models_testing import test_model; print(test_model('partial_fine_tuning'))"
```

This writes `results.json`, `predictions.csv`, and `confusion_matrix.png` to
`artifacts/testing/partial_fine_tuning/`. Evaluate the test set only after the
model design and hyperparameters have been chosen from the training and
validation splits.

## When to Use It

Use partial fine-tuning when a frozen encoder plus head does not capture
domain-specific routing distinctions, but full fine-tuning is unnecessarily
expensive or the labeled dataset is too small to safely update every encoder
layer. Compare its validation metrics with the frozen baseline, then reserve
the test split for the final selected model.
