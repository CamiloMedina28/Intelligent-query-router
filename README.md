<div align="center">

# Intelligent Router

### Route multilingual customer queries to the right workflow—fast, consistently, and with measurable confidence.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-powered-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Sentence%20Transformers](https://img.shields.io/badge/Sentence%20Transformers-multilingual-1F6FEB)](https://www.sbert.net/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-baseline-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

</div>

> A practical text-classification laboratory for deciding whether an incoming
> query belongs on an **LLM**, **API**, or **undetermined** route.

Intelligent Router compares a lightweight lexical baseline with progressively
more adaptable multilingual embedding models. It is designed to make model
selection an evidence-based process: train on one split, compare on validation
data, and evaluate the selected configuration once on a held-out test set.

## The Routing Decision

```text
Incoming query
      |
      v
┌─────────────────────┐
│  Intelligent Router │
└──────────┬──────────┘
           |
     ┌─────┼───────────────┐
     v     v               v
   LLM    API       UNDETERMINED
    0      1               2
```

| Route | Label | Intended outcome |
| --- | ---: | --- |
| **LLM** | `0` | Send the query to the language-model workflow. |
| **API** | `1` | Send the query to an API-oriented workflow. |
| **UNDETERMINED** | `2` | Use a fallback or request further handling. |

## Four Paths to Better Routing

Choose the smallest model that solves the problem well on validation data.

| Approach | What learns | Best fit | Guide |
| --- | --- | --- | --- |
| **TF-IDF + Logistic Regression** | Word and bigram weights | A fast, interpretable lexical baseline | [Explore →](docs/tf_idf.md) |
| **Linear Probing** | Classification head only | Fast semantic routing with a frozen encoder | [Explore →](docs/linear_probing.md) |
| **Partial Fine-Tuning** | Classification head + final Transformer blocks | Domain adaptation with controlled cost | [Explore →](docs/partial_fine_tuning.md) |
| **Full Fine-Tuning** | Classification head + entire encoder | Maximum adaptation for a representative labeled dataset | [Explore →](docs/full_fine_tuning.md) |

<details>
<summary><strong>How the approaches scale</strong></summary>

```text
Lowest compute                                                Highest adaptation
TF-IDF  ──────>  Frozen encoder  ──────>  Partial FT  ──────>  Full FT
```

Moving right increases the number of trainable parameters, training cost, and
the need for careful validation. It can also capture more domain-specific
language.

</details>

## Quick Start

Run these commands from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The training splits belong in `data/processed/`. Each CSV requires a text
column named `query` and an integer route label named `path`.

| File | Role |
| --- | --- |
| `training.csv` | Fits model parameters. |
| `validation.csv` | Guides model and hyperparameter choices. |
| `test.csv` | Final, held-out evaluation only. |

Train one model at a time:

```powershell
# Fast lexical baseline
python -c "from src.router.models_training import train_tfidf; train_tfidf()"

# Frozen multilingual encoder + classification head
python -c "from src.router.models_training import train_frozen; train_frozen()"

# Final Transformer blocks + classification head
python -c "from src.router.models_training import train_partial_fine_tuning; train_partial_fine_tuning()"

# Entire multilingual encoder + classification head
python -c "from src.router.models_training import train_full_fine_tuning; train_full_fine_tuning()"
```

## Evaluate with Discipline

Use validation results to choose the approach and its settings. Once selected,
evaluate the final model on the untouched test set:

```powershell
python -c "from src.router.models_testing import test_model; print(test_model('full_fine_tuning'))"
```

Replace `full_fine_tuning` with `tfidf`, `frozen`, or
`partial_fine_tuning` as needed. Each evaluation saves a metrics report,
per-query predictions, and a confusion matrix under
`artifacts/testing/<model-name>/`.

## Project Map

```text
data/
├── raw/                         # Source queries
├── processors/                  # Dataset preparation utilities
└── processed/                   # Training, validation, and test splits

src/router/
├── models/                      # Model architectures
├── trainer/                     # Shared and model-specific training logic
├── tester/                      # Evaluation adapters and metrics
├── models_training.py           # Public training entry points
└── models_testing.py            # Public testing entry points

artifacts/                       # Checkpoints, configurations, histories, metrics
docs/                            # Model guides and architecture diagrams
```

## What Every Run Produces

Training records the model checkpoint, configuration, and loss history in
`artifacts/<model-name>/`. Test evaluation adds:

- `results.json` — aggregate metrics and per-class report
- `predictions.csv` — predicted route and class probabilities for each query
- `confusion_matrix.png` — a visual view of model errors

This makes it easy to compare experiments without treating a single accuracy
number as the whole story.

## Documentation

For architecture details, default hyperparameters, inference behavior, and
trade-offs, start with the [documentation hub](docs/README.md) or open a model
guide directly:

- [TF-IDF baseline](docs/tf_idf.md)
- [Frozen encoder / linear probing](docs/linear_probing.md)
- [Partial fine-tuning](docs/partial_fine_tuning.md)
- [Full fine-tuning](docs/full_fine_tuning.md)

---

<div align="center">
  <sub>Built to make routing-model experiments understandable, comparable, and ready for production decisions.</sub>
</div>
