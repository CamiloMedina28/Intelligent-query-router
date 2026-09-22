# Intelligent Router Documentation

Welcome to the technical documentation for **Intelligent Router**, a
multilingual text-classification project that directs queries to an LLM, an
API workflow, or an undetermined fallback.

> Looking for the project overview, setup instructions, and experiment
> workflow? Start at the [repository README](../README.md).

## Choose a Model Guide

| Approach | Trainable components | Why use it? | Guide |
| --- | --- | --- | --- |
| TF-IDF + Logistic Regression | Lexical classifier | Fast and interpretable baseline | [Open guide →](tf_idf.md) |
| Linear Probing | Classification head | Semantic classification with a frozen encoder | [Open guide →](linear_probing.md) |
| Partial Fine-Tuning | Head + final Transformer blocks | Controlled encoder adaptation | [Open guide →](partial_fine_tuning.md) |
| Full Fine-Tuning | Head + entire encoder | Maximum domain adaptation | [Open guide →](full_fine_tuning.md) |

## Model Selection at a Glance

```text
Fastest and lightest                                      Most adaptable
TF-IDF  ──────>  Frozen encoder  ──────>  Partial FT  ──────>  Full FT
```

Start with the simpler approaches and use validation metrics to determine
whether additional trainable encoder layers improve generalization enough to
justify their cost.

## Shared Contract

Every model predicts one of these routes:

| Label | Route | Meaning |
| --- | --- | --- |
| `0` | LLM | Handle the query in the language-model flow. |
| `1` | API | Handle the query in an API-oriented flow. |
| `2` | UNDETERMINED | Use a fallback or request further handling. |

All processed datasets use `query` for the input text and `path` for the
integer route label. Keep the three standard splits separate:

| Split | Purpose |
| --- | --- |
| `training.csv` | Fits model parameters. |
| `validation.csv` | Compares models and chooses settings. |
| `test.csv` | Final evaluation of the chosen configuration. |

## Evaluation Outputs

The shared test runner writes three artifacts per model to
`artifacts/testing/<model-name>/`:

- `results.json` — aggregate and per-class metrics
- `predictions.csv` — predictions and route probabilities
- `confusion_matrix.png` — class-level error visualization

Use the test set only after model selection is complete; this keeps the final
evaluation meaningful.
