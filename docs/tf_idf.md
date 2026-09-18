# TF-IDF: Multiclass Lexical Baseline

## Objective

This model routes a query into one of three classes:

| Label | Route |
| --- | --- |
| `0` | LLM |
| `1` | API |
| `2` | UNDETERMINED |

It is a lightweight baseline built from TF-IDF features and logistic
regression. Unlike the frozen-encoder approach, it does not use contextual
embeddings: it learns from words and two-word sequences that occur in the
training queries.

## Pipeline

![](./media/Tf-idf%20with%20multiclass%20linear%20regression.jpeg)

The implementation is in `src/router/models/tfidf_model.py`.

## Feature Extraction

`TfidfVectorizer` converts each query into a sparse numeric vector. With the
current defaults, a term `t` in document `d` receives the following TF-IDF
weight before L2 normalization:

$$
\operatorname{tfidf}(t, d) =
\left(1 + \log(\operatorname{tf}(t, d))\right)
\left(\log\left(\frac{1 + N}{1 + \operatorname{df}(t)}\right) + 1\right)
$$

where `tf` is the term frequency in the query, `df` is the number of training
queries containing the term, and `N` is the total number of training queries.
The `+1` terms are scikit-learn's IDF smoothing; `sublinear_tf=True` replaces
raw term frequency with its logarithmic form. Terms that are frequent in one
query but uncommon across the dataset receive a higher weight.

The current vectorizer configuration is:

| Setting | Value | Effect |
| --- | --- | --- |
| `lowercase` | `True` | Makes matching case-insensitive |
| `strip_accents` | `"unicode"` | Normalizes accented characters |
| `ngram_range` | `(1, 2)` | Uses unigrams and bigrams |
| `min_df` | `1` | Retains terms seen in at least one query |
| `max_df` | `0.98` | Drops terms appearing in more than 98% of queries |
| `sublinear_tf` | `True` | Applies logarithmic term-frequency scaling |
| `smooth_idf` | `True` | Applies additive-one IDF smoothing |
| `norm` | `"l2"` | Normalizes every query vector to unit L2 norm |

For example, the bigram `"create ticket"` can be a more useful routing signal
than either word alone.

## Classifier

The feature vector is passed to scikit-learn's `LogisticRegression`. The model
computes one score per class and normalizes the scores with softmax to produce
probabilities. `predict()` returns the class with the highest probability;
`predict_proba()` returns the three probabilities in `classes_` order.

The classifier is configured with:

| Setting | Value |
| --- | --- |
| Maximum iterations | `1000` |
| Random seed | `42` |
| Class weighting | `balanced` |

Balanced class weighting compensates for uneven class frequencies when they
occur. The current dataset is balanced, but this setting makes retraining more
robust if the distribution changes.

## Data Requirements

Training data is read from `data/processed/training.csv`. It must contain:

| Column | Description |
| --- | --- |
| `query` | Text to be routed |
| `path` | Integer route label: `0`, `1`, or `2` |

The training wrapper explicitly requires all three labels. It raises an error
when a class is absent, preventing an accidental binary artifact from being
saved.

## Training

`models_training.py` provides the `train_tfidf()` function. From the repository
root, train and save the model with:

```powershell
python -c "from src.router.models_training import train_tfidf; train_tfidf()"
```

The resulting artifact is saved to:

```text
artifacts/tfidf/tfidf.joblib
```

The default `python -m src.router.models_training` entry point trains the
frozen-encoder model, not TF-IDF. Use the command above when retraining this
baseline.

## Inference

Load the artifact and make predictions as follows:

```python
from src.router.models.tfidf_model import TfIdfLogisticRegressionTrain

model = TfIdfLogisticRegressionTrain.load("artifacts/tfidf/tfidf.joblib")
queries = ["I need to check my order status"]

route = model.predict(queries)
probabilities = model.predict_proba(queries)

print(model.classes_)       # [0, 1, 2]
print(route)
print(probabilities)
```

## Validation Results

The current validation artifact contains 1,378 balanced samples. Its results
are stored in `artifacts/validation/tfidf/results.json`:

| Metric | Result |
| --- | ---: |
| Accuracy | 0.9586 |
| Balanced accuracy | 0.9586 |
| Macro F1 | 0.9585 |
| Log loss | 0.2418 |

The confusion matrix is:

| Actual \ Predicted | LLM | API | UNDETERMINED |
| --- | ---: | ---: | ---: |
| LLM | 449 | 4 | 6 |
| API | 2 | 447 | 11 |
| UNDETERMINED | 20 | 14 | 425 |

Metrics must be regenerated after changing the data, preprocessing, or model
configuration.

## Strengths and Limitations

TF-IDF is fast to train, inexpensive to run, interpretable, and useful as a
baseline. It works especially well when route-specific keywords and phrases
are stable.

Its main limitation is that it does not represent context or semantic
similarity as well as a sentence-embedding model. It may struggle when two
queries have the same meaning but use different vocabulary. The frozen encoder
documented in [linear_probing.md](linear_probing.md) is the semantic alternative
in this project.
