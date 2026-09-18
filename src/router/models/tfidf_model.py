"""Modelo sencillo TF-IDF + Logistic Regression."""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TfIdfLogisticRegression:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            sublinear_tf=True,
        )
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
        )

    def predict(self, queries):
        X = self.vectorizer.transform(queries)
        return self.model.predict(X)

    def predict_proba(self, queries):
        X = self.vectorizer.transform(queries)
        return self.model.predict_proba(X)


class TfIdfLogisticRegressionTrain:
    def __init__(self, model):
        self.model = model

    def train(self, queries, labels):
        X = self.model.vectorizer.fit_transform(queries)
        self.model.model.fit(X, labels)
        return self.model

    def save(self, path):
        joblib.dump(self.model, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
