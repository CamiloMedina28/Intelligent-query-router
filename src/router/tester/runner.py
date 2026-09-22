"""Convenience functions for testing one or all persisted router models."""

from __future__ import annotations

import json
from pathlib import Path

from src.router.tester.adapters import load_model_adapter
from src.router.tester.model_tester import ModelTester


MODEL_NAMES = ("tfidf", "frozen", "partial_fine_tuning", "full_fine_tuning")


def test_model(model_name, dataset="data/processed/test.csv", output_dir="artifacts/testing"):
    """Tests one persisted model and returns its metrics."""

    model = load_model_adapter(model_name)
    return ModelTester(model, model_name, dataset, output_dir).run()


def test_all_models(dataset="data/processed/test.csv", output_dir="artifacts/testing"):
    """Tests every model independently and records failed evaluations too."""

    output_dir = Path(output_dir)
    summary = {}
    for model_name in MODEL_NAMES:
        try:
            results = test_model(model_name, dataset, output_dir)
            summary[model_name] = {
                "status": "completed",
                "metrics": results["metrics"],
            }
        except Exception as error:
            summary[model_name] = {"status": "failed", "error": str(error)}

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary
