"""Command-line entry points for the shared router model tester."""

from src.router.tester.runner import test_all_models, test_model


if __name__ == "__main__":
    print(test_all_models())
