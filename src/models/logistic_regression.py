"""Fast, interpretable baseline for candidate-road classification."""
from sklearn.linear_model import LogisticRegression

from src.models.train_utils import run_training


def main() -> None:
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=200, solver="lbfgs")
    run_training("logistic_regression", model, scale_features=True)


if __name__ == "__main__":
    main()
