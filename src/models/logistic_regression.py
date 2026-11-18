"""Fast, interpretable baseline for candidate-road classification."""
from sklearn.linear_model import LogisticRegression
from skopt.space import Categorical, Real

from src.models.train_utils import run_training


def main() -> None:
    model = LogisticRegression(max_iter=250, solver="lbfgs")
    search_spaces = {
        "model__C": Real(1e-3, 100.0, prior="log-uniform"),
        "model__class_weight": Categorical([None, "balanced"]),
    }
    run_training("logistic_regression", model, search_spaces, scale_features=True)


if __name__ == "__main__":
    main()
