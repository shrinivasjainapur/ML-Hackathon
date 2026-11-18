"""Linear margin classifier for fast candidate ranking."""
from sklearn.svm import LinearSVC
from skopt.space import Categorical, Real

from src.models.train_utils import run_training


def main() -> None:
    model = LinearSVC(dual="auto", max_iter=3000, random_state=42)
    search_spaces = {
        "model__C": Real(1e-3, 10.0, prior="log-uniform"),
        "model__class_weight": Categorical([None, "balanced"]),
        "model__tol": Real(1e-5, 1e-3, prior="log-uniform"),
    }
    run_training("linear_svm", model, search_spaces, scale_features=True)


if __name__ == "__main__":
    main()
