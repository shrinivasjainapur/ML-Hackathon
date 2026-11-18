"""Histogram gradient boosting model with bounded iterations and early stopping."""
from sklearn.ensemble import HistGradientBoostingClassifier
from skopt.space import Integer, Real

from src.models.train_utils import run_training


def main() -> None:
    model = HistGradientBoostingClassifier(
        max_iter=50,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
    )
    search_spaces = {
        "model__max_iter": Integer(30, 60),
        "model__learning_rate": Real(0.03, 0.2, prior="log-uniform"),
        "model__max_leaf_nodes": Integer(15, 31),
        "model__l2_regularization": Real(1e-4, 10.0, prior="log-uniform"),
    }
    run_training("hist_gradient_boosting", model, search_spaces)


if __name__ == "__main__":
    main()
