"""Histogram gradient boosting model with bounded iterations and early stopping."""
from sklearn.ensemble import HistGradientBoostingClassifier

from src.models.train_utils import run_training


def main() -> None:
    model = HistGradientBoostingClassifier(
        max_iter=120,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
    )
    run_training("hist_gradient_boosting", model, balanced_sample_weight=True)


if __name__ == "__main__":
    main()
