"""Bounded random-forest model for nonlinear candidate scoring."""
from sklearn.ensemble import RandomForestClassifier

from src.models.train_utils import run_training


def main() -> None:
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=18,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    run_training("random_forest", model)


if __name__ == "__main__":
    main()
