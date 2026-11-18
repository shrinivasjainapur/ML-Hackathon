"""Bounded random-forest model for nonlinear candidate scoring."""
from sklearn.ensemble import RandomForestClassifier
from skopt.space import Categorical, Integer, Real

from src.models.train_utils import run_training


def main() -> None:
    model = RandomForestClassifier(
        n_estimators=60,
        n_jobs=-1,
        random_state=42,
    )
    search_spaces = {
        "model__n_estimators": Integer(40, 80),
        "model__max_depth": Integer(8, 24),
        "model__min_samples_leaf": Integer(2, 8),
        "model__max_features": Real(0.4, 0.95),
        "model__class_weight": Categorical(["balanced", "balanced_subsample"]),
    }
    run_training("random_forest", model, search_spaces)


if __name__ == "__main__":
    main()
