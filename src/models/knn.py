"""Scaled K-nearest-neighbors candidate scorer with a smaller fit cap."""
from sklearn.neighbors import KNeighborsClassifier
from skopt.space import Categorical, Integer

from src.models.train_utils import run_training


def main() -> None:
    model = KNeighborsClassifier(
        algorithm="kd_tree",
        n_jobs=-1,
    )
    search_spaces = {
        "model__n_neighbors": Integer(5, 31),
        "model__weights": Categorical(["uniform", "distance"]),
        "model__p": Integer(1, 2),
        "model__leaf_size": Integer(20, 60),
    }
    # KNN prediction cost grows with stored examples; use a smaller trip-safe sample.
    run_training("knn", model, search_spaces, scale_features=True, max_train_observations=4_000)


if __name__ == "__main__":
    main()
