"""Scaled K-nearest-neighbors candidate scorer with a smaller fit cap."""
from sklearn.neighbors import KNeighborsClassifier

from src.models.train_utils import run_training


def main() -> None:
    model = KNeighborsClassifier(
        n_neighbors=15,
        weights="distance",
        algorithm="kd_tree",
        leaf_size=40,
        n_jobs=-1,
    )
    # KNN prediction cost grows with the stored examples; cap the fit at 12k GPS observations.
    run_training("knn", model, scale_features=True, max_train_observations=12_000)


if __name__ == "__main__":
    main()
