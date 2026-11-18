"""Linear margin classifier for fast candidate ranking."""
from sklearn.svm import LinearSVC

from src.models.train_utils import run_training


def main() -> None:
    model = LinearSVC(C=0.5, class_weight="balanced", dual="auto", max_iter=2000, random_state=42)
    run_training("linear_svm", model, scale_features=True)


if __name__ == "__main__":
    main()
