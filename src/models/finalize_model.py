"""Refit the selected HGB configuration on all permitted training trips.

Run from the repository root:
    .venv/bin/python -m src.models.finalize_model

Only outputs/features/train_candidates.csv.gz is used for fitting. The separate
student_test partition is used afterward for development evaluation; evaluator
data is never read by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import skopt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.evaluation.evaluate_student_test import _evaluate_one
from src.features.candidate_features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CANDIDATES = ROOT / "outputs/features/train_candidates.csv.gz"
DEFAULT_STUDENT_CANDIDATES = ROOT / "outputs/features/student_test_candidates.csv.gz"
DEFAULT_MODEL_DIR = ROOT / "outputs/models"
LABEL = "is_reference_candidate"
KEY_COLUMNS = ["trip_id", "obs_idx"]
MODEL_NAMES = (
    "logistic_regression",
    "linear_svm",
    "random_forest",
    "knn",
    "hist_gradient_boosting",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-candidates", type=Path, default=DEFAULT_TRAIN_CANDIDATES)
    parser.add_argument("--student-candidates", type=Path, default=DEFAULT_STUDENT_CANDIDATES)
    parser.add_argument("--data-root", type=Path, default=ROOT / "Student folder")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    selection_metrics_path = args.model_dir / "hist_gradient_boosting_metrics.json"
    if not selection_metrics_path.is_file():
        raise FileNotFoundError(f"Run Bayesian tuning first; missing {selection_metrics_path}")
    selection_metrics = json.loads(selection_metrics_path.read_text(encoding="utf-8"))
    validation_comparison = []
    for model_name in MODEL_NAMES:
        metrics_path = args.model_dir / f"{model_name}_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        validation_comparison.append({
            "model": model_name,
            "best_cv_average_precision": metrics.get("best_cv_average_precision"),
            "validation_top1_observation_accuracy": metrics.get("validation_top1_observation_accuracy"),
            "validation_roc_auc": metrics.get("validation_roc_auc"),
            "validation_average_precision": metrics.get("validation_average_precision"),
        })
    student_comparison_path = ROOT / "outputs/evaluation/student_test_model_comparison.json"
    student_comparison = json.loads(student_comparison_path.read_text(encoding="utf-8"))
    selection_comparison = {
        "selected_model": "hist_gradient_boosting",
        "validation_metrics": validation_comparison,
        "student_test_metrics": student_comparison,
        "selection_rationale": (
            "Histogram Gradient Boosting led student_test on Top-1 accuracy, route-arc Macro F1, "
            "candidate-row Macro F1, candidate-row Log Loss, and direct continuity. It also led "
            "the available validation ROC AUC and average precision; Logistic Regression and "
            "Linear SVM had slightly higher validation Top-1 accuracy."
        ),
    }
    comparison_path = args.model_dir / "final_model_selection_comparison.json"
    comparison_path.write_text(json.dumps(selection_comparison, indent=2) + "\n", encoding="utf-8")
    tuned_parameters = {
        key.removeprefix("model__"): value
        for key, value in selection_metrics["best_hyperparameters"].items()
    }
    classifier_parameters = {
        **tuned_parameters,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 10,
        "random_state": 42,
    }

    columns = FEATURE_COLUMNS + KEY_COLUMNS + [
        LABEL,
        "reference_alignment_confident",
        "has_reference_candidate",
    ]
    candidates = pd.read_csv(
        args.train_candidates,
        usecols=columns,
        dtype={"trip_id": "string"},
    )
    candidates[LABEL] = pd.to_numeric(candidates[LABEL], errors="coerce")
    fit_mask = (
        candidates["reference_alignment_confident"].eq(1)
        & candidates["has_reference_candidate"].eq(1)
        & candidates[LABEL].notna()
    )
    fit_rows = candidates.loc[fit_mask].copy()
    if fit_rows.empty or fit_rows[LABEL].nunique() != 2:
        raise ValueError("Eligible training rows must contain both candidate classes.")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(**classifier_parameters)),
    ])
    X_fit = fit_rows[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    y_fit = fit_rows[LABEL].astype(np.int8).to_numpy()
    started = time.perf_counter()
    pipeline.fit(X_fit, y_fit)
    fit_seconds = time.perf_counter() - started

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "final_model.joblib"
    config_path = args.model_dir / "final_model_config.json"
    metrics_path = args.model_dir / "final_model_metrics.json"
    per_trip_path = args.model_dir / "final_model_student_test_per_trip.csv"
    joblib.dump(pipeline, model_path)

    config = {
        "selected_model": "HistGradientBoostingClassifier",
        "selection_reason": selection_comparison["selection_rationale"],
        "selection_comparison_file": comparison_path.name,
        "bayesian_search": {
            "method": selection_metrics["tuning_method"],
            "objective": selection_metrics["tuning_scoring"],
            "cross_validation": selection_metrics["tuning_cv"],
            "evaluations": selection_metrics["bayesian_iterations"],
            "initial_design": selection_metrics["bayesian_initial_design"],
            "initial_design_points": selection_metrics["bayesian_initial_design_points"],
            "best_cv_average_precision": selection_metrics["best_cv_average_precision"],
            "search_results_file": "hist_gradient_boosting_bayes_search.csv",
        },
        "classifier_parameters": classifier_parameters,
        "preprocessing": {
            "feature_columns": FEATURE_COLUMNS,
            "imputer": "median, fitted on the full eligible training rows",
            "scaling": "none for tree-based model",
            "training_filter": (
                "reference_alignment_confident == 1, has_reference_candidate == 1, "
                "and non-missing is_reference_candidate"
            ),
        },
        "training_data": {
            "partition": "train only",
            "candidate_file": str(args.train_candidates.resolve()),
            "candidate_file_sha256": _sha256(args.train_candidates),
            "trip_count": int(fit_rows["trip_id"].nunique()),
            "observation_count": int(fit_rows[KEY_COLUMNS].drop_duplicates().shape[0]),
            "candidate_row_count": int(len(fit_rows)),
            "seed": 42,
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scikit_optimize": skopt.__version__,
        },
        "development_evaluation": {
            "student_test_results_file": "final_model_metrics.json",
            "evaluator_test_used": False,
        },
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    # Evaluate the final all-training-trip refit on the permitted development
    # partition. Reference labels are used only by the metric code, not features.
    student_columns = FEATURE_COLUMNS + KEY_COLUMNS + [
        "candidate_arc_id",
        "reference_arc_id",
        "reference_alignment_confident",
        "has_reference_candidate",
        LABEL,
        "time_gap_seconds",
    ]
    student_candidates = pd.read_csv(
        args.student_candidates,
        usecols=student_columns,
        dtype={"trip_id": "string"},
    )
    student_metrics, per_trip_metrics = _evaluate_one(
        "final_hist_gradient_boosting",
        model_path,
        student_candidates,
        args.data_root,
    )
    final_metrics = {
        "model": "HistGradientBoostingClassifier",
        "model_artifact": model_path.name,
        "fit_seconds": float(fit_seconds),
        "actual_boosting_iterations": int(pipeline.named_steps["model"].n_iter_),
        "training": config["training_data"],
        "bayesian_selection": config["bayesian_search"],
        "model_selection_comparison_file": comparison_path.name,
        "heldout_fit_validation_before_full_refit": {
            key: selection_metrics[key]
            for key in (
                "validation_observation_count",
                "validation_top1_observation_accuracy",
                "validation_roc_auc",
                "validation_average_precision",
            )
        },
        "student_test_after_full_refit": student_metrics,
        "data_integrity_note": (
            "Only train candidates were used for fitting. student_test was used for evaluation. "
            "evaluator_test.zip was not read."
        ),
    }
    metrics_path.write_text(json.dumps(final_metrics, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(per_trip_metrics).to_csv(per_trip_path, index=False)

    print(f"Saved final model: {model_path}")
    print(f"Saved configuration: {config_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved per-trip metrics: {per_trip_path}")
    print(f"Fit on {len(fit_rows):,} candidate rows from {fit_rows['trip_id'].nunique()} train trips")
    print(
        "Final student_test: "
        f"Top-1={student_metrics['top1_observation_accuracy']:.4f}, "
        f"route Macro-F1={student_metrics['observation_route_arc_macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
