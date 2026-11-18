"""Shared loading, trip-safe sampling, validation, and saving for model scripts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.features.candidate_features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "outputs/features/train_candidates.csv.gz"
DEFAULT_SPLITS = ROOT / "outputs/eda/trip_split_assignments.csv"
DEFAULT_OUTPUT = ROOT / "outputs/models"
KEY_COLUMNS = ["trip_id", "obs_idx"]
LABEL = "is_reference_candidate"


def _cap_observations(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Sample whole GPS observations so candidate rows never get split apart."""
    groups = frame[KEY_COLUMNS].drop_duplicates()
    if len(groups) <= limit:
        return frame
    selected = groups.sample(n=limit, random_state=seed)
    return frame.merge(selected, on=KEY_COLUMNS, how="inner", sort=False)


def run_training(
    model_name: str,
    estimator,
    *,
    scale_features: bool = False,
    max_train_observations: int = 30_000,
    balanced_sample_weight: bool = False,
) -> None:
    parser = argparse.ArgumentParser(description=f"Train and validate {model_name}.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Training candidate CSV or CSV.GZ")
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS, help="Trip-level split assignment CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-train-observations", type=int, default=max_train_observations)
    parser.add_argument("--max-validation-observations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_train_observations < 1 or args.max_validation_observations < 1:
        parser.error("observation limits must be positive")

    needed = FEATURE_COLUMNS + KEY_COLUMNS + [LABEL, "reference_alignment_confident", "has_reference_candidate"]
    candidates = pd.read_csv(args.data, usecols=needed, dtype={"trip_id": "string"})
    split_table = pd.read_csv(args.splits, dtype={"trip_id": "string"})
    assignments = split_table.loc[split_table["partition"].eq("train")].set_index("trip_id")["holdout_role"]
    candidates[LABEL] = pd.to_numeric(candidates[LABEL], errors="coerce")
    eligible = (
        candidates["reference_alignment_confident"].eq(1)
        & candidates["has_reference_candidate"].eq(1)
        & candidates[LABEL].notna()
    )
    candidates = candidates.loc[eligible].copy()
    candidates["_role"] = candidates["trip_id"].map(assignments)
    fit = candidates.loc[candidates["_role"].eq("fit")].drop(columns="_role")
    validation = candidates.loc[candidates["_role"].eq("validation")].drop(columns="_role")
    if fit.empty or validation.empty:
        raise ValueError("No eligible fit/validation rows. Generate candidates and trip splits first.")

    fit = _cap_observations(fit, args.max_train_observations, args.seed)
    validation = _cap_observations(validation, args.max_validation_observations, args.seed)
    y_fit = fit[LABEL].astype("int8").to_numpy()
    y_valid = validation[LABEL].astype("int8").to_numpy()
    if np.unique(y_fit).size != 2:
        raise ValueError("Training sample must contain both positive and negative candidate labels.")

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    pipeline = Pipeline(steps)
    X_fit = fit[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    X_valid = validation[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    if balanced_sample_weight:
        weights = compute_sample_weight(class_weight="balanced", y=y_fit)
        pipeline.fit(X_fit, y_fit, model__sample_weight=weights)
    else:
        pipeline.fit(X_fit, y_fit)

    if hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(X_valid)[:, 1]
    else:
        scores = pipeline.decision_function(X_valid)
    metrics = {
        "model": model_name,
        "fit_trip_count": int(fit["trip_id"].nunique()),
        "fit_observation_count": int(fit[KEY_COLUMNS].drop_duplicates().shape[0]),
        "fit_candidate_row_count": int(len(fit)),
        "validation_trip_count": int(validation["trip_id"].nunique()),
        "validation_observation_count": int(validation[KEY_COLUMNS].drop_duplicates().shape[0]),
        "validation_candidate_row_count": int(len(validation)),
        "validation_top1_observation_accuracy": None,
        "validation_roc_auc": None,
        "validation_average_precision": None,
        "note": "Only confident aligned observations with an available positive candidate are included.",
    }
    if np.unique(y_valid).size == 2:
        metrics["validation_roc_auc"] = float(roc_auc_score(y_valid, scores))
        metrics["validation_average_precision"] = float(average_precision_score(y_valid, scores))
    ranked = validation[KEY_COLUMNS].copy()
    ranked["target"] = y_valid
    ranked["score"] = scores
    best = ranked.sort_values("score", kind="stable").groupby(KEY_COLUMNS, sort=False).tail(1)
    metrics["validation_top1_observation_accuracy"] = float(
        accuracy_score(np.ones(len(best), dtype=np.int8), best["target"])
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"{model_name}.joblib"
    metrics_path = args.output_dir / f"{model_name}_metrics.json"
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved model:   {model_path}")
    print(f"Saved metrics: {metrics_path}")
