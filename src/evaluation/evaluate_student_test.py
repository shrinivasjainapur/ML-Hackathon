"""Evaluate saved candidate classifiers on the permitted student_test partition.

Run from the repository root:
    .venv/bin/python -m src.evaluation.evaluate_student_test

This script never reads evaluator_test data. It uses student_test reference
alignments only for development-stage scoring and does not include route labels
in the model inputs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

from src.features.candidate_features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = ROOT / "outputs/features/student_test_candidates.csv.gz"
DEFAULT_MODEL_DIR = ROOT / "outputs/models"
DEFAULT_DATA_ROOT = ROOT / "Student folder"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/evaluation"
MODEL_NAMES = (
    "logistic_regression",
    "linear_svm",
    "random_forest",
    "knn",
    "hist_gradient_boosting",
)
KEYS = ["trip_id", "obs_idx"]
LABEL = "is_reference_candidate"


def _arc_endpoints(data_root: Path, trip_id: str) -> np.ndarray:
    """Load directed arc endpoints; arc row index is its arc ID."""
    path = data_root / "student_test" / trip_id / f"{trip_id}.arcs"
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"],
        engine="python",
    )
    return frame.to_numpy(dtype=np.int64)


def _continuity_for_trip(
    trip_rows: pd.DataFrame,
    endpoints: np.ndarray,
) -> tuple[int, int, int]:
    """Count direct connected transitions between consecutive GPS observations.

    Pairs separated by a missing observation index or a timestamp gap over
    1.5 seconds are skipped. Repeating the same directed arc is continuous.
    """
    rows = trip_rows.sort_values("obs_idx", kind="stable")
    connected = checked = skipped = 0
    records = list(rows[["obs_idx", "candidate_arc_id", "time_gap_seconds"]].itertuples(index=False, name=None))
    for current, following in zip(records, records[1:]):
        obs_a, arc_a, _ = current
        obs_b, arc_b, gap_b = following
        if int(obs_b) != int(obs_a) + 1 or (pd.notna(gap_b) and float(gap_b) > 1.5):
            skipped += 1
            continue
        checked += 1
        arc_a, arc_b = int(arc_a), int(arc_b)
        if arc_a == arc_b:
            connected += 1
        elif (
            0 <= arc_a < len(endpoints)
            and 0 <= arc_b < len(endpoints)
            and int(endpoints[arc_a, 1]) == int(endpoints[arc_b, 0])
        ):
            connected += 1
    return connected, checked, skipped


def _evaluate_one(
    model_name: str,
    model_path: Path,
    candidates: pd.DataFrame,
    data_root: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing saved model: {model_path}")
    model = joblib.load(model_path)
    X = candidates[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

    started = time.perf_counter()
    probabilities: np.ndarray | None = None
    if hasattr(model, "predict_proba"):
        class_values = np.asarray(model.classes_)
        positive_column = int(np.flatnonzero(class_values == 1)[0])
        probabilities = model.predict_proba(X)[:, positive_column]
        scores = probabilities
        predicted_labels = (scores >= 0.5).astype(np.int8)
    else:
        scores = model.decision_function(X)
        predicted_labels = (scores >= 0.0).astype(np.int8)
    inference_seconds = time.perf_counter() - started

    scored = candidates[KEYS + [
        "candidate_arc_id",
        "reference_arc_id",
        "reference_alignment_confident",
        "has_reference_candidate",
        LABEL,
        "time_gap_seconds",
    ]].copy()
    scored["_score"] = scores
    scored["_predicted_label"] = predicted_labels
    if probabilities is not None:
        scored["_probability"] = probabilities

    observation_meta = (
        scored.groupby(KEYS, as_index=False, sort=False)
        .agg(
            reference_alignment_confident=("reference_alignment_confident", "max"),
            has_reference_candidate=("has_reference_candidate", "max"),
            reference_arc_id=("reference_arc_id", "first"),
        )
    )
    confident = observation_meta["reference_alignment_confident"].eq(1)
    confident_meta = observation_meta.loc[confident]
    miss_meta = confident_meta[confident_meta["has_reference_candidate"].eq(0)]
    eligible_meta = confident_meta[
        confident_meta["has_reference_candidate"].eq(1)
        & confident_meta["reference_arc_id"].notna()
    ]
    eligible_keys = eligible_meta[KEYS]
    eligible_rows = scored.merge(eligible_keys, on=KEYS, how="inner", sort=False)
    if eligible_rows.empty:
        raise ValueError(f"{model_name}: no confidently labelled student_test candidates to score")

    top_rows = (
        scored.sort_values("_score", kind="stable")
        .groupby(KEYS, sort=False, as_index=False)
        .tail(1)
    )
    top_eligible = top_rows.drop(columns="reference_arc_id").merge(
        eligible_meta[KEYS + ["reference_arc_id"]], on=KEYS, how="inner"
    )
    y_observation = top_eligible["reference_arc_id"].astype(np.int64).to_numpy()
    pred_observation = top_eligible["candidate_arc_id"].astype(np.int64).to_numpy()
    arc_labels = np.union1d(y_observation, pred_observation)
    observation_macro_f1 = f1_score(
        y_observation,
        pred_observation,
        labels=arc_labels,
        average="macro",
        zero_division=0,
    )

    y_candidate = eligible_rows[LABEL].astype(np.int8).to_numpy()
    pred_candidate = eligible_rows["_predicted_label"].astype(np.int8).to_numpy()
    candidate_macro_f1 = f1_score(y_candidate, pred_candidate, average="macro", zero_division=0)
    candidate_log_loss: float | None = None
    observation_candidate_log_loss: float | None = None
    if probabilities is not None:
        candidate_log_loss = float(log_loss(
            y_candidate,
            eligible_rows["_probability"].to_numpy(),
            labels=[0, 1],
        ))
        group_losses: list[float] = []
        for _, group in eligible_rows.groupby(KEYS, sort=False):
            scores_for_observation = group["_probability"].to_numpy(dtype=float)
            total = float(scores_for_observation.sum())
            true_score = float(group.loc[group[LABEL].eq(1), "_probability"].sum())
            probability_of_true_candidate = (
                true_score / total if total > 0 else 1.0 / len(group)
            )
            group_losses.append(-np.log(np.clip(probability_of_true_candidate, 1e-15, 1.0)))
        observation_candidate_log_loss = float(np.mean(group_losses))

    top_rows_all = top_rows.copy()
    per_trip: list[dict[str, object]] = []
    total_connected = total_checked = total_skipped = 0
    for trip_id, trip_predictions in top_rows_all.groupby("trip_id", sort=True):
        endpoints = _arc_endpoints(data_root, str(trip_id))
        connected, checked, skipped = _continuity_for_trip(trip_predictions, endpoints)
        total_connected += connected
        total_checked += checked
        total_skipped += skipped
        trip_eligible = top_eligible[top_eligible["trip_id"].eq(trip_id)]
        trip_confident = confident_meta[confident_meta["trip_id"].eq(trip_id)]
        trip_misses = miss_meta[miss_meta["trip_id"].eq(trip_id)]
        per_trip.append({
            "model": model_name,
            "trip_id": str(trip_id),
            "confident_observations": int(len(trip_confident)),
            "candidate_misses": int(len(trip_misses)),
            "candidate_miss_rate": (
                float(len(trip_misses) / len(trip_confident)) if len(trip_confident) else None
            ),
            "top1_observation_count": int(len(trip_eligible)),
            "top1_accuracy": float(
                (trip_eligible["candidate_arc_id"] == trip_eligible["reference_arc_id"]).mean()
            ) if len(trip_eligible) else None,
            "directly_connected_transitions": connected,
            "checked_transitions": checked,
            "skipped_transitions": skipped,
            "route_continuity_rate": float(connected / checked) if checked else None,
        })

    metrics: dict[str, object] = {
        "model": model_name,
        "evaluation_partition": "student_test",
        "student_test_trip_count": int(candidates["trip_id"].nunique()),
        "confident_observation_count": int(len(confident_meta)),
        "candidate_miss_count": int(len(miss_meta)),
        "candidate_miss_rate": (
            float(len(miss_meta) / len(confident_meta)) if len(confident_meta) else None
        ),
        "evaluated_observation_count": int(len(top_eligible)),
        "top1_observation_accuracy": float((pred_observation == y_observation).mean()),
        "observation_route_arc_macro_f1": float(observation_macro_f1),
        "candidate_row_macro_f1": float(candidate_macro_f1),
        "candidate_row_log_loss": candidate_log_loss,
        "observation_candidate_log_loss": observation_candidate_log_loss,
        "log_loss_note": (
            "N/A for LinearSVC because it does not produce probabilities."
            if probabilities is None
            else "Candidate-row binary loss and per-observation normalized candidate-set loss."
        ),
        "directly_connected_transitions": int(total_connected),
        "checked_transitions": int(total_checked),
        "skipped_transitions": int(total_skipped),
        "route_continuity_rate": float(total_connected / total_checked) if total_checked else None,
        "inference_seconds": float(inference_seconds),
        "inference_candidate_rows_per_second": float(len(candidates) / inference_seconds)
        if inference_seconds > 0 else None,
        "notes": (
            "Top-1 and route metrics exclude confident observations where the reference arc was absent "
            "from the candidate set; those are reported in candidate_miss_rate. Route continuity is "
            "direct arc-to-arc connectivity for consecutive observation indices with <=1.5s gaps."
        ),
    }
    return metrics, per_trip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    required = FEATURE_COLUMNS + KEYS + [
        "candidate_arc_id",
        "reference_arc_id",
        "reference_alignment_confident",
        "has_reference_candidate",
        LABEL,
        "time_gap_seconds",
    ]
    candidates = pd.read_csv(args.candidates, usecols=required, dtype={"trip_id": "string"})
    metrics_rows: list[dict[str, object]] = []
    per_trip_rows: list[dict[str, object]] = []
    for model_name in MODEL_NAMES:
        metrics, per_trip = _evaluate_one(
            model_name,
            args.model_dir / f"{model_name}.joblib",
            candidates,
            args.data_root,
        )
        metrics_rows.append(metrics)
        per_trip_rows.extend(per_trip)
        print(
            f"{model_name}: Top-1={metrics['top1_observation_accuracy']:.4f}, "
            f"Macro-F1={metrics['observation_route_arc_macro_f1']:.4f}, "
            f"candidate miss={metrics['candidate_miss_rate']:.4%}, "
            f"continuity={metrics['route_continuity_rate']:.4f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "student_test_model_comparison.csv"
    per_trip_path = args.output_dir / "student_test_per_trip_metrics.csv"
    json_path = args.output_dir / "student_test_model_comparison.json"
    pd.DataFrame(metrics_rows).to_csv(summary_path, index=False)
    pd.DataFrame(per_trip_rows).to_csv(per_trip_path, index=False)
    json_path.write_text(json.dumps(metrics_rows, indent=2) + "\n", encoding="utf-8")
    print(f"Saved summary: {summary_path}")
    print(f"Saved per-trip metrics: {per_trip_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
