"""Tune HMM/Viterbi decoder parameters with sequential Bayesian optimization.

The objective uses only trip-grouped ``train`` validation trips. The classifier
must be fitted on the corresponding fit trips. No evaluator data is read.

Run from the repository root:
    .venv/bin/python -m src.optimization.tune_hmm_parameters --calls 8
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from skopt import Optimizer
from skopt.space import Real, Integer
from sklearn.metrics import f1_score

from src.evaluation.compare_hmm_student_test import _load_trip
from src.features.candidate_features import FEATURE_COLUMNS
from src.map_matching.hmm_viterbi import decode_trip_candidates

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = ROOT / "outputs/features/train_candidates.csv.gz"
DEFAULT_SPLITS = ROOT / "outputs/eda/trip_split_assignments.csv"
DEFAULT_MODEL = ROOT / "outputs/models/hist_gradient_boosting.joblib"
DEFAULT_DATA_ROOT = ROOT / "Student folder"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/evaluation"
DEFAULT_CONFIG = ROOT / "outputs/models/hmm_decoder_config.json"
KEYS = ["trip_id", "obs_idx"]
PARAMETER_NAMES = [
    "emission_weight", "transition_weight", "max_gap_seconds",
    "disconnected_penalty", "sigma_floor_m", "sigma_time_scale_m",
    "sigma_distance_fraction",
]


def _macro_f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    labels = np.union1d(truth, prediction)
    return float(f1_score(truth, prediction, labels=labels, average="macro", zero_division=0))


def _as_parameters(values: list[float | int]) -> dict[str, float | int]:
    return dict(zip(PARAMETER_NAMES, values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-output", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calls", type=int, default=8, help="Total BO evaluations; at least 6 recommended")
    args = parser.parse_args()
    if args.calls < 6:
        parser.error("--calls must be at least 6 (4 Sobol seed trials plus Bayesian proposals)")

    started = time.perf_counter()
    split = pd.read_csv(args.splits, dtype={"trip_id": "string"})
    validation_ids = set(
        split.loc[split["partition"].eq("train") & split["holdout_role"].eq("validation"), "trip_id"]
        .astype(str)
    )
    if not validation_ids:
        raise RuntimeError("No train validation trips found in split file")

    selected_columns = FEATURE_COLUMNS + KEYS + [
        "candidate_arc_id", "reference_arc_id", "reference_alignment_confident",
        "has_reference_candidate",
    ]
    candidates = pd.read_csv(
        args.candidates, usecols=selected_columns, dtype={"trip_id": "string"}
    )
    candidates = candidates[candidates["trip_id"].isin(validation_ids)].copy()
    validation_ids &= set(candidates["trip_id"].unique().astype(str))
    if not validation_ids:
        raise RuntimeError("No validation trips have candidate rows")

    model = joblib.load(args.model)
    features = candidates[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    class_values = np.asarray(model.classes_)
    positive_column = int(np.flatnonzero(class_values == 1)[0])
    candidates["_model_probability"] = model.predict_proba(features)[:, positive_column]

    # Cache each validation trip's raw files and scored candidate probabilities;
    # Bayesian trials then vary only the Viterbi decoder.
    payloads = []
    for trip_id in sorted(validation_ids):
        rows = candidates.loc[candidates["trip_id"].eq(trip_id)].copy()
        disk_trip_id = str(trip_id).zfill(8)
        track, nodes, endpoints = _load_trip(args.data_root, disk_trip_id, partition="train")
        truth_meta = rows.groupby("obs_idx", sort=False).agg(
            reference_arc_id=("reference_arc_id", "first"),
            confident=("reference_alignment_confident", "max"),
            has_reference=("has_reference_candidate", "max"),
        )
        scored = truth_meta.loc[
            truth_meta["confident"].eq(1)
            & truth_meta["has_reference"].eq(1)
            & truth_meta["reference_arc_id"].notna()
        ]
        if scored.empty:
            continue
        payloads.append({
            "trip_id": str(trip_id), "rows": rows, "track": track,
            "nodes": nodes, "endpoints": endpoints,
            "obs_idx": scored.index.to_numpy(dtype=np.int64),
            "truth": scored["reference_arc_id"].to_numpy(dtype=np.int64),
        })
    if not payloads:
        raise RuntimeError("No confident validation observations with reference candidates")

    bounds = [
        Real(0.5, 2.5, name="emission_weight", prior="log-uniform"),
        Real(0.25, 4.0, name="transition_weight", prior="log-uniform"),
        Integer(15, 120, name="max_gap_seconds"),
        Real(2.0, 20.0, name="disconnected_penalty"),
        Real(5.0, 50.0, name="sigma_floor_m", prior="log-uniform"),
        Real(2.0, 20.0, name="sigma_time_scale_m", prior="log-uniform"),
        Real(0.05, 0.75, name="sigma_distance_fraction", prior="log-uniform"),
    ]
    optimizer = Optimizer(
        dimensions=bounds,
        base_estimator="GP",
        acq_func="EI",
        n_initial_points=4,
        initial_point_generator="sobol",
        random_state=173,
    )
    # Classifier-only validation metrics provide a guardrail: the decoder should
    # improve macro-F1 without sacrificing Top-1 accuracy on the tuning split.
    classifier_pred, truth_values = [], []
    for payload in payloads:
        classifier = (
            payload["rows"].sort_values("_model_probability", kind="stable")
            .groupby("obs_idx", sort=False).tail(1)
            .set_index("obs_idx")["candidate_arc_id"]
        )
        classifier_pred.extend(classifier.reindex(payload["obs_idx"]).fillna(-1).to_numpy(dtype=np.int64))
        truth_values.extend(payload["truth"])
    truth_values = np.asarray(truth_values, dtype=np.int64)
    classifier_pred = np.asarray(classifier_pred, dtype=np.int64)
    baseline = _macro_f1(truth_values, classifier_pred)
    baseline_accuracy = float(np.mean(truth_values == classifier_pred))

    trial_rows: list[dict[str, object]] = []
    for trial_index in range(args.calls):
        point = optimizer.ask()
        parameters = _as_parameters(point)
        truths, predictions = [], []
        for payload in payloads:
            decoded, _ = decode_trip_candidates(
                payload["rows"], payload["track"], payload["nodes"], payload["endpoints"],
                **parameters,
            )
            aligned = pd.DataFrame({"obs_idx": payload["obs_idx"]}).merge(
                decoded[["obs_idx", "hmm_arc_id"]], on="obs_idx", how="left"
            )
            truths.append(payload["truth"])
            predictions.append(aligned["hmm_arc_id"].fillna(-1).to_numpy(dtype=np.int64))
        y_true, y_pred = np.concatenate(truths), np.concatenate(predictions)
        score = _macro_f1(y_true, y_pred)
        top1 = float(np.mean(y_true == y_pred))
        feasible = top1 >= baseline_accuracy
        # Constrained Bayesian objective: feasible trials rank by macro-F1;
        # infeasible trials are worse than every feasible point.
        optimization_loss = -score if feasible else 1.0 + (baseline_accuracy - top1)
        optimizer.tell(point, optimization_loss)
        record: dict[str, object] = {
            "trial": trial_index + 1,
            "search_phase": "Sobol initialization" if trial_index < 4 else "Gaussian-process Bayesian proposal",
            "validation_trips": len(payloads),
            "evaluated_observations": int(len(y_true)),
            "route_arc_macro_f1": score,
            "top1_accuracy": top1,
            "accuracy_guardrail_met": feasible,
            "optimization_loss": optimization_loss,
            "elapsed_seconds": time.perf_counter() - started,
            **parameters,
        }
        trial_rows.append(record)
        eligible = [row for row in trial_rows if row["accuracy_guardrail_met"]]
        best_score = max((float(row["route_arc_macro_f1"]) for row in eligible), default=float("nan"))
        print(
            f"Trial {trial_index + 1}/{args.calls} [{record['search_phase']}]: "
            f"macro-F1={score:.5f}, top1={top1:.5f}, "
            f"guardrail={'pass' if feasible else 'fail'}, best-feasible={best_score:.5f}",
            flush=True,
        )

    trials = pd.DataFrame(trial_rows)
    eligible_trials = trials.loc[trials["accuracy_guardrail_met"]]
    if eligible_trials.empty:
        raise RuntimeError("No Bayesian trial met the classifier-only validation Top-1 guardrail")
    best = eligible_trials.loc[eligible_trials["route_arc_macro_f1"].idxmax()].to_dict()
    best_parameters = {name: best[name].item() if hasattr(best[name], "item") else best[name] for name in PARAMETER_NAMES}
    result = {
        "method": "sequential Gaussian-process Bayesian optimization (Sobol initialization; expected-improvement acquisition)",
        "objective": "maximize pooled route-arc macro F1 subject to Top-1 accuracy >= classifier-only on train validation",
        "validation_trip_count": len(payloads),
        "evaluated_observations": int(len(truth_values)),
        "calls": args.calls,
        "sobol_initial_trials": 4,
        "bayesian_trials": args.calls - 4,
        "classifier_only_validation_route_arc_macro_f1": baseline,
        "classifier_only_validation_top1_accuracy": baseline_accuracy,
        "best_validation_route_arc_macro_f1": float(best["route_arc_macro_f1"]),
        "best_validation_top1_accuracy": float(best["top1_accuracy"]),
        "best_trial": int(best["trial"]),
        "feasible_trials": int(trials["accuracy_guardrail_met"].sum()),
        "decoder_parameters": best_parameters,
        "total_elapsed_seconds": time.perf_counter() - started,
        "data_policy": "Only train fit model and trip-grouped train validation observations were used; evaluator data was not read.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    trials.to_csv(args.output_dir / "hmm_bayes_search.csv", index=False)
    (args.output_dir / "hmm_bayes_search.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.config_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Best train-validation macro-F1: {result['best_validation_route_arc_macro_f1']:.5f}")
    print(f"Classifier-only train-validation macro-F1: {baseline:.5f}")
    print(f"Saved trials: {args.output_dir / 'hmm_bayes_search.csv'}")
    print(f"Saved decoder configuration: {args.config_output}")


if __name__ == "__main__":
    main()
