"""Compare classifier-only predictions with HMM/Viterbi on student_test.

Run from the repository root:
    .venv/bin/python -m src.evaluation.compare_hmm_student_test

This script scores only the development student_test partition and never
reads evaluator_test data or route files.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.features.candidate_features import FEATURE_COLUMNS
from src.map_matching.hmm_viterbi import decode_trip_candidates
from src.preprocessing.dataset import FILE_COLUMNS, read_trip_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATES = ROOT / "outputs/features/student_test_candidates.csv.gz"
DEFAULT_MODEL = ROOT / "outputs/models/final_model.joblib"
DEFAULT_DATA_ROOT = ROOT / "Student folder"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/evaluation"
KEYS = ["trip_id", "obs_idx"]


def _route_metrics(truth: np.ndarray, predicted: np.ndarray) -> tuple[float | None, float | None]:
    if not len(truth):
        return None, None
    accuracy = float(np.mean(truth == predicted))
    labels = np.union1d(truth, predicted)
    macro_f1 = float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0))
    return accuracy, macro_f1


def _continuity(track: pd.DataFrame, predictions: pd.DataFrame, endpoints: np.ndarray):
    """Direct arc connectivity for adjacent track rows with at most 1.5s gap."""
    rows = track[["obs_idx", "timestamp"]].merge(predictions, on="obs_idx", how="left")
    arc_ids = pd.to_numeric(rows["predicted_arc_id"], errors="coerce").fillna(-1).to_numpy(int)
    times = pd.to_numeric(rows["timestamp"], errors="coerce").to_numpy(float)
    connected = checked = skipped = 0
    for index in range(1, len(rows)):
        dt = times[index] - times[index - 1]
        if not np.isfinite(dt) or dt <= 0 or dt > 1.5:
            skipped += 1
            continue
        arc_a, arc_b = int(arc_ids[index - 1]), int(arc_ids[index])
        if arc_a < 0 or arc_b < 0:
            skipped += 1
            continue
        checked += 1
        if arc_a == arc_b or (
            0 <= arc_a < len(endpoints)
            and 0 <= arc_b < len(endpoints)
            and int(endpoints[arc_a, 1]) == int(endpoints[arc_b, 0])
        ):
            connected += 1
    return connected, checked, skipped


def _load_trip(data_root: Path, trip_id: str):
    folder = data_root / "student_test" / trip_id
    track = read_trip_file(folder / f"{trip_id}.track", FILE_COLUMNS[".track"])
    nodes = read_trip_file(folder / f"{trip_id}.nodes", FILE_COLUMNS[".nodes"])
    arcs = read_trip_file(folder / f"{trip_id}.arcs", FILE_COLUMNS[".arcs"])
    assert track is not None and nodes is not None and arcs is not None
    track.insert(0, "obs_idx", np.arange(len(track), dtype=np.int64))
    track["timestamp"] = pd.to_numeric(track["timestamp"], errors="coerce")
    node_lonlat = nodes[["longitude", "latitude"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    endpoints = arcs[["from_node_id", "to_node_id"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return track, node_lonlat, endpoints


def _evaluate_trip(trip_id, all_candidates, model, data_root, max_gap_seconds, disconnected_penalty):
    rows = all_candidates.loc[all_candidates["trip_id"].eq(trip_id)].copy()
    X = rows[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    class_values = np.asarray(model.classes_)
    positive_column = int(np.flatnonzero(class_values == 1)[0])
    started = time.perf_counter()
    rows["_model_probability"] = model.predict_proba(X)[:, positive_column]
    inference_seconds = time.perf_counter() - started

    classifier_top = (
        rows.sort_values("_model_probability", kind="stable")
        .groupby("obs_idx", sort=False, as_index=False)
        .tail(1)[["obs_idx", "candidate_arc_id"]]
        .rename(columns={"candidate_arc_id": "classifier_only_arc_id"})
    )
    track, nodes, endpoints = _load_trip(data_root, str(trip_id))
    decode_started = time.perf_counter()
    hmm_rows, diagnostics = decode_trip_candidates(
        rows, track, nodes, endpoints,
        max_gap_seconds=max_gap_seconds,
        disconnected_penalty=disconnected_penalty,
    )
    decode_seconds = time.perf_counter() - decode_started

    predictions = track[["obs_idx", "timestamp", "longitude", "latitude"]].copy()
    predictions = predictions.merge(classifier_top, on="obs_idx", how="left")
    predictions = predictions.merge(hmm_rows, on="obs_idx", how="left")
    for column in ("classifier_only_arc_id", "hmm_arc_id"):
        predictions[column] = pd.to_numeric(predictions[column], errors="coerce").fillna(-1).astype(int)

    truth_meta = (
        rows.groupby("obs_idx", as_index=False, sort=False)
        .agg(
            reference_arc_id=("reference_arc_id", "first"),
            reference_alignment_confident=("reference_alignment_confident", "max"),
            has_reference_candidate=("has_reference_candidate", "max"),
        )
    )
    confident = truth_meta["reference_alignment_confident"].eq(1)
    misses = confident & truth_meta["has_reference_candidate"].eq(0)
    eligible = confident & truth_meta["has_reference_candidate"].eq(1) & truth_meta["reference_arc_id"].notna()
    eligible_truth = truth_meta.loc[eligible, ["obs_idx", "reference_arc_id"]]
    scored = predictions.merge(eligible_truth, on="obs_idx", how="inner")
    truth = scored["reference_arc_id"].astype(np.int64).to_numpy()
    classifier_pred = scored["classifier_only_arc_id"].to_numpy(np.int64)
    hmm_pred = scored["hmm_arc_id"].to_numpy(np.int64)
    classifier_accuracy, classifier_f1 = _route_metrics(truth, classifier_pred)
    hmm_accuracy, hmm_f1 = _route_metrics(truth, hmm_pred)

    classifier_connected, classifier_checked, classifier_skipped = _continuity(
        track,
        predictions[["obs_idx", "classifier_only_arc_id"]].rename(columns={"classifier_only_arc_id": "predicted_arc_id"}),
        endpoints,
    )
    hmm_connected, hmm_checked, hmm_skipped = _continuity(
        track,
        predictions[["obs_idx", "hmm_arc_id"]].rename(columns={"hmm_arc_id": "predicted_arc_id"}),
        endpoints,
    )
    confident_count, miss_count = int(confident.sum()), int(misses.sum())
    trip_metrics = {
        "trip_id": str(trip_id),
        "confident_observations": confident_count,
        "candidate_misses": miss_count,
        "candidate_miss_rate": miss_count / confident_count if confident_count else None,
        "evaluated_observations": int(len(scored)),
        "classifier_only_top1_accuracy": classifier_accuracy,
        "hmm_viterbi_top1_accuracy": hmm_accuracy,
        "top1_accuracy_delta": hmm_accuracy - classifier_accuracy,
        "classifier_only_route_arc_macro_f1": classifier_f1,
        "hmm_viterbi_route_arc_macro_f1": hmm_f1,
        "route_arc_macro_f1_delta": hmm_f1 - classifier_f1,
        "classifier_only_direct_continuity": classifier_connected / classifier_checked if classifier_checked else None,
        "hmm_viterbi_direct_continuity": hmm_connected / hmm_checked if hmm_checked else None,
        "classifier_connected_transitions": classifier_connected,
        "classifier_checked_transitions": classifier_checked,
        "hmm_connected_transitions": hmm_connected,
        "hmm_checked_transitions": hmm_checked,
        "skipped_transitions_classifier": classifier_skipped,
        "skipped_transitions_hmm": hmm_skipped,
        "missing_candidate_observations": diagnostics["missing_candidate_observations"],
        "large_gap_resets": diagnostics["large_gap_resets"],
        "classifier_inference_seconds": inference_seconds,
        "hmm_decode_seconds": decode_seconds,
    }
    predictions.insert(0, "trip_id", str(trip_id))
    predictions = predictions.merge(truth_meta, on="obs_idx", how="left")
    predictions["reference_arc_id"] = pd.to_numeric(predictions["reference_arc_id"], errors="coerce")
    predictions["reference_alignment_confident"] = predictions["reference_alignment_confident"].fillna(0).astype(int)
    predictions["has_reference_candidate"] = predictions["has_reference_candidate"].fillna(0).astype(int)
    totals = {
        "truth": truth,
        "classifier_pred": classifier_pred,
        "hmm_pred": hmm_pred,
        "confident": confident_count,
        "misses": miss_count,
        "classifier_connected": classifier_connected,
        "classifier_checked": classifier_checked,
        "hmm_connected": hmm_connected,
        "hmm_checked": hmm_checked,
        "classifier_skipped": classifier_skipped,
        "hmm_skipped": hmm_skipped,
        "inference_seconds": inference_seconds,
        "decode_seconds": decode_seconds,
        "missing_candidates": diagnostics["missing_candidate_observations"],
        "large_gap_resets": diagnostics["large_gap_resets"],
    }
    return trip_metrics, predictions, totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-gap-seconds", type=float, default=60.0)
    parser.add_argument("--disconnected-penalty", type=float, default=8.0)
    args = parser.parse_args()
    model = joblib.load(args.model)
    columns = FEATURE_COLUMNS + KEYS + [
        "candidate_arc_id", "reference_arc_id", "reference_alignment_confident",
        "has_reference_candidate", "is_reference_candidate",
    ]
    candidates = pd.read_csv(args.candidates, usecols=columns, dtype={"trip_id": "string"})
    trip_results, prediction_tables, totals = [], [], []
    for trip_id in sorted(candidates["trip_id"].unique()):
        result, predictions, counts = _evaluate_trip(
            str(trip_id), candidates, model, args.data_root,
            args.max_gap_seconds, args.disconnected_penalty,
        )
        trip_results.append(result)
        prediction_tables.append(predictions)
        totals.append(counts)
        print(
            f"{trip_id}: Top-1 {result['classifier_only_top1_accuracy']:.4f} -> "
            f"{result['hmm_viterbi_top1_accuracy']:.4f}; continuity "
            f"{result['classifier_only_direct_continuity']:.4f} -> "
            f"{result['hmm_viterbi_direct_continuity']:.4f}"
        )

    truth = np.concatenate([item["truth"] for item in totals])
    classifier_pred = np.concatenate([item["classifier_pred"] for item in totals])
    hmm_pred = np.concatenate([item["hmm_pred"] for item in totals])
    classifier_accuracy, classifier_f1 = _route_metrics(truth, classifier_pred)
    hmm_accuracy, hmm_f1 = _route_metrics(truth, hmm_pred)
    total = lambda key: int(sum(item[key] for item in totals))
    classifier_connected, classifier_checked = total("classifier_connected"), total("classifier_checked")
    hmm_connected, hmm_checked = total("hmm_connected"), total("hmm_checked")
    comparison = {
        "evaluation_partition": "student_test",
        "trip_count": int(len(trip_results)),
        "evaluated_observations": int(len(truth)),
        "confident_observations": total("confident"),
        "candidate_misses": total("misses"),
        "candidate_miss_rate": total("misses") / total("confident") if total("confident") else None,
        "classifier_only": {
            "top1_accuracy": classifier_accuracy,
            "route_arc_macro_f1": classifier_f1,
            "direct_continuity": classifier_connected / classifier_checked if classifier_checked else None,
            "connected_transitions": classifier_connected,
            "checked_transitions": classifier_checked,
            "inference_seconds": float(sum(item["inference_seconds"] for item in totals)),
        },
        "hmm_viterbi": {
            "top1_accuracy": hmm_accuracy,
            "route_arc_macro_f1": hmm_f1,
            "direct_continuity": hmm_connected / hmm_checked if hmm_checked else None,
            "connected_transitions": hmm_connected,
            "checked_transitions": hmm_checked,
            "decode_seconds": float(sum(item["decode_seconds"] for item in totals)),
            "missing_candidate_observations": total("missing_candidates"),
            "large_gap_resets": total("large_gap_resets"),
        },
        "decoder_configuration": {
            "state": "directed candidate road arc",
            "emission": "log model probability normalized within each observation",
            "transition": "directed shortest-path distance versus geodesic GPS displacement; finite penalty if disconnected",
            "max_gap_seconds": args.max_gap_seconds,
            "disconnected_penalty": args.disconnected_penalty,
            "large_gap_behavior": "reset Viterbi chain for timestamp gaps above max_gap_seconds",
            "missing_candidate_behavior": "arc_id -1 for missing observations; bridge candidate-bearing observations when possible",
        },
        "notes": "Reference labels are used for student_test scoring only. No evaluator_test data was read.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "student_test_hmm_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([
        {"method": "classifier_only", **comparison["classifier_only"]},
        {"method": "hmm_viterbi", **comparison["hmm_viterbi"]},
    ]).to_csv(args.output_dir / "student_test_hmm_comparison.csv", index=False)
    pd.DataFrame(trip_results).to_csv(args.output_dir / "student_test_hmm_per_trip.csv", index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        args.output_dir / "student_test_hmm_predictions.csv.gz", index=False, compression="gzip"
    )
    print(f"Overall Top-1: {classifier_accuracy:.4f} -> {hmm_accuracy:.4f}")
    print(f"Overall Macro-F1: {classifier_f1:.4f} -> {hmm_f1:.4f}")
    print(
        "Overall continuity: "
        f"{comparison['classifier_only']['direct_continuity']:.4f} -> "
        f"{comparison['hmm_viterbi']['direct_continuity']:.4f}"
    )


if __name__ == "__main__":
    main()
