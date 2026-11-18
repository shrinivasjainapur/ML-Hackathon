"""Create deterministic, trip-level holdout and cross-validation assignments.

No GPS rows are split independently: every observation from one trip remains
in the same group. The student-test partition is listed separately and is
never included in the training or cross-validation assignments.

Run from the repository root:
    .venv/bin/python -m src.preprocessing.make_splits
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .dataset import default_data_root


def _trip_ids(data_root: Path, partition: str) -> list[str]:
    folder = data_root / partition
    if not folder.is_dir():
        raise FileNotFoundError(f"Partition folder not found: {folder}")
    trip_dirs = sorted(p for p in folder.iterdir() if p.is_dir())
    if not trip_dirs:
        raise ValueError(f"No trip folders found under {folder}")
    ids = [p.name for p in trip_dirs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate trip IDs found in {partition}")
    required_exts = (".track", ".nodes", ".arcs", ".route")
    for trip_dir in trip_dirs:
        missing = [ext for ext in required_exts if not (trip_dir / f"{trip_dir.name}{ext}").is_file()]
        if missing:
            raise FileNotFoundError(
                f"Cannot make a complete split: {partition}/{trip_dir.name} is missing {missing}"
            )
    return ids


def create_assignments(
    data_root: Path,
    *,
    seed: int = 42,
    validation_fraction: float = 0.2,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return one assignment row per trip plus a JSON-serializable manifest."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    train_ids = _trip_ids(data_root, "train")
    student_ids = _trip_ids(data_root, "student_test")
    if len(train_ids) != 80 or len(student_ids) != 10:
        raise ValueError(
            "Expected the documented 80 train and 10 student_test trips; "
            f"found {len(train_ids)} and {len(student_ids)}. Check the data root before splitting."
        )
    overlap = set(train_ids) & set(student_ids)
    if overlap:
        raise ValueError(f"Trip IDs appear in both train and student_test: {sorted(overlap)}")
    if n_splits < 2 or n_splits > len(train_ids):
        raise ValueError(f"n_splits must be in [2, {len(train_ids)}]")

    ids = np.asarray(train_ids, dtype=object)
    groups = ids.copy()  # One record per trip; rows later join by trip_id.
    holdout = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    fit_idx, val_idx = next(holdout.split(np.zeros(len(ids)), groups=groups))
    holdout_roles = np.full(len(ids), "fit", dtype=object)
    holdout_roles[val_idx] = "validation"

    folds = np.full(len(ids), -1, dtype=np.int16)
    cv = GroupKFold(n_splits=n_splits)
    for fold_id, (_, fold_val_idx) in enumerate(cv.split(np.zeros(len(ids)), groups=groups)):
        folds[fold_val_idx] = fold_id
    if (folds < 0).any() or len(set(train_ids[i] for i in fit_idx) & set(train_ids[i] for i in val_idx)):
        raise RuntimeError("Internal error: a trip is missing a CV fold or leaked across holdout roles")

    train_frame = pd.DataFrame({
        "partition": "train",
        "trip_id": train_ids,
        "holdout_role": holdout_roles,
        "cv_fold": folds.astype(int),
    })
    student_frame = pd.DataFrame({
        "partition": "student_test",
        "trip_id": student_ids,
        "holdout_role": "student_test",
        "cv_fold": pd.Series([pd.NA] * len(student_ids), dtype="Int64"),
    })
    assignments = pd.concat([train_frame, student_frame], ignore_index=True)

    manifest: dict[str, object] = {
        "seed": seed,
        "group_key": "trip_id",
        "training_partition": "train",
        "development_partition": "student_test",
        "evaluator_partition_included": False,
        "train_trip_count": len(train_ids),
        "student_test_trip_count": len(student_ids),
        "holdout_method": "GroupShuffleSplit over one row per training trip",
        "validation_fraction_requested": validation_fraction,
        "fit_trip_count": int((holdout_roles == "fit").sum()),
        "validation_trip_count": int((holdout_roles == "validation").sum()),
        "cross_validation_method": "GroupKFold",
        "cross_validation_folds": n_splits,
        "cv_fold_trip_counts": {
            str(fold): int((folds == fold).sum()) for fold in range(n_splits)
        },
        "assignment_file": "trip_split_assignments.csv",
    }
    return assignments, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root(),
                        help='Folder containing train/ and student_test/ (default: "Student folder")')
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    assignments, manifest = create_assignments(
        args.data_root,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        n_splits=args.folds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = args.output_dir / "trip_split_assignments.csv"
    manifest_path = args.output_dir / "split_manifest.json"
    assignments.to_csv(assignment_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(assignments.groupby(["partition", "holdout_role"], dropna=False)
          .size().rename("trip_count").to_string())
    print("CV trips per fold:", manifest["cv_fold_trip_counts"])
    print(f"Saved: {assignment_path.resolve()}")
    print(f"Saved: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
