# PathSnap — distributed map matching and route reconstruction

PathSnap maps noisy GPS observations to likely road segments and reconstructs
the vehicle's route. This repository currently contains the dataset inventory
and candidate-feature preparation stages.

## Repository data layout

In this checkout, the supplied data is in `Student folder/`:

```text
Student folder/
├── train/<trip_id>/<trip_id>.(track|nodes|arcs|route)
└── student_test/<trip_id>/<trip_id>.(track|nodes|arcs|route)
```

The input files are headerless and whitespace-delimited:

- `.track`: longitude, latitude, timestamp in seconds
- `.nodes`: longitude, latitude; row index is the node ID
- `.arcs`: from-node ID, to-node ID; row index is the arc ID
- `.route`: ordered reference arc IDs, when available

The 80 `train` trips support fitting and trip-aware cross-validation. The 10
`student_test` trips may be used for development-stage comparisons. Evaluator
trips and labels are not part of this workflow and must not be added to it.

## Setup

Python 3.10 or newer is recommended. From this repository's root directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

On Windows PowerShell, use `.venv\Scripts\python.exe` in place of
`.venv/bin/python` in the install and run commands below. Activating the
environment is optional when calling its Python executable directly.

## Person 1: inspect and validate the data

The default data path is detected from this checkout. Generate the inventory:

```bash
.venv/bin/python -m src.preprocessing.inspect_dataset
```

Optional explicit paths:

```bash
.venv/bin/python -m src.preprocessing.inspect_dataset \
  --data-root "Student folder" \
  --output-dir outputs/eda \
  --partitions train student_test
```

Outputs:

- `outputs/eda/trip_inventory.csv`: one row per trip with file presence/size,
  point/node/arc/route counts, coordinate checks, timestamp-gap checks,
  arc/node index-integrity counts, track/route row-count comparison, and
  broken consecutive route-arc transitions.
- `outputs/eda/partition_summary.csv`: aggregate counts by partition.
- `outputs/eda/data_quality_summary.md`: concise findings and route-label caveats.

Malformed or missing files are reported in the inventory so one bad trip does
not prevent inspection of the rest.

Create a reproducible trip-level holdout and five GroupKFold assignments:

```bash
.venv/bin/python -m src.preprocessing.make_splits
```

This writes `outputs/eda/trip_split_assignments.csv` and
`outputs/eda/split_manifest.json`. The default split is 64 fit trips and 16
validation trips from `train`; the five CV folds cover all 80 training trips.
Every trip stays wholly inside one split/fold. `student_test` is listed
separately and never enters the training split. Change the deterministic seed
with `--seed` if needed and record it with the experiments.

## Person 2: build candidate features

```bash
.venv/bin/python -m src.features.candidate_features
```

The default command processes both `train` and `student_test`. For a smaller
development run, cap each partition to one trip:

```bash
.venv/bin/python -m src.features.candidate_features --max-trips 1
```

Or process only training trips:

```bash
.venv/bin/python -m src.features.candidate_features --partitions train
```

Optional controls are `--search-radius-m` (default 80 metres),
`--max-candidates` (default 32 candidates per GPS observation),
`--alignment-confidence-m` (default 50 metres), `--data-root`, and
`--output-dir`. `--max-trips 1` overwrites each partition output with a
one-trip subset; rerun without that option before model training.

Outputs:

- `outputs/features/train_candidates.csv.gz`
- `outputs/features/student_test_candidates.csv.gz` (unless excluded with
  `--partitions train`)
- `outputs/features/train_candidate_summary.csv` and
  `student_test_candidate_summary.csv`
- `outputs/features/feature_manifest.json`
- `outputs/features/feature_summary.md`

Each candidate row pairs one GPS observation with one nearby directed road
segment. The model feature columns are listed in the manifest and in
`src/features/candidate_features.py`. Keep identifiers and reference-target
columns out of the model input. For candidate-classifier training, start with
rows where `reference_alignment_confident == 1` and
`has_reference_candidate == 1`. This evaluates the classifier conditional on
the correct road being in the candidate set. Report candidate recall
separately; do not treat an observation with no positive candidate as a normal
all-negative training example. Keep every candidate from a trip in the same
fold and fit any learned scaling within each fold.

### Reference-route alignment caveat

The `.route` sequence is shorter than the GPS track and must not be joined to
GPS rows by row number. Candidate preparation uses a monotone, spatial,
Viterbi-style alignment to derive an arc label for each observation. This is a
documented modelling assumption, not an organizer-provided one-to-one label
mapping. Targets are left blank when the aligned route is farther than the
configured confidence threshold. The default 50 m cutoff is a conservative,
provisional data-quality rule, not a threshold supplied or validated by the
organizers. Check its sensitivity using training-trip cross-validation and
record any change; do not tune it on evaluator data. Review
`reference_alignment_distance_m`, `reference_alignment_confident`, candidate
misses, and per-trip candidate recall before using the labels. Do not report
these derived targets as independently verified ground truth.

Candidate generation uses a per-trip local azimuthal-equidistant projection so
distances are measured in metres rather than degrees. Model scaling, if needed,
must be fitted within each training fold to avoid leakage.

## Source layout

- `src/preprocessing/dataset.py`: shared file readers, trip container, and
  index validation.
- `src/preprocessing/inspect_dataset.py`: Person 1 inventory command.
- `src/preprocessing/make_splits.py`: deterministic trip-level holdout and
  GroupKFold assignments.
- `src/preprocessing/label_alignment.py`: monotone reference-route alignment.
- `src/features/candidate_features.py`: Person 2 candidate generation,
  feature calculation, CLI, and outputs.
- `src/models/`, `src/optimization/`, `src/evaluation/`, and
  `src/map_matching/`: reserved for the remaining experiment and decoder work.

## Data integrity and reproducibility

Keep all observations from a trip in the same cross-validation fold. Never use
evaluator-testing labels for preprocessing choices, feature selection, model
selection, or tuning. Record the exact command, settings, and actual outputs
for every experiment; do not invent metrics or hardcode predictions.
