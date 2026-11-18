# PathSnap — Distributed Map-Matching & Route Reconstruction

PathSnap is a machine-learning system for matching noisy GPS observations to a road network and reconstructing a plausible vehicle route. The planned pipeline includes data exploration, preprocessing, feature engineering, model comparison, Bayesian hyperparameter optimisation, HMM/Viterbi decoding, a FastAPI backend, and an evaluator-facing UI.

## Dataset

Extract the supplied archive locally so the layout is:

```text
data/
├── train/<trip_id>/<trip_id>.(track|nodes|arcs|route)
└── student_test/<trip_id>/<trip_id>.(track|nodes|arcs|route)
```

Keep the dataset out of this public repository. Do not add evaluator-testing data or expose evaluator ground-truth labels.

Files are whitespace-separated and headerless:
- `.track`: longitude, latitude, timestamp (seconds)
- `.nodes`: longitude, latitude
- `.arcs`: from-node ID, to-node ID
- `.route`: chronological reference arc IDs, where supplied

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## First milestone: dataset inventory

After extracting the archive into `data/`, run:

```bash
python -m src.preprocessing.inspect_dataset --data-root data --output-dir outputs/eda
```

The script creates `trip_inventory.csv` and `partition_summary.csv`.

## Planned structure

- `src/preprocessing/`: raw-data inspection and preprocessing
- `src/features/`: candidate generation and feature engineering
- `src/models/`: candidate-scoring classifiers
- `src/optimization/`: Bayesian hyperparameter search
- `src/map_matching/`: HMM and Viterbi decoding
- `src/evaluation/`: metrics and model comparisons
- `backend/`: FastAPI service
- `frontend/`: evaluator interface
- `notebooks/`: exploratory and experimental notebooks
- `visualizations/`: Folium maps and plots
- `experiments/`: configurations and experiment logs
- `reports/`: technical report source and evidence

## Data integrity

Use the 80 training trips for fitting and group-aware cross-validation. Use the 10 student-testing trips for development-stage comparisons as permitted by the guidelines. Keep evaluator-testing data and labels out of development and model-selection workflows. Record actual experiment configurations, metrics, and outputs. Never invent results or hardcode evaluation predictions.
