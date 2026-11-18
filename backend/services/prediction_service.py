"""Run the saved candidate classifier and HMM on a development trip."""
from __future__ import annotations

import json
import logging
import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.config import DATA_DIR, HMM_CONFIG_PATH, MODEL_PATH, STUDENT_TEST_DATA_DIR
from backend.services.trip_service import normalize_trip_id
from src.features.candidate_features import FEATURE_COLUMNS, iter_candidate_features
from src.map_matching.hmm_viterbi import decode_trip_candidates
from src.preprocessing.dataset import load_trip

logger = logging.getLogger(__name__)


class ModelNotReadyError(RuntimeError):
    """Raised when required model artifacts or input data are unavailable."""


@lru_cache(maxsize=1)
def _load_runtime(model_path: str, config_path: str):
    model_file, config_file = Path(model_path), Path(config_path)
    if not model_file.is_file():
        raise ModelNotReadyError(
            f"Model artifact not found at {model_file}. Train or provide PATHSNAP_MODEL_PATH."
        )
    try:
        model = joblib.load(model_file)
    except Exception as exc:
        raise ModelNotReadyError(f"Could not load model artifact: {model_file}") from exc

    decoder_parameters: dict[str, float] = {}
    if config_file.is_file():
        try:
            raw_config = json.loads(config_file.read_text(encoding="utf-8"))
            decoder_parameters = raw_config.get("decoder_parameters", raw_config)
            allowed = {
                "max_gap_seconds", "disconnected_penalty", "emission_weight",
                "transition_weight", "sigma_floor_m", "sigma_time_scale_m",
                "sigma_distance_fraction", "same_arc_bonus", "connected_arc_bonus",
            }
            decoder_parameters = {
                key: float(value)
                for key, value in decoder_parameters.items()
                if key in allowed and isinstance(value, (int, float))
            }
        except (OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
            raise ModelNotReadyError(f"Could not read HMM configuration: {config_file}") from exc
    return model, decoder_parameters


def prediction_readiness() -> dict[str, object]:
    """Return runtime readiness details without loading model binaries."""
    model_available = MODEL_PATH.is_file()
    data_available = STUDENT_TEST_DATA_DIR.is_dir()
    ready = model_available and data_available
    return {
        "ready": ready,
        "model_available": model_available,
        "data_available": data_available,
        "model_path": str(MODEL_PATH),
        "data_path": str(STUDENT_TEST_DATA_DIR),
    }


def generate_prediction(trip_id: str) -> dict[str, object] | None:
    """Generate road-matched points without reading any reference route labels."""
    normalized = normalize_trip_id(trip_id)
    if normalized is None:
        return None
    trip_dir = STUDENT_TEST_DATA_DIR / normalized
    if not trip_dir.is_dir():
        return None

    model, decoder_parameters = _load_runtime(str(MODEL_PATH), str(HMM_CONFIG_PATH))
    if not DATA_DIR.is_dir():
        raise ModelNotReadyError(f"Dataset directory not found: {DATA_DIR}")

    started = time.perf_counter()
    try:
        trip = load_trip(
            trip_dir,
            "student_test",
            require_route=False,
            include_route=False,
        )
        gps_points = []
        for row in trip.track.itertuples(index=False):
            lon, lat, timestamp = float(row.longitude), float(row.latitude), float(row.timestamp)
            if not np.isfinite([lon, lat, timestamp]).all():
                raise ValueError("Trip contains a non-finite GPS coordinate or timestamp")
            gps_points.append({"lat": lat, "lng": lon, "timestamp": timestamp})
        if not gps_points:
            raise ValueError("Trip has no GPS observations")

        chunks = list(iter_candidate_features(trip, summary={}))
        if not chunks:
            raise ValueError("No road candidates could be generated for this trip")
        candidates = pd.concat(chunks, ignore_index=True)
        features = candidates[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
        class_values = np.asarray(model.classes_)
        positive_columns = np.flatnonzero(class_values == 1)
        if len(positive_columns) != 1 or not hasattr(model, "predict_proba"):
            raise ModelNotReadyError("The configured classifier must provide class-1 probabilities")
        candidates["_model_probability"] = model.predict_proba(features)[:, int(positive_columns[0])]

        node_lonlat = trip.nodes[["longitude", "latitude"]].to_numpy(dtype=float)
        arc_endpoints = trip.arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=float)
        decoded, diagnostics = decode_trip_candidates(
            candidates,
            trip.track,
            node_lonlat,
            arc_endpoints,
            **decoder_parameters,
        )
        chosen = decoded.merge(
            candidates[["obs_idx", "candidate_arc_id", "projected_fraction"]],
            left_on=["obs_idx", "hmm_arc_id"],
            right_on=["obs_idx", "candidate_arc_id"],
            how="left",
            validate="one_to_one",
        ).sort_values("obs_idx", kind="stable")

        route_points = []
        for row in chosen.itertuples(index=False):
            arc_id = int(row.hmm_arc_id)
            fraction = float(row.projected_fraction)
            if not (0 <= arc_id < len(arc_endpoints)) or not np.isfinite(fraction):
                continue
            endpoints = arc_endpoints[arc_id]
            if not np.isfinite(endpoints).all() or not np.equal(endpoints, np.floor(endpoints)).all():
                continue
            start_node, end_node = map(int, endpoints)
            if not (0 <= start_node < len(node_lonlat) and 0 <= end_node < len(node_lonlat)):
                continue
            start_lonlat, end_lonlat = node_lonlat[start_node], node_lonlat[end_node]
            if not np.isfinite(start_lonlat).all() or not np.isfinite(end_lonlat).all():
                continue
            lonlat = start_lonlat + np.clip(fraction, 0.0, 1.0) * (end_lonlat - start_lonlat)
            route_points.append({"lat": float(lonlat[1]), "lng": float(lonlat[0])})

        if not route_points:
            raise ValueError("The decoder did not return any valid matched road points")
        elapsed = time.perf_counter() - started
        return {
            "trip_id": normalized,
            "gps_points": gps_points,
            "predicted_route": route_points,
            "road_network": [],
            "metrics": {
                "inference_seconds": round(float(elapsed), 4),
                "matched_observations": int(len(route_points)),
                "total_observations": int(len(gps_points)),
                "coverage_ratio": float(len(route_points) / len(gps_points)),
                "large_gap_resets": int(diagnostics["large_gap_resets"]),
            },
        }
    except ModelNotReadyError:
        raise
    except Exception as exc:
        logger.exception("Prediction failed for trip %s", normalized)
        raise ValueError(f"Could not generate a route for trip {normalized}: {exc}") from exc
