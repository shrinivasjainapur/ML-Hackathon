"""Build point-to-road candidate features from raw PathSnap trips.

Run from the repository root:
    python -m src.features.candidate_features

The output is a candidate-level table: each row is one GPS observation paired
with one nearby directed road arc. Reference targets are derived by the
monotone route-to-GPS alignment in ``src.preprocessing.label_alignment``.
Identifiers and target columns are metadata; model code must select only the
numeric columns listed in ``FEATURE_COLUMNS``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from src.preprocessing.dataset import TripData, default_data_root, load_trip
from src.preprocessing.label_alignment import align_reference_route


FEATURE_COLUMNS = [
    "distance_to_segment_m",
    "projected_fraction",
    "segment_length_m",
    "distance_to_start_m",
    "distance_to_end_m",
    "heading_cosine",
    "heading_available",
    "gps_speed_mps",
    "time_gap_seconds",
    "candidate_rank",
    "candidate_count",
    "nearest_fallback_used",
]
METADATA_COLUMNS = [
    "partition", "trip_id", "obs_idx", "candidate_arc_id", "reference_arc_id",
    "reference_route_position", "reference_alignment_distance_m", "reference_alignment_confident",
    "is_reference_candidate", "has_reference_candidate",
]
OUTPUT_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS


def _local_transformer(lon: np.ndarray, lat: np.ndarray) -> Transformer:
    valid = np.isfinite(lon) & np.isfinite(lat) & (lon >= -180) & (lon <= 180) & (lat >= -90) & (lat <= 90)
    if not valid.any():
        raise ValueError("Trip has no valid coordinates from which to define a local projection")
    lon0 = float(np.median(lon[valid]))
    lat0 = float(np.median(lat[valid]))
    local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0:.8f} +lon_0={lon0:.8f} +datum=WGS84 +units=m +no_defs"
    )
    return Transformer.from_crs("EPSG:4326", local, always_xy=True)


def _project(lon: np.ndarray, lat: np.ndarray, transformer: Transformer) -> np.ndarray:
    xy = np.full((len(lon), 2), np.nan, dtype=float)
    valid = np.isfinite(lon) & np.isfinite(lat) & (lon >= -180) & (lon <= 180) & (lat >= -90) & (lat <= 90)
    if valid.any():
        x, y = transformer.transform(lon[valid], lat[valid])
        xy[valid, 0], xy[valid, 1] = x, y
    return xy


def _road_geometry(
    trip: TripData, node_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, STRtree, np.ndarray]:
    """Return valid arc IDs/endpoints and an STRtree of projected segments."""
    endpoints_all = trip.arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=float)
    integral = np.isfinite(endpoints_all).all(axis=1) & np.equal(endpoints_all, np.floor(endpoints_all)).all(axis=1)
    valid = integral.copy()
    valid &= (endpoints_all[:, 0] >= 0) & (endpoints_all[:, 1] >= 0)
    valid &= (endpoints_all[:, 0] < len(node_xy)) & (endpoints_all[:, 1] < len(node_xy))
    arc_ids = np.flatnonzero(valid).astype(np.int64)
    endpoints = endpoints_all[valid].astype(np.int64)
    finite_nodes = np.isfinite(node_xy).all(axis=1)
    keep = finite_nodes[endpoints].all(axis=1) if len(endpoints) else np.zeros(0, dtype=bool)
    arc_ids, endpoints = arc_ids[keep], endpoints[keep]
    if not len(arc_ids):
        raise ValueError(f"Trip {trip.trip_id}: no arcs have valid endpoint coordinates")
    starts, ends = node_xy[endpoints[:, 0]], node_xy[endpoints[:, 1]]
    lengths = np.linalg.norm(ends - starts, axis=1)
    nonzero = lengths > 1e-6
    arc_ids, endpoints = arc_ids[nonzero], endpoints[nonzero]
    starts, ends, lengths = starts[nonzero], ends[nonzero], lengths[nonzero]
    geometries = np.asarray([LineString((a, b)) for a, b in zip(starts, ends)], dtype=object)
    return arc_ids, endpoints, lengths, STRtree(geometries), geometries


def _route_alignment(
    trip: TripData,
    track_xy: np.ndarray,
    node_xy: np.ndarray,
    *,
    confidence_threshold_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str | None]:
    n = len(track_xy)
    no_route = (np.full(n, -1, dtype=np.int64), np.full(n, -1, dtype=np.int64),
                np.full(n, np.nan, dtype=float), np.zeros(n, dtype=bool), "route file unavailable")
    if trip.route is None or trip.route.empty:
        return no_route
    try:
        all_endpoints = trip.arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=float)
        route_values = pd.to_numeric(trip.route["arc_id"], errors="coerce").to_numpy(dtype=float)
        if (not np.isfinite(route_values).all()
                or not np.equal(route_values, np.floor(route_values)).all()
                or (route_values < 0).any()
                or (route_values >= len(all_endpoints)).any()):
            raise ValueError("route has missing, non-integer, or out-of-range arc IDs")
        if not np.isfinite(all_endpoints).all() or not np.equal(all_endpoints, np.floor(all_endpoints)).all():
            raise ValueError(".arcs has missing or non-integer endpoint IDs")
        endpoints = all_endpoints.astype(np.int64)
        positions, arc_ids, distances = align_reference_route(
            track_xy, node_xy, endpoints, route_values.astype(np.int64)
        )
        confident = np.isfinite(distances) & (distances <= confidence_threshold_m)
        return positions, arc_ids, distances, confident, None
    except Exception as exc:
        return no_route[0], no_route[1], no_route[2], no_route[3], str(exc)


def iter_candidate_features(
    trip: TripData,
    *,
    search_radius_m: float = 80.0,
    max_candidates: int = 32,
    alignment_confidence_m: float = 50.0,
    chunk_rows: int = 8000,
    summary: dict[str, object] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield candidate rows in bounded chunks and populate per-trip summary."""
    if search_radius_m <= 0 or max_candidates < 1 or alignment_confidence_m <= 0 or chunk_rows < 1:
        raise ValueError("search radius, candidate limit, confidence threshold, and chunk size must be positive")
    summary = summary if summary is not None else {}
    track = trip.track
    node_lon = trip.nodes["longitude"].to_numpy(dtype=float)
    node_lat = trip.nodes["latitude"].to_numpy(dtype=float)
    track_lon = track["longitude"].to_numpy(dtype=float)
    track_lat = track["latitude"].to_numpy(dtype=float)
    transformer = _local_transformer(
        np.concatenate([track_lon, node_lon]), np.concatenate([track_lat, node_lat])
    )
    track_xy = _project(track_lon, track_lat, transformer)
    node_xy = _project(node_lon, node_lat, transformer)
    arc_ids, endpoints, _, tree, geometries = _road_geometry(trip, node_xy)
    starts, ends = node_xy[endpoints[:, 0]], node_xy[endpoints[:, 1]]
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    lengths_sq = np.maximum(np.einsum("ij,ij->i", vectors, vectors), 1e-12)

    route_pos, reference_arc, alignment_distance, alignment_confident, alignment_error = _route_alignment(
        trip, track_xy, node_xy, confidence_threshold_m=alignment_confidence_m
    )
    timestamps = pd.to_numeric(track["timestamp"], errors="coerce").to_numpy(dtype=float)
    obs_idx = track["obs_idx"].to_numpy(dtype=np.int64)
    valid_track = np.isfinite(track_xy).all(axis=1)
    rows: list[dict[str, object]] = []
    candidate_observations = 0
    candidate_rows = 0
    matched_observations = 0
    candidate_misses = 0
    raw_alignment_hits = 0
    raw_alignment_misses = 0
    no_road_candidates = 0
    reference_alignment_observations = 0
    confident_reference_observations = 0
    unconfident_reference_observations = 0
    confident_candidate_rows = 0
    unconfident_candidate_rows = 0
    eligible_training_candidate_rows = 0
    radius_fallback_observations = 0
    alignment_distances: list[float] = []

    for i in range(len(track)):
        if not valid_track[i]:
            continue
        point_xy = track_xy[i]
        point = Point(float(point_xy[0]), float(point_xy[1]))
        found = np.asarray(tree.query(point.buffer(search_radius_m), predicate="intersects"), dtype=np.int64)
        used_fallback = not len(found)
        if not len(found):
            nearest = tree.nearest(point)
            if nearest is None:
                no_road_candidates += 1
                continue
            found = np.asarray([int(nearest)], dtype=np.int64)
        if used_fallback:
            radius_fallback_observations += 1
        candidate_arc_ids = arc_ids[found]
        a = starts[found]
        v = vectors[found]
        l2 = lengths_sq[found]
        delta = point_xy[None, :] - a
        fraction = np.clip(np.einsum("ij,ij->i", delta, v) / l2, 0.0, 1.0)
        closest = a + fraction[:, None] * v
        distance = np.linalg.norm(point_xy[None, :] - closest, axis=1)
        order = np.argsort(distance, kind="stable")[:max_candidates]
        found = found[order]
        candidate_arc_ids = candidate_arc_ids[order]
        fraction = fraction[order]
        distance = distance[order]
        candidate_count = len(order)
        candidate_observations += 1
        candidate_rows += candidate_count

        has_alignment = reference_arc[i] >= 0
        is_confident = bool(has_alignment and alignment_confident[i])
        if has_alignment:
            reference_alignment_observations += 1
            alignment_distances.append(float(alignment_distance[i]))
            if np.any(candidate_arc_ids == reference_arc[i]):
                raw_alignment_hits += 1
            else:
                raw_alignment_misses += 1
        if is_confident:
            confident_reference_observations += 1
            confident_candidate_rows += candidate_count
            if np.any(candidate_arc_ids == reference_arc[i]):
                matched_observations += 1
                eligible_training_candidate_rows += candidate_count
            else:
                candidate_misses += 1
        elif has_alignment:
            unconfident_reference_observations += 1
            unconfident_candidate_rows += candidate_count

        has_previous = i > 0 and valid_track[i - 1] and np.isfinite(timestamps[i]) and np.isfinite(timestamps[i - 1])
        dt = float(timestamps[i] - timestamps[i - 1]) if has_previous else 0.0
        if has_previous and dt > 0:
            movement = point_xy - track_xy[i - 1]
            movement_len = float(np.linalg.norm(movement))
            speed = movement_len / dt
            has_heading = movement_len > 1e-6
        else:
            movement = np.zeros(2, dtype=float)
            speed, has_heading = 0.0, False

        selected_starts = starts[found]
        selected_vectors = vectors[found]
        selected_lengths = lengths[found]
        headings = np.arctan2(selected_vectors[:, 1], selected_vectors[:, 0])
        if has_heading:
            observed_heading = math.atan2(float(movement[1]), float(movement[0]))
            heading_cosine = np.cos(headings - observed_heading)
        else:
            heading_cosine = np.zeros(candidate_count, dtype=float)
        start_distance = np.linalg.norm(point_xy[None, :] - selected_starts, axis=1)
        end_distance = np.linalg.norm(point_xy[None, :] - (selected_starts + selected_vectors), axis=1)
        reference_id = int(reference_arc[i]) if has_alignment else np.nan
        ref_position = int(route_pos[i]) if has_alignment else np.nan
        ref_distance = float(alignment_distance[i]) if has_alignment else np.nan
        confidence_value = int(is_confident) if has_alignment else np.nan
        hit = bool(np.any(candidate_arc_ids == reference_arc[i])) if is_confident else np.nan

        for rank, (arc_id, dist, frac, d_start, d_end, heading_cos) in enumerate(
            zip(candidate_arc_ids, distance, fraction, start_distance, end_distance, heading_cosine),
            start=1,
        ):
            rows.append({
                "partition": trip.partition,
                "trip_id": trip.trip_id,
                "obs_idx": int(obs_idx[i]),
                "candidate_arc_id": int(arc_id),
                "reference_arc_id": reference_id,
                "reference_route_position": ref_position,
                "reference_alignment_distance_m": ref_distance,
                "reference_alignment_confident": confidence_value,
                "is_reference_candidate": int(arc_id == reference_arc[i]) if is_confident else np.nan,
                "has_reference_candidate": int(hit) if is_confident else np.nan,
                "distance_to_segment_m": float(dist),
                "projected_fraction": float(frac),
                "segment_length_m": float(selected_lengths[rank - 1]),
                "distance_to_start_m": float(d_start),
                "distance_to_end_m": float(d_end),
                "heading_cosine": float(heading_cos),
                "heading_available": int(has_heading),
                "gps_speed_mps": float(speed),
                "time_gap_seconds": max(dt, 0.0),
                "candidate_rank": rank,
                "candidate_count": candidate_count,
                "nearest_fallback_used": int(used_fallback),
            })
        if len(rows) >= chunk_rows:
            yield pd.DataFrame.from_records(rows, columns=OUTPUT_COLUMNS)
            rows.clear()

    if rows:
        yield pd.DataFrame.from_records(rows, columns=OUTPUT_COLUMNS)
    summary.update({
        "partition": trip.partition,
        "trip_id": trip.trip_id,
        "gps_observations": int(len(track)),
        "valid_gps_observations": int(valid_track.sum()),
        "candidate_observations": int(candidate_observations),
        "candidate_rows": int(candidate_rows),
        "reference_alignment_observations": int(reference_alignment_observations),
        "confident_reference_observations": int(confident_reference_observations),
        "unconfident_reference_observations": int(unconfident_reference_observations),
        "confident_candidate_rows": int(confident_candidate_rows),
        "unconfident_candidate_rows": int(unconfident_candidate_rows),
        "eligible_training_candidate_rows": int(eligible_training_candidate_rows),
        "reference_candidate_hits": int(matched_observations),
        "reference_candidate_misses": int(candidate_misses),
        "raw_reference_candidate_hits": int(raw_alignment_hits),
        "raw_reference_candidate_misses": int(raw_alignment_misses),
        "reference_candidate_recall": (
            float(matched_observations / confident_reference_observations)
            if confident_reference_observations else np.nan
        ),
        "raw_reference_candidate_recall": (
            float(raw_alignment_hits / reference_alignment_observations)
            if reference_alignment_observations else np.nan
        ),
        "reference_alignment_confidence_coverage": (
            float(confident_reference_observations / reference_alignment_observations)
            if reference_alignment_observations else np.nan
        ),
        "median_reference_alignment_distance_m": (
            float(np.median(alignment_distances)) if alignment_distances else np.nan
        ),
        "p95_reference_alignment_distance_m": (
            float(np.percentile(alignment_distances, 95)) if alignment_distances else np.nan
        ),
        "alignment_error": alignment_error or "",
        "observations_without_road_candidates": int(no_road_candidates),
        "observations_using_nearest_fallback": int(radius_fallback_observations),
        "alignment_confidence_threshold_m": float(alignment_confidence_m),
    })


def build_partition(
    data_root: Path,
    output_dir: Path,
    partition: str,
    *,
    max_trips: int | None,
    search_radius_m: float,
    max_candidates: int,
    alignment_confidence_m: float,
) -> tuple[Path, Path]:
    folder = data_root / partition
    if not folder.is_dir():
        raise FileNotFoundError(f"Partition directory not found: {folder}")
    trip_dirs = sorted(p for p in folder.iterdir() if p.is_dir())
    if max_trips is not None:
        trip_dirs = trip_dirs[:max_trips]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{partition}_candidates.csv.gz"
    if output_path.exists():
        output_path.unlink()
    summaries: list[dict[str, object]] = []
    wrote_header = False
    for index, trip_dir in enumerate(trip_dirs, start=1):
        try:
            trip = load_trip(trip_dir, partition, require_route=False)
            trip_summary: dict[str, object] = {}
            for chunk in iter_candidate_features(
                trip,
                search_radius_m=search_radius_m,
                max_candidates=max_candidates,
                alignment_confidence_m=alignment_confidence_m,
                summary=trip_summary,
            ):
                chunk.to_csv(
                    output_path,
                    index=False,
                    mode="a" if wrote_header else "w",
                    header=not wrote_header,
                    compression="gzip",
                )
                wrote_header = True
            summaries.append(trip_summary)
            print(f"[{index}/{len(trip_dirs)}] {partition}/{trip.trip_id}: "
                  f"{trip_summary.get('candidate_rows', 0)} candidate rows; "
                  f"confident recall={trip_summary.get('reference_candidate_recall', np.nan):.3f}; "
                  f"label coverage={trip_summary.get('reference_alignment_confidence_coverage', np.nan):.3f}")
        except Exception as exc:
            summaries.append({"partition": partition, "trip_id": trip_dir.name, "error": str(exc)})
            print(f"WARNING: skipped {partition}/{trip_dir.name}: {exc}")
    if not wrote_header:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_path, index=False, compression="gzip")
    summary_path = output_dir / f"{partition}_candidate_summary.csv"
    pd.DataFrame.from_records(summaries).to_csv(summary_path, index=False)
    return output_path, summary_path


def render_feature_summary(
    summary_paths: list[Path],
    *,
    search_radius_m: float,
    max_candidates: int,
    alignment_confidence_m: float,
) -> str:
    """Summarize candidate coverage and the selected feature schema."""
    sections: list[str] = []
    for summary_path in summary_paths:
        if not summary_path.is_file():
            continue
        table = pd.read_csv(summary_path, dtype={"trip_id": str})
        partition = summary_path.name.removesuffix("_candidate_summary.csv")
        if table.empty:
            sections.append(f"### {partition}\n\nNo trip summaries were produced.\n")
            continue
        def total(column: str) -> float:
            if column not in table:
                return 0.0
            return float(pd.to_numeric(table[column], errors="coerce").fillna(0).sum())

        confident = total("confident_reference_observations")
        aligned = total("reference_alignment_observations")
        hits = total("reference_candidate_hits")
        misses = total("reference_candidate_misses")
        candidate_rows = total("candidate_rows")
        unconfident = total("unconfident_reference_observations")
        eligible_rows = total("eligible_training_candidate_rows")
        fallback = total("observations_using_nearest_fallback")
        errors = int(table.get("error", pd.Series(dtype=str)).notna().sum()) if "error" in table else 0
        recall = hits / confident if confident else np.nan
        coverage = confident / aligned if aligned else np.nan
        sections.append(
            f"### {partition}\n\n"
            f"- Trips processed: **{table['trip_id'].nunique()}**; trip errors: **{errors}**.\n"
            f"- Candidate rows: **{int(candidate_rows):,}**.\n"
            f"- Route-aligned observations: **{int(aligned):,}**; confident labels: **{int(confident):,}**; "
            f"low-confidence observations left unlabeled: **{int(unconfident):,}**.\n"
            f"- Candidate rows retained for candidate-classifier training after confidence/candidate filtering: **{int(eligible_rows):,}**.\n"
            f"- Candidate recall among confident alignments: **{recall:.4%}** "
            f"({int(hits):,} hits, {int(misses):,} misses).\n"
            f"- Confidence coverage among route-aligned observations: **{coverage:.4%}**.\n"
            f"- Observations requiring nearest-road fallback: **{int(fallback):,}**.\n"
        )

    feature_lines = [
        "| Feature | Meaning |",
        "| --- | --- |",
    ]
    for name, meaning in {
        "distance_to_segment_m": "GPS-to-road perpendicular distance in metres",
        "projected_fraction": "position along the directed segment, from 0 to 1",
        "segment_length_m": "candidate road-segment length in metres",
        "distance_to_start_m": "distance to candidate segment's start node in metres",
        "distance_to_end_m": "distance to candidate segment's end node in metres",
        "heading_cosine": "cosine agreement between GPS movement and directed road bearing",
        "heading_available": "whether the previous valid GPS point supports a bearing",
        "gps_speed_mps": "consecutive-point speed in metres per second",
        "time_gap_seconds": "elapsed time from the preceding GPS observation",
        "candidate_rank": "distance rank among returned candidates",
        "candidate_count": "number of returned candidates for this observation",
        "nearest_fallback_used": "1 when no road intersects the search radius and nearest road is used",
    }.items():
        feature_lines.append(f"| `{name}` | {meaning} |")

    return f"""# Candidate feature preparation summary

Generated from the raw trips by `src.features.candidate_features`.

## Configuration

- Candidate search radius: **{search_radius_m:g} m**.
- Maximum returned candidates per observation: **{max_candidates}**.
- Route-alignment confidence threshold: **{alignment_confidence_m:g} m**.
- Distances and bearings are computed in a per-trip local azimuthal-equidistant projection.
- Candidate features are hand-designed geometry/motion features; no learned scaler is fitted here.

## Results

{''.join(sections)}

## Feature definitions

{chr(10).join(feature_lines)}

`is_reference_candidate`, `has_reference_candidate`, arc IDs, route positions,
and trip/observation IDs are labels or metadata. Do not pass them as model
features. For training, retain confident observations with an available
positive candidate; report candidate recall separately from classifier
accuracy. The route-to-observation alignment is an assumption, not an
organizer-provided pointwise ground truth.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root(),
                        help='Folder containing train/ and student_test/ (default: "Student folder")')
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/features"))
    parser.add_argument("--partitions", nargs="+", choices=["train", "student_test"],
                        default=["train", "student_test"])
    parser.add_argument("--max-trips", type=int, default=None,
                        help="Optional per-partition cap for a quicker development run")
    parser.add_argument("--search-radius-m", type=float, default=80.0)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--alignment-confidence-m", type=float, default=50.0,
                        help="Do not emit training targets when aligned route distance exceeds this threshold")
    args = parser.parse_args()
    if args.max_trips is not None and args.max_trips < 1:
        parser.error("--max-trips must be positive")
    all_summaries: list[Path] = []
    for partition in args.partitions:
        feature_path, summary_path = build_partition(
            args.data_root,
            args.output_dir,
            partition,
            max_trips=args.max_trips,
            search_radius_m=args.search_radius_m,
            max_candidates=args.max_candidates,
            alignment_confidence_m=args.alignment_confidence_m,
        )
        all_summaries.append(summary_path)
        print(f"Saved features: {feature_path.resolve()}")
        print(f"Saved summary:  {summary_path.resolve()}")

    manifest = {
        "feature_columns": FEATURE_COLUMNS,
        "metadata_columns": METADATA_COLUMNS,
        "label_column": "is_reference_candidate",
        "candidate_generation": {
            "method": "projected Euclidean distance to directed road segments",
            "search_radius_m": args.search_radius_m,
            "max_candidates_per_observation": args.max_candidates,
            "fallback": "nearest road segment when no segment intersects the search radius",
            "nearest_fallback_used_feature": "nearest_fallback_used",
        },
        "reference_alignment": {
            "method": "monotone Viterbi-style alignment of ordered route arcs to GPS observations",
            "emission_scale_m": 25.0,
            "advance_penalty": 1.0,
            "confidence_threshold_m": args.alignment_confidence_m,
            "low_confidence_target_behavior": "reference candidate target is left blank when route alignment distance exceeds threshold",
            "caution": "Derived alignment is an assumption; review route alignment distances and candidate misses.",
        },
        "feature_selection": {
            "method": "hand-designed geometry and motion features",
            "scaling": "none during feature generation; fit any scaler within each training fold",
            "feature_rationales": {
                "distance_to_segment_m": "GPS-to-road perpendicular distance in metres",
                "projected_fraction": "location along directed road segment from its start, from 0 to 1",
                "segment_length_m": "length of the candidate segment in metres",
                "distance_to_start_m": "GPS distance to the segment's start node in metres",
                "distance_to_end_m": "GPS distance to the segment's end node in metres",
                "heading_cosine": "alignment of GPS movement direction and directed segment orientation",
                "heading_available": "whether a valid preceding GPS point supplies a movement direction",
                "gps_speed_mps": "speed between consecutive GPS observations in metres per second",
                "time_gap_seconds": "elapsed timestamp difference from the preceding observation",
                "candidate_rank": "distance rank among the returned candidates",
                "candidate_count": "number of candidates returned for this observation",
                "nearest_fallback_used": "whether only the nearest segment was available outside the search radius",
            },
        },
        "generated_summaries": [p.name for p in all_summaries],
        "student_test_is_development_data": True,
        "evaluator_test_included": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved manifest: {manifest_path.resolve()}")
    summary_markdown_path = args.output_dir / "feature_summary.md"
    summary_markdown_path.write_text(
        render_feature_summary(
            all_summaries,
            search_radius_m=args.search_radius_m,
            max_candidates=args.max_candidates,
            alignment_confidence_m=args.alignment_confidence_m,
        ),
        encoding="utf-8",
    )
    print(f"Saved feature summary: {summary_markdown_path.resolve()}")


if __name__ == "__main__":
    main()
