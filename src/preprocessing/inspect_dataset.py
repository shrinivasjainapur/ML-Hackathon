"""Create a per-trip inventory and partition summary for PathSnap.

Run from the repository root:
    python -m src.preprocessing.inspect_dataset
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import FILE_COLUMNS, default_data_root, read_trip_file


def _safe_read(path: Path, columns: list[str]) -> pd.DataFrame | None:
    try:
        return read_trip_file(path, columns, required=False)
    except Exception:
        return None


def inspect_trip(trip_dir: Path, partition: str) -> dict[str, object]:
    """Return quality and size fields for a trip, retaining malformed trips."""
    trip_id = trip_dir.name
    files = {ext: trip_dir / f"{trip_id}{ext}" for ext in FILE_COLUMNS}
    present = {ext: path.is_file() for ext, path in files.items()}
    row: dict[str, object] = {
        "partition": partition,
        "trip_id": trip_id,
        **{f"has_{ext[1:]}": present[ext] for ext in FILE_COLUMNS},
        **{f"{ext[1:]}_bytes": files[ext].stat().st_size if present[ext] else np.nan
           for ext in FILE_COLUMNS},
    }
    track = _safe_read(files[".track"], FILE_COLUMNS[".track"])
    nodes = _safe_read(files[".nodes"], FILE_COLUMNS[".nodes"])
    arcs = _safe_read(files[".arcs"], FILE_COLUMNS[".arcs"])
    route = _safe_read(files[".route"], FILE_COLUMNS[".route"])
    parse_errors = [
        ext[1:] for ext, frame in ((".track", track), (".nodes", nodes),
                                  (".arcs", arcs), (".route", route))
        if present[ext] and frame is None
    ]
    row["parse_errors"] = ";".join(parse_errors)

    if track is None:
        row.update({"gps_observations": np.nan, "missing_gps_cells": np.nan,
                    "invalid_coordinate_rows": np.nan, "timestamp_monotonic": np.nan,
                    "nonpositive_time_gaps": np.nan, "non_1s_time_gaps": np.nan,
                    "median_time_gap_s": np.nan, "longitude_min": np.nan,
                    "longitude_max": np.nan, "latitude_min": np.nan, "latitude_max": np.nan})
    else:
        for col in track.columns:
            track[col] = pd.to_numeric(track[col], errors="coerce")
        lon, lat, ts = track["longitude"], track["latitude"], track["timestamp"]
        invalid = (~np.isfinite(lon) | ~np.isfinite(lat)
                   | ~lon.between(-180, 180) | ~lat.between(-90, 90))
        gaps = ts.diff().dropna()
        row.update({
            "gps_observations": len(track),
            "missing_gps_cells": int(track.isna().sum().sum()),
            "invalid_coordinate_rows": int(invalid.sum()),
            "timestamp_monotonic": bool(ts.is_monotonic_increasing),
            "nonpositive_time_gaps": int((gaps <= 0).sum()),
            "non_1s_time_gaps": int((~np.isclose(gaps, 1.0, equal_nan=False)).sum()),
            "median_time_gap_s": float(gaps.median()) if len(gaps) else np.nan,
            "longitude_min": float(lon.min()) if lon.notna().any() else np.nan,
            "longitude_max": float(lon.max()) if lon.notna().any() else np.nan,
            "latitude_min": float(lat.min()) if lat.notna().any() else np.nan,
            "latitude_max": float(lat.max()) if lat.notna().any() else np.nan,
        })

    row["road_nodes"] = len(nodes) if nodes is not None else np.nan
    row["road_arcs"] = len(arcs) if arcs is not None else np.nan
    row["reference_route_length"] = len(route) if route is not None else np.nan
    invalid_node_coordinates = np.nan
    if nodes is not None and len(nodes):
        node_values = nodes[["longitude", "latitude"]].apply(pd.to_numeric, errors="coerce")
        node_lon, node_lat = node_values["longitude"], node_values["latitude"]
        invalid_node_coordinates = int((
            ~np.isfinite(node_lon) | ~np.isfinite(node_lat)
            | ~node_lon.between(-180, 180) | ~node_lat.between(-90, 90)
        ).sum())

    invalid_arc_endpoints = np.nan
    route_disconnected_transitions = np.nan
    if nodes is not None and arcs is not None and len(arcs):
        arc_values = arcs[["from_node_id", "to_node_id"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        finite_arcs = np.isfinite(arc_values).all(axis=1)
        integral_arcs = finite_arcs & np.equal(arc_values, np.floor(np.nan_to_num(arc_values))).all(axis=1)
        bad = (~integral_arcs
               | (arc_values[:, 0] < 0) | (arc_values[:, 1] < 0)
               | (arc_values[:, 0] >= len(nodes)) | (arc_values[:, 1] >= len(nodes)))
        invalid_arc_endpoints = int(bad.sum())
    if route is not None and arcs is not None and len(route):
        ids = pd.to_numeric(route["arc_id"], errors="coerce")
        route_values = ids.to_numpy(dtype=float)
        valid_route_ids = (
            np.isfinite(route_values)
            & np.equal(route_values, np.floor(np.nan_to_num(route_values)))
            & (route_values >= 0)
            & (route_values < len(arcs))
        )
        invalid_route_arc_ids = int((~valid_route_ids).sum())
        if valid_route_ids.all() and arcs is not None:
            arc_values = arcs[["from_node_id", "to_node_id"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            if np.isfinite(arc_values).all() and np.equal(arc_values, np.floor(arc_values)).all():
                route_endpoints = arc_values[route_values.astype(np.int64)]
                route_disconnected_transitions = (
                    int((route_endpoints[:-1, 1] != route_endpoints[1:, 0]).sum())
                    if len(route_endpoints) > 1 else 0
                )
    row["invalid_node_coordinate_rows"] = invalid_node_coordinates
    row["invalid_arc_endpoints"] = invalid_arc_endpoints
    row["invalid_route_arc_ids"] = invalid_route_arc_ids
    row["route_disconnected_transitions"] = route_disconnected_transitions
    track_count, route_count = row.get("gps_observations"), row.get("reference_route_length")
    if pd.notna(track_count) and pd.notna(route_count):
        row["track_route_count_equal"] = bool(track_count == route_count)
        row["route_to_track_count_ratio"] = float(route_count / track_count) if track_count else np.nan
    else:
        row["track_route_count_equal"] = np.nan
        row["route_to_track_count_ratio"] = np.nan
    return row


def build_inventory(data_root: Path, partitions: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for partition in partitions:
        folder = data_root / partition
        if not folder.is_dir():
            print(f"WARNING: partition directory not found: {folder}")
            continue
        for trip_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            records.append(inspect_trip(trip_dir, partition))
    if not records:
        raise FileNotFoundError(
            f"No trip folders found under {data_root}. Expected train/ and/or student_test/."
        )
    return pd.DataFrame.from_records(records)


def partition_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    return (inventory.groupby("partition", dropna=False)
            .agg(
                trip_count=("trip_id", "nunique"),
                trips_missing_track=("has_track", lambda s: int((~s).sum())),
                trips_missing_nodes=("has_nodes", lambda s: int((~s).sum())),
                trips_missing_arcs=("has_arcs", lambda s: int((~s).sum())),
                trips_missing_route=("has_route", lambda s: int((~s).sum())),
                total_gps_observations=("gps_observations", "sum"),
                median_gps_observations=("gps_observations", "median"),
                total_road_nodes=("road_nodes", "sum"),
                total_road_arcs=("road_arcs", "sum"),
                total_reference_arcs=("reference_route_length", "sum"),
                trips_with_parse_errors=("parse_errors", lambda s: int(s.ne("").sum())),
                invalid_node_coordinate_rows=("invalid_node_coordinate_rows", "sum"),
                invalid_arc_endpoints=("invalid_arc_endpoints", "sum"),
                invalid_route_arc_ids=("invalid_route_arc_ids", "sum"),
                disconnected_route_transitions=("route_disconnected_transitions", "sum"),
                trips_with_equal_track_route_counts=("track_route_count_equal", "sum"),
            ).reset_index())


def render_quality_summary(inventory: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Create a concise, data-derived narrative summary for the report."""
    total_trips = int(inventory["trip_id"].nunique())
    non_1s_gaps = int(pd.to_numeric(inventory["non_1s_time_gaps"], errors="coerce").sum())
    trips_with_non_1s_gaps = int(
        (pd.to_numeric(inventory["non_1s_time_gaps"], errors="coerce").fillna(0) > 0).sum()
    )
    equal_count_trips = int(
        pd.to_numeric(inventory["track_route_count_equal"], errors="coerce").fillna(0).sum()
    )
    route_breaks = int(
        pd.to_numeric(inventory["route_disconnected_transitions"], errors="coerce").sum()
    )
    total_route_transitions = int(
        (pd.to_numeric(inventory["reference_route_length"], errors="coerce").fillna(0) - 1)
        .clip(lower=0).sum()
    )
    broken_routes = inventory.loc[
        pd.to_numeric(inventory["route_disconnected_transitions"], errors="coerce").fillna(0) > 0,
        ["partition", "trip_id", "route_disconnected_transitions"],
    ]
    broken_route_lines = (
        "\n".join(
            f"- `{row.partition}/{row.trip_id}`: {int(row.route_disconnected_transitions)} transition(s)."
            for row in broken_routes.itertuples(index=False)
        )
        if len(broken_routes) else "- None found."
    )
    columns = list(summary.columns)
    table_lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for record in summary.fillna("").to_dict(orient="records"):
        table_lines.append("| " + " | ".join(str(record[column]) for column in columns) + " |")
    table = "\n".join(table_lines)
    return f"""# Dataset inventory and quality summary

Generated from the raw trip files by `src.preprocessing.inspect_dataset`.
This is a structural audit, not a model evaluation.

## Partition counts

{table}

## Data-quality findings

- Trips inventoried: **{total_trips}**.
- Missing required trip files and parse errors are counted in the partition table.
- Missing GPS cells: **{int(pd.to_numeric(inventory['missing_gps_cells'], errors='coerce').sum())}**.
- Invalid GPS coordinate rows: **{int(pd.to_numeric(inventory['invalid_coordinate_rows'], errors='coerce').sum())}**.
- Invalid road-node coordinate rows: **{int(pd.to_numeric(inventory['invalid_node_coordinate_rows'], errors='coerce').sum())}**.
- Invalid arc endpoint references: **{int(pd.to_numeric(inventory['invalid_arc_endpoints'], errors='coerce').sum())}**.
- Invalid reference-route arc IDs: **{int(pd.to_numeric(inventory['invalid_route_arc_ids'], errors='coerce').sum())}**.
- Non-positive timestamp gaps: **{int(pd.to_numeric(inventory['nonpositive_time_gaps'], errors='coerce').sum())}**.
- Timestamp gaps other than one second: **{non_1s_gaps}**, across **{trips_with_non_1s_gaps}** trips.
- Trips with equal `.track` and `.route` row counts: **{equal_count_trips}/{total_trips}**.
- Disconnected consecutive route-arc transitions: **{route_breaks}/{total_route_transitions}** transitions.
{broken_route_lines}

## Interpretation and limits

The `.route` file is an ordered sequence of road-arc IDs, while `.track` is a
sequence of GPS observations. Their row counts differ, so they cannot be
paired by row position. The route/observation association used later is a
derived spatial alignment and must be treated as an assumption. A few route
transitions also do not connect in the directed graph; inspect those trips
before treating every reference sequence as a continuous path.

Use `src.preprocessing.make_splits` for the deterministic trip-level holdout
and GroupKFold assignments. Do not split individual GPS observations across
training and validation.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root(),
                        help='Folder containing train/ and student_test/ (default: "Student folder")')
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eda"))
    parser.add_argument("--partitions", nargs="+", default=["train", "student_test"],
                        choices=["train", "student_test"])
    args = parser.parse_args()
    inventory = build_inventory(args.data_root, args.partitions)
    summary = partition_summary(inventory)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = args.output_dir / "trip_inventory.csv"
    summary_path = args.output_dir / "partition_summary.csv"
    narrative_path = args.output_dir / "data_quality_summary.md"
    inventory.to_csv(inventory_path, index=False)
    summary.to_csv(summary_path, index=False)
    narrative_path.write_text(render_quality_summary(inventory, summary), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved: {inventory_path.resolve()}")
    print(f"Saved: {summary_path.resolve()}")
    print(f"Saved: {narrative_path.resolve()}")


if __name__ == "__main__":
    main()
