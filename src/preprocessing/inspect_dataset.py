"""Inventory the PathSnap development dataset without modifying source data.

Expected layout:
    data/train/<trip_id>/<trip_id>.(track|nodes|arcs|route)
    data/student_test/<trip_id>/<trip_id>.(track|nodes|arcs|route)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EXTENSIONS = (".track", ".nodes", ".arcs", ".route")


def read_table(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=columns, engine="python")


def inspect_trip(trip_dir: Path, partition: str) -> dict:
    trip_id = trip_dir.name
    paths = {ext: trip_dir / f"{trip_id}{ext}" for ext in EXTENSIONS}
    present = {ext: path.is_file() for ext, path in paths.items()}
    row = {
        "partition": partition,
        "trip_id": trip_id,
        **{f"has_{ext[1:]}": present[ext] for ext in EXTENSIONS},
        **{f"{ext[1:]}_bytes": paths[ext].stat().st_size if present[ext] else None
           for ext in EXTENSIONS},
    }

    if present[".track"]:
        gps = read_table(paths[".track"], ["longitude", "latitude", "timestamp"])
        invalid = (
            ~gps["longitude"].between(-180, 180)
            | ~gps["latitude"].between(-90, 90)
        )
        gaps = gps["timestamp"].diff().dropna()
        row.update({
            "gps_observations": len(gps),
            "missing_gps_values": int(gps.isna().sum().sum()),
            "invalid_coordinate_rows": int(invalid.sum()),
            "timestamp_monotonic": bool(gps["timestamp"].is_monotonic_increasing),
            "timestamp_gap_count_nonpositive": int((gaps <= 0).sum()),
            "timestamp_gap_count_not_1s": int((gaps != 1).sum()),
            "median_timestamp_gap": float(gaps.median()) if len(gaps) else None,
            "longitude_min": float(gps["longitude"].min()) if len(gps) else None,
            "longitude_max": float(gps["longitude"].max()) if len(gps) else None,
            "latitude_min": float(gps["latitude"].min()) if len(gps) else None,
            "latitude_max": float(gps["latitude"].max()) if len(gps) else None,
        })
    else:
        row["gps_observations"] = None

    specs = {
        ".nodes": ("road_nodes", ["longitude", "latitude"]),
        ".arcs": ("road_arcs", ["from_node_id", "to_node_id"]),
        ".route": ("reference_route_length", ["arc_id"]),
    }
    for ext, (field, columns) in specs.items():
        row[field] = len(read_table(paths[ext], columns)) if present[ext] else None
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eda"))
    args = parser.parse_args()

    records = []
    for partition in ("train", "student_test"):
        folder = args.data_root / partition
        if not folder.is_dir():
            print(f"WARNING: missing partition directory: {folder}")
            continue
        for trip_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            records.append(inspect_trip(trip_dir, partition))

    if not records:
        raise SystemExit(f"No trips found under {args.data_root}; expected train/ and student_test/.")

    inventory = pd.DataFrame(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(args.output_dir / "trip_inventory.csv", index=False)

    summary = (
        inventory.groupby("partition")
        .agg(
            trip_count=("trip_id", "nunique"),
            total_gps_observations=("gps_observations", "sum"),
            mean_gps_observations=("gps_observations", "mean"),
            total_road_nodes=("road_nodes", "sum"),
            total_road_arcs=("road_arcs", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / "partition_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved exploration outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
