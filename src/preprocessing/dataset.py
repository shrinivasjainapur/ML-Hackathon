"""Typed readers and validation helpers for the PathSnap trip files.

The supplied files are headerless whitespace-delimited tables.  This module
keeps row order intact because arc IDs and node IDs are zero-based row indices.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


class DatasetFormatError(ValueError):
    """Raised when a trip file cannot be interpreted safely."""


@dataclass(frozen=True)
class TripData:
    trip_id: str
    partition: str
    directory: Path
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: pd.DataFrame | None


FILE_COLUMNS: dict[str, list[str]] = {
    ".track": ["longitude", "latitude", "timestamp"],
    ".nodes": ["longitude", "latitude"],
    ".arcs": ["from_node_id", "to_node_id"],
    ".route": ["arc_id"],
}


def project_root() -> Path:
    """Return repository root (the parent of ``src``)."""
    return Path(__file__).resolve().parents[2]


def default_data_root() -> Path:
    """Find the dataset root used in this checkout, with README layout fallback."""
    root = project_root()
    for candidate in (root / "Student folder", root / "data"):
        if (candidate / "train").is_dir():
            return candidate
    return root / "Student folder"


def read_trip_file(path: Path, columns: list[str], *, required: bool = True) -> pd.DataFrame | None:
    """Read a headerless whitespace-separated table, preserving its row order."""
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required trip file is missing: {path}")
        return None
    try:
        table = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=columns,
            usecols=range(len(columns)),
            engine="python",
            on_bad_lines="error",
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)
    except Exception as exc:
        raise DatasetFormatError(f"Could not parse {path}: {exc}") from exc

    # usecols can hide extra columns. Check the raw field count explicitly so a
    # malformed file is not silently accepted.
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip() and len(line.split()) != len(columns):
                raise DatasetFormatError(
                    f"{path}:{line_number} has {len(line.split())} fields; expected {len(columns)}"
                )
    return table


def load_trip(
    trip_dir: Path,
    partition: str,
    *,
    require_route: bool = False,
    validate_indices: bool = True,
) -> TripData:
    """Load one trip and validate node/arc/route index relationships.

    ``.route`` is optional for inference data. It is loaded whenever present;
    callers that need supervised labels should set ``require_route=True``.
    """
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name
    if partition not in {"train", "student_test", "evaluator_test"}:
        raise ValueError(f"Unexpected partition name: {partition!r}")
    track = read_trip_file(trip_dir / f"{trip_id}.track", FILE_COLUMNS[".track"])
    nodes = read_trip_file(trip_dir / f"{trip_id}.nodes", FILE_COLUMNS[".nodes"])
    arcs = read_trip_file(trip_dir / f"{trip_id}.arcs", FILE_COLUMNS[".arcs"])
    route = read_trip_file(
        trip_dir / f"{trip_id}.route", FILE_COLUMNS[".route"], required=require_route
    )
    assert track is not None and nodes is not None and arcs is not None

    for frame in (track, nodes):
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for frame in (arcs, route):
        if frame is not None:
            for column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if validate_indices:
        arc_values = arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=float)
        if np.isfinite(arc_values).all() and len(arc_values):
            if not np.equal(arc_values, np.floor(arc_values)).all():
                raise DatasetFormatError(f"Non-integer node IDs in {trip_dir}")
            if (arc_values < 0).any() or (arc_values >= len(nodes)).any():
                raise DatasetFormatError(f"Arc references a node outside .nodes in {trip_dir}")
        if route is not None and len(route):
            ids = route["arc_id"].to_numpy(dtype=float)
            if np.isfinite(ids).all():
                if not np.equal(ids, np.floor(ids)).all():
                    raise DatasetFormatError(f"Non-integer arc IDs in {trip_dir}")
                if (ids < 0).any() or (ids >= len(arcs)).any():
                    raise DatasetFormatError(f"Route references an arc outside .arcs in {trip_dir}")

    # Stable row IDs make it possible to trace output features to source rows.
    track.insert(0, "obs_idx", np.arange(len(track), dtype=np.int64))
    nodes.insert(0, "node_id", np.arange(len(nodes), dtype=np.int64))
    arcs.insert(0, "arc_id", np.arange(len(arcs), dtype=np.int64))
    return TripData(trip_id, partition, trip_dir, track, nodes, arcs, route)
