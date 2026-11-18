"""Read PathSnap trips without mutating their source files.

The loader keeps reference routes separate from model inputs.  A ``.route``
file is loaded only as reference information; callers must explicitly access
``TripData.route`` when evaluating a prediction.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Literal

import pandas as pd


PartitionName = Literal["train", "student_test"]
Severity = Literal["error", "warning"]

PARTITIONS: tuple[PartitionName, ...] = ("train", "student_test")
TRIP_ID_PATTERN = re.compile(r"^\d{8}$")
FILE_COLUMNS: Mapping[str, tuple[str, ...]] = {
    ".track": ("longitude", "latitude", "timestamp"),
    ".nodes": ("longitude", "latitude"),
    ".arcs": ("from_node_id", "to_node_id"),
    ".route": ("arc_id",),
}


@dataclass(frozen=True)
class DataIssue:
    """A non-fatal problem encountered while loading or validating a trip."""

    severity: Severity
    code: str
    message: str
    partition: str | None = None
    trip_id: str | None = None
    file_type: str | None = None
    path: Path | None = None
    count: int = 1
    sample_rows: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return a serialisable representation for logs or reports."""
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "partition": self.partition,
            "trip_id": self.trip_id,
            "file_type": self.file_type,
            "path": str(self.path) if self.path else None,
            "count": self.count,
            "sample_rows": list(self.sample_rows),
        }


@dataclass
class TripData:
    """The tables for one trip, plus any recoverable loading issues.

    ``route`` is deliberately not included by :meth:`model_inputs`; it is a
    reference label and must remain outside feature-generation code.
    """

    partition: str
    trip_id: str
    trip_dir: Path
    track: pd.DataFrame | None
    nodes: pd.DataFrame | None
    arcs: pd.DataFrame | None
    route: pd.DataFrame | None
    source_paths: dict[str, Path]
    issues: list[DataIssue] = field(default_factory=list)

    def model_inputs(self, *, copy: bool = True) -> dict[str, pd.DataFrame | None]:
        """Return GPS and road-network inputs only, never reference routes."""
        frames = {"track": self.track, "nodes": self.nodes, "arcs": self.arcs}
        if not copy:
            return frames
        return {
            name: frame.copy(deep=True) if frame is not None else None
            for name, frame in frames.items()
        }


@dataclass
class DatasetLoadResult:
    """Trips and discovery/loading issues returned by :func:`load_dataset`."""

    trips: list[TripData]
    issues: list[DataIssue] = field(default_factory=list)

    @property
    def trips_by_partition(self) -> dict[str, list[TripData]]:
        grouped: dict[str, list[TripData]] = {}
        for trip in self.trips:
            grouped.setdefault(trip.partition, []).append(trip)
        return grouped

    @property
    def all_issues(self) -> list[DataIssue]:
        """Include dataset-level issues and issues attached to individual trips."""
        return [*self.issues, *(issue for trip in self.trips for issue in trip.issues)]


def is_valid_trip_id(trip_id: str) -> bool:
    """Return whether ``trip_id`` matches the verified eight-digit format."""
    return bool(TRIP_ID_PATTERN.fullmatch(trip_id))


def _issue(
    severity: Severity,
    code: str,
    message: str,
    *,
    partition: str | None = None,
    trip_id: str | None = None,
    file_type: str | None = None,
    path: Path | None = None,
) -> DataIssue:
    return DataIssue(
        severity=severity,
        code=code,
        message=message,
        partition=partition,
        trip_id=trip_id,
        file_type=file_type,
        path=path,
    )


def _coerce_numeric_columns(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Convert expected numeric columns while retaining every parsed row."""
    result = frame.copy()
    for column in result.columns:
        values = pd.to_numeric(result[column], errors="coerce")
        if suffix in {".arcs", ".route"}:
            integral = values.isna() | values.mod(1).eq(0)
            values = values.where(integral, pd.NA)
            result[column] = values.astype("Int64")
        else:
            result[column] = values.astype("float64")
    return result


def _read_table(
    path: Path,
    suffix: str,
    *,
    partition: str,
    trip_id: str,
) -> tuple[pd.DataFrame | None, DataIssue | None]:
    """Read one whitespace-delimited, headerless file without skipping rows."""
    columns = FILE_COLUMNS[suffix]
    try:
        frame = pd.read_csv(path, sep=r"\s+", header=None, on_bad_lines="error")
    except pd.errors.EmptyDataError:
        return None, _issue(
            "error",
            "empty_file",
            f"{path.name} is empty.",
            partition=partition,
            trip_id=trip_id,
            file_type=suffix,
            path=path,
        )
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
        return None, _issue(
            "error",
            "malformed_file",
            f"Could not parse {path.name}: {exc}",
            partition=partition,
            trip_id=trip_id,
            file_type=suffix,
            path=path,
        )

    if frame.shape[1] != len(columns):
        return None, _issue(
            "error",
            "unexpected_column_count",
            f"{path.name} has {frame.shape[1]} columns; expected {len(columns)}.",
            partition=partition,
            trip_id=trip_id,
            file_type=suffix,
            path=path,
        )

    frame.columns = list(columns)
    return _coerce_numeric_columns(frame, suffix), None


def load_trip(trip_dir: str | Path, partition: str) -> TripData:
    """Load one trip directory and record missing or malformed files safely."""
    directory = Path(trip_dir)
    trip_id = directory.name
    issues: list[DataIssue] = []

    if not is_valid_trip_id(trip_id):
        issues.append(
            _issue(
                "error",
                "invalid_trip_id",
                f"Trip directory name {trip_id!r} is not an eight-digit trip ID.",
                partition=partition,
                trip_id=trip_id,
                path=directory,
            )
        )

    source_paths = {suffix: directory / f"{trip_id}{suffix}" for suffix in FILE_COLUMNS}
    tables: dict[str, pd.DataFrame | None] = {}
    for suffix, path in source_paths.items():
        if not path.is_file():
            tables[suffix] = None
            issues.append(
                _issue(
                    "error",
                    "missing_file",
                    f"Required file is missing: {path.name}",
                    partition=partition,
                    trip_id=trip_id,
                    file_type=suffix,
                    path=path,
                )
            )
            continue

        table, issue = _read_table(
            path,
            suffix,
            partition=partition,
            trip_id=trip_id,
        )
        tables[suffix] = table
        if issue is not None:
            issues.append(issue)

    return TripData(
        partition=partition,
        trip_id=trip_id,
        trip_dir=directory,
        track=tables[".track"],
        nodes=tables[".nodes"],
        arcs=tables[".arcs"],
        route=tables[".route"],
        source_paths=source_paths,
        issues=issues,
    )


def load_dataset(
    data_root: str | Path = "data",
    *,
    partitions: Iterable[str] = PARTITIONS,
) -> DatasetLoadResult:
    """Discover and load requested partitions from a PathSnap dataset root.

    The function only reads files.  It reports missing partitions, invalid
    directory names and malformed trip files instead of terminating at the
    first problem.
    """
    root = Path(data_root)
    result = DatasetLoadResult(trips=[])

    if not root.is_dir():
        result.issues.append(
            _issue(
                "error",
                "missing_data_root",
                f"Dataset root does not exist or is not a directory: {root}",
                path=root,
            )
        )
        return result

    for partition in partitions:
        partition_dir = root / partition
        if not partition_dir.is_dir():
            result.issues.append(
                _issue(
                    "error",
                    "missing_partition",
                    f"Partition directory is missing: {partition_dir}",
                    partition=partition,
                    path=partition_dir,
                )
            )
            continue

        trip_dirs = sorted(path for path in partition_dir.iterdir() if path.is_dir())
        for trip_dir in trip_dirs:
            result.trips.append(load_trip(trip_dir, partition))

    return result
