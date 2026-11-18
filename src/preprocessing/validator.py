"""Validation for PathSnap GPS observations and road-network tables."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from .data_loader import DataIssue, TripData


SAMPLE_ROW_LIMIT = 10


@dataclass
class ValidationReport:
    """Issues found while validating one or more trips."""

    issues: list[DataIssue]

    @property
    def errors(self) -> list[DataIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[DataIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def counts_by_code(self) -> dict[str, int]:
        """Return the total affected-row count for each issue code."""
        totals: Counter[str] = Counter()
        for issue in self.issues:
            totals[issue.code] += issue.count
        return dict(totals)


def _rows(mask: pd.Series) -> tuple[int, ...]:
    return tuple(int(index) for index in mask[mask].index[:SAMPLE_ROW_LIMIT])


def _trip_issue(
    trip: TripData,
    severity: str,
    code: str,
    message: str,
    *,
    file_type: str | None = None,
    count: int = 1,
    sample_rows: tuple[int, ...] = (),
) -> DataIssue:
    return DataIssue(
        severity=severity,  # type: ignore[arg-type]
        code=code,
        message=message,
        partition=trip.partition,
        trip_id=trip.trip_id,
        file_type=file_type,
        path=trip.source_paths.get(file_type) if file_type else trip.trip_dir,
        count=count,
        sample_rows=sample_rows,
    )


def _validate_coordinates(
    trip: TripData,
    frame: pd.DataFrame,
    *,
    file_type: str,
    label: str,
) -> list[DataIssue]:
    issues: list[DataIssue] = []
    missing = frame[["longitude", "latitude"]].isna().any(axis=1)
    if missing.any():
        issues.append(
            _trip_issue(
                trip,
                "error",
                f"invalid_{label}_coordinates",
                f"{int(missing.sum())} {label} row(s) have missing or non-numeric coordinates.",
                file_type=file_type,
                count=int(missing.sum()),
                sample_rows=_rows(missing),
            )
        )

    out_of_range = (~missing) & (
        ~frame["longitude"].between(-180, 180)
        | ~frame["latitude"].between(-90, 90)
    )
    if out_of_range.any():
        issues.append(
            _trip_issue(
                trip,
                "error",
                f"out_of_range_{label}_coordinates",
                f"{int(out_of_range.sum())} {label} row(s) fall outside longitude/latitude bounds.",
                file_type=file_type,
                count=int(out_of_range.sum()),
                sample_rows=_rows(out_of_range),
            )
        )
    return issues


def validate_trip(trip: TripData) -> ValidationReport:
    """Validate one trip while retaining all observations and labels unchanged."""
    issues = list(trip.issues)

    if trip.track is not None:
        if trip.track.empty:
            issues.append(_trip_issue(trip, "error", "empty_track", "Track has no observations.", file_type=".track"))
        else:
            issues.extend(_validate_coordinates(trip, trip.track, file_type=".track", label="gps"))
            missing_timestamps = trip.track["timestamp"].isna()
            if missing_timestamps.any():
                issues.append(
                    _trip_issue(
                        trip,
                        "error",
                        "invalid_timestamps",
                        f"{int(missing_timestamps.sum())} GPS row(s) have missing or non-numeric timestamps.",
                        file_type=".track",
                        count=int(missing_timestamps.sum()),
                        sample_rows=_rows(missing_timestamps),
                    )
                )

            valid_timestamps = trip.track.loc[~missing_timestamps, "timestamp"]
            if len(valid_timestamps) > 1:
                gaps = valid_timestamps.diff().dropna()
                nonpositive = gaps <= 0
                if nonpositive.any():
                    issues.append(
                        _trip_issue(
                            trip,
                            "warning",
                            "timestamps_not_strictly_increasing",
                            f"{int(nonpositive.sum())} timestamp gap(s) are zero or negative.",
                            file_type=".track",
                            count=int(nonpositive.sum()),
                            sample_rows=tuple(int(index) for index in gaps[nonpositive].index[:SAMPLE_ROW_LIMIT]),
                        )
                    )
                irregular = gaps != 1
                if irregular.any():
                    issues.append(
                        _trip_issue(
                            trip,
                            "warning",
                            "irregular_timestamp_gaps",
                            f"{int(irregular.sum())} timestamp gap(s) are not one second.",
                            file_type=".track",
                            count=int(irregular.sum()),
                            sample_rows=tuple(int(index) for index in gaps[irregular].index[:SAMPLE_ROW_LIMIT]),
                        )
                    )
                if not valid_timestamps.is_monotonic_increasing:
                    issues.append(
                        _trip_issue(
                            trip,
                            "warning",
                            "timestamps_out_of_order",
                            "GPS timestamps are not in chronological order; preprocessing can sort a copy.",
                            file_type=".track",
                        )
                    )

    if trip.nodes is not None:
        if trip.nodes.empty:
            issues.append(_trip_issue(trip, "error", "empty_nodes", "Road network has no nodes.", file_type=".nodes"))
        else:
            issues.extend(_validate_coordinates(trip, trip.nodes, file_type=".nodes", label="node"))

    if trip.arcs is not None:
        if trip.arcs.empty:
            issues.append(_trip_issue(trip, "error", "empty_arcs", "Road network has no arcs.", file_type=".arcs"))
        elif trip.nodes is not None and not trip.nodes.empty:
            missing_endpoint = trip.arcs[["from_node_id", "to_node_id"]].isna().any(axis=1)
            if missing_endpoint.any():
                issues.append(
                    _trip_issue(
                        trip,
                        "error",
                        "invalid_arc_node_ids",
                        f"{int(missing_endpoint.sum())} arc row(s) have missing, non-numeric, or non-integer node IDs.",
                        file_type=".arcs",
                        count=int(missing_endpoint.sum()),
                        sample_rows=_rows(missing_endpoint),
                    )
                )
            valid_endpoint = ~missing_endpoint
            out_of_range = valid_endpoint & (
                ~trip.arcs["from_node_id"].between(0, len(trip.nodes) - 1)
                | ~trip.arcs["to_node_id"].between(0, len(trip.nodes) - 1)
            )
            if out_of_range.any():
                issues.append(
                    _trip_issue(
                        trip,
                        "error",
                        "arc_node_id_out_of_range",
                        f"{int(out_of_range.sum())} arc row(s) reference a node outside the node table.",
                        file_type=".arcs",
                        count=int(out_of_range.sum()),
                        sample_rows=_rows(out_of_range),
                    )
                )

    if trip.route is not None:
        if trip.route.empty:
            issues.append(_trip_issue(trip, "warning", "empty_route", "Reference route has no arc IDs.", file_type=".route"))
        elif trip.arcs is not None and not trip.arcs.empty:
            invalid_route_id = trip.route["arc_id"].isna() | ~trip.route["arc_id"].between(0, len(trip.arcs) - 1)
            if invalid_route_id.any():
                issues.append(
                    _trip_issue(
                        trip,
                        "error",
                        "invalid_route_arc_ids",
                        f"{int(invalid_route_id.sum())} route row(s) reference an invalid arc ID.",
                        file_type=".route",
                        count=int(invalid_route_id.sum()),
                        sample_rows=_rows(invalid_route_id),
                    )
                )

    return ValidationReport(issues=issues)


def validate_dataset(
    trips: Sequence[TripData],
    *,
    expected_trip_counts: Mapping[str, int] | None = None,
) -> ValidationReport:
    """Validate all trips and optionally verify expected partition sizes."""
    issues = [issue for trip in trips for issue in validate_trip(trip).issues]

    if expected_trip_counts is not None:
        actual_counts = Counter(trip.partition for trip in trips)
        for partition, expected_count in expected_trip_counts.items():
            actual_count = actual_counts.get(partition, 0)
            if actual_count != expected_count:
                issues.append(
                    DataIssue(
                        severity="error",
                        code="unexpected_trip_count",
                        message=(
                            f"Partition {partition!r} has {actual_count} discovered trip(s); "
                            f"expected {expected_count}."
                        ),
                        partition=partition,
                        count=abs(actual_count - expected_count),
                    )
                )

    return ValidationReport(issues=issues)
