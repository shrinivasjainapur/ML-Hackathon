"""Non-destructive preprocessing shared by every PathSnap model branch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from .data_loader import TripData


TRACK_COLUMNS = ("longitude", "latitude", "timestamp")


@dataclass
class ModelInputs:
    """Data allowed to reach feature and prediction code; route labels are absent."""

    track: pd.DataFrame | None
    nodes: pd.DataFrame | None
    arcs: pd.DataFrame | None


@dataclass
class PreprocessedTrip:
    """A trip with original observations retained alongside a sorted working copy."""

    trip_id: str
    partition: str
    original_track: pd.DataFrame | None
    track: pd.DataFrame | None
    nodes: pd.DataFrame | None
    arcs: pd.DataFrame | None
    reference_route: pd.DataFrame | None

    def model_inputs(self, *, copy: bool = True) -> ModelInputs:
        """Return feature inputs only; reference-route labels cannot leak in."""
        frames = (self.track, self.nodes, self.arcs)
        if copy:
            frames = tuple(frame.copy(deep=True) if frame is not None else None for frame in frames)
        return ModelInputs(track=frames[0], nodes=frames[1], arcs=frames[2])


def sort_gps_observations(track: pd.DataFrame) -> pd.DataFrame:
    """Return a stable chronological copy without dropping or modifying rows.

    Rows with invalid timestamps remain present and are placed after rows with
    valid timestamps.  The original DataFrame and its index remain untouched;
    the returned index therefore preserves each observation's source position.
    """
    missing_columns = set(TRACK_COLUMNS).difference(track.columns)
    if missing_columns:
        raise ValueError(f"Track is missing required columns: {sorted(missing_columns)}")
    return track.sort_values("timestamp", kind="stable", na_position="last").copy(deep=True)


def preprocess_trip(trip: TripData) -> PreprocessedTrip:
    """Create non-destructive working copies for one trip.

    Validation is intentionally separate: this function does not silently
    remove invalid observations, arcs, or route labels.
    """
    original_track = trip.track.copy(deep=True) if trip.track is not None else None
    sorted_track = sort_gps_observations(trip.track) if trip.track is not None else None
    return PreprocessedTrip(
        trip_id=trip.trip_id,
        partition=trip.partition,
        original_track=original_track,
        track=sorted_track,
        nodes=trip.nodes.copy(deep=True) if trip.nodes is not None else None,
        arcs=trip.arcs.copy(deep=True) if trip.arcs is not None else None,
        reference_route=trip.route.copy(deep=True) if trip.route is not None else None,
    )


def preprocess_dataset(trips: Sequence[TripData]) -> list[PreprocessedTrip]:
    """Preprocess a collection of trips without writing to the dataset."""
    return [preprocess_trip(trip) for trip in trips]
