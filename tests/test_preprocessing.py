"""Unit tests for the non-destructive PathSnap preprocessing layer."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.preprocessing.data_loader import load_dataset, load_trip
from src.preprocessing.preprocess import preprocess_trip
from src.preprocessing.validator import validate_dataset, validate_trip


def _write_trip(root: Path, partition: str, trip_id: str, *, invalid: bool = False) -> Path:
    trip_dir = root / partition / trip_id
    trip_dir.mkdir(parents=True, exist_ok=True)
    track = "8.5 49.8 2\n8.6 49.9 1\n"
    nodes = "8.5 49.8\n8.6 49.9\n"
    arcs = "0 1\n"
    route = "0\n"
    if invalid:
        track = "181.0 49.8 2\n8.6 49.9 not-a-time\n"
        nodes = "8.5 95.0\n8.6 49.9\n"
        arcs = "0 7\n"
        route = "5\n"
    (trip_dir / f"{trip_id}.track").write_text(track, encoding="utf-8")
    (trip_dir / f"{trip_id}.nodes").write_text(nodes, encoding="utf-8")
    (trip_dir / f"{trip_id}.arcs").write_text(arcs, encoding="utf-8")
    (trip_dir / f"{trip_id}.route").write_text(route, encoding="utf-8")
    return trip_dir


class DataLoadingTests(unittest.TestCase):
    def test_loads_verified_layout_and_keeps_route_out_of_model_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_trip(root, "train", "00000001")

            result = load_dataset(root, partitions=("train",))

            self.assertEqual(len(result.trips), 1)
            self.assertEqual(result.all_issues, [])
            trip = result.trips[0]
            self.assertEqual(list(trip.track.columns), ["longitude", "latitude", "timestamp"])
            self.assertEqual(list(trip.route.columns), ["arc_id"])
            self.assertNotIn("route", trip.model_inputs())
            self.assertEqual(trip.model_inputs()["arcs"].iloc[0, 1], 1)

    def test_reports_invalid_trip_id_and_missing_file_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trip_dir = _write_trip(root, "train", "bad-trip")
            (trip_dir / "bad-trip.route").unlink()

            trip = load_trip(trip_dir, "train")
            codes = {issue.code for issue in trip.issues}

            self.assertIn("invalid_trip_id", codes)
            self.assertIn("missing_file", codes)
            self.assertIsNone(trip.route)

    def test_reports_a_malformed_column_count_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trip_dir = _write_trip(root, "train", "00000001")
            (trip_dir / "00000001.track").write_text("8.5 49.8 1 extra\n", encoding="utf-8")

            trip = load_trip(trip_dir, "train")

            self.assertIsNone(trip.track)
            self.assertIn("unexpected_column_count", {issue.code for issue in trip.issues})


class ValidationAndPreprocessingTests(unittest.TestCase):
    def test_reports_invalid_coordinates_timestamps_arcs_and_route_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trip_dir = _write_trip(root, "train", "00000001", invalid=True)
            trip = load_trip(trip_dir, "train")

            report = validate_trip(trip)
            codes = report.counts_by_code()

            self.assertIn("out_of_range_gps_coordinates", codes)
            self.assertIn("out_of_range_node_coordinates", codes)
            self.assertIn("invalid_timestamps", codes)
            self.assertIn("arc_node_id_out_of_range", codes)
            self.assertIn("invalid_route_arc_ids", codes)

    def test_preprocessing_sorts_a_copy_and_preserves_original_order_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trip_dir = _write_trip(root, "train", "00000001")
            trip = load_trip(trip_dir, "train")
            before = trip.track.copy(deep=True)

            processed = preprocess_trip(trip)

            pd.testing.assert_frame_equal(trip.track, before)
            self.assertEqual(processed.original_track["timestamp"].tolist(), [2.0, 1.0])
            self.assertEqual(processed.track["timestamp"].tolist(), [1.0, 2.0])
            self.assertEqual(processed.track.index.tolist(), [1, 0])
            self.assertEqual(processed.reference_route["arc_id"].tolist(), [0])
            self.assertFalse(hasattr(processed.model_inputs(), "reference_route"))

    def test_dataset_trip_count_validation_is_optional_and_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_trip(root, "train", "00000001")
            trips = load_dataset(root, partitions=("train",)).trips

            report = validate_dataset(trips, expected_trip_counts={"train": 80})

            self.assertIn("unexpected_trip_count", report.counts_by_code())


if __name__ == "__main__":
    unittest.main()
