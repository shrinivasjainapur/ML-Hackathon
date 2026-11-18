"""Read development GPS trips without opening their reference routes."""
from __future__ import annotations

import re
import logging

from backend.config import STUDENT_TEST_DATA_DIR
from src.preprocessing.dataset import FILE_COLUMNS, read_trip_file

logger = logging.getLogger(__name__)
TRIP_ID_PATTERN = re.compile(r"^[0-9]{1,32}$")


def normalize_trip_id(trip_id: str) -> str | None:
    """Accept only numeric directory identifiers; never allow path fragments."""
    value = str(trip_id).strip()
    return value if TRIP_ID_PATTERN.fullmatch(value) else None


def get_available_trips() -> list[dict[str, str | int]]:
    """Return trips with all files required for GPS-to-road inference."""
    if not STUDENT_TEST_DATA_DIR.is_dir():
        return []

    trips = []
    for trip_dir in sorted(STUDENT_TEST_DATA_DIR.iterdir()):
        if not trip_dir.is_dir() or normalize_trip_id(trip_dir.name) is None:
            continue
        trip_id = trip_dir.name
        required = [trip_dir / f"{trip_id}{suffix}" for suffix in (".track", ".nodes", ".arcs")]
        if not all(path.is_file() for path in required):
            continue
        try:
            track = read_trip_file(required[0], FILE_COLUMNS[".track"])
        except (OSError, ValueError) as exc:
            logger.warning("Skipping unreadable development trip %s: %s", trip_id, exc)
            continue
        assert track is not None
        trips.append({"trip_id": trip_id, "point_count": int(len(track))})
    return trips


def get_trip_gps_points(trip_id: str) -> dict[str, object] | None:
    """Load track points only; route labels and arcs are not returned here."""
    normalized = normalize_trip_id(trip_id)
    if normalized is None:
        return None

    track_file = STUDENT_TEST_DATA_DIR / normalized / f"{normalized}.track"
    if not track_file.is_file():
        return None
    track = read_trip_file(track_file, FILE_COLUMNS[".track"])
    assert track is not None

    gps_points = []
    for row in track.itertuples(index=False):
        try:
            longitude, latitude, timestamp = float(row.longitude), float(row.latitude), float(row.timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Trip {normalized} contains an invalid GPS observation") from exc
        if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
            raise ValueError(f"Trip {normalized} contains coordinates outside valid ranges")
        gps_points.append({"lat": latitude, "lng": longitude, "timestamp": timestamp})

    return {"trip_id": normalized, "point_count": len(gps_points), "gps_points": gps_points}
