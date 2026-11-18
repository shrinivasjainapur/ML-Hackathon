from backend.config import STUDENT_TEST_DATA_DIR


def get_available_trips():
    """Return basic metadata for available student-test trips."""

    trips = []

    if not STUDENT_TEST_DATA_DIR.exists():
        return trips

    for trip_dir in sorted(STUDENT_TEST_DATA_DIR.iterdir()):

        if not trip_dir.is_dir():
            continue

        track_file = trip_dir / f"{trip_dir.name}.track"

        if not track_file.exists():
            continue

        with track_file.open("r", encoding="utf-8") as file:
            point_count = sum(1 for line in file if line.strip())

        trips.append({
            "trip_id": trip_dir.name,
            "point_count": point_count,
        })

    return trips


def get_trip_gps_points(trip_id: str):
    """Read GPS observations for one trip without accessing route labels."""

    # Prevent invalid paths and directory traversal
    if not trip_id.isdigit():
        return None

    trip_dir = STUDENT_TEST_DATA_DIR / trip_id
    track_file = trip_dir / f"{trip_id}.track"

    if not track_file.is_file():
        return None

    gps_points = []

    with track_file.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            values = line.split()

            if not values:
                continue

            if len(values) < 3:
                raise ValueError(
                    f"Invalid GPS data in {trip_id}.track at line {line_number}"
                )

            try:
                longitude = float(values[0])
                latitude = float(values[1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid coordinates in {trip_id}.track at line {line_number}"
                ) from exc

            gps_points.append({
                "lat": latitude,
                "lng": longitude,
                "timestamp": values[2],
            })

    return {
        "trip_id": trip_id,
        "point_count": len(gps_points),
        "gps_points": gps_points,
    }