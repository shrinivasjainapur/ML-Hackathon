from fastapi import APIRouter, HTTPException

from backend.services.trip_service import (
    get_available_trips,
    get_trip_gps_points,
)

router = APIRouter()


@router.get("/trips")
def list_trips():
    """Get available student-test trips."""

    trips = get_available_trips()

    return {
        "trips": trips,
        "count": len(trips),
    }


@router.get("/trips/{trip_id}")
def get_trip_details(trip_id: str):
    """Get GPS observations for a selected trip."""

    try:
        trip = get_trip_gps_points(trip_id)

    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    if trip is None:
        raise HTTPException(
            status_code=404,
            detail="Trip not found",
        )

    return trip