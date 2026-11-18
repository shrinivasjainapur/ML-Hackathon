from backend.services.trip_service import get_trip_gps_points


class ModelNotReadyError(Exception):
    """Raised when the trained prediction model is unavailable."""


def generate_prediction(trip_id: str):
    """
    Prediction service entry point.

    The actual classifier and HMM/Viterbi integration
    will be added after model tuning is complete.
    """

    trip = get_trip_gps_points(trip_id)

    if trip is None:
        return None

    # Do not generate dummy predictions.
    raise ModelNotReadyError(
        "Prediction model is not configured yet. "
        "Waiting for the finalized classifier and HMM."
    )