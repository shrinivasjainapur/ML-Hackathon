from pydantic import BaseModel
from typing import Optional


class TripResponse(BaseModel):
    trip_id: str
    point_count: int


class TripsListResponse(BaseModel):
    trips: list[TripResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    message: str


class GPSPoint(BaseModel):
    lat: float
    lng: float
    timestamp: str


class RoutePoint(BaseModel):
    lat: float
    lng: float


class PredictionMetrics(BaseModel):
    inference_seconds: Optional[float] = None
    matched_observations: Optional[int] = None
    total_observations: Optional[int] = None


class PredictionResponse(BaseModel):
    trip_id: str
    gps_points: list[GPSPoint]
    predicted_route: list[RoutePoint]
    road_network: Optional[list[RoutePoint]] = None
    metrics: Optional[PredictionMetrics] = None