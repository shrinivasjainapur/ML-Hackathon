"""Stable response schemas shared by the frontend and API."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    message: str


class ReadinessResponse(BaseModel):
    status: str
    ready: bool
    model_available: bool
    data_available: bool


class TripResponse(BaseModel):
    trip_id: str
    point_count: int


class TripsListResponse(BaseModel):
    trips: list[TripResponse]
    count: int


class GPSPoint(BaseModel):
    lat: float
    lng: float
    timestamp: float


class TripDetailsResponse(BaseModel):
    trip_id: str
    point_count: int
    gps_points: list[GPSPoint]


class RoutePoint(BaseModel):
    lat: float
    lng: float


class PredictionMetrics(BaseModel):
    inference_seconds: float
    matched_observations: int
    total_observations: int
    coverage_ratio: float = Field(ge=0, le=1)
    large_gap_resets: int


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trip_id: str
    gps_points: list[GPSPoint]
    predicted_route: list[RoutePoint]
    road_network: list[RoutePoint] = Field(default_factory=list)
    metrics: PredictionMetrics
