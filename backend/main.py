"""PathSnap FastAPI application."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import APP_NAME, APP_VERSION, CORS_ORIGINS
from backend.routes.trips import router as trips_router
from backend.schemas.responses import HealthResponse, PredictionResponse, ReadinessResponse
from backend.services.prediction_service import (
    ModelNotReadyError,
    generate_prediction,
    prediction_readiness,
)

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="PathSnap road matching and route reconstruction API",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/", response_model=HealthResponse)
def home() -> HealthResponse:
    return HealthResponse(status="running", message="PathSnap API is running")


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(status="healthy", message="Backend process is operational")


@app.get("/health/ready", response_model=ReadinessResponse)
def readiness_check() -> ReadinessResponse:
    details = prediction_readiness()
    return ReadinessResponse(
        status="ready" if details["ready"] else "not_ready",
        ready=bool(details["ready"]),
        model_available=bool(details["model_available"]),
        data_available=bool(details["data_available"]),
    )


app.include_router(trips_router)


@app.post("/predict/{trip_id}", response_model=PredictionResponse)
def predict_trip(trip_id: str) -> PredictionResponse:
    try:
        result = generate_prediction(trip_id)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ModelNotReadyError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    if result is None:
        raise HTTPException(status_code=404, detail="Trip not found")
    return PredictionResponse.model_validate(result)
