
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import (
    APP_NAME,
    APP_VERSION,
    CORS_ORIGINS,
)

from backend.routes.trips import router as trips_router
from backend.schemas.responses import HealthResponse

from backend.services.prediction_service import (
    generate_prediction,
    ModelNotReadyError,
)


# Create FastAPI application
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="PathSnap - Distributed Map-Matching and Route Reconstruction API",
)


# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Home endpoint
@app.get("/", response_model=HealthResponse)
def home():
    return {
        "status": "running",
        "message": "PathSnap API is running successfully",
    }


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy",
        "message": "Backend is operational",
    }


# Register trip routes
app.include_router(trips_router)


# Prediction endpoint
@app.post("/predict/{trip_id}")
def predict_trip(trip_id: str):

    try:
        result = generate_prediction(trip_id)

    except ValueError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )

    except ModelNotReadyError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Trip not found",
        )

    return result
