from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Student folder"

TRAIN_DATA_DIR = DATA_DIR / "train"
STUDENT_TEST_DATA_DIR = DATA_DIR / "student_test"

# Backend settings
APP_NAME = "PathSnap API"
APP_VERSION = "1.0.0"

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Frontend development origin
CORS_ORIGINS = [
    "http://localhost:5173",
]