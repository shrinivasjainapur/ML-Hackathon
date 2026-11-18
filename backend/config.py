"""Runtime paths and API settings, overridable for local deployments."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("PATHSNAP_DATA_DIR", BASE_DIR / "Student folder")).expanduser().resolve()
STUDENT_TEST_DATA_DIR = DATA_DIR / "student_test"
MODEL_PATH = Path(
    os.getenv("PATHSNAP_MODEL_PATH", BASE_DIR / "outputs/models/final_model.joblib")
).expanduser().resolve()
HMM_CONFIG_PATH = Path(
    os.getenv("PATHSNAP_HMM_CONFIG_PATH", BASE_DIR / "outputs/models/hmm_decoder_config.json")
).expanduser().resolve()

APP_NAME = "PathSnap API"
APP_VERSION = "1.1.0"
API_HOST = os.getenv("PATHSNAP_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("PATHSNAP_API_PORT", "8000"))

_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("PATHSNAP_CORS_ORIGINS", _default_origins).split(",")
    if origin.strip()
]
