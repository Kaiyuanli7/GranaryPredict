from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Root directory assumed as two levels up from this file
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

# Create dirs at import time if they don't exist
for _dir in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Alert threshold (°C) for potential spoilage
ALERT_TEMP_THRESHOLD = float(os.getenv("ALERT_TEMP_THRESHOLD", "28"))

# Optional per-grain safe temperature thresholds (°C)
# If a grain type isn't listed, fall back to ALERT_TEMP_THRESHOLD.
GRAIN_ALERT_THRESHOLDS: dict[str, float] = {
    "Mid-to-late indica rice": 28,
    "Early indica rice": 28,
    "Japonica rice": 26,
    "Yellow corn": 30,
    "Soybeans": 25,
    "Wheat": 27,
}

# API endpoint placeholders
METEOROLOGY_API_BASE = os.getenv("METEOROLOGY_API_BASE", "https://api.example.com/weather")
COMPANY_API_BASE = os.getenv("COMPANY_API_BASE", "https://api.example.com/granary")


__all__ = [
    "ROOT_DIR",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "MODELS_DIR",
    "ALERT_TEMP_THRESHOLD",
    "METEOROLOGY_API_BASE",
    "COMPANY_API_BASE",
    "GRAIN_ALERT_THRESHOLDS",
] 