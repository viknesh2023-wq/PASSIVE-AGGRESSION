# PassiveAggression Detector — Project Configuration
# backend/app/config.py

import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent  # d:\python_folder

DATA_DIR         = PROJECT_ROOT / "data"
MODELS_DIR       = PROJECT_ROOT / "models"
REPORTS_DIR      = PROJECT_ROOT / "reports"
REGISTRY_DIR     = MODELS_DIR / "registry"
ANNOTATIONS_DIR  = DATA_DIR / "annotations"

# ─── API ──────────────────────────────────────────────────────────────────────
API_VERSION        = "v1"
API_PREFIX         = "/api"
ALLOWED_ORIGINS    = [
    "http://localhost:5173",   # Vite dev server
    "http://localhost:4173",   # Vite preview
    "http://127.0.0.1:5173",
    "http://127.0.0.1:4173",
]

# ─── Model ────────────────────────────────────────────────────────────────────
MODEL_VERSION      = "pa-rule-logreg-v1.0"
MAX_TEXT_LENGTH    = 2000     # characters
MAX_BATCH_SIZE     = 50       # messages per batch request

# ─── Labels ───────────────────────────────────────────────────────────────────
LABELS = ["neutral", "passive_aggressive", "overt_hostile"]
LABEL_TO_INT = {l: i for i, l in enumerate(LABELS)}
INT_TO_LABEL = {i: l for l, i in LABEL_TO_INT.items()}

# ─── Sensitivity multipliers ──────────────────────────────────────────────────
SENSITIVITY_THRESHOLDS = {
    "lenient":  {"neutral_max": 35, "pa_max": 70},
    "balanced": {"neutral_max": 30, "pa_max": 60},
    "strict":   {"neutral_max": 22, "pa_max": 50},
}

# ─── Tactics ──────────────────────────────────────────────────────────────────
TACTICS = [
    "sarcasm",
    "backhanded_compliment",
    "guilt_tripping",
    "mock_politeness",
    "blame_shifting",
    "curt_dismissal",
]

# ─── SQLite DB ────────────────────────────────────────────────────────────────
DB_PATH = PROJECT_ROOT / "backend" / "pa_detector.db"

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL         = os.getenv("LOG_LEVEL", "INFO")
LOG_FULL_TEXT     = False   # Never log full private messages by default

# ─── Env overrides ────────────────────────────────────────────────────────────
MODEL_VERSION = os.getenv("PA_MODEL_VERSION", MODEL_VERSION)
