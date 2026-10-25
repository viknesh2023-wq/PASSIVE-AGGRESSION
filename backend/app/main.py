"""
FastAPI application entry point.
Passive Aggression Detector — Backend API
"""

from __future__ import annotations
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import ALLOWED_ORIGINS, API_PREFIX
from .api.routes import router
from .api.dashboard_routes import dashboard_router
from .database.db import init_db
from .ml.classifier import get_classifier

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Startup / Shutdown ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("PassiveAggression Detector API starting up...")
    # Initialize DB
    init_db()
    # Pre-load classifier
    clf = get_classifier()
    logger.info("Classifier ready — mode: %s, version: %s", clf.inference_mode, clf.model_version)
    yield
    logger.info("API shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PassiveAggression Detector API",
    description=(
        "Explainable detection of passive-aggressive and overtly hostile language in text. "
        "Provides per-message analysis, conversation thread analysis, tactic detection, "
        "text highlighting, calmer rewrite suggestions, and user feedback."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Accept"],
)

# ─── Routes ───────────────────────────────────────────────────────────────────
app.include_router(router, prefix=API_PREFIX)
app.include_router(dashboard_router, prefix=API_PREFIX)
