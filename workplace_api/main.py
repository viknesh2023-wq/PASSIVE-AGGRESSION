"""
workplace_api/main.py — Production-grade FastAPI HTTP API for the Workplace Passive-Aggression Model.
Exposes:
  - GET  /health   -> Service and model readiness check
  - POST /predict  -> Single-statement prediction using trained model
  - GET  /         -> Service metadata & docs redirect
"""

import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from workplace_api.predictor import predictor

app = FastAPI(
    title="Workplace Passive-Aggression Detection API",
    description="Dedicated microservice exposing the workplace passive-aggression classification model.",
    version="1.0.0",
)

# Enable CORS for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=4000,
        description="The workplace communication statement to analyze.",
        examples=["Per my previous email, as I already explained twice yesterday."],
    )


class PredictResponse(BaseModel):
    model: str = Field(default="workplace_logistic_regression_v1", description="Model architecture identifier")
    text: str = Field(..., description="Normalized input text")
    prediction: int = Field(..., description="Raw binary prediction (0: Not flagged, 1: Potentially PA)")
    label: str = Field(..., description="Human-readable model label")
    score: float = Field(..., description="Estimated model probability for the predicted class [0.0 - 1.0]")
    sentiment_label: str = Field(..., description="Supporting VADER sentiment label (Positive, Negative, Neutral)")
    sentiment_score: float = Field(..., description="VADER compound polarity score [-1.0 to 1.0]")
    cues: List[str] = Field(default_factory=list, description="Model-weighted n-gram cue terms")
    latency_ms: float = Field(..., description="Inference execution duration in milliseconds")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status ('ready' or 'degraded')")
    model_loaded: bool = Field(..., description="Whether the classifier pipeline is loaded in memory")
    model_name: str = Field(default="workplace_logistic_regression_v1")
    version: str = Field(default="1.0.0")
    classes: List[str] = Field(
        default=["Not flagged as passive-aggressive", "Potentially passive-aggressive"]
    )


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Workplace Passive-Aggression Detection API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Healthcheck endpoint.
    Returns 200 with status='ready' only if model is verified loaded in memory.
    Returns 503 Service Unavailable if model artifact is missing or failed to load.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready. Pipeline artifact failed to load.",
        )
    return HealthResponse(
        status="ready",
        model_loaded=True,
        model_name="workplace_logistic_regression_v1",
        version="1.0.0",
        classes=["Not flagged as passive-aggressive", "Potentially passive-aggressive"],
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_statement(payload: PredictRequest):
    """
    Execute passive-aggression detection on a single workplace statement.
    Validates input length, evaluates TF-IDF Logistic Regression, and computes VADER sentiment.
    """
    clean_text = payload.text.strip()
    if len(clean_text) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text must contain at least 3 non-whitespace characters.",
        )

    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workplace detection model is not ready.",
        )

    t0 = time.perf_counter()
    try:
        res = predictor.predict(clean_text)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return PredictResponse(
        model="workplace_logistic_regression_v1",
        text=clean_text,
        prediction=res["prediction"],
        label=res["label"],
        score=round(res["score"], 4),
        sentiment_label=res["sentiment_label"],
        sentiment_score=round(res["sentiment_score"], 4),
        cues=res["cues"],
        latency_ms=round(latency_ms, 2),
    )
