"""
tests/test_workplace_api.py — Unit tests for the Workplace Passive-Aggression Detection API.
"""

import pytest
from fastapi.testclient import TestClient
from workplace_api.main import app

client = TestClient(app)


def test_workplace_api_health():
    """Verify health endpoint reports service readiness and model loaded in memory."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert "workplace_logistic_regression" in data["model_name"]
    assert len(data["classes"]) == 2


def test_workplace_predict_passive_aggressive():
    """Verify prediction on clear passive-aggressive workplace statement."""
    statement = "Per my previous email, as I already explained twice yesterday."
    response = client.post("/predict", json={"text": statement})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in (0, 1)
    assert data["label"] in ("Potentially passive-aggressive", "Not flagged as passive-aggressive")
    assert 0.0 <= data["score"] <= 1.0
    assert data["sentiment_label"] in ("Positive", "Negative", "Neutral")
    assert -1.0 <= data["sentiment_score"] <= 1.0
    assert isinstance(data["cues"], list)
    assert data["latency_ms"] >= 0.0


def test_workplace_predict_benign():
    """Verify prediction on benign constructive workplace statement."""
    statement = "Great job completing the project deliverables on time team!"
    response = client.post("/predict", json={"text": statement})
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["score"] <= 1.0
    assert data["model"] == "workplace_logistic_regression_v1"


def test_workplace_predict_short_input():
    """Verify validation rejection on undersized input (< 3 chars)."""
    response = client.post("/predict", json={"text": "ok"})
    assert response.status_code in (400, 422)


def test_workplace_predict_empty_input():
    """Verify validation rejection on empty input."""
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code in (400, 422)
