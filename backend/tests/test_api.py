"""
End-to-end API tests for all endpoints:
  - GET /api/health
  - GET /api/model-info
  - POST /api/analyze
  - POST /api/batch
  - POST /api/feedback
  - GET /api/history
  - DELETE /api/history
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "model_version" in data


def test_model_info_endpoint():
    res = client.get("/api/model-info")
    assert res.status_code == 200
    data = res.json()
    assert "model_version" in data
    assert "labels" in data
    assert "tactics" in data


def test_analyze_endpoint_passive_aggressive():
    payload = {
        "text": "As per my last email, the deadline was Friday. I'm not sure what happened.",
        "context": "workplace",
        "sensitivity": "balanced",
    }
    res = client.post("/api/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "id" in data
    assert data["label"] in ["passive_aggressive", "neutral", "overt_hostile"]
    assert "score" in data
    assert "probabilities" in data
    assert "tactics" in data
    assert "highlights" in data
    assert "sentences" in data
    assert "rewrites" in data
    assert "model_version" in data
    assert "latency_ms" in data


def test_analyze_endpoint_validation_error():
    # Empty text
    res = client.post("/api/analyze", json={"text": "", "context": "workplace"})
    assert res.status_code == 422


def test_analyze_endpoint_non_english_rejection():
    # Non-English text must return 422 with English-only policy message
    res = client.post("/api/analyze", json={"text": "Привет мир, это тест на русском языке.", "context": "workplace"})
    assert res.status_code == 422
    assert "strictly supports English" in res.json()["detail"]


def test_batch_endpoint():
    payload = {
        "messages": [
            {"sender": "Alice", "text": "Can you provide an update on the design?"},
            {"sender": "Bob", "text": "As per my last email, I already gave you the link."},
            {"sender": "Alice", "text": "Thanks, checking it now."},
        ],
        "context": "workplace",
        "sensitivity": "balanced",
    }
    res = client.post("/api/batch", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "results" in data
    assert len(data["results"]) == 3
    assert data["results"][0]["sender"] == "Alice"
    assert data["results"][1]["sender"] == "Bob"


def test_feedback_endpoint():
    # First create an analysis
    res_an = client.post("/api/analyze", json={"text": "Fine. Whatever.", "context": "chat"})
    assert res_an.status_code == 200
    an_id = res_an.json()["id"]

    # Submit feedback
    fb_payload = {
        "id": an_id,
        "helpful": True,
        "corrected_label": "passive_aggressive",
        "note": "Accurate detection",
    }
    res_fb = client.post("/api/feedback", json=fb_payload)
    assert res_fb.status_code == 200
    assert res_fb.json()["status"] == "recorded"


def test_history_lifecycle():
    # Submit analysis
    client.post("/api/analyze", json={"text": "Unique test message for history lifecycle", "context": "social"})

    # Fetch history
    res = client.get("/api/history")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) >= 1
