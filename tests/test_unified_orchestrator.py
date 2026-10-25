"""
tests/test_unified_orchestrator.py — Integration tests for Unified Orchestration Engine.
"""

import pytest
from inference.unified_orchestrator import (
    analyze_unified_statement,
    check_workplace_api_health,
)


def test_orchestrator_social_routing():
    """Verify social media statement dispatches to Social-Media Detector."""
    text = "Imagine having this take in 2026. Bold of you to post this publicly, enjoy the downvotes."
    result = analyze_unified_statement(text)
    assert result["is_valid"] is True
    assert result["status"] == "completed"
    assert result["selected_domain"] == "social_media"
    assert "Social-Media" in result["detector_info"]["detector_name"]
    assert "score" in result["prediction"]
    assert "threat" in result["prediction"]


def test_orchestrator_workplace_routing():
    """Verify workplace statement dispatches to Workplace Detector."""
    text = "Per my previous email, as I already explained twice in the standup meeting."
    result = analyze_unified_statement(text)
    assert result["is_valid"] is True
    assert result["status"] == "completed"
    assert result["selected_domain"] == "workplace"
    assert "Workplace" in result["detector_info"]["detector_name"]
    assert result["prediction"]["primary_label"] in (
        "Potentially passive-aggressive",
        "Not flagged as passive-aggressive",
    )
    assert "sentiment_score" in result["prediction"]


def test_orchestrator_uncertain_clarification():
    """Verify uncertain statement triggers clarification request."""
    text = "Fine. Do whatever you want."
    result = analyze_unified_statement(text)
    assert result["is_valid"] is True
    assert result["status"] == "needs_clarification"
    assert result["selected_domain"] == "uncertain"
    assert "clarification_message" in result


def test_orchestrator_manual_override():
    """Verify manual domain selection bypasses router."""
    text = "Fine. Do whatever you want."
    result = analyze_unified_statement(text, manual_domain="workplace")
    assert result["is_valid"] is True
    assert result["status"] == "completed"
    assert result["selected_domain"] == "workplace"
    assert "Workplace" in result["detector_info"]["detector_name"]


def test_orchestrator_short_text():
    """Verify rejection of statement under 3 characters."""
    result = analyze_unified_statement("no")
    assert result["is_valid"] is False
    assert "error" in result


def test_workplace_api_health_check():
    """Verify workplace API health check logic returns valid response."""
    health = check_workplace_api_health()
    assert "is_available" in health
    assert "status" in health
