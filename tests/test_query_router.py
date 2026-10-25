"""
tests/test_query_router.py — Unit tests for the Gemini Context Query Router and Fallback.
"""

import pytest
from inference.query_router import route_statement, RouteResult, _fallback_heuristic_router


def test_router_workplace_context():
    """Verify workplace statements route to 'workplace'."""
    text = "Per my previous email, as I already explained twice in the sprint standup meeting."
    result = route_statement(text)
    assert result.route == "workplace"
    assert result.needs_clarification is False
    assert len(result.detected_domain_cues) > 0


def test_router_social_media_context():
    """Verify social-media statements route to 'social_media'."""
    text = "Tell me you got ratioed on twitter without telling me you got ratioed."
    result = route_statement(text)
    assert result.route == "social_media"
    assert result.needs_clarification is False
    assert len(result.detected_domain_cues) > 0


def test_router_ambiguous_context():
    """Verify context-neutral statements return 'uncertain' and request clarification."""
    text = "Sure, do whatever you want."
    result = route_statement(text)
    assert result.route == "uncertain"
    assert result.needs_clarification is True


def test_router_empty_input():
    """Verify empty input returns uncertain route with empty cues."""
    result = route_statement("   ")
    assert result.route == "uncertain"
    assert result.needs_clarification is True


def test_router_result_dictionary():
    """Verify RouteResult serialization structure."""
    res = RouteResult(
        route="workplace",
        explanation="Testing rationale",
        confidence=0.92,
        needs_clarification=False,
        detected_domain_cues=["email", "standup"],
        is_fallback=True,
    )
    d = res.to_dict()
    assert d["route"] == "workplace"
    assert d["confidence"] == 0.92
    assert d["needs_clarification"] is False
    assert "email" in d["detected_domain_cues"]
