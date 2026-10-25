"""
tests/test_rewrite_engine.py
Comprehensive test suite for the constructive rewriting engine.
Validates:
- Meaning preservation and core concern retention
- Removal of indirect hostility without changing the core message
- No invented facts, accusations, or threats
- Appropriate handling of already respectful comments
- Cautious treatment of ambiguous or terse comments
- Handling of empty and malformed inputs
- Safety refusal on flagged threats
- Serialization to dictionary
"""

import pytest
from inference.rewrite_engine import rewrite, RewriteResponse, RewriteCandidate


class TestRewriteEngine:

    def test_empty_and_whitespace_input(self):
        """Empty or whitespace text should return a valid response without candidates."""
        resp1 = rewrite("")
        assert not resp1.has_rewrites
        assert resp1.candidates == []
        assert "empty" in resp1.rationale.lower()

        resp2 = rewrite("    ")
        assert not resp2.has_rewrites
        assert resp2.candidates == []

    def test_already_respectful_comments(self):
        """Direct, polite comments should be marked already respectful without modifications."""
        respectful_samples = [
            "Thank you for sharing the updated roadmap, I will review it today.",
            "I agree with this proposal, let's schedule a call tomorrow morning.",
            "Could you please clarify how the database migration handles existing indexes?",
        ]
        for sample in respectful_samples:
            resp = rewrite(sample)
            assert resp.is_already_respectful
            assert not resp.has_rewrites
            assert len(resp.candidates) == 0
            assert resp.original_text == sample

    def test_fake_politeness_removal(self):
        """Corporate passive-aggressive recall phrases should be softened into polite reminders."""
        text = "As per my last email, the client deadline is Friday."
        resp = rewrite(text)
        assert resp.has_rewrites
        assert resp.primary_tactic == "fake_politeness"
        assert len(resp.candidates) > 0
        sugg = resp.candidates[0].suggested_text
        assert "as per my last email" not in sugg.lower()
        assert "client deadline is Friday" in sugg  # preserves core fact/concern

    def test_backhanded_compliment_reframing(self):
        """Backhanded compliments should be reframed to straightforward commendation."""
        text = "Better late than never on submitting the report."
        resp = rewrite(text)
        assert resp.has_rewrites
        assert resp.primary_tactic == "backhanded_compliment"
        sugg = resp.candidates[0].suggested_text
        assert "better late than never" not in sugg.lower()
        assert "report" in sugg.lower()  # preserves core task

        text2 = "For once your script worked without breaking production."
        resp2 = rewrite(text2)
        assert resp2.has_rewrites
        sugg2 = resp2.candidates[0].suggested_text
        assert "for once" not in sugg2.lower()
        assert "production" in sugg2.lower()

    def test_implied_blame_neutralization(self):
        """Sharp accusatory recall should be reframed to collaborative review."""
        text = "I already told you how to configure the API credentials."
        resp = rewrite(text)
        assert resp.has_rewrites
        assert resp.primary_tactic == "implied_blame"
        sugg = resp.candidates[0].suggested_text
        assert "i already told you" not in sugg.lower()
        assert "API credentials" in sugg

    def test_dismissiveness_deescalation(self):
        """Hostile disengagement should be transformed into respectful delegation."""
        text = "Whatever you say, let's just do it."
        resp = rewrite(text)
        assert resp.has_rewrites
        assert resp.primary_tactic == "dismissiveness"
        sugg = resp.candidates[0].suggested_text
        assert "whatever you say" not in sugg.lower()

    def test_sarcastic_agreement(self):
        """Mock deference should be transformed into collegial alignment."""
        text = "Clearly you know best, so I won't bother speaking up."
        resp = rewrite(text)
        assert resp.has_rewrites
        assert resp.primary_tactic in ("sarcastic_agreement", "dismissiveness")
        sugg = resp.candidates[0].suggested_text
        assert "clearly you know best" not in sugg.lower()

    def test_guilt_tripping(self):
        """Passive resentment should be reframed as an actionable request."""
        text = "Must be nice to leave at 4 PM while the rest of us fix bugs."
        resp = rewrite(text)
        assert resp.has_rewrites
        assert resp.primary_tactic == "guilt_tripping"
        sugg = resp.candidates[0].suggested_text
        assert "must be nice to" not in sugg.lower()

    def test_ambiguous_terse_replies(self):
        """Terse single-word replies like 'fine', 'sure', 'k' should be expanded with caution."""
        terse_words = ["fine.", "sure", "k", "whatever", "okay"]
        for word in terse_words:
            resp = rewrite(word)
            assert resp.is_ambiguous
            assert resp.has_rewrites
            assert len(resp.candidates) >= 1
            # Check candidates offer full sentence expansions
            assert len(resp.candidates[0].suggested_text.split()) >= 3

    def test_terse_context_awareness(self):
        """Context from parent comment should influence terse reply suggestions."""
        parent = "Can you deploy the staging build before 5 PM?"
        resp = rewrite("Fine", parent_comment=parent)
        assert resp.is_ambiguous
        assert any("take care of that" in c.suggested_text for c in resp.candidates)

    def test_threat_refusal_safety(self):
        """Threats should not be sanitized into polite requests."""
        detection_with_threat = {
            "combined": {"threat": "direct_threat", "verdict": "passive_aggressive"},
            "threat_result": {"label": "direct_threat"},
        }
        resp = rewrite("I will find where you live and destroy you.", detection_result=detection_with_threat)
        assert resp.is_threat_flagged
        assert not resp.has_rewrites
        assert len(resp.candidates) == 0
        assert "safety" in resp.rationale.lower()

    def test_serialization_to_dict(self):
        """Response should serialize cleanly to a Python dictionary."""
        resp = rewrite("As per my last email, please review the doc.")
        d = resp.to_dict()
        assert isinstance(d, dict)
        assert "original_text" in d
        assert "candidates" in d
        assert isinstance(d["candidates"], list)
        assert len(d["candidates"]) > 0
        assert "suggested_text" in d["candidates"][0]
        assert "disclaimer" in d
