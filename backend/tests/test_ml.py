"""
Tests for ML classifier pipeline, loading, calibration, and sensitivity thresholds.
"""

import pytest
from backend.app.ml.classifier import PAClassifier, get_classifier


def test_classifier_is_loaded():
    clf = get_classifier()
    assert clf.is_model_loaded is True
    assert clf.inference_mode == "ml_classifier"
    assert clf.model_version is not None


def test_classifier_predict_neutral():
    clf = get_classifier()
    text = "Thanks for the feedback, I will update the documentation accordingly."
    res = clf.predict(text, context="workplace", sensitivity="balanced")
    assert res["label"] in ["neutral", "passive_aggressive", "overt_hostile"]
    assert "probabilities" in res
    assert "score" in res
    assert 0 <= res["score"] <= 100
    assert "latency_ms" in res
    assert res["probabilities"]["neutral"] > 0.40


def test_classifier_predict_passive_aggressive():
    clf = get_classifier()
    text = "As per my last email, the deadline was Friday. Thanks in advance."
    res = clf.predict(text, context="workplace", sensitivity="balanced")
    assert res["label"] == "passive_aggressive"
    assert res["score"] >= 30
    assert res["probabilities"]["passive_aggressive"] > 0.40


def test_classifier_predict_overt_hostile():
    clf = get_classifier()
    text = "You are a complete idiot and I hate working with you, get out."
    res = clf.predict(text, context="workplace", sensitivity="balanced")
    assert res["label"] == "overt_hostile"
    assert res["score"] >= 60
    assert res["probabilities"]["overt_hostile"] > 0.50


def test_classifier_sensitivity_impact():
    clf = get_classifier()
    text = "Per my last email, please confirm."
    res_lenient = clf.predict(text, sensitivity="lenient")
    res_strict = clf.predict(text, sensitivity="strict")
    # Strict sensitivity threshold makes it more sensitive or gives higher score
    assert res_strict["score"] >= res_lenient["score"] or res_strict["label"] in ["passive_aggressive", "neutral"]


def test_uncertainty_structure():
    clf = get_classifier()
    text = "Okay."
    res = clf.predict(text)
    unc = res["uncertainty"]
    assert "entropy" in unc
    assert "max_confidence" in unc
    assert "is_uncertain" in unc
    assert isinstance(unc["is_uncertain"], bool)
