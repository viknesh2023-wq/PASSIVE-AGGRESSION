"""
tests/test_v2_inference.py
Automated tests for PassiveAggression Detector 2.0:
- Standalone inference module (inference/pa_inference.py)
- Combinatorial rule-based detector (inference/rule_detector.py)
- SQLite history persistence (inference/history_db.py)
- Synthetic dataset schema & label integrity
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
import pandas as pd
from inference.pa_inference import analyze, get_model_info, _load_models
from inference.rule_detector import detect as rule_detect
from inference.history_db import save as save_history, get_all as get_history, delete as delete_history, init_db


@pytest.fixture(scope="session", autouse=True)
def setup_models():
    _load_models()
    init_db()


def test_model_info_loaded():
    info = get_model_info()
    assert info["pa_model_loaded"] is True
    assert "v2" in info["pa_model_version"]
    assert info["threat_model_loaded"] is True


def test_analyze_passive_aggressive():
    text = "As per my last email, which you clearly didn't read, the deadline has passed."
    res = analyze(text=text, sensitivity="balanced")
    assert "ml_result" in res
    assert "rule_result" in res
    assert "threat_result" in res
    assert res["ml_result"]["label"] == "passive_aggressive"
    assert res["rule_result"]["pa_label"] == "passive_aggressive"
    assert len(res["rule_result"]["tactics"]) >= 1


def test_analyze_non_passive_aggressive():
    text = "Great job on the presentation! The slides were clear and well organized."
    res = analyze(text=text, sensitivity="balanced")
    assert res["ml_result"]["label"] == "non_passive_aggressive"
    assert res["rule_result"]["pa_label"] == "non_passive_aggressive"


def test_analyze_ambiguous():
    text = "Fine."
    res = analyze(text=text, sensitivity="balanced")
    assert res["ml_result"]["label"] in ["ambiguous", "non_passive_aggressive", "passive_aggressive"]
    assert res["ml_result"]["uncertainty"]["is_uncertain"] is True or res["rule_result"]["is_ambiguous"] is True


def test_figurative_threat_not_flagged():
    text = "You are going down, I will destroy you in this video game match!"
    res = analyze(text=text)
    assert res["rule_result"]["threat_label"] == "no_threat"
    assert res["combined"]["threat"] == "no_threat"


def test_direct_threat_detection():
    text = "I know where you live and you will pay for what you did."
    res = analyze(text=text)
    assert res["combined"]["threat"] in ["direct_threat", "implied_threat"]
    assert res["rule_result"]["threat_label"] in ["direct_threat", "implied_threat"]


def test_rule_detector_benign_override():
    text = "Per your request, here is the updated document."
    rule_res = rule_detect(text)
    # Benign override should prevent fake_politeness accusation
    assert rule_res.pa_label == "non_passive_aggressive"


def test_history_db_crud():
    test_id = "test-uuid-12345"
    test_text = "Testing history SQLite database."
    dummy_result = {
        "analysis_id": test_id,
        "ml_result": {"label": "non_passive_aggressive", "pa_score": 5.0, "inference_mode": "ml_test"},
        "rule_result": {"pa_label": "non_passive_aggressive", "pa_score": 0.0, "tactics": []},
        "threat_result": {"label": "no_threat", "threat_score": 0.0},
        "combined": {"primary": "non_passive_aggressive", "threat": "no_threat"},
    }
    save_history(test_id, test_text, "social", "balanced", dummy_result)
    
    records = get_history(limit=50)
    found = [r for r in records if r["id"] == test_id]
    assert len(found) == 1
    assert found[0]["text_snippet"].startswith("Testing history")

    # Clean up
    deleted = delete_history(test_id)
    assert deleted is True


def test_synthetic_dataset_schema():
    csv_path = _ROOT / "data" / "synthetic" / "pa_synthetic_merged.csv"
    assert csv_path.exists(), "Merged synthetic dataset must exist"
    df = pd.read_csv(csv_path, nrows=50)

    canonical_cols = [
        "id", "text", "primary_label", "threat_label", "post_text",
        "conversation_id", "turn_index", "parent_comment_id", "parent_comment",
        "thread_context", "target_message_id", "tactic_labels", "sarcasm_label",
        "sentiment_label", "generation_source", "generator_model",
        "generation_batch", "review_status", "quality_flags"
    ]
    for col in canonical_cols:
        assert col in df.columns, f"Missing canonical column: {col}"

    assert df["id"].str.startswith("SYNPA-").all(), "All synthetic IDs must start with SYNPA-"
    assert set(df["primary_label"].unique()).issubset({"passive_aggressive", "non_passive_aggressive", "ambiguous"})
    assert set(df["threat_label"].unique()).issubset({"no_threat", "direct_threat", "implied_threat", "ambiguous"})
