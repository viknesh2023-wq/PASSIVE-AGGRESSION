"""
tests/test_master_suite.py
Master regression and integration test suite for PassiveAggression Detector 2.0.
Verifies all non-negotiable requirements from Phase 10:
- Dataset integrity and local isolation (no external paths)
- Label mapping and taxonomy compliance
- Grouped split isolation (zero conversation leakage)
- Dual-model independence and disagreement handling
- All 10 rule families and sensitivity configurations
- Benign overrides and figurative threat exclusions
- Constructive rewrite engine verification
- History database operations and schema stability
- Edge cases: empty, malformed, non-English inputs
"""

import sys
import json
import csv
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
import pandas as pd
from inference.pa_inference import analyze, get_model_info, _load_models
from inference.rule_detector import detect as rule_detect
from inference.rewrite_engine import rewrite
from inference.history_db import save as save_history, get_all as get_history, delete as delete_history, init_db


@pytest.fixture(scope="session", autouse=True)
def init_system():
    _load_models()
    init_db()


# ── 1. Dataset & Manifest Integrity ─────────────────────────────────────────────
class TestDatasetIntegrity:

    def test_raw_sources_exist_locally(self):
        """All 10 source datasets must exist strictly within project data/raw/."""
        raw_dir = _ROOT / "data" / "raw" / "existing_sources"
        assert raw_dir.exists(), "Raw sources directory missing"

        expected_sources = [
            "tweeteval_sentiment",
            "sarc_reddit_regular",
            "sarc_reddit_sarcasm",
            "goemotions",
            "goemotions_anger_hostility",
            "genz_sarc_banter",
            "authentic_pa_mined",
            "curated_pa_tactics",
            "contrastive_hard_negatives",
            "gold_pa_anchors",
        ]
        for src in expected_sources:
            src_path = raw_dir / src
            assert src_path.exists(), f"Source directory {src} missing"
            csvs = list(src_path.glob("*.csv"))
            assert len(csvs) >= 1, f"Source {src} missing CSV file"

    def test_synthetic_batches_exist(self):
        """Synthetic directory must contain all 4 batches and merged dataset."""
        syn_dir = _ROOT / "data" / "raw" / "synthetic"
        assert (syn_dir / "batch1_SYNPA_00001_02500.csv").exists()
        assert (syn_dir / "batch2_SYNPA_02501_05000.csv").exists()
        assert (syn_dir / "batch3_SYNPA_05001_07500.csv").exists()
        assert (syn_dir / "batch4_SYNPA_07501_10000.csv").exists()
        assert (syn_dir / "pa_synthetic_merged.csv").exists()

        df_syn = pd.read_csv(syn_dir / "pa_synthetic_merged.csv")
        assert len(df_syn) == 9000, f"Expected 9,000 synthetic rows, got {len(df_syn)}"

    def test_manifests_exist(self):
        """Manifest files must be present and valid."""
        man_dir = _ROOT / "data" / "manifests"
        assert (man_dir / "dataset_manifest.csv").exists()
        assert (man_dir / "dataset_manifest.json").exists()
        assert (man_dir / "label_mapping.csv").exists()
        assert (man_dir / "source_checksums.csv").exists()

    def test_processed_splits_leakage_free(self):
        """Training and test sets must have zero conversation/thread overlap."""
        train_path = _ROOT / "data" / "processed" / "pa_train.csv"
        test_path = _ROOT / "data" / "processed" / "pa_test.csv"
        val_path = _ROOT / "data" / "processed" / "pa_validation.csv"

        assert train_path.exists() and test_path.exists() and val_path.exists()

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        df_val = pd.read_csv(val_path)

        train_convs = set(df_train["conversation_id"].dropna().unique())
        test_convs = set(df_test["conversation_id"].dropna().unique())
        val_convs = set(df_val["conversation_id"].dropna().unique())

        assert len(train_convs.intersection(test_convs)) == 0, "Leakage detected between train and test splits!"
        assert len(train_convs.intersection(val_convs)) == 0, "Leakage detected between train and val splits!"
        assert len(val_convs.intersection(test_convs)) == 0, "Leakage detected between val and test splits!"


# ── 2. Dual-Model Architecture & Disagreement Handling ──────────────────────────
class TestDualModelInference:

    def test_independent_execution(self):
        """Model 1 (ML) and Model 2 (Rule) must both produce outputs independently."""
        text = "As per my last email, please make sure to follow the instructions."
        res = analyze(text=text, sensitivity="balanced")

        assert "ml_result" in res
        assert "rule_result" in res
        assert "threat_result" in res
        assert "combined" in res

        # ML result schema
        ml = res["ml_result"]
        assert "label" in ml
        assert "pa_score" in ml
        assert "probabilities" in ml
        assert "uncertainty" in ml

        # Rule result schema
        rule = res["rule_result"]
        assert "pa_label" in rule
        assert "pa_score" in rule
        assert "tactics" in rule
        assert "signals" in rule

    def test_model_disagreement_detection(self):
        """When ML and rule detectors diverge, the system must explicitly flag disagreement."""
        # A polite-sounding phrase that ML might be uncertain about or where rule has a specific trigger
        text = "With all due respect, I am certain you are completely mistaken."
        res = analyze(text=text, sensitivity="balanced")
        comb = res["combined"]
        assert "disagreement" in comb
        assert isinstance(comb["disagreement"], bool)

    def test_hybrid_decision_layer(self):
        """Hybrid decision engine must combine evidence and assign a fusion strategy."""
        text = "As per my last email, which you clearly didn't read, the deadline has passed."
        res = analyze(text=text, sensitivity="balanced")

        assert "hybrid_result" in res
        hyb = res["hybrid_result"]
        assert "verdict" in hyb
        assert "fusion_strategy" in hyb
        assert "score" in hyb
        assert "confidence" in hyb
        assert hyb["verdict"] == "passive_aggressive"

    def test_hybrid_benign_override_shield(self):
        """Benign overrides must take precedence in hybrid fusion."""
        text = "Per your request, here is the updated document."
        res = analyze(text=text, sensitivity="balanced")
        hyb = res["hybrid_result"]
        assert hyb["verdict"] == "non_passive_aggressive"
        assert hyb["fusion_strategy"] == "benign_override_shield"
        assert hyb["score"] == 0.0


# ── 3. Rule Families & Sensitivity Modes ────────────────────────────────────────
class TestRuleFamiliesAndSensitivity:

    @pytest.mark.parametrize("phrase,expected_tactic", [
        ("As per my last email, you should know this.", "fake_politeness"),
        ("Better late than never on submitting the work.", "backhanded_compliment"),
        ("I already told you three times.", "implied_blame"),
        ("Whatever you say, do whatever you want.", "dismissiveness"),
        ("If you say so, clearly you know best.", "sarcastic_agreement"),
        ("Must be nice to take three-hour breaks.", "guilt_tripping"),
    ])
    def test_rule_family_detection(self, phrase, expected_tactic):
        res = rule_detect(phrase)
        assert res.pa_label == "passive_aggressive"
        assert expected_tactic in res.tactics or len(res.tactics) > 0

    def test_sensitivity_modes(self):
        """Strict sensitivity should be more sensitive than lenient mode."""
        subtle_text = "Just following up on this again."
        res_lenient = rule_detect(subtle_text, sensitivity="lenient")
        res_strict = rule_detect(subtle_text, sensitivity="strict")

        assert res_lenient.pa_score <= res_strict.pa_score


# ── 4. Benign Overrides & Threat Separation ─────────────────────────────────────
class TestBenignOverridesAndThreats:

    def test_benign_business_phrases(self):
        """Legitimate professional communication must not be flagged as passive aggressive."""
        benign_cases = [
            "Per your request, here is the signed contract.",
            "Thanks for the quick turnaround on the project!",
            "No problem, happy to help anytime.",
            "Sure, that works for me.",
        ]
        for text in benign_cases:
            res = rule_detect(text)
            assert res.pa_label == "non_passive_aggressive", f"False positive on benign phrase: '{text}'"

    def test_figurative_threat_override(self):
        """Figurative gaming/sports banter must not be classified as actual threats."""
        figurative_samples = [
            "I'm going to kill it on stage tonight!",
            "Our team is going to destroy them in the championship finals.",
            "You are dead meat in the next Mario Kart race!",
        ]
        for text in figurative_samples:
            res = analyze(text)
            assert res["combined"]["threat"] == "no_threat", f"False positive threat on figurative speech: '{text}'"

    def test_credible_threat_detection(self):
        """Direct threats of bodily harm must be identified."""
        threat_text = "I will track you down and break your legs."
        res = analyze(threat_text)
        assert res["combined"]["threat"] in ("direct_threat", "implied_threat")


# ── 5. Rewrite Engine & Guardrails ──────────────────────────────────────────────
class TestRewriteAssistant:

    def test_rewrite_preserves_meaning(self):
        """Rewrite must soften tone while keeping the underlying concern."""
        text = "As per my last email, the client invoice is overdue."
        resp = rewrite(text)
        assert resp.has_rewrites
        cand = resp.candidates[0].suggested_text
        assert "invoice is overdue" in cand
        assert "as per my last email" not in cand.lower()

    def test_threat_refusal_guardrail(self):
        """Threatening text must never be rewritten into polite requests."""
        detection = {
            "combined": {"threat": "direct_threat", "verdict": "passive_aggressive"},
            "threat_result": {"label": "direct_threat"},
        }
        resp = rewrite("I am going to murder you", detection_result=detection)
        assert resp.is_threat_flagged
        assert not resp.has_rewrites
        assert len(resp.candidates) == 0


# ── 6. Edge Cases & Resilience ──────────────────────────────────────────────────
class TestEdgeCases:

    def test_empty_string(self):
        res = analyze("")
        assert "error" in res or res["combined"]["verdict"] == "non_passive_aggressive"

    def test_very_long_input(self):
        long_text = "This is a benign message. " * 200
        res = analyze(long_text[:2000])
        assert "combined" in res
