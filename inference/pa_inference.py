"""
inference/pa_inference.py
Standalone PA + threat inference module.
No FastAPI dependency — used directly by Streamlit.

Loads:
  - models/pa_classifier_v2.pkl  (new 3-class model)
  - models/threat_classifier_v1.pkl
  - Falls back to models/pa_classifier.pkl (legacy) if v2 not available

Architecture decision: direct Python inference, no HTTP layer.
Model state is module-level singleton to avoid repeated loading.
"""

from __future__ import annotations
import json
import logging
import math
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from .preprocess import validate_english, model_ready, normalize, split_sentences
from .rule_detector import detect as rule_detect, format_signals_for_display

logger = logging.getLogger(__name__)

# ── Paths (portable — resolved relative to this file's project root) ───────────
_HERE          = Path(__file__).resolve().parent.parent
MODELS_DIR     = _HERE / "models"
PA_MODEL_V2    = MODELS_DIR / "pa_classifier_v2.pkl"
THREAT_MODEL   = MODELS_DIR / "threat_classifier_v1.pkl"
PA_MODEL_V1    = MODELS_DIR / "pa_classifier.pkl"   # legacy SVM (fallback)
REGISTRY_FILE  = MODELS_DIR / "registry" / "active_model.json"

# ── New v2 label set ────────────────────────────────────────────────────────────
PA_LABELS_V2    = ["passive_aggressive", "non_passive_aggressive", "ambiguous"]
THREAT_LABELS   = ["no_threat", "direct_threat", "implied_threat", "ambiguous"]

# ── Legacy label mapping (v1 model uses different labels) ────────────────────────
LEGACY_LABEL_MAP = {
    "neutral":            "non_passive_aggressive",
    "passive_aggressive": "passive_aggressive",
    "overt_hostile":      "passive_aggressive",  # closest v2 equivalent
}

_pa_pipeline     = None
_threat_pipeline = None
_model_version   = "rule-only"
_inference_mode  = "rule_only"
_pa_labels_active: list[str] = PA_LABELS_V2


def _load_models() -> None:
    global _pa_pipeline, _threat_pipeline, _model_version, _inference_mode, _pa_labels_active

    import joblib

    # ── Try v2 PA model first ──
    if PA_MODEL_V2.exists():
        try:
            _pa_pipeline = joblib.load(PA_MODEL_V2)
            _model_version  = "pa-logreg-v2.0"
            _inference_mode = "ml_v2"
            _pa_labels_active = PA_LABELS_V2
            logger.info("Loaded PA model v2: %s", PA_MODEL_V2)
        except Exception as e:
            logger.warning("Failed to load v2 model: %s", e)
            _pa_pipeline = None

    # ── Fallback to legacy v1 model ──
    if _pa_pipeline is None and PA_MODEL_V1.exists():
        try:
            _pa_pipeline = joblib.load(PA_MODEL_V1)
            _model_version  = "pa-tfidf-svm-v1"
            _inference_mode = "ml_v1_legacy"
            _pa_labels_active = ["neutral", "passive_aggressive", "overt_hostile"]
            logger.info("Loaded legacy PA model v1: %s (will map labels to v2)", PA_MODEL_V1)
        except Exception as e:
            logger.warning("Failed to load legacy model: %s", e)

    # ── Load threat model ──
    if THREAT_MODEL.exists():
        try:
            _threat_pipeline = joblib.load(THREAT_MODEL)
            logger.info("Loaded threat model: %s", THREAT_MODEL)
        except Exception as e:
            logger.warning("Failed to load threat model: %s", e)


def _ensure_loaded() -> None:
    global _pa_pipeline
    if _pa_pipeline is None and _model_version == "rule-only":
        _load_models()


def _entropy(probs: list[float]) -> float:
    n = len(probs)
    max_ent = math.log2(n) if n > 1 else 1.0
    return -sum(p * math.log2(p + 1e-12) for p in probs) / max_ent


# ── Public API ─────────────────────────────────────────────────────────────────

def get_model_info() -> dict:
    """Return information about the currently loaded models."""
    _ensure_loaded()
    return {
        "pa_model_version":   _model_version,
        "inference_mode":     _inference_mode,
        "pa_model_loaded":    _pa_pipeline is not None,
        "threat_model_loaded": _threat_pipeline is not None,
        "pa_labels":          _pa_labels_active,
        "threat_labels":      THREAT_LABELS,
    }


def analyze(
    text: str,
    context: str = "social",
    sensitivity: str = "balanced",
    post_text: Optional[str] = None,
    parent_comment: Optional[str] = None,
) -> dict:
    """
    Full analysis pipeline. Returns a structured result dict.

    Args:
        text:           The message to analyse.
        context:        Context hint (social | workplace | chat).
        sensitivity:    Sensitivity level (lenient | balanced | strict).
        post_text:      Original post text for context.
        parent_comment: Parent comment for context.

    Returns:
        {
          analysis_id, text_snippet, timestamp,
          ml_result: {...},
          rule_result: {...},
          threat_result: {...},
          combined_summary: {...},
        }
    """
    t0 = time.perf_counter()
    _ensure_loaded()

    # Validate & preprocess
    if not text or not text.strip():
        return {"error": "Input text cannot be empty or whitespace.", "analysis_id": str(uuid.uuid4())[:8]}

    try:
        validate_english(text)
    except ValueError as e:
        return {"error": str(e), "analysis_id": str(uuid.uuid4())[:8]}

    norm = normalize(text)
    model_input = model_ready(text)

    # Enrich with context
    if parent_comment and len(parent_comment.strip()) > 5:
        parent_clean = model_ready(parent_comment)
        model_input_ctx = f"[context]: {parent_clean} [reply]: {model_input}"
    else:
        model_input_ctx = model_input

    analysis_id = str(uuid.uuid4())[:8]

    # ── ML PA Inference ────────────────────────────────────────────────────────
    ml_result = _ml_pa_predict(model_input_ctx, sensitivity)

    # ── Threat Inference ───────────────────────────────────────────────────────
    threat_result = _threat_predict(model_input_ctx)

    # ── Rule-Based Inference ───────────────────────────────────────────────────
    rule_res = rule_detect(norm, context=context, parent_comment=parent_comment, sensitivity=sensitivity)
    rule_result = {
        "pa_label":          rule_res.pa_label,
        "primary_label":     rule_res.primary_label,
        "threat_label":      rule_res.threat_label,
        "pa_score":          rule_res.pa_score,
        "threat_score":      rule_res.threat_score,
        "strength_score":    rule_res.strength_score,
        "pa_confidence":     rule_res.pa_confidence,
        "threat_confidence": rule_res.threat_confidence,
        "tactics":           rule_res.tactics,
        "tactic_labels":     rule_res.tactic_labels,
        "signals":           format_signals_for_display(rule_res.signals),
        "threat_signals":    format_signals_for_display(rule_res.threat_signals),
        "matched_spans":     rule_res.matched_spans,
        "rule_ids":          rule_res.rule_ids,
        "rationale":         rule_res.rationale,
        "is_ambiguous":      rule_res.is_ambiguous,
        "needs_review":      rule_res.needs_review,
        "inference_mode":    "rule_based",
    }

    # ── Per-sentence scores ────────────────────────────────────────────────────
    sentences = split_sentences(norm)
    sentence_scores = []
    if _pa_pipeline is not None and len(sentences) > 1:
        for sent in sentences[:8]:  # max 8 sentences
            if len(sent.strip()) < 5:
                continue
            s_pred = _ml_pa_predict(model_ready(sent), sensitivity)
            sentence_scores.append({
                "text":  sent,
                "label": s_pred.get("label", ""),
                "score": s_pred.get("pa_score", 0),
            })

    # ── Combined Summary ───────────────────────────────────────────────────────
    combined = _combine_results(ml_result, rule_result, threat_result, sensitivity)

    total_latency = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "analysis_id":     analysis_id,
        "text_snippet":    norm[:120] + ("..." if len(norm) > 120 else ""),
        "ml_result":       ml_result,
        "rule_result":     rule_result,
        "hybrid_result":   combined,
        "threat_result":   threat_result,
        "sentence_scores": sentence_scores,
        "combined":        combined,
        "model_info":      get_model_info(),
        "total_latency_ms": total_latency,
    }


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _ml_pa_predict(text: str, sensitivity: str) -> dict:
    """Run ML PA prediction. Returns result dict."""
    if _pa_pipeline is None:
        return {
            "label":       "non_passive_aggressive",
            "pa_score":    0,
            "probabilities": {},
            "uncertainty": {"is_uncertain": True, "note": "No ML model loaded — rule-only mode"},
            "inference_mode": "rule_only",
        }

    try:
        probs_raw = _pa_pipeline.predict_proba([text])[0]
        classes   = _pa_pipeline.classes_

        probs_dict = {str(c): round(float(p), 4) for c, p in zip(classes, probs_raw)}

        # Map legacy labels to v2 labels if needed
        if _inference_mode == "ml_v1_legacy":
            v2_probs: dict[str, float] = {lbl: 0.0 for lbl in PA_LABELS_V2}
            for old_lbl, prob in probs_dict.items():
                new_lbl = LEGACY_LABEL_MAP.get(old_lbl, "non_passive_aggressive")
                v2_probs[new_lbl] = v2_probs.get(new_lbl, 0.0) + prob
            probs_dict = v2_probs

        label_idx = int(np.argmax(probs_raw))
        raw_label  = str(classes[label_idx])
        label = LEGACY_LABEL_MAP.get(raw_label, raw_label) if _inference_mode == "ml_v1_legacy" else raw_label

        # Sensitivity threshold adjustment
        pa_prob = probs_dict.get("passive_aggressive", 0.0)
        thresholds = {
            "lenient":  0.35,
            "balanced": 0.40,
            "strict":   0.30,
        }
        pa_threshold = thresholds.get(sensitivity, 0.40)

        if label == "non_passive_aggressive" and pa_prob >= pa_threshold:
            label = "passive_aggressive"
        elif label == "passive_aggressive" and pa_prob < pa_threshold and sensitivity == "lenient":
            label = "ambiguous"

        max_prob = float(max(probs_raw))
        entropy  = _entropy(probs_raw.tolist())

        # PA score (0-100): probability × 100 for PA class, with secondary weighting
        pa_score = round(pa_prob * 100, 1)

        return {
            "label":         label,
            "pa_score":      pa_score,
            "probabilities": probs_dict,
            "uncertainty": {
                "entropy":        round(entropy, 4),
                "max_confidence": round(max_prob, 4),
                "is_uncertain":   entropy > 0.75 or max_prob < 0.50,
                "note": (
                    "Model is uncertain — context may improve accuracy."
                    if (entropy > 0.75 or max_prob < 0.50)
                    else "Model has reasonable confidence."
                ),
            },
            "inference_mode": _inference_mode,
        }

    except Exception as e:
        logger.error("ML PA predict error: %s", e)
        return {
            "label": "non_passive_aggressive",
            "pa_score": 0,
            "probabilities": {},
            "uncertainty": {"is_uncertain": True, "note": f"Prediction error: {e}"},
            "inference_mode": "error",
        }


def _threat_predict(text: str) -> dict:
    """Run threat classification."""
    if _threat_pipeline is None:
        # Use rule-based fallback
        return {
            "label":         "no_threat",
            "probabilities": {},
            "inference_mode": "rule_fallback",
            "note": "Threat ML model not loaded — using rule-based detection",
        }

    try:
        probs_raw = _threat_pipeline.predict_proba([text])[0]
        classes   = _threat_pipeline.classes_
        probs_dict = {str(c): round(float(p), 4) for c, p in zip(classes, probs_raw)}
        label_idx  = int(np.argmax(probs_raw))
        label      = str(classes[label_idx])
        max_prob   = float(max(probs_raw))

        return {
            "label":         label,
            "probabilities": probs_dict,
            "confidence":    round(max_prob, 4),
            "is_uncertain":  max_prob < 0.50,
            "inference_mode": "ml_threat",
        }
    except Exception as e:
        logger.error("Threat predict error: %s", e)
        return {
            "label": "no_threat",
            "probabilities": {},
            "inference_mode": "error",
        }


def _combine_results(
    ml: dict,
    rule: dict,
    threat: dict,
    sensitivity: str,
) -> dict:
    """
    Evidence-based Hybrid Decision Layer:
    Combines Model 1 (ML statistical probability) with Model 2 (deterministic linguistic rules).
    Resolves disagreements with defensible, transparent fusion policies:
      1. Benign Override Shield (deterministic non-PA takes precedence over statistical noise)
      2. Concordant Consensus (when ML and Rules align)
      3. Rule-Supported Escalation (when high-confidence tactic pattern is verified)
      4. ML Statistical Generalization (when strong statistical syntax is detected without dictionary match)
      5. Terse Ambiguity Arbitration (when isolated short replies lack sufficient contextual signals)
      6. Conflicting Evidence Arbitration (flags disagreement and returns ambiguous/needs-review)
    """
    ml_label   = ml.get("label", "non_passive_aggressive")
    rule_label = rule.get("pa_label", "non_passive_aggressive")
    ml_score   = float(ml.get("pa_score", 0.0))
    rule_score = float(rule.get("pa_score", 0.0))
    agreement  = (ml_label == rule_label)

    # 1. Threat Fusion: Independent threat evaluation
    rule_threat = rule.get("threat_label", "no_threat")
    rule_threat_score = float(rule.get("threat_score", 0.0))
    ml_threat_label = threat.get("label", "no_threat")

    if rule_threat in ["direct_threat", "implied_threat"] and rule_threat_score >= 30:
        threat_label = rule_threat
    elif threat.get("inference_mode") == "rule_fallback" or threat.get("is_uncertain", False):
        threat_label = rule_threat
    else:
        threat_label = ml_threat_label

    flags = []
    rule_ids = rule.get("rule_ids", [])

    # 2. Benign Override Shield:
    if "RULE_BENIGN_OVERRIDE" in rule_ids or (rule.get("primary_label") == "non_passive_aggressive" and any("override" in str(r).lower() for r in rule.get("rationale", []))):
        verdict = "non_passive_aggressive"
        fusion_strategy = "benign_override_shield"
        hybrid_score = 0.0
        confidence = max(0.90, rule.get("pa_confidence", 0.90))
        if ml_label == "passive_aggressive":
            flags.append("Benign conversational override suppressed a statistical ML false positive.")

    # 3. Terse Ambiguity Guard:
    elif rule.get("is_ambiguous") and len(rule.get("signals", [])) == 0 and ml.get("uncertainty", {}).get("is_uncertain", False):
        verdict = "ambiguous"
        fusion_strategy = "terse_context_boundary"
        hybrid_score = round((ml_score + rule_score) / 2.0, 1)
        confidence = 0.50
        flags.append("Message is terse and lacks clear conversational context — flagged as ambiguous.")

    # 4. Concordant Consensus:
    elif agreement:
        verdict = ml_label
        fusion_strategy = "concordant_consensus"
        hybrid_score = round(0.60 * ml_score + 0.40 * rule_score, 1)
        confidence = min(0.99, round(0.60 * ml.get("probabilities", {}).get(ml_label, 0.70) + 0.40 * rule.get("pa_confidence", 0.70), 3))

    # 5. Rule-Supported Escalation:
    elif rule_label == "passive_aggressive" and rule_score >= 35.0 and len(rule.get("tactics", [])) >= 1:
        verdict = "passive_aggressive"
        fusion_strategy = "rule_supported_escalation"
        hybrid_score = max(ml_score, rule_score)
        confidence = min(0.92, round(0.50 * (rule.get("pa_confidence", 0.75) + 0.80), 3))
        flags.append(f"Specific passive-aggression tactic detected by rules ({', '.join(rule.get('tactics', []))}).")

    # 6. ML Statistical Generalization:
    elif ml_label == "passive_aggressive" and ml.get("probabilities", {}).get("passive_aggressive", 0.0) >= 0.75 and not ml.get("uncertainty", {}).get("is_uncertain", False):
        verdict = "passive_aggressive"
        fusion_strategy = "ml_statistical_generalization"
        hybrid_score = ml_score
        confidence = round(ml.get("probabilities", {}).get("passive_aggressive", 0.75), 3)
        flags.append("Statistical n-gram patterns indicate passive aggression without exact lexicon match.")

    # 7. Disagreement / Ambiguity Arbitration:
    else:
        if sensitivity == "strict":
            verdict = "passive_aggressive" if (ml_label == "passive_aggressive" or rule_label == "passive_aggressive") else "ambiguous"
            fusion_strategy = "strict_sensitivity_precaution"
        elif sensitivity == "lenient":
            verdict = "non_passive_aggressive" if (ml_label == "non_passive_aggressive" or rule_label == "non_passive_aggressive") else "ambiguous"
            fusion_strategy = "lenient_sensitivity_filtering"
        else:
            verdict = "ambiguous"
            fusion_strategy = "conflict_uncertainty_arbitration"
        hybrid_score = round((ml_score + rule_score) / 2.0, 1)
        confidence = 0.50
        flags.append(f"Model 1 (ML: {ml_label}) and Model 2 (Rule: {rule_label}) diverged.")

    if not agreement:
        flags.append(f"Divergence detected: ML predicted '{ml_label}', Rules predicted '{rule_label}'.")

    if threat_label != "no_threat":
        flags.append(f"⚠️ Threat detected: {threat_label.replace('_', ' ')}")

    needs_review = (not agreement) or (threat_label != "no_threat") or ml.get("uncertainty", {}).get("is_uncertain", False) or rule.get("needs_review", False) or verdict == "ambiguous"

    return {
        "verdict":         verdict,
        "score":           hybrid_score,
        "pa_score":        hybrid_score,
        "confidence":      round(confidence, 3),
        "threat":          threat_label,
        "fusion_strategy": fusion_strategy,
        "agreement":       agreement,
        "disagreement":    not agreement,
        "flags":           flags,
        "needs_review":    needs_review,
        "explanation":     _build_explanation(verdict, rule, threat_label, fusion_strategy),
    }


def _build_explanation(verdict: str, rule: dict, threat_label: str, fusion_strategy: str = "") -> str:
    lines = []

    label_explanations = {
        "passive_aggressive": "The message shows evidence of indirect hostility — phrases or syntactic patterns that express negativity covertly rather than directly.",
        "non_passive_aggressive": "The message does not show clear signs of passive aggression. It expresses direct communication, genuine sentiment, or neutral information.",
        "ambiguous": "The message is ambiguous — intent cannot be definitively established without additional conversational context.",
    }
    lines.append(label_explanations.get(verdict, ""))

    tactics = rule.get("tactics", [])
    if tactics:
        tactic_names = ", ".join(t.replace("_", " ") for t in tactics[:3])
        lines.append(f"Identified tactics: {tactic_names}.")

    if fusion_strategy == "benign_override_shield":
        lines.append("A verified benign conversational override confirmed this phrasing is constructive/polite.")
    elif fusion_strategy == "rule_supported_escalation":
        lines.append("A deterministic linguistic tactic pattern confirmed covert hostility.")

    if threat_label != "no_threat":
        lines.append(f"Threat classification: {threat_label.replace('_', ' ')} (monitored separately from passive aggression).")

    return " ".join(l for l in lines if l)
