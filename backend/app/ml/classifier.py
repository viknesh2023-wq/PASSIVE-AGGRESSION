"""
ML Classifier pipeline.
Wraps scikit-learn TF-IDF + LogisticRegression as the primary trained model.
The pipeline is trained on a gold-label dataset — see training/train.py.

If no trained model file exists, falls back to a rule-only scorer.
"""

from __future__ import annotations
import os
import math
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
LABELS = ["neutral", "passive_aggressive", "overt_hostile"]
MODEL_FILE = Path(__file__).parent.parent.parent.parent / "models" / "pa_classifier.pkl"
REGISTRY_FILE = Path(__file__).parent.parent.parent.parent / "models" / "registry" / "active_model.json"


class PAClassifier:
    """
    Passive Aggression classifier.

    Inference priority:
      1. Trained scikit-learn pipeline (if model file exists)
      2. Rule-only fallback (phrase bank + style signals)
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._model_version = "rule-only-v1.0"
        self._inference_mode = "rule_only"
        self._loaded = False
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained sklearn pipeline if available."""
        if MODEL_FILE.exists():
            try:
                import joblib
                self._pipeline = joblib.load(MODEL_FILE)
                self._inference_mode = "ml_classifier"
                # Read registry for version
                if REGISTRY_FILE.exists():
                    with open(REGISTRY_FILE) as f:
                        reg = json.load(f)
                    self._model_version = reg.get("model_version", "pa-logreg-v1")
                else:
                    self._model_version = "pa-logreg-v1"
                self._loaded = True
                logger.info("Loaded trained PA classifier: %s", self._model_version)
            except Exception as e:
                logger.warning("Could not load trained model %s: %s — using rule-only fallback", MODEL_FILE, e)
                self._pipeline = None
                self._loaded = False
        else:
            logger.info("No trained model file found at %s — using rule-only scoring", MODEL_FILE)
            self._loaded = False

    def predict(
        self,
        text: str,
        context: str = "workplace",
        sensitivity: str = "balanced",
    ) -> dict:
        """
        Run inference and return a prediction dict.

        Returns:
            label, score (0-100), probabilities dict, uncertainty, model_version, inference_mode
        """
        t0 = time.perf_counter()

        if self._pipeline is not None and self._loaded:
            result = self._ml_predict(text, sensitivity)
        else:
            result = self._rule_predict(text, sensitivity)

        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result["model_version"] = self._model_version
        result["inference_mode"] = self._inference_mode
        return result

    def _ml_predict(self, text: str, sensitivity: str) -> dict:
        """Use the trained sklearn pipeline."""
        from ..nlp.preprocess import model_ready_text
        ready = model_ready_text(text)
        probs_raw = self._pipeline.predict_proba([ready])[0]

        # Apply sensitivity-based threshold adjustment
        thresholds = _sensitivity_thresholds(sensitivity)
        probs = _calibrate_probs(probs_raw, thresholds)

        label_idx = int(np.argmax(probs))
        label = LABELS[label_idx]
        score = _probs_to_score(probs)
        uncertainty = _compute_uncertainty(probs)

        return {
            "label": label,
            "score": score,
            "probabilities": {
                "neutral":            round(float(probs[0]), 4),
                "passive_aggressive": round(float(probs[1]), 4),
                "overt_hostile":      round(float(probs[2]), 4),
            },
            "uncertainty": uncertainty,
        }

    def _rule_predict(self, text: str, sensitivity: str) -> dict:
        """
        Rule-only fallback using phrase bank + style signals.
        Clearly labeled as rule_only inference.
        """
        from ..nlp.phrase_matcher import find_phrase_matches, compute_style_signals

        matches = find_phrase_matches(text)
        style   = compute_style_signals(text)

        # Accumulate raw evidence weight
        raw_score = sum(m.weight for m in matches)
        raw_score += style["ellipsis_bonus"] + style["caps_bonus"] + style["rhet_bonus"]

        # Sensitivity multiplier
        mult = {"lenient": 0.80, "balanced": 1.0, "strict": 1.20}.get(sensitivity, 1.0)
        adjusted = min(100.0, max(0.0, raw_score * mult))
        score = int(round(adjusted))

        # Map score to label using sensitivity-aware thresholds
        thresholds = _sensitivity_thresholds(sensitivity)
        if adjusted < thresholds["neutral_max"]:
            label = "neutral"
            probs = _score_to_rule_probs(adjusted, "neutral", thresholds)
        elif adjusted <= thresholds["pa_max"]:
            label = "passive_aggressive"
            probs = _score_to_rule_probs(adjusted, "passive_aggressive", thresholds)
        else:
            label = "overt_hostile"
            probs = _score_to_rule_probs(adjusted, "overt_hostile", thresholds)

        uncertainty = _compute_uncertainty(list(probs.values()))

        return {
            "label": label,
            "score": score,
            "probabilities": probs,
            "uncertainty": uncertainty,
        }

    @property
    def is_model_loaded(self) -> bool:
        return self._loaded

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def inference_mode(self) -> str:
        return self._inference_mode


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _sensitivity_thresholds(sensitivity: str) -> dict[str, float]:
    return {
        "lenient":  {"neutral_max": 35.0, "pa_max": 70.0},
        "balanced": {"neutral_max": 30.0, "pa_max": 60.0},
        "strict":   {"neutral_max": 22.0, "pa_max": 50.0},
    }.get(sensitivity, {"neutral_max": 30.0, "pa_max": 60.0})


def _probs_to_score(probs: "np.ndarray | list[float]") -> int:
    """Convert probability array to 0-100 display score."""
    if hasattr(probs, 'tolist'):
        probs = probs.tolist()
    # Weighted sum: overt_hostile = 100, pa = 60, neutral = 0
    weights = [0.0, 60.0, 100.0]
    score = sum(p * w for p, w in zip(probs, weights))
    return int(round(min(100, max(0, score))))


def _calibrate_probs(
    raw: "np.ndarray",
    thresholds: dict,
) -> "np.ndarray":
    """Return probabilities as-is (calibration from training)."""
    # Temperature scaling could go here post-training
    return raw


def _score_to_rule_probs(
    score: float,
    label: str,
    thresholds: dict,
) -> dict[str, float]:
    """
    Construct pseudo-probabilities from a rule score.
    These are NOT calibrated probabilities — they are heuristic mappings.
    Clearly labeled as rule_only inference mode.
    """
    n_max = thresholds["neutral_max"]
    pa_max = thresholds["pa_max"]

    if label == "neutral":
        conf = max(0.4, 1.0 - (score / n_max) * 0.5)
        pa   = (1 - conf) * 0.7
        oh   = (1 - conf) * 0.3
    elif label == "passive_aggressive":
        t = (score - n_max) / max(1, pa_max - n_max)
        conf = 0.4 + t * 0.4
        n    = max(0.05, 1 - conf - 0.1)
        oh   = max(0.05, 1 - conf - n)
        conf = min(0.85, conf)
        pa   = conf
        n_   = max(0.05, 1 - pa - oh)
        return {
            "neutral":            round(n_, 4),
            "passive_aggressive": round(pa, 4),
            "overt_hostile":      round(oh, 4),
        }
    else:  # overt_hostile
        t = min(1.0, (score - pa_max) / max(1, 100 - pa_max))
        conf = 0.45 + t * 0.45
        pa   = max(0.05, (1 - conf) * 0.6)
        n_   = max(0.05, 1 - conf - pa)
        return {
            "neutral":            round(n_, 4),
            "passive_aggressive": round(pa, 4),
            "overt_hostile":      round(conf, 4),
        }

    return {
        "neutral":            round(conf, 4),
        "passive_aggressive": round(pa, 4),
        "overt_hostile":      round(oh, 4),
    }


def _compute_uncertainty(probs) -> dict:
    """Compute Shannon entropy and max confidence for uncertainty reporting."""
    if hasattr(probs, 'tolist'):
        p_list = probs.tolist()
    elif isinstance(probs, dict):
        p_list = list(probs.values())
    else:
        p_list = list(probs)

    # Shannon entropy (base 2), normalized to [0,1]
    n = len(p_list)
    max_entropy = math.log2(n) if n > 1 else 1.0
    entropy = -sum(p * math.log2(p + 1e-12) for p in p_list) / max_entropy
    max_conf = max(p_list)

    return {
        "entropy":        round(entropy, 4),
        "max_confidence": round(max_conf, 4),
        "is_uncertain":   entropy > 0.75 or max_conf < 0.50,
        "note": (
            "Model is uncertain — result may be unreliable. Context may be required."
            if (entropy > 0.75 or max_conf < 0.50)
            else "Model has reasonable confidence in this prediction."
        ),
    }


# ─── Module-level singleton ───────────────────────────────────────────────────
_classifier: Optional[PAClassifier] = None


def get_classifier() -> PAClassifier:
    global _classifier
    if _classifier is None:
        _classifier = PAClassifier()
    return _classifier
