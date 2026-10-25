"""
Feature extraction pipeline.
Produces a feature dict suitable for ML classifiers and explainability.
All features are computed on training data only — no test-set contamination.
"""

from __future__ import annotations
import re
from typing import Any

from .preprocess import (
    normalize_text,
    model_ready_text,
    split_sentences,
    compute_text_stats,
)
from .phrase_matcher import find_phrase_matches, compute_style_signals
from .sentiment import get_sentiment_scores, detect_polarity_contrast


# ─── Feature groups ─────────────────────────────────────────────────────────────


def extract_lexical_features(text: str) -> dict[str, Any]:
    """
    Lexical surface features — character-level signals.
    """
    lower = text.lower()
    stats = compute_text_stats(text)
    return {
        "char_count":       stats["char_count"],
        "word_count":       stats["word_count"],
        "sentence_count":   stats["sentence_count"],
        "avg_word_len":     stats["avg_word_len"],
        "avg_sent_len":     stats["avg_sent_len"],
        # Negation markers
        "has_negation":     int(bool(re.search(r"\b(not|never|no|nobody|nothing|neither|nor|nowhere|hardly|barely|scarcely)\b", lower))),
        # Contraction count (informal register)
        "contraction_count": len(re.findall(r"\b\w+'(t|re|ve|ll|d|s|m)\b", lower)),
    }


def extract_style_features(text: str) -> dict[str, Any]:
    """
    Stylistic signals: capitalization, punctuation patterns, emojis.
    These are weak heuristic signals, not standalone classifiers.
    """
    signals = compute_style_signals(text)
    lower   = text.lower()

    # Emoji presence (rough Unicode range)
    emoji_pattern = re.compile(
        "[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001FA00-\U0001FA9F]",
        flags=re.UNICODE,
    )
    emoji_count = len(emoji_pattern.findall(text))

    return {
        "ellipsis_count":       signals["ellipsis_count"],
        "caps_word_count":      signals["caps_word_count"],
        "rhetorical_question":  signals["rhetorical_question"],
        "exclamation_count":    signals["exclamation_count"],
        "question_count":       signals["question_count"],
        "emoji_count":          emoji_count,
        # Punctuation density
        "punct_density":        len(re.findall(r'[!?.]', text)) / max(1, len(text.split())),
    }


def extract_phrase_features(text: str) -> dict[str, Any]:
    """
    Phrase-bank match counts by tactic, and total evidence score.
    """
    matches = find_phrase_matches(text)

    tactic_counts = {
        "phrase_sarcasm":               0,
        "phrase_backhanded_compliment": 0,
        "phrase_guilt_tripping":        0,
        "phrase_mock_politeness":       0,
        "phrase_blame_shifting":        0,
        "phrase_curt_dismissal":        0,
    }
    total_weight = 0.0
    for m in matches:
        key = f"phrase_{m.tactic}"
        if key in tactic_counts:
            tactic_counts[key] += 1
        total_weight += m.weight

    return {
        **tactic_counts,
        "phrase_total_weight":  total_weight,
        "phrase_match_count":   len(matches),
    }


def extract_sentiment_features(text: str) -> dict[str, Any]:
    """
    VADER sentiment features.
    """
    sentences = split_sentences(text)
    sent_scores = get_sentiment_scores(text)
    contrast    = detect_polarity_contrast(sentences)

    return {
        "vader_compound":   sent_scores["compound"],
        "vader_pos":        sent_scores["pos"],
        "vader_neg":        sent_scores["neg"],
        "vader_neu":        sent_scores["neu"],
        "polarity_contrast": contrast,
        "sentiment_available": int(sent_scores.get("available", 0)),
    }


def extract_all_features(text: str) -> dict[str, Any]:
    """
    Full feature vector — union of all feature groups.
    Call this for ML classifier training and inference.
    """
    norm = normalize_text(text)
    return {
        **extract_lexical_features(norm),
        **extract_style_features(norm),
        **extract_phrase_features(norm),
        **extract_sentiment_features(norm),
    }


def feature_names() -> list[str]:
    """Return a stable list of feature names (same order as extract_all_features)."""
    dummy = extract_all_features("hello")
    return list(dummy.keys())
