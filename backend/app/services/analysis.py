"""
Core analysis service.
Orchestrates preprocessing, ML inference, phrase matching,
tactic scoring, explainability, and rewrite generation.
"""

from __future__ import annotations
import uuid
import time
import logging
from typing import Optional

from ..nlp.preprocess import normalize_text, split_sentences_with_offsets, validate_english_language
from ..nlp.phrase_matcher import find_phrase_matches, PhraseMatch
from ..nlp.sentiment import get_sentiment_scores
from ..nlp.rewriter import generate_rewrites
from ..ml.classifier import get_classifier
from ..schemas import (
    AnalyzeResponse,
    BatchResult,
    Highlight,
    TacticScore,
    SentenceScore,
    Probabilities,
    UncertaintyInfo,
    Label,
    Tactic,
)
from ..config import MODEL_VERSION, TACTICS

logger = logging.getLogger(__name__)


def analyze_text(
    text: str,
    context: str = "workplace",
    sensitivity: str = "balanced",
) -> AnalyzeResponse:
    """
    Full analysis pipeline for a single message.

    Steps:
      1. Normalize text
      2. Rule-based phrase matching (character offsets)
      3. ML classification (or rule-only fallback)
      4. Tactic aggregation
      5. Per-sentence scoring
      6. Rewrite generation
      7. Assemble response
    """
    t0 = time.perf_counter()
    validate_english_language(text)
    norm_text = normalize_text(text)

    # ── Step 1: Phrase matching ────────────────────────────────────────────────
    phrase_matches = find_phrase_matches(norm_text)

    # ── Step 2: ML / rule inference ───────────────────────────────────────────
    clf = get_classifier()
    pred = clf.predict(norm_text, context=context, sensitivity=sensitivity)

    label_str    = pred["label"]
    score        = pred["score"]
    raw_probs    = pred["probabilities"]
    uncertainty  = pred["uncertainty"]
    model_ver    = pred.get("model_version", clf.model_version)
    inf_mode     = pred.get("inference_mode", clf.inference_mode)

    # ── Step 3: Highlights from phrase matches ────────────────────────────────
    highlights = _build_highlights(phrase_matches)

    # ── Step 4: Tactic scores ─────────────────────────────────────────────────
    tactic_scores = _aggregate_tactics(phrase_matches, raw_probs)

    # ── Step 5: Per-sentence scores ───────────────────────────────────────────
    sentence_scores = _score_sentences(norm_text, context, sensitivity)

    # ── Step 6: Rewrite suggestions ───────────────────────────────────────────
    rewrites = generate_rewrites(norm_text)

    # ── Step 7: Assemble response ─────────────────────────────────────────────
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    try:
        probs_obj = Probabilities(**raw_probs)
    except Exception as e:
        logger.warning("Probability validation failed: %s — normalizing", e)
        probs_obj = _normalize_probs(raw_probs)

    return AnalyzeResponse(
        id=str(uuid.uuid4())[:8],
        label=Label(label_str),
        score=score,
        probabilities=probs_obj,
        tactics=tactic_scores,
        highlights=highlights,
        sentences=sentence_scores,
        rewrites=rewrites,
        uncertainty=UncertaintyInfo(**uncertainty),
        model_version=model_ver,
        latency_ms=latency_ms,
        inference_mode=inf_mode,
    )


def analyze_batch(
    messages: list[dict],
    context: str = "chat",
    sensitivity: str = "balanced",
) -> list[BatchResult]:
    """
    Analyze a list of messages (thread/conversation).
    Each message: {"sender": str, "text": str, "time": str | None}
    """
    results: list[BatchResult] = []
    for msg in messages:
        try:
            ar = analyze_text(msg["text"], context=context, sensitivity=sensitivity)
            br = BatchResult(sender=msg["sender"], **ar.model_dump())
            results.append(br)
        except Exception as e:
            logger.error("Error analyzing message from %s: %s", msg.get("sender", "?"), e)
            # Return a safe error result for this turn
            results.append(_error_batch_result(msg.get("sender", "Unknown"), str(e)))
    return results


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _build_highlights(matches: list[PhraseMatch]) -> list[Highlight]:
    seen_spans: set[tuple[int, int]] = set()
    highlights: list[Highlight] = []
    for m in sorted(matches, key=lambda x: x.start):
        span = (m.start, m.end)
        if span not in seen_spans:
            seen_spans.add(span)
            highlights.append(Highlight(
                start=m.start,
                end=m.end,
                text=m.text,
                weight=min(100.0, m.weight * 2),
                reason=m.reason,
            ))
    return highlights


def _aggregate_tactics(
    matches: list[PhraseMatch],
    probs: dict,
) -> list[TacticScore]:
    """
    Combine rule-match evidence by tactic.
    Confidence is computed from accumulated weight, not invented.
    """
    tactic_weights: dict[str, float] = {}
    for m in matches:
        tactic_weights[m.tactic] = tactic_weights.get(m.tactic, 0) + m.weight

    results: list[TacticScore] = []
    for tactic_name in TACTICS:
        weight = tactic_weights.get(tactic_name, 0)
        if weight > 0:
            # Confidence = accumulated weight / 60 (empirical scale), capped at 1
            confidence = round(min(1.0, weight / 60.0), 3)
            try:
                t = Tactic(tactic_name)
                results.append(TacticScore(name=t, confidence=confidence))
            except ValueError:
                pass

    return sorted(results, key=lambda x: x.confidence, reverse=True)


def _score_sentences(
    text: str,
    context: str,
    sensitivity: str,
) -> list[SentenceScore]:
    """Per-sentence hostility scores using the full classifier."""
    clf = get_classifier()
    sent_spans = split_sentences_with_offsets(text)
    results: list[SentenceScore] = []

    for sent, start, end in sent_spans:
        if len(sent.strip()) < 3:
            continue
        pred = clf.predict(sent, context=context, sensitivity=sensitivity)
        results.append(SentenceScore(
            text=sent,
            score=float(pred["score"]),
        ))

    return results


def _normalize_probs(raw: dict) -> Probabilities:
    """Force-normalize probabilities to sum to 1.0."""
    keys = ["neutral", "passive_aggressive", "overt_hostile"]
    vals = [max(0.0, raw.get(k, 0.0)) for k in keys]
    total = sum(vals) or 1.0
    normed = [v / total for v in vals]
    return Probabilities(**dict(zip(keys, normed)))


def _error_batch_result(sender: str, error_msg: str) -> BatchResult:
    """Return a safe placeholder result for a failed batch message."""
    return BatchResult(
        sender=sender,
        id=str(uuid.uuid4())[:8],
        label=Label.neutral,
        score=0,
        probabilities=Probabilities(neutral=1.0, passive_aggressive=0.0, overt_hostile=0.0),
        tactics=[],
        highlights=[],
        sentences=[],
        rewrites=[],
        uncertainty=UncertaintyInfo(
            entropy=1.0, max_confidence=0.0, is_uncertain=True,
            note=f"Analysis failed: {error_msg}",
        ),
        model_version="error",
        latency_ms=0.0,
        inference_mode="error",
    )
