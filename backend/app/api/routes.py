"""
API route: /api/analyze, /api/batch, /api/feedback, /api/history, /api/health, /api/model-info
"""

from __future__ import annotations
import logging
from fastapi import APIRouter, HTTPException, status

from ..schemas import (
    AnalyzeRequest, AnalyzeResponse,
    BatchRequest, BatchResponse,
    FeedbackRequest,
    HealthResponse, ModelInfoResponse,
)
from ..services.analysis import analyze_text, analyze_batch
from ..database.db import save_analysis, get_all_analyses, get_analysis, delete_all_analyses, save_feedback
from ..ml.classifier import get_classifier
from ..config import MODEL_VERSION, LABELS, TACTICS, MAX_TEXT_LENGTH, MAX_BATCH_SIZE

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Health ──────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    clf = get_classifier()
    return HealthResponse(
        status="ok",
        model_loaded=clf.is_model_loaded,
        model_version=clf.model_version,
    )


@router.get("/model-info", response_model=ModelInfoResponse, tags=["system"])
def model_info() -> ModelInfoResponse:
    clf = get_classifier()
    return ModelInfoResponse(
        model_version=clf.model_version,
        model_type="TF-IDF + LogisticRegression (rule-only fallback available)",
        labels=LABELS,
        tactics=TACTICS,
        max_text_length=MAX_TEXT_LENGTH,
        inference_mode=clf.inference_mode,
    )


# ─── Analyze ─────────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse, tags=["inference"])
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze a single message for passive aggression.

    - Text is normalized before analysis.
    - Full text is NOT logged by default (privacy-preserving).
    - Results are saved to local SQLite history.
    """
    try:
        result = analyze_text(
            text=req.text,
            context=req.context.value,
            sensitivity=req.sensitivity.value,
        )
        # Persist to history (stores snippet only)
        try:
            save_analysis(
                analysis_id=result.id,
                text=req.text,
                context=req.context.value,
                sensitivity=req.sensitivity.value,
                result_dict=result.model_dump(mode="json"),
            )
        except Exception as db_err:
            logger.warning("Failed to save analysis to DB: %s", db_err)

        return result

    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        logger.error("Analyze error: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Analysis failed")


# ─── Batch ───────────────────────────────────────────────────────────────────

@router.post("/batch", response_model=BatchResponse, tags=["inference"])
def batch(req: BatchRequest) -> BatchResponse:
    """
    Analyze a thread/conversation of multiple messages.
    Returns per-message analysis results.
    """
    if len(req.messages) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch size {len(req.messages)} exceeds limit of {MAX_BATCH_SIZE}",
        )
    try:
        msgs = [
            {"sender": m.sender, "text": m.text, "time": m.time}
            for m in req.messages
        ]
        results = analyze_batch(
            messages=msgs,
            context=req.context.value,
            sensitivity=req.sensitivity.value,
        )
        return BatchResponse(results=results)
    except Exception as e:
        logger.error("Batch error: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Batch analysis failed")


# ─── Feedback ────────────────────────────────────────────────────────────────

@router.post("/feedback", status_code=status.HTTP_200_OK, tags=["feedback"])
def feedback(req: FeedbackRequest) -> dict:
    """
    Record user feedback for a prediction.
    Feedback is stored as review data — NOT used for automatic retraining.
    """
    clf = get_classifier()
    try:
        fb_id = save_feedback(
            analysis_id=req.id,
            helpful=req.helpful,
            corrected_label=req.corrected_label.value if req.corrected_label else None,
            note=req.note,
            model_version=clf.model_version,
        )
        return {"status": "recorded", "feedback_id": fb_id}
    except Exception as e:
        logger.error("Feedback error: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to record feedback")


# ─── History ─────────────────────────────────────────────────────────────────

@router.get("/history", tags=["history"])
def history() -> list:
    """Return saved analysis history (most recent first, up to 200 records)."""
    try:
        return get_all_analyses(limit=200)
    except Exception as e:
        logger.error("History error: %s", e, exc_info=True)
        return []


@router.get("/history/{analysis_id}", tags=["history"])
def get_one_history(analysis_id: str) -> dict:
    """Return a single analysis by ID."""
    result = get_analysis(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id!r} not found")
    return result


@router.delete("/history", status_code=status.HTTP_200_OK, tags=["history"])
def clear_history() -> dict:
    """Delete all saved analyses."""
    count = delete_all_analyses()
    return {"deleted": count}
