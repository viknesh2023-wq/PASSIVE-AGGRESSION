"""
Pydantic schemas matching the TypeScript contract in d:/python_folder/src/lib/types.ts
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Enums ─────────────────────────────────────────────────────────────────────

class Label(str, Enum):
    neutral            = "neutral"
    passive_aggressive = "passive_aggressive"
    overt_hostile      = "overt_hostile"


class Tactic(str, Enum):
    sarcasm                = "sarcasm"
    backhanded_compliment  = "backhanded_compliment"
    guilt_tripping         = "guilt_tripping"
    mock_politeness        = "mock_politeness"
    blame_shifting         = "blame_shifting"
    curt_dismissal         = "curt_dismissal"


class Context(str, Enum):
    workplace = "workplace"
    chat      = "chat"
    social    = "social"


class Sensitivity(str, Enum):
    lenient  = "lenient"
    balanced = "balanced"
    strict   = "strict"


# ─── Nested result structures ───────────────────────────────────────────────────

class Highlight(BaseModel):
    start:  int   = Field(..., description="Start char offset in original text")
    end:    int   = Field(..., description="End char offset in original text (exclusive)")
    text:   str   = Field(..., description="Matched substring")
    weight: float = Field(..., description="Hostility weight 0-100")
    reason: str   = Field(..., description="Human-readable explanation")


class TacticScore(BaseModel):
    name:       Tactic = Field(..., description="Tactic identifier")
    confidence: float  = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")


class SentenceScore(BaseModel):
    text:  str   = Field(..., description="Sentence text")
    score: float = Field(..., ge=0.0, le=100.0, description="Per-sentence hostility score 0-100")


class Probabilities(BaseModel):
    neutral:            float = Field(..., ge=0.0, le=1.0)
    passive_aggressive: float = Field(..., ge=0.0, le=1.0)
    overt_hostile:      float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def probabilities_sum_to_one(self) -> "Probabilities":
        total = self.neutral + self.passive_aggressive + self.overt_hostile
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Probabilities must sum to ~1.0, got {total:.4f}")
        return self


class UncertaintyInfo(BaseModel):
    entropy:        float  = Field(..., description="Prediction entropy (higher = less certain)")
    max_confidence: float  = Field(..., description="Highest class probability")
    is_uncertain:   bool   = Field(..., description="True if model is uncertain")
    note:           str    = Field("", description="Human-readable uncertainty explanation")


# ─── Request schemas ────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text:        str         = Field(..., min_length=1, max_length=2000)
    context:     Context     = Field(Context.workplace)
    sensitivity: Sensitivity = Field(Sensitivity.balanced)

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class BatchMessage(BaseModel):
    sender: str       = Field(..., min_length=1)
    text:   str       = Field(..., min_length=1, max_length=2000)
    time:   Optional[str] = None


class BatchRequest(BaseModel):
    messages:    list[BatchMessage] = Field(..., min_length=1, max_length=50)
    context:     Context            = Field(Context.chat)
    sensitivity: Sensitivity        = Field(Sensitivity.balanced)


class FeedbackRequest(BaseModel):
    id:              str            = Field(..., min_length=1)
    helpful:         bool
    corrected_label: Optional[Label] = None
    note:            Optional[str]   = Field(None, max_length=500)


# ─── Response schemas ───────────────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    id:            str             = Field(..., description="Unique analysis ID")
    label:         Label           = Field(..., description="Top-level verdict")
    score:         int             = Field(..., ge=0, le=100, description="0-100 display score")
    probabilities: Probabilities   = Field(..., description="Calibrated class probabilities")
    tactics:       list[TacticScore]   = Field(default_factory=list)
    highlights:    list[Highlight]     = Field(default_factory=list)
    sentences:     list[SentenceScore] = Field(default_factory=list)
    rewrites:      list[str]           = Field(default_factory=list)
    uncertainty:   UncertaintyInfo     = Field(..., description="Model uncertainty information")
    model_version: str             = Field(..., description="Active model version string")
    latency_ms:    float           = Field(..., description="Inference latency in milliseconds")
    inference_mode: str            = Field("real", description="'real' or 'rule_only'")


class BatchResult(AnalyzeResponse):
    sender: str = Field(..., description="Message sender name")


class BatchResponse(BaseModel):
    results: list[BatchResult]


class HealthResponse(BaseModel):
    status:        str  = "ok"
    model_loaded:  bool = False
    model_version: str  = ""


class ModelInfoResponse(BaseModel):
    model_version: str
    model_type:    str
    labels:        list[str]
    tactics:       list[str]
    max_text_length: int
    inference_mode:  str
    trained_on:    str = "See models/registry/ for dataset manifest"
