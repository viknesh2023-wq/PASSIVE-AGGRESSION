"""
inference/query_router.py — Gemini-Powered Context Query Router.

Analyzes a single user-submitted statement and determines whether it belongs to:
  - "social_media" -> Routed to the social-media passive-aggression detector
  - "workplace"    -> Routed to the workplace passive-aggression detector API
  - "uncertain"    -> Prompts user for clarification / manual selection

Uses google-genai SDK with structured output enforcement (Pydantic schema).
Gracefully falls back to heuristic domain analysis and manual clarification if
GEMINI_API_KEY is not configured or network/quota error occurs.
"""

import os
import re
import json
import time
from typing import Literal, List, Optional, Dict, Any
from pydantic import BaseModel, Field

# Schema definition for Gemini Structured Outputs
class RoutingDecision(BaseModel):
    route: Literal["social_media", "workplace", "uncertain"] = Field(
        ...,
        description="The target domain: 'social_media', 'workplace', or 'uncertain'."
    )
    explanation: str = Field(
        ...,
        description="Concise rationale explaining the contextual linguistic signals that motivated the routing decision."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score in the domain classification [0.0 - 1.0]."
    )
    needs_clarification: bool = Field(
        ...,
        description="True if context is ambiguous or insufficient, requiring user clarification."
    )
    detected_domain_cues: List[str] = Field(
        default_factory=list,
        description="Key domain indicators or terms extracted from the statement."
    )


class RouteResult:
    """Standardized result returned by the query router."""

    def __init__(
        self,
        route: Literal["social_media", "workplace", "uncertain"],
        explanation: str,
        confidence: float,
        needs_clarification: bool,
        detected_domain_cues: List[str],
        is_fallback: bool = False,
        router_model: str = "gemini-2.0-flash",
        latency_ms: float = 0.0,
    ):
        self.route = route
        self.explanation = explanation
        self.confidence = confidence
        self.needs_clarification = needs_clarification
        self.detected_domain_cues = detected_domain_cues
        self.is_fallback = is_fallback
        self.router_model = router_model
        self.latency_ms = latency_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route": self.route,
            "explanation": self.explanation,
            "confidence": round(self.confidence, 3),
            "needs_clarification": self.needs_clarification,
            "detected_domain_cues": self.detected_domain_cues,
            "is_fallback": self.is_fallback,
            "router_model": self.router_model,
            "latency_ms": round(self.latency_ms, 2),
        }


def _get_gemini_api_key() -> Optional[str]:
    """Retrieve Gemini API key from environment or Streamlit secrets."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()

    # Attempt to retrieve from Streamlit secrets if running inside Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass

    return None


def _fallback_heuristic_router(text: str, reason: str) -> RouteResult:
    """
    Deterministic context classifier used when Gemini API is unavailable or unconfigured.
    Never silently guesses — defaults to 'uncertain' unless strong domain markers are present.
    """
    lower = text.lower()

    workplace_cues = [
        "per my last email", "per my previous email", "as discussed", "standup", "sprint",
        "deadline", "deliverable", "client", "quarterly", "report", "manager", "boss",
        "meeting agenda", "colleague", "coworker", "deck", "slide deck", "cc'd", "jira",
        "slack channel", "teams call", "presentation", "synced with", "performance review"
    ]
    social_cues = [
        "ratio", "tweet", "retweet", "quote tweet", "reddit", "subthread", "downvote",
        "upvote", "discord", "streamer", "stream chat", "subscribers", "follower", "vlog",
        "post", "insta", "tiktok", "algorithm", "gaming", "ranked match", "lfg", "tournament",
        "bronze", "diamond rank", "slayed", "queen", "unpopular opinion"
    ]

    matched_workplace = [w for w in workplace_cues if w in lower]
    matched_social = [s for s in social_cues if s in lower]

    if matched_workplace and not matched_social:
        return RouteResult(
            route="workplace",
            explanation=f"{reason} Detected clear workplace cues: {', '.join(matched_workplace[:3])}.",
            confidence=0.85,
            needs_clarification=False,
            detected_domain_cues=matched_workplace[:4],
            is_fallback=True,
            router_model="heuristic_fallback",
        )
    elif matched_social and not matched_workplace:
        return RouteResult(
            route="social_media",
            explanation=f"{reason} Detected clear social-media cues: {', '.join(matched_social[:3])}.",
            confidence=0.85,
            needs_clarification=False,
            detected_domain_cues=matched_social[:4],
            is_fallback=True,
            router_model="heuristic_fallback",
        )
    else:
        # Ambiguous, mixed, or lacking strong markers
        cues = (matched_workplace + matched_social)[:4]
        return RouteResult(
            route="uncertain",
            explanation=f"{reason} The statement lacks unambiguous domain markers. Please select whether this is a workplace or social-media statement.",
            confidence=0.50,
            needs_clarification=True,
            detected_domain_cues=cues,
            is_fallback=True,
            router_model="heuristic_fallback",
        )


def route_statement(text: str, timeout_seconds: float = 5.0) -> RouteResult:
    """
    Route a statement to 'social_media', 'workplace', or 'uncertain'.
    First attempts Gemini 2.0 Flash structured routing.
    Gracefully falls back to heuristic routing if API key is missing or call fails.
    """
    clean_text = text.strip()
    if not clean_text:
        return RouteResult(
            route="uncertain",
            explanation="Empty input provided.",
            confidence=0.0,
            needs_clarification=True,
            detected_domain_cues=[],
            is_fallback=True,
            router_model="validator",
        )

    api_key = _get_gemini_api_key()
    if not api_key:
        return _fallback_heuristic_router(
            clean_text,
            "GEMINI_API_KEY is not configured in environment or Streamlit secrets."
        )

    t0 = time.perf_counter()
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        system_instruction = (
            "You are a domain classifier for passive-aggressive communication analysis.\n"
            "Your task is to classify whether a given statement belongs to:\n"
            "1. 'social_media': Public posts, tweets, replies, forum comments (Reddit), video remarks (YouTube/TikTok), "
            "Discord banter, streaming, follower interactions, or gaming chats.\n"
            "2. 'workplace': Professional business emails, Slack/Teams work chats, manager-employee feedback, "
            "sprint/project meetings, deadlines, deliverables, office politics, or formal corporate communication.\n"
            "3. 'uncertain': When the statement is genuinely context-neutral, terse (e.g. 'Fine.', 'Whatever you say.'), "
            "mixed, or equally plausible in both domains without surrounding context.\n\n"
            "CRITICAL RULES:\n"
            "- Do NOT generate passive-aggression predictions. You ONLY route the domain.\n"
            "- If domain clues are weak or ambiguous, choose 'uncertain' with needs_clarification=true.\n"
            "- Explain your reasoning in 1-2 concise sentences."
        )

        prompt = f"Analyze the communication domain of this statement:\n\"{clean_text}\""

        # Call Gemini with structured output constraint
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=RoutingDecision,
                temperature=0.1,
            ),
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not response or not response.text:
            return _fallback_heuristic_router(clean_text, "Gemini returned empty response.")

        parsed_data = json.loads(response.text)
        decision = RoutingDecision(**parsed_data)

        return RouteResult(
            route=decision.route,
            explanation=decision.explanation,
            confidence=decision.confidence,
            needs_clarification=decision.needs_clarification,
            detected_domain_cues=decision.detected_domain_cues,
            is_fallback=False,
            router_model="gemini-2.0-flash",
            latency_ms=latency_ms,
        )

    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return _fallback_heuristic_router(
            clean_text,
            f"Gemini API request failed ({type(e).__name__})."
        )
