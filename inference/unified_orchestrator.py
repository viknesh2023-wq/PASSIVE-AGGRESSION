"""
inference/unified_orchestrator.py — Unified Orchestration Engine.

Connects:
  1. The Gemini Query Router (inference/query_router.py)
  2. The Social-Media Passive-Aggression Detector (inference/pa_inference.py)
  3. The Workplace Passive-Aggression Detector API (workplace_api or remote HTTP service)

Standardizes outputs while preserving individual detector identities, authentic labels,
and distinct score semantics. Never redirects the user to external sites.
"""

import os
import time
from typing import Dict, Any, Optional, Literal
import httpx

from inference.query_router import route_statement, RouteResult
from inference.pa_inference import analyze as analyze_social

DEFAULT_WORKPLACE_API_URL = "http://localhost:8001"


def get_workplace_api_url() -> str:
    """Retrieve the configured workplace API base URL."""
    url = os.environ.get("WORKPLACE_API_URL")
    if url:
        return url.strip().rstrip("/")
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "WORKPLACE_API_URL" in st.secrets:
            return str(st.secrets["WORKPLACE_API_URL"]).strip().rstrip("/")
    except Exception:
        pass
    return DEFAULT_WORKPLACE_API_URL


def check_workplace_api_health(api_url: Optional[str] = None, timeout: float = 3.0) -> Dict[str, Any]:
    """Check health and model readiness of the workplace API."""
    base_url = (api_url or get_workplace_api_url()).rstrip("/")
    health_url = f"{base_url}/health"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(health_url)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "is_available": True,
                    "model_loaded": data.get("model_loaded", True),
                    "status": "ready",
                    "url": base_url,
                    "details": data,
                }
            else:
                return {
                    "is_available": False,
                    "model_loaded": False,
                    "status": f"HTTP {resp.status_code}",
                    "url": base_url,
                    "error": resp.text,
                }
    except Exception as e:
        # Check if local fallback workplace predictor exists
        try:
            from workplace_api.predictor import predictor as local_workplace_pred
            if local_workplace_pred.is_loaded:
                return {
                    "is_available": True,
                    "model_loaded": True,
                    "status": "local_fallback_ready",
                    "url": "local_in_process",
                    "note": f"Remote API unreachable ({type(e).__name__}); local workplace model fallback is active.",
                }
        except Exception:
            pass

        return {
            "is_available": False,
            "model_loaded": False,
            "status": "unreachable",
            "url": base_url,
            "error": str(e),
        }


def call_workplace_detector(text: str, api_url: Optional[str] = None, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Call the Workplace Passive-Aggression Detection API.
    Gracefully falls back to local in-process predictor if remote endpoint is unreachable.
    """
    base_url = (api_url or get_workplace_api_url()).rstrip("/")
    predict_url = f"{base_url}/predict"
    payload = {"text": text}

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(predict_url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                data["invoked_via"] = "http_api"
                data["endpoint"] = predict_url
                return data
            else:
                raise RuntimeError(f"Workplace API error (HTTP {resp.status_code}): {resp.text}")
    except Exception as http_err:
        # Check if in-process local workplace predictor is available
        try:
            from workplace_api.predictor import predictor as local_wp
            if local_wp.is_loaded:
                t0 = time.perf_counter()
                res = local_wp.predict(text)
                lat = (time.perf_counter() - t0) * 1000.0
                return {
                    "model": "workplace_logistic_regression_v1 (local fallback)",
                    "text": text,
                    "prediction": res["prediction"],
                    "label": res["label"],
                    "score": round(res["score"], 4),
                    "sentiment_label": res["sentiment_label"],
                    "sentiment_score": round(res["sentiment_score"], 4),
                    "cues": res["cues"],
                    "latency_ms": round(lat, 2),
                    "invoked_via": "local_fallback",
                    "api_error": str(http_err),
                }
        except Exception:
            pass
        raise http_err


def analyze_unified_statement(
    text: str,
    manual_domain: Optional[Literal["social_media", "workplace"]] = None,
    workplace_api_url: Optional[str] = None,
    sensitivity: str = "balanced",
) -> Dict[str, Any]:
    """
    Unified Single-Statement Analysis Workflow:
      1. Validates input statement.
      2. If manual_domain is specified, skips router.
         Otherwise, routes statement via Gemini Context Query Router.
      3. Dispatches to Social Media Detector (local) or Workplace Detector (API).
      4. Standardizes response structure while preserving detector identities and labels.
    """
    clean_text = text.strip()
    if len(clean_text) < 3:
        return {
            "error": "Statement must be at least 3 characters long.",
            "is_valid": False,
        }

    t0 = time.perf_counter()

    # Step 1: Routing Decision
    if manual_domain in ("social_media", "workplace"):
        router_result = RouteResult(
            route=manual_domain,
            explanation=f"Manual user selection: evaluated as {manual_domain.replace('_', ' ').title()}.",
            confidence=1.0,
            needs_clarification=False,
            detected_domain_cues=["user_selection"],
            is_fallback=False,
            router_model="manual_override",
        )
    else:
        router_result = route_statement(clean_text)

    # Step 2: Handle Clarification / Uncertain Domain
    if router_result.route == "uncertain" or router_result.needs_clarification:
        total_latency = (time.perf_counter() - t0) * 1000.0
        return {
            "is_valid": True,
            "status": "needs_clarification",
            "text": clean_text,
            "selected_domain": "uncertain",
            "router_info": router_result.to_dict(),
            "clarification_message": (
                "The statement could appear in either social media or a workplace setting. "
                "Please choose which context you would like to evaluate."
            ),
            "total_latency_ms": round(total_latency, 2),
        }

    # Step 3: Invoke Targeted Detector
    detector_output: Dict[str, Any] = {}
    detector_identity: Dict[str, Any] = {}

    if router_result.route == "social_media":
        try:
            raw_social = analyze_social(
                text=clean_text,
                context="social",
                sensitivity=sensitivity,
            )
            detector_identity = {
                "detector_name": "Social-Media Dual ML + Rule Engine",
                "domain": "social_media",
                "model_version": "pa-logreg-v2.0 (TF-IDF + FeatureUnion)",
                "architecture_type": "Dual Independent (Supervised ML + 10-Family Rule Engine)",
            }
            # Standardize social output
            ml_res = raw_social.get("ml_result", {})
            rule_res = raw_social.get("rule_result", {})
            threat_res = raw_social.get("threat_result", {})

            detector_output = {
                "primary_label": ml_res.get("label", "non_passive_aggressive"),
                "display_label": ml_res.get("label", "non_passive_aggressive").replace("_", " ").title(),
                "score": ml_res.get("pa_score", 0.0),
                "score_semantics": "Normalized PA Score [0-100] derived from 3-class ML probabilities",
                "rule_label": rule_res.get("pa_label", "non_passive_aggressive"),
                "rule_score": rule_res.get("pa_score", 0.0),
                "tactics": rule_res.get("tactics", []),
                "signals": rule_res.get("signals", []),
                "threat": {
                    "label": threat_res.get("label", "no_threat"),
                    "score": threat_res.get("score", 0.0),
                    "is_flagged": threat_res.get("label", "no_threat") != "no_threat",
                },
                "models_agree": ml_res.get("label") == rule_res.get("pa_label"),
                "explanation": raw_social.get("combined", {}).get("explanation", ""),
            }
        except Exception as e:
            return {
                "error": f"Social-Media Detector failed: {str(e)}",
                "router_info": router_result.to_dict(),
            }

    elif router_result.route == "workplace":
        try:
            raw_workplace = call_workplace_detector(clean_text, api_url=workplace_api_url)
            detector_identity = {
                "detector_name": "Workplace TF-IDF Logistic Regression",
                "domain": "workplace",
                "model_version": raw_workplace.get("model", "workplace_logistic_regression_v1"),
                "architecture_type": "Binary Scikit-Learn Pipeline + VADER Sentiment",
                "invoked_via": raw_workplace.get("invoked_via", "http_api"),
            }

            pred_bin = raw_workplace.get("prediction", 0)
            raw_label = raw_workplace.get("label", "Not flagged as passive-aggressive")
            prob_score = raw_workplace.get("score", 0.0)

            detector_output = {
                "primary_label": raw_label,
                "display_label": raw_label,
                "binary_prediction": pred_bin,
                "score": round(prob_score * 100.0, 1),
                "score_semantics": f"Estimated class probability ({prob_score:.1%}) for '{raw_label}'",
                "sentiment_label": raw_workplace.get("sentiment_label", "Neutral"),
                "sentiment_score": raw_workplace.get("sentiment_score", 0.0),
                "cues": raw_workplace.get("cues", []),
                "threat": {
                    "label": "not_applicable",
                    "note": "Workplace model does not evaluate threat independently.",
                    "is_flagged": False,
                },
                "explanation": (
                    f"Workplace model evaluated statement as '{raw_label}' with {prob_score:.1%} probability. "
                    f"Sentiment: {raw_workplace.get('sentiment_label')} (compound: {raw_workplace.get('sentiment_score', 0):.3f})."
                ),
            }
        except Exception as e:
            return {
                "error": f"Workplace Detector API failed: {str(e)}",
                "router_info": router_result.to_dict(),
                "status": "detector_error",
                "details": "Could not connect to Workplace API endpoint. Ensure the service is running at WORKPLACE_API_URL.",
            }

    total_latency = (time.perf_counter() - t0) * 1000.0

    return {
        "is_valid": True,
        "status": "completed",
        "text": clean_text,
        "selected_domain": router_result.route,
        "router_info": router_result.to_dict(),
        "detector_info": detector_identity,
        "prediction": detector_output,
        "total_latency_ms": round(total_latency, 2),
    }
