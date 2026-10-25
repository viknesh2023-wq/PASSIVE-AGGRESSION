"""
inference/rewrite_engine.py
Constructive, meaning-preserving rewrite engine for PassiveAggression Detector 2.0.
Rephrases covertly hostile, indirect, or dismissive communication into clear, direct,
respectful alternatives without inventing facts, adding accusations, or injecting threats.

Core Principles:
1. Preserve core concern and communicative intent.
2. Remove indirect hostility, sarcasm, and guilt-tripping framing.
3. Keep the original text intact if already direct and constructive.
4. Provide cautious, context-aware suggestions for ambiguous/terse comments.
5. Never hallucinate facts or add unstated claims.
6. Provide a clear disclaimer that rewrites are recommendations, not absolute requirements.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import Optional, Any


@dataclass
class RewriteCandidate:
    suggested_text: str
    tactic: str
    explanation: str
    confidence: float
    transformation_type: str  # "direct_clarification" | "neutral_reframing" | "constructive_deescalation"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RewriteResponse:
    original_text: str
    candidates: list[RewriteCandidate]
    primary_tactic: Optional[str]
    has_rewrites: bool
    disclaimer: str
    rationale: str
    is_already_respectful: bool = False
    is_ambiguous: bool = False
    is_threat_flagged: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "candidates": [c.to_dict() for c in self.candidates],
            "primary_tactic": self.primary_tactic,
            "has_rewrites": self.has_rewrites,
            "disclaimer": self.disclaimer,
            "rationale": self.rationale,
            "is_already_respectful": self.is_already_respectful,
            "is_ambiguous": self.is_ambiguous,
            "is_threat_flagged": self.is_threat_flagged,
        }


# ── Curated Tactic Substitutions ────────────────────────────────────────────────
# Format: (pattern_re, replacement_str, tactic, explanation, transform_type)
_REWRITE_RULES: list[tuple[re.Pattern, str, str, str, str]] = [
    # 1. Fake politeness
    (
        re.compile(r"\bas per my (last|previous) (email|message|note)\b", re.IGNORECASE),
        "as I mentioned earlier",
        "fake_politeness",
        "Replaces adversarial recall phrasing with a neutral, professional reference.",
        "direct_clarification",
    ),
    (
        re.compile(r"\bper my last (email|message|note)\b", re.IGNORECASE),
        "as noted previously",
        "fake_politeness",
        "Replaces curt recall acronym/phrase with a courteous reminder.",
        "direct_clarification",
    ),
    (
        re.compile(r"\bclearly you didn't read\b", re.IGNORECASE),
        "to clarify the key details",
        "fake_politeness",
        "Reframes accusatory tone into collaborative clarification.",
        "neutral_reframing",
    ),
    (
        re.compile(r"\bi thought i made it clear\b", re.IGNORECASE),
        "let me clarify my perspective",
        "fake_politeness",
        "Removes patronizing implication and offers clear re-explanation.",
        "neutral_reframing",
    ),
    (
        re.compile(r"\bwith all due respect\b", re.IGNORECASE),
        "from my perspective",
        "fake_politeness",
        "Replaces adversarial preamble with an honest, subjective viewpoint.",
        "direct_clarification",
    ),
    (
        re.compile(r"\b(i'm )?sorry you feel that way\b", re.IGNORECASE),
        "I understand your frustration and want to find a solution",
        "fake_politeness",
        "Replaces invalidating non-apology with empathetic acknowledgment.",
        "constructive_deescalation",
    ),
    (
        re.compile(r"\bbless your heart\b", re.IGNORECASE),
        "thank you for sharing your thoughts",
        "fake_politeness",
        "Replaces condescending regionalism with sincere acknowledgement.",
        "neutral_reframing",
    ),
    (
        re.compile(r"\bhope that helps[!.]?\b", re.IGNORECASE),
        "let me know if you have any questions",
        "fake_politeness",
        "Replaces dismissive sign-off with an open invitation for follow-up.",
        "constructive_deescalation",
    ),
    (
        re.compile(r"\bno offense,?( but)?\b|\bno offence,?( but)?\b", re.IGNORECASE),
        "to be direct,",
        "fake_politeness",
        "Replaces defensive pre-emptive justification with honest directness.",
        "direct_clarification",
    ),
    (
        re.compile(r"\bnot to be rude,?( but)?\b", re.IGNORECASE),
        "from my perspective,",
        "fake_politeness",
        "Replaces negative framing with straightforward perspective.",
        "direct_clarification",
    ),

    # 2. Backhanded compliments
    (
        re.compile(r"\bbetter late than never\b", re.IGNORECASE),
        "thank you for getting this completed",
        "backhanded_compliment",
        "Focuses on task completion rather than emphasizing past delay.",
        "constructive_deescalation",
    ),
    (
        re.compile(r"\bfor once\b", re.IGNORECASE),
        "this time",
        "backhanded_compliment",
        "Removes the implication of habitual past failure.",
        "neutral_reframing",
    ),
    (
        re.compile(r"\bsurprisingly (good|well|competent)\b", re.IGNORECASE),
        r"really \1",
        "backhanded_compliment",
        "Removes low-expectation bias and provides straightforward praise.",
        "direct_clarification",
    ),
    (
        re.compile(r"\bat least you tried\b", re.IGNORECASE),
        "thank you for putting in the effort on this",
        "backhanded_compliment",
        "Converts condescending consolation into genuine appreciation.",
        "constructive_deescalation",
    ),
    (
        re.compile(r"\bnot bad for (a beginner|someone new|your first time)\b", re.IGNORECASE),
        "great job on this",
        "backhanded_compliment",
        "Removes qualifying disparagement to deliver sincere commendation.",
        "direct_clarification",
    ),
    (
        re.compile(r"\bglad you finally (noticed|understood|decided to show up)\b", re.IGNORECASE),
        "thank you for looking into this with us",
        "backhanded_compliment",
        "Replaces resentful relief with collaborative appreciation.",
        "constructive_deescalation",
    ),

    # 3. Implied blame
    (
        re.compile(r"\bi already told you\b", re.IGNORECASE),
        "as we discussed earlier",
        "implied_blame",
        "Replaces sharp blame with neutral reference to prior discussion.",
        "neutral_reframing",
    ),
    (
        re.compile(r"\bi thought we agreed\b", re.IGNORECASE),
        "let's review what we had agreed upon",
        "implied_blame",
        "Invites constructive alignment instead of assigning fault.",
        "constructive_deescalation",
    ),
    (
        re.compile(r"\bif you had (actually )?checked\b", re.IGNORECASE),
        "could you please verify whether",
        "implied_blame",
        "Replaces counterfactual blame with a polite verification request.",
        "direct_clarification",
    ),
    (
        re.compile(r"\bi warned you\b", re.IGNORECASE),
        "this was one of the concerns we noted",
        "implied_blame",
        "Depersonalizes the issue and focuses on objective risk management.",
        "neutral_reframing",
    ),
    (
        re.compile(r"\blike i said (before|earlier)\b", re.IGNORECASE),
        "to reiterate",
        "implied_blame",
        "Replaces exasperated repetition with clean summary phrasing.",
        "direct_clarification",
    ),

    # 4. Dismissiveness
    (
        re.compile(r"\bwhatever you say\b", re.IGNORECASE),
        "I will go with your decision on this",
        "dismissiveness",
        "Replaces hostile disengagement with respectful delegation.",
        "constructive_deescalation",
    ),
    (
        re.compile(r"\bdo whatever you want\b", re.IGNORECASE),
        "I trust your judgment on this approach",
        "dismissiveness",
        "Converts cynical withdrawal into positive empowerment.",
        "constructive_deescalation",
    ),
    (
        re.compile(r"\bdo what you want\b", re.IGNORECASE),
        "I am open to whichever approach works best for you",
        "dismissiveness",
        "Frames autonomy positively without implied resentment.",
        "constructive_deescalation",
    ),
    (
        re.compile(r"\bnot that it matters\b", re.IGNORECASE),
        "to share my perspective for consideration",
        "dismissiveness",
        "Affirms the value of the input rather than minimizing it.",
        "direct_clarification",
    ),
    (
        re.compile(r"\bdon'?t bother\b", re.IGNORECASE),
        "no worries, I have this covered",
        "dismissiveness",
        "Replaces resentful refusal of help with cordial self-reliance.",
        "neutral_reframing",
    ),
    (
        re.compile(r"\bgood luck with that\b", re.IGNORECASE),
        "I hope that approach goes smoothly",
        "dismissiveness",
        "Transforms cynical skepticism into courteous encouragement.",
        "constructive_deescalation",
    ),

    # 5. Sarcastic agreement
    (
        re.compile(r"\bif you say so\b", re.IGNORECASE),
        "understood, let's proceed with that",
        "sarcastic_agreement",
        "Replaces skeptical acquiescence with clear, professional alignment.",
        "direct_clarification",
    ),
    (
        re.compile(r"\bclearly you know best\b", re.IGNORECASE),
        "I respect your expertise on this topic",
        "sarcastic_agreement",
        "Transforms mock deference into sincere, collegial respect.",
        "neutral_reframing",
    ),
    (
        re.compile(r"\blove that for you\b", re.IGNORECASE),
        "glad to hear that worked out for you",
        "sarcastic_agreement",
        "Replaces internet snark with straightforward goodwill.",
        "neutral_reframing",
    ),
    (
        re.compile(r"\bsure,?( because)? that always works\b", re.IGNORECASE),
        "I have reservations about whether that approach will be effective",
        "sarcastic_agreement",
        "Replaces sarcastic dismissal with clear, actionable technical skepticism.",
        "direct_clarification",
    ),

    # 6. Indirect criticism
    (
        re.compile(r"\binteresting choice\b", re.IGNORECASE),
        "could you share the reasoning behind this choice?",
        "indirect_criticism",
        "Converts cryptic eyebrow-raising into an open, constructive inquiry.",
        "direct_clarification",
    ),
    (
        re.compile(r"\ban interesting decision\b", re.IGNORECASE),
        "what factors guided this decision?",
        "indirect_criticism",
        "Replaces judgment with curious, collaborative exploration.",
        "direct_clarification",
    ),
    (
        re.compile(r"\bnot sure why you thought that would work\b", re.IGNORECASE),
        "let's examine what didn't work and how we can adapt",
        "indirect_criticism",
        "Converts personal critique into collective problem-solving.",
        "constructive_deescalation",
    ),
    (
        re.compile(r"\bsome people just don't get it\b", re.IGNORECASE),
        "I see this differently and would like to share my view",
        "indirect_criticism",
        "Replaces broad condescension with personal ownership of perspective.",
        "direct_clarification",
    ),

    # 7. Guilt-tripping / Martyrdom
    (
        re.compile(r"\bmust be nice to\b", re.IGNORECASE),
        "I would appreciate some balance or assistance with",
        "guilt_tripping",
        "Converts passive resentment into a direct, actionable request.",
        "direct_clarification",
    ),
    (
        re.compile(r"\bwhile the rest of us\b", re.IGNORECASE),
        "so the workload is shared equitably",
        "guilt_tripping",
        "Focuses on fair collaboration rather than collective grievance.",
        "constructive_deescalation",
    ),
    (
        re.compile(r"\bi'll just do it myself( like always)?\b", re.IGNORECASE),
        "could someone please assist me with this task?",
        "guilt_tripping",
        "Replaces martyrdom with a clear, direct request for support.",
        "direct_clarification",
    ),
    (
        re.compile(r"\bdon't worry about me\b", re.IGNORECASE),
        "I could actually use some support when you have a moment",
        "guilt_tripping",
        "Replaces feigned independence with honest communication.",
        "direct_clarification",
    ),
    (
        re.compile(r"\bnobody asked for my opinion anyway\b", re.IGNORECASE),
        "I would like to share my thoughts on this if helpful",
        "guilt_tripping",
        "Replaces sulking exclusion with proactive engagement.",
        "direct_clarification",
    ),
    (
        re.compile(r"\bthanks for nothing\b", re.IGNORECASE),
        "I still need assistance resolving this issue",
        "guilt_tripping",
        "Reframes bitter rejection into an actionable help request.",
        "direct_clarification",
    ),
]

# ── Terse / Ambiguous Replacements ──────────────────────────────────────────────
_TERSE_REPLACEMENTS: dict[str, list[tuple[str, str]]] = {
    "fine": [
        ("Understood, that works for me.", "Affirms agreement cleanly without curtness."),
        ("Confirmed, I will proceed on that basis.", "Professional, unambiguous confirmation."),
    ],
    "sure": [
        ("Sure, sounds good to me.", "Adds warmth to avoid sounding reluctant."),
        ("Yes, I can do that.", "Clear, positive commitment."),
    ],
    "k": [
        ("Got it, thank you.", "Expands minimal single-letter reply into respectful acknowledgement."),
        ("Acknowledged.", "Direct, clear confirmation."),
    ],
    "ok": [
        ("Okay, thank you for the update.", "Courteous, unambiguous acknowledgement."),
        ("Sounds good.", "Positive confirmation."),
    ],
    "okay": [
        ("Okay, that makes sense. Thank you.", "Courteous, constructive validation."),
        ("Understood, thank you.", "Clean professional acknowledgement."),
    ],
    "whatever": [
        ("I am comfortable deferring to your judgment on this.", "Respectfully delegates the decision."),
        ("Whichever option you prefer works well for me.", "Constructive, collaborative flexibility."),
    ],
    "no problem": [
        ("You're very welcome!", "Warm, polite affirmation."),
        ("Happy to help anytime.", "Enthusiastic support."),
    ],
}


def rewrite(
    text: str,
    parent_comment: Optional[str] = None,
    detection_result: Optional[dict] = None,
) -> RewriteResponse:
    """
    Generate constructive, direct rewrite alternatives for social media text.

    Args:
        text: The user message to analyze and potentially rewrite.
        parent_comment: Optional preceding comment context.
        detection_result: Optional output dictionary from pa_inference.analyze().

    Returns:
        RewriteResponse dataclass instance with candidates and explanation.
    """
    disclaimer = (
        "Rewrites are suggestions intended to provide clearer, more direct, and respectful communication. "
        "They are not guaranteed to be semantically identical, and human judgment should always be exercised. "
        "The original message can always be retained unchanged."
    )

    if not text or not text.strip():
        return RewriteResponse(
            original_text="",
            candidates=[],
            primary_tactic=None,
            has_rewrites=False,
            disclaimer=disclaimer,
            rationale="Input text is empty or whitespace.",
        )

    clean_text = text.strip()
    clean_lower = clean_text.lower().rstrip(".!?,")

    # Threat check: Do NOT sanitize genuine threats into polite requests
    is_threat = False
    if detection_result:
        comb_threat = detection_result.get("combined", {}).get("threat")
        threat_lbl = detection_result.get("threat_result", {}).get("label")
        if comb_threat in ("direct_threat", "implied_threat") or threat_lbl in ("direct_threat", "implied_threat"):
            is_threat = True

    if is_threat:
        return RewriteResponse(
            original_text=clean_text,
            candidates=[],
            primary_tactic="flagged_threat",
            has_rewrites=False,
            disclaimer=disclaimer,
            rationale=(
                "This message contains threatening language or severe hostility. "
                "Constructive rewrites are withheld because threatening content cannot be safely rephrased "
                "into polite social-media discourse. Consider reviewing platform safety guidelines."
            ),
            is_threat_flagged=True,
        )

    # Check for terse one- or two-word ambiguous cases
    words = clean_text.split()
    if len(words) <= 2 and clean_lower in _TERSE_REPLACEMENTS:
        candidates = []
        for alt_text, alt_exp in _TERSE_REPLACEMENTS[clean_lower]:
            # Context sensitivity: if parent asks a question or gives an order
            if parent_comment and ("?" in parent_comment or "can you" in parent_comment.lower()):
                alt_text = alt_text.replace("that works for me", "I will take care of that")

            candidates.append(
                RewriteCandidate(
                    suggested_text=alt_text,
                    tactic="curt_dismissal",
                    explanation=alt_exp,
                    confidence=0.85,
                    transformation_type="constructive_deescalation",
                )
            )
        return RewriteResponse(
            original_text=clean_text,
            candidates=candidates,
            primary_tactic="dismissiveness",
            has_rewrites=True,
            disclaimer=disclaimer,
            rationale=(
                "Terse one-word replies often convey covert annoyance or disinterest without vocal inflection. "
                "Expanding them into a complete sentence prevents misinterpretation and establishes goodwill."
            ),
            is_ambiguous=True,
        )

    # Check for rule substitutions
    candidates = []
    seen_texts: set[str] = set()
    primary_tactic = None

    for pattern, replacement, tactic, exp, transform in _REWRITE_RULES:
        if pattern.search(clean_text):
            if not primary_tactic:
                primary_tactic = tactic

            # Apply replacement with case preservation
            def _sub(match):
                orig_span = match.group(0)
                if orig_span and orig_span[0].isupper():
                    return replacement[0].upper() + replacement[1:]
                return replacement

            candidate_text = pattern.sub(_sub, clean_text)
            candidate_text = re.sub(r"\s+", " ", candidate_text).strip()

            if candidate_text != clean_text and candidate_text not in seen_texts:
                seen_texts.add(candidate_text)
                candidates.append(
                    RewriteCandidate(
                        suggested_text=candidate_text,
                        tactic=tactic,
                        explanation=exp,
                        confidence=0.90,
                        transformation_type=transform,
                    )
                )

            if len(candidates) >= 3:
                break

    # If rules found rewrites
    if candidates:
        tactic_str = primary_tactic.replace('_', ' ') if primary_tactic else "indirect"
        return RewriteResponse(
            original_text=clean_text,
            candidates=candidates,
            primary_tactic=primary_tactic,
            has_rewrites=True,
            disclaimer=disclaimer,
            rationale=(
                f"Identified {tactic_str} phrasing. "
                "Suggested direct, respectful alternatives that maintain your core concern without covert hostility."
            ),
        )

    # If detection_result indicates passive aggression but no specific phrase matched
    if detection_result:
        is_pa = (
            detection_result.get("combined", {}).get("verdict") == "passive_aggressive"
            or detection_result.get("ml_result", {}).get("label") == "passive_aggressive"
        )
        if is_pa:
            # Provide general de-escalation suggestion
            generic_suggestion = "I would like to discuss this directly so we can find a constructive path forward."
            candidates.append(
                RewriteCandidate(
                    suggested_text=f"{clean_text} — {generic_suggestion}",
                    tactic="general_deescalation",
                    explanation="Appends an explicit constructive intention to clarify tone.",
                    confidence=0.75,
                    transformation_type="constructive_deescalation",
                )
            )
            return RewriteResponse(
                original_text=clean_text,
                candidates=candidates,
                primary_tactic="indirect_criticism",
                has_rewrites=True,
                disclaimer=disclaimer,
                rationale="The system detected passive aggression. Appending a collaborative intent reduces ambiguity.",
            )

    # If nothing was triggered, message is already respectful/neutral
    return RewriteResponse(
        original_text=clean_text,
        candidates=[],
        primary_tactic=None,
        has_rewrites=False,
        disclaimer=disclaimer,
        rationale="No indirect or passive-aggressive phrasing detected. The message is already direct, respectful, or neutral.",
        is_already_respectful=True,
    )
