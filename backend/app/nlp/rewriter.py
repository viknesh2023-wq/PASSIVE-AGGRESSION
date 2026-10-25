"""
Deterministic rewrite engine using curated phrase substitutions.
No generative AI — all rewrites use a manually curated substitution dictionary.

Rules:
- Preserves original meaning as closely as possible.
- Does NOT automatically apply rewrites.
- Returns 1-3 calmer alternatives when possible.
- Returns empty list when no reliable rewrite can be produced.
- A rewrite is NOT guaranteed to be semantically equivalent.
"""

from __future__ import annotations
import re
from .phrase_matcher import find_phrase_matches


def generate_rewrites(text: str, max_rewrites: int = 3) -> list[str]:
    """
    Generate calmer rewrite suggestions for the given text.

    Returns a list of up to `max_rewrites` alternatives.
    Returns [] if no reliable substitution can be made.
    """
    matches = find_phrase_matches(text)

    # Filter to matches that have a rewrite suggestion
    rewritable = [m for m in matches if m.rewrite and m.rewrite.strip()]

    if not rewritable:
        return []

    rewrite_set: list[str] = []
    seen: set[str] = set()

    # Apply each rewrite independently (one substitution per candidate)
    for m in rewritable:
        if not m.rewrite:
            continue
        # Case-preserving replacement
        original_span = text[m.start:m.end]
        # Preserve sentence case: if original starts with capital, capitalize rewrite
        rewrite_text = m.rewrite
        if original_span and original_span[0].isupper() and rewrite_text:
            rewrite_text = rewrite_text[0].upper() + rewrite_text[1:]

        candidate = text[:m.start] + rewrite_text + text[m.end:]
        candidate = candidate.strip()

        # Avoid duplicates and trivially unchanged texts
        if candidate not in seen and candidate != text.strip():
            seen.add(candidate)
            rewrite_set.append(candidate)

        if len(rewrite_set) >= max_rewrites:
            break

    return rewrite_set


def explain_rewrite(original: str, rewrite: str) -> str:
    """
    Produce a brief human-readable explanation of what changed in the rewrite.
    """
    if not rewrite:
        return "No calmer rewrite could be produced for this text."
    return (
        "This version rephrases the potentially indirect or dismissive language "
        "into more direct, neutral phrasing. The original message's meaning has "
        "been approximately preserved but the communication style has been softened."
    )
