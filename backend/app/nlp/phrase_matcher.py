"""
Phrase-bank matcher with exact character-offset tracking.
Returns matched spans, tactic labels, weights, and human-readable reasons.

This is a RULE-BASED component — evidence is clearly labeled as rule-based.
It is NOT a trained PA classifier.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhraseMatch:
    start:   int
    end:     int
    text:    str       # matched text in original case
    phrase:  str       # matched pattern (lower)
    weight:  float     # raw evidence weight (0–50)
    tactic:  str
    reason:  str
    rewrite: Optional[str] = None


# ─── Phrase bank ────────────────────────────────────────────────────────────────
# Format: (pattern, weight, tactic, rewrite_suggestion | None)
# Weight represents evidence strength — not probability.
# Context is required for definitive labeling; these are signals only.
#
# Patterns are LOWERCASE plain strings or compiled regexes.
# All phrases were selected based on established pragmatics literature on
# indirect hostility (see annotation_guidelines.md for rationale).

_PHRASE_BANK: list[tuple[str, float, str, Optional[str]]] = [
    # Blame-shifting — redirecting responsibility
    ("as per my last email",          30, "blame_shifting",        "As I mentioned earlier"),
    ("as per my previous email",      28, "blame_shifting",        "As I previously noted"),
    ("per my previous",               22, "blame_shifting",        "As I previously noted"),
    ("as mentioned",                  10, "blame_shifting",        None),
    ("as i mentioned",                14, "blame_shifting",        "As I noted earlier"),
    ("as i said",                     16, "blame_shifting",        "To reiterate"),
    ("as stated before",              18, "blame_shifting",        "To reiterate"),
    ("i already told you",            24, "blame_shifting",        "I mentioned earlier"),
    ("i thought i made it clear",     26, "blame_shifting",        "Let me clarify"),
    ("clearly you didn't read",       30, "blame_shifting",        None),
    ("i thought we agreed",           18, "blame_shifting",        "We had discussed"),

    # Mock politeness — polite form with condescending/dismissive intent
    ("no worries",                    14, "mock_politeness",       "That is completely fine"),
    ("no problem at all",             10, "mock_politeness",       "That is fine"),
    ("with all due respect",          20, "mock_politeness",       "I respectfully disagree"),
    ("not to be rude but",            24, "mock_politeness",       None),
    ("sorry you feel that way",       26, "mock_politeness",       "I'm sorry — let me understand your concern"),
    ("i'm sorry you feel that",       26, "mock_politeness",       "I'm sorry — let me understand your concern"),
    ("not that i'm complaining",      18, "mock_politeness",       None),
    ("bless your heart",              20, "mock_politeness",       None),
    ("that's a nice try",             22, "mock_politeness",       None),
    ("good for you",                  12, "mock_politeness",       None),
    ("how nice for you",              18, "mock_politeness",       None),

    # Curt dismissal — disengagement signals
    ("fine.",                         22, "curt_dismissal",        "Okay, that works for me."),
    ("fine,",                         16, "curt_dismissal",        None),
    ("whatever you say",              26, "curt_dismissal",        "I will go with your decision."),
    ("whatever you think",            22, "curt_dismissal",        "I trust your judgment."),
    ("whatever you want",             24, "curt_dismissal",        "Your call — I am flexible."),
    ("do whatever you want",          26, "curt_dismissal",        "I am flexible — whatever works best."),
    ("do what you want",              22, "curt_dismissal",        None),
    ("not that it matters",           24, "curt_dismissal",        None),
    ("forget it",                     18, "curt_dismissal",        None),
    ("never mind",                    16, "curt_dismissal",        None),
    ("don't bother",                  20, "curt_dismissal",        "Please do not worry about it"),

    # Sarcasm / irony
    ("if you say so",                 22, "sarcasm",               "I understand."),
    ("oh really",                     16, "sarcasm",               None),
    ("wow thanks",                    14, "sarcasm",               None),
    ("how helpful",                   18, "sarcasm",               None),
    ("how convenient",                20, "sarcasm",               None),
    ("that's just great",             20, "sarcasm",               None),
    ("oh great",                      14, "sarcasm",               None),
    ("oh perfect",                    14, "sarcasm",               None),
    ("just what i needed",            16, "sarcasm",               None),

    # Guilt-tripping — implied burden or ingratitude
    ("must be nice",                  22, "guilt_tripping",        None),
    ("thanks in advance",             12, "guilt_tripping",        "I would appreciate your help with this"),
    ("just checking in again",        18, "guilt_tripping",        "Following up on my previous message"),
    ("just checking in",              10, "guilt_tripping",        "Following up on this"),
    ("i've been waiting",             20, "guilt_tripping",        None),
    ("i've been trying to reach you", 22, "guilt_tripping",        "I have attempted to reach you"),
    ("i was expecting",               16, "guilt_tripping",        None),
    ("after everything i've done",    28, "guilt_tripping",        None),
    ("i guess i'll just",             22, "guilt_tripping",        None),
    ("don't worry about me",          20, "guilt_tripping",        None),

    # Backhanded compliment — apparent praise with implicit criticism
    ("finally",                       14, "backhanded_compliment", None),
    ("i'm sure you were busy",        20, "backhanded_compliment", "I know you have a lot on your plate"),
    ("i'm surprised you",             22, "backhanded_compliment", "I appreciate you getting this done"),
    ("honestly i'm surprised",        22, "backhanded_compliment", "I appreciate you getting this done"),
    ("better late than never",        20, "backhanded_compliment", None),
    ("at least you tried",            20, "backhanded_compliment", None),
    ("for once",                      18, "backhanded_compliment", None),
    ("you actually did it",           16, "backhanded_compliment", None),
    ("i didn't think you'd",          18, "backhanded_compliment", None),
    ("you almost got it right",       24, "backhanded_compliment", None),
    ("not bad for",                   16, "backhanded_compliment", None),
]

# Compile lookup for efficiency
_LOWER_PATTERNS = [(p.lower(), w, t, r) for (p, w, t, r) in _PHRASE_BANK]


# ─── Style-signal patterns ────────────────────────────────────────────────────
_ELLIPSIS_RE   = re.compile(r'\.{3,}')
_CAPS_WORD_RE  = re.compile(r'\b[A-Z]{3,}\b')
_RHET_QUEST_RE = re.compile(r'\b(seriously|really|honestly|right)\?\s*$', re.IGNORECASE)


def find_phrase_matches(text: str) -> list[PhraseMatch]:
    """
    Scan text for all phrase-bank patterns, returning exact character offsets.
    All offsets reference the original `text` string (Unicode-safe).
    """
    lower = text.lower()
    matches: list[PhraseMatch] = []
    seen_ranges: list[tuple[int, int]] = []

    for phrase, weight, tactic, rewrite in _LOWER_PATTERNS:
        idx = 0
        while True:
            pos = lower.find(phrase, idx)
            if pos == -1:
                break
            end = pos + len(phrase)

            # Word-boundary check: ensure we don't match mid-word
            before_ok = (pos == 0 or not lower[pos - 1].isalpha())
            after_ok  = (end == len(lower) or not lower[end].isalpha())

            if before_ok and after_ok:
                # Avoid overlapping spans
                overlaps = any(s < end and e > pos for s, e in seen_ranges)
                if not overlaps:
                    original_text = text[pos:end]
                    reason = _build_reason(tactic, phrase)
                    matches.append(PhraseMatch(
                        start=pos,
                        end=end,
                        text=original_text,
                        phrase=phrase,
                        weight=weight,
                        tactic=tactic,
                        reason=reason,
                        rewrite=rewrite,
                    ))
                    seen_ranges.append((pos, end))
            idx = pos + 1

    return matches


def compute_style_signals(text: str) -> dict[str, float]:
    """
    Compute scalar style features from raw text.
    These are weak signals — not standalone classifiers.
    """
    ellipsis_count  = len(_ELLIPSIS_RE.findall(text))
    caps_count      = len(_CAPS_WORD_RE.findall(text))
    rhet_quest      = 1 if _RHET_QUEST_RE.search(text) else 0
    exclamation_ct  = text.count('!')
    question_ct     = text.count('?')

    return {
        "ellipsis_count":    ellipsis_count,
        "caps_word_count":   caps_count,
        "rhetorical_question": rhet_quest,
        "exclamation_count": exclamation_ct,
        "question_count":    question_ct,
        # Bonus weights for scoring
        "ellipsis_bonus":   ellipsis_count * 4.0,
        "caps_bonus":       caps_count     * 6.0,
        "rhet_bonus":       rhet_quest     * 8.0,
    }


def _build_reason(tactic: str, phrase: str) -> str:
    reason_map: dict[str, str] = {
        "sarcasm":               f'"{phrase}" is often used sarcastically to express doubt or dissatisfaction without stating it directly.',
        "backhanded_compliment": f'"{phrase}" appears complimentary but carries a subtle sting or implied criticism.',
        "guilt_tripping":        f'"{phrase}" implies the reader has failed in some obligation or is being burdensome.',
        "mock_politeness":       f'"{phrase}" uses polite phrasing to deliver a dismissive or condescending message.',
        "blame_shifting":        f'"{phrase}" subtly redirects responsibility or fault toward the reader.',
        "curt_dismissal":        f'"{phrase}" signals disengagement and an unwillingness to continue the conversation.',
    }
    return reason_map.get(tactic, f'Rule match: "{phrase}" is a known indirect-hostility signal.')
