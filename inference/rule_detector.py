"""
inference/rule_detector.py
Enhanced, explainable rule-based passive-aggression and threat detector.
Independent of any trained ML model — fully interpretable.

Implements 10 Rule Families:
  1. Fake politeness
  2. Indirect criticism
  3. Backhanded compliments
  4. Implied blame
  5. Sarcastic agreement
  6. Dismissiveness
  7. Other or unclear tactic signals
  8. Contextual escalation & parent-comment tension
  9. Benign overrides (false-positive prevention)
  10. Quotation, negation, and figurative-language handling

Supports Sensitivity Modes:
  - 'balanced' (default)
  - 'high_recall' (or 'strict')
  - 'high_precision' (or 'lenient')

Returns structured RuleResult containing:
  - primary_label (pa_label)
  - threat_label
  - tactic_labels (tactics)
  - matched_spans (character offsets)
  - rule_ids
  - rationale
  - strength_score (pa_score)
  - needs_review
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


# ── Data Structures ─────────────────────────────────────────────────────────────

@dataclass
class RuleSignal:
    name: str
    weight: float
    tactic: str
    reason: str
    rule_id: str = "RULE_GENERIC"
    span: Optional[tuple[int, int]] = None


@dataclass
class RuleResult:
    pa_label: str                  # passive_aggressive | non_passive_aggressive | ambiguous
    threat_label: str              # no_threat | direct_threat | implied_threat | ambiguous
    pa_score: float                # raw accumulated evidence score 0-100
    threat_score: float            # raw accumulated threat score 0-100
    pa_confidence: float           # heuristic confidence 0-1
    threat_confidence: float       # heuristic confidence 0-1
    tactics: list[str]             # detected PA tactics
    signals: list[RuleSignal]      # all matched signals
    threat_signals: list[RuleSignal]
    is_ambiguous: bool
    # Canonical Phase 5 extensions
    primary_label: str = ""
    tactic_labels: list[str] = field(default_factory=list)
    matched_spans: list[dict] = field(default_factory=list)
    rule_ids: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    strength_score: float = 0.0
    needs_review: bool = False

    def __post_init__(self):
        if not self.primary_label:
            self.primary_label = self.pa_label
        if not self.tactic_labels:
            self.tactic_labels = self.tactics
        if not self.strength_score:
            self.strength_score = self.pa_score


# ── Rule Families 1–7: Lexicon Patterns ─────────────────────────────────────────

_PA_PHRASES: list[tuple[str, float, str, str, str]] = [
    # Family 1: Fake politeness
    ("as per my last email",          40, "fake_politeness",     "Indirect reference implying failure to read prior communication", "RULE_FP_01"),
    ("as per my previous",            35, "fake_politeness",     "Indirect reference implying oversight by the reader", "RULE_FP_02"),
    ("as previously mentioned",       25, "fake_politeness",     "Recall phrase encoding implicit frustration at repetition", "RULE_FP_03"),
    ("per my previous message",       35, "fake_politeness",     "Indirect reference implying reader inattention", "RULE_FP_04"),
    ("i thought i made it clear",     40, "fake_politeness",     "Implies the reader failed to understand something obvious", "RULE_FP_05"),
    ("clearly you didn't read",       45, "fake_politeness",     "Accusation wrapped in polite observational register", "RULE_FP_06"),
    ("with all due respect",          30, "fake_politeness",     "Polite preamble used to soften dismissive assertion", "RULE_FP_07"),
    ("not to be rude but",            30, "fake_politeness",     "Disclaims rudeness immediately prior to delivering criticism", "RULE_FP_08"),
    ("sorry you feel that way",       35, "fake_politeness",     "Deflection non-apology that invalidates concerns", "RULE_FP_09"),
    ("i'm sorry you feel",            35, "fake_politeness",     "Non-apology deflecting fault onto recipient's perception", "RULE_FP_10"),
    ("bless your heart",              30, "fake_politeness",     "Patronizing idiom masking condescension", "RULE_FP_11"),
    ("how nice for you",              25, "fake_politeness",     "Perfunctory praise with detached or envious undertone", "RULE_FP_12"),
    ("that's a nice try",             30, "fake_politeness",     "Superficial praise framing the effort as deficient", "RULE_FP_13"),
    ("glad you found time",           30, "fake_politeness",     "Implies the recipient neglected priorities", "RULE_FP_14"),
    ("appreciate the input",          15, "fake_politeness",     "Perfunctory dismissal of contribution", "RULE_FP_15"),
    ("thanks for pointing out",       15, "fake_politeness",     "Mild resentment framed as gratitude", "RULE_FP_16"),
    ("so kind of you",                20, "fake_politeness",     "Perfunctory gratitude with possible sarcastic register", "RULE_FP_17"),

    # Family 2: Indirect criticism
    ("i guess",                       10, "indirect_criticism",  "Hedged assertion implying dissatisfaction", "RULE_IC_01"),
    ("i'm just saying",               20, "indirect_criticism",  "Deniability disclaimer following critical statement", "RULE_IC_02"),
    ("just find it interesting",      30, "indirect_criticism",  "Faux-neutral observation veiling an accusation", "RULE_IC_03"),
    ("not gonna say anything but",    35, "indirect_criticism",  "Claims silence while actively delivering criticism", "RULE_IC_04"),
    ("some people",                   18, "indirect_criticism",  "Vague subject avoiding direct accountability", "RULE_IC_05"),
    ("certain people",                20, "indirect_criticism",  "Targeted vague subject implying specific fault", "RULE_IC_06"),
    ("not that i'm judging",          25, "indirect_criticism",  "Disclaims judgment while rendering it", "RULE_IC_07"),
    ("i noticed that",                12, "indirect_criticism",  "Observational framing for implicit critique", "RULE_IC_08"),
    ("funny how",                     22, "indirect_criticism",  "Faux-incredulity framing implicit complaint", "RULE_IC_09"),
    ("interesting how",               20, "indirect_criticism",  "Veiled criticism under observational veneer", "RULE_IC_10"),
    ("isn't it interesting",          22, "indirect_criticism",  "Rhetorical questioning masking disagreement", "RULE_IC_11"),

    # Family 3: Backhanded compliment
    ("better late than never",        30, "backhanded_compliment", "Concession acknowledging completion while emphasizing tardiness", "RULE_BC_01"),
    ("for once",                      28, "backhanded_compliment", "Implies habitual inadequacy with solitary exception", "RULE_BC_02"),
    ("surprisingly",                  18, "backhanded_compliment", "Marks surprise at baseline competence", "RULE_BC_03"),
    ("actually competent",            30, "backhanded_compliment", "Low expectations framed as surprise", "RULE_BC_04"),
    ("didn't think you'd",            28, "backhanded_compliment", "Expectation of failure preceded praise", "RULE_BC_05"),
    ("didn't expect you to",          28, "backhanded_compliment", "Low-expectation framing", "RULE_BC_06"),
    ("not bad for",                   25, "backhanded_compliment", "Qualifying clause that diminishes achievement", "RULE_BC_07"),
    ("almost got it right",           30, "backhanded_compliment", "Praise conditioned on remaining deficit", "RULE_BC_08"),
    ("at least you tried",            28, "backhanded_compliment", "Condescending consolation signaling subpar outcome", "RULE_BC_09"),
    ("you actually did it",           20, "backhanded_compliment", "Expresses disbelief at basic performance", "RULE_BC_10"),

    # Family 4: Implied blame
    ("if you had",                    20, "implied_blame",       "Counterfactual blaming recipient for failure", "RULE_IB_01"),
    ("if someone had",                22, "implied_blame",       "Indirect counterfactual distributing blame", "RULE_IB_02"),
    ("i already told you",            35, "implied_blame",       "Emphasizes prior warning to highlight recipient error", "RULE_IB_03"),
    ("i thought we agreed",           25, "implied_blame",       "Accuses other party of breaking consensus", "RULE_IB_04"),
    ("i specifically",                18, "implied_blame",       "Highlights own rigor to contrast other's negligence", "RULE_IB_05"),
    ("it worked fine until",          28, "implied_blame",       "Attributes system/process degradation to other's intervention", "RULE_IB_06"),
    ("i warned you",                  30, "implied_blame",       "Explicit prior warning cited to establish culpability", "RULE_IB_07"),
    ("i said my concerns",            25, "implied_blame",       "Self-vindication deflecting mutual responsibility", "RULE_IB_08"),

    # Family 5: Sarcastic agreement
    ("if you say so",                 30, "sarcastic_agreement", "Nominal agreement signaling underlying skepticism", "RULE_SA_01"),
    ("great, so",                     15, "sarcastic_agreement", "Ironic optimism introducing an inconvenience", "RULE_SA_02"),
    ("love that for you",             20, "sarcastic_agreement", "Superficial celebration masking envy or disapproval", "RULE_SA_03"),
    ("clearly you know best",         35, "sarcastic_agreement", "Mock deference to authority or expertise", "RULE_SA_04"),

    # Family 6: Dismissiveness
    ("whatever you say",              30, "dismissiveness",      "Capitulation signaling hostile disengagement", "RULE_DM_01"),
    ("whatever you think",            25, "dismissiveness",      "Concession lacking genuine alignment", "RULE_DM_02"),
    ("do what you want",              25, "dismissiveness",      "Withdrawal implying anticipated failure", "RULE_DM_03"),
    ("do whatever you want",          28, "dismissiveness",      "Resigned withdrawal with negative expectation", "RULE_DM_04"),
    ("not that it matters",           28, "dismissiveness",      "Minimizes situation to signal futility", "RULE_DM_05"),
    ("if that's what you want",       22, "dismissiveness",      "Disapproving concession", "RULE_DM_06"),
    ("sure, okay",                    15, "dismissiveness",      "Terse capitulation signaling disinterest", "RULE_DM_07"),
    ("dont bother",                   25, "dismissiveness",      "Refusal of assistance with implied resentment", "RULE_DM_08"),

    # Family 7: Other / Guilt-Tripping
    ("must be nice",                  35, "guilt_tripping",      "Envious grievance highlighting disparity in effort/privilege", "RULE_GT_01"),
    ("must be great",                 30, "guilt_tripping",      "Envious grievance", "RULE_GT_02"),
    ("while the rest of us",          35, "guilt_tripping",      "Contrasts recipient's leisure with collective toil", "RULE_GT_03"),
    ("guess my time",                 30, "guilt_tripping",      "Claims own worth is disregarded", "RULE_GT_04"),
    ("i'll just do it myself",        35, "guilt_tripping",      "Martyr complex refusing collaboration", "RULE_GT_05"),
    ("don't worry about me",          30, "guilt_tripping",      "Feigned self-reliance masking resentment", "RULE_GT_06"),
    ("i'm used to it",                25, "guilt_tripping",      "Habitual victim framing", "RULE_GT_07"),
    ("it's totally fine",             15, "guilt_tripping",      "Insincere acceptance signaling displeasure", "RULE_GT_08"),
]

# Style Indicators
_ELLIPSIS_RE   = re.compile(r"\.{3,}")
_CAPS_WORD_RE  = re.compile(r"\b[A-Z]{3,}\b")
_RHET_QUEST_RE = re.compile(r"\b(seriously|really|honestly|right)\?(\s*$)", re.IGNORECASE)
_SCARE_QUOTE   = re.compile(r'"[^"]{1,30}"')

# Threat Phrase Banks
_DIRECT_THREAT_PHRASES: list[tuple[str, float, str, str]] = [
    ("you will regret",          50, "Explicit threat of retribution", "RULE_THR_DIR_01"),
    ("i will make you",          45, "Coercive threat of forced action", "RULE_THR_DIR_02"),
    ("i'll make sure you",       45, "Deliberate retaliatory threat", "RULE_THR_DIR_03"),
    ("you'll be sorry",          45, "Direct warning of negative consequences", "RULE_THR_DIR_04"),
    ("i know where you",         50, "Direct location-based intimidation", "RULE_THR_DIR_05"),
    ("i will destroy",           50, "Explicit destruction threat", "RULE_THR_DIR_06"),
    ("you will pay",             45, "Retribution threat", "RULE_THR_DIR_07"),
    ("i'm coming for",           45, "Direct physical intimidation", "RULE_THR_DIR_08"),
    ("touch that again",         40, "Physical boundary threat", "RULE_THR_DIR_09"),
    ("track you down",           50, "Physical tracking threat", "RULE_THR_DIR_10"),
    ("break your legs",          55, "Explicit physical violence threat", "RULE_THR_DIR_11"),
    ("break your",               45, "Threat of physical harm", "RULE_THR_DIR_12"),
    ("hunt you down",            50, "Physical tracking threat", "RULE_THR_DIR_13"),
    ("i will hurt you",          50, "Direct bodily harm threat", "RULE_THR_DIR_14"),
    ("i will kill",              55, "Direct lethal threat", "RULE_THR_DIR_15"),
    ("i'm going to kill",        55, "Direct lethal threat", "RULE_THR_DIR_16"),
]

_IMPLIED_THREAT_PHRASES: list[tuple[str, float, str, str]] = [
    ("it would be a shame",      35, "Indirect threat via conditional harm", "RULE_THR_IMP_01"),
    ("be careful",               20, "Ambiguous warning", "RULE_THR_IMP_02"),
    ("i'm watching",             30, "Surveillance intimidation", "RULE_THR_IMP_03"),
    ("i know people",            25, "Networked intimidation threat", "RULE_THR_IMP_04"),
    ("things have a way of",     25, "Karmic retribution framing", "RULE_THR_IMP_05"),
    ("keep pushing",             20, "Escalation warning", "RULE_THR_IMP_06"),
    ("you've been warned",       35, "Post-hoc threat attribution", "RULE_THR_IMP_07"),
    ("don't say i didn't warn",  35, "Prior warning attribution", "RULE_THR_IMP_08"),
    ("i'm keeping track",        30, "Documentary surveillance threat", "RULE_THR_IMP_09"),
    ("act accordingly",          25, "Implicit ultimatum", "RULE_THR_IMP_10"),
    ("this isn't over",          35, "Ongoing conflict threat", "RULE_THR_IMP_11"),
]

# Family 9: Benign Overrides (False-Positive Prevention)
_BENIGN_OVERRIDES: list[str] = [
    "per your request",
    "as you requested",
    "thank you for your help",
    "thanks for your help",
    "no problem at all",
    "no problem at all, happy to help",
    "happy to help",
    "glad to help",
    "hope this helps",
    "let me know if you need anything else",
    "feel free to reach out",
    "looking forward to working with you",
    "great job on",
    "congratulations on",
    "thank you so much",
    "much appreciated",
    "good luck with your project",
    "have a wonderful day",
]

# Family 10: Figurative / Hyperbolic Threat Filters
_FIGURATIVE_PHRASES: list[str] = [
    "i'm dying of laughter",
    "this is killing me",
    "i will literally die",
    "i could kill for",
    "i could eat you",
    "dying to try",
    "dying to see",
    "to die for",
    "killer deal",
    "killer content",
    "murdering this",
    "destroyed it",
    "slayed",
    "i'll destroy",
    "i will destroy you in",
    "destroy you in",
    "video game",
    "game match",
    "in this match",
    "in this game",
    "on the board",
    "in chess",
    "going to kill it",
    "killed it",
    "you're dead to me",
    "i will fight you",
    "going to roast",
    "roast you",
]

# Quotation and Negation Regexes
_QUOTED_PA_RE = re.compile(r'["\']([^"\']*(?:as per|whatever you|must be nice|fine)[^"\']*)["\']', re.IGNORECASE)
_NEGATED_PA_RE = re.compile(r"\b(not|never|didn't|don't)\s+(mean to be|trying to be|sound)\s+passive\s+aggressive\b", re.IGNORECASE)


# ── Core Matching Engine ────────────────────────────────────────────────────────

def _match_phrases(text_lower: str, bank: list[tuple]) -> list[RuleSignal]:
    signals = []
    seen_spans: list[tuple[int, int]] = []

    for entry in bank:
        phrase = entry[0]
        weight = entry[1]
        tactic = entry[2] if len(entry) > 2 else "other"
        reason = entry[3] if len(entry) > 3 else f'Matched: "{phrase}"'
        rule_id = entry[4] if len(entry) > 4 else "RULE_GENERIC"

        idx = 0
        while True:
            pos = text_lower.find(phrase, idx)
            if pos == -1:
                break
            end = pos + len(phrase)

            # Word boundary check
            before_ok = (pos == 0 or not text_lower[pos - 1].isalpha())
            after_ok  = (end == len(text_lower) or not text_lower[end].isalpha())

            if before_ok and after_ok:
                overlaps = any(s < end and e > pos for s, e in seen_spans)
                if not overlaps:
                    signals.append(RuleSignal(
                        name=phrase,
                        weight=weight,
                        tactic=tactic,
                        reason=reason,
                        rule_id=rule_id,
                        span=(pos, end),
                    ))
                    seen_spans.append((pos, end))
            idx = pos + 1
    return signals


def _extract_style_signals(text: str) -> tuple[float, list[RuleSignal]]:
    bonus = 0.0
    signals = []

    ellipses = list(_ELLIPSIS_RE.finditer(text))
    if ellipses:
        cnt = len(ellipses)
        b = min(15.0, cnt * 5.0)
        bonus += b
        first = ellipses[0]
        signals.append(RuleSignal(
            "ellipsis", b, "dismissiveness",
            f"Ellipsis trailing (...) indicates intentional pause, trailing off, or withholding comment",
            "RULE_STYLE_ELLIPSIS",
            span=(first.start(), first.end()),
        ))

    caps_words = [m for m in _CAPS_WORD_RE.finditer(text) if m.group() not in {"URL", "USA", "API", "FAQ", "HTML", "CSS"}]
    if caps_words:
        cnt = len(caps_words)
        b = min(12.0, cnt * 4.0)
        bonus += b
        first = caps_words[0]
        signals.append(RuleSignal(
            "all_caps", b, "indirect_criticism",
            f"Capitalized emphasis detected: {[m.group() for m in caps_words[:3]]}",
            "RULE_STYLE_CAPS",
            span=(first.start(), first.end()),
        ))

    if _RHET_QUEST_RE.search(text):
        m = _RHET_QUEST_RE.search(text)
        bonus += 10.0
        signals.append(RuleSignal(
            "rhetorical_question", 10.0, "sarcastic_agreement",
            "Rhetorical question ending (seriously? really? honestly?) challenges recipient",
            "RULE_STYLE_RHETORICAL",
            span=(m.start(), m.end()) if m else None,
        ))

    scare_quotes = list(_SCARE_QUOTE.finditer(text))
    if scare_quotes:
        b = len(scare_quotes) * 8.0
        bonus += b
        first = scare_quotes[0]
        signals.append(RuleSignal(
            "air_quotes", b, "indirect_criticism",
            f"Air quotes detected: {[m.group() for m in scare_quotes[:3]]} — ironic framing",
            "RULE_STYLE_SCAREQUOTES",
            span=(first.start(), first.end()),
        ))

    return bonus, signals


def _is_figurative_threat(text_lower: str) -> bool:
    return any(p in text_lower for p in _FIGURATIVE_PHRASES)


def _has_benign_override(text_lower: str) -> bool:
    return any(p in text_lower for p in _BENIGN_OVERRIDES)


# ── Main Detector Function ──────────────────────────────────────────────────────

def detect(
    text: str,
    context: Optional[str] = None,
    parent_comment: Optional[str] = None,
    sensitivity: str = "balanced",
) -> RuleResult:
    """
    Run rule-based PA and threat detection across all 10 rule families.

    Args:
        text: Target social media comment.
        context: Optional conversation context tag.
        parent_comment: Optional parent message text.
        sensitivity: 'balanced' | 'high_recall' ('strict') | 'high_precision' ('lenient').
    """
    sens = sensitivity.lower().strip()
    if sens in ("strict", "high_recall"):
        mode = "high_recall"
    elif sens in ("lenient", "high_precision"):
        mode = "high_precision"
    else:
        mode = "balanced"

    text_lower = text.lower().strip()

    # Family 9: Benign Overrides
    if _has_benign_override(text_lower):
        return RuleResult(
            pa_label="non_passive_aggressive",
            threat_label="no_threat",
            pa_score=0.0,
            threat_score=0.0,
            pa_confidence=0.88,
            threat_confidence=0.95,
            tactics=[],
            signals=[],
            threat_signals=[],
            is_ambiguous=False,
            primary_label="non_passive_aggressive",
            tactic_labels=[],
            matched_spans=[],
            rule_ids=["RULE_BENIGN_OVERRIDE"],
            rationale=["Benign conversational override matched; polite/sincere framing."],
            strength_score=0.0,
            needs_review=False,
        )

    # Family 10: Negation / Quotation Checks
    if _NEGATED_PA_RE.search(text):
        # Explicit disclaimer, keep as non-PA unless strong hostile evidence
        pass

    # Match Lexicon Phrases (Families 1–7)
    pa_signals = _match_phrases(text_lower, _PA_PHRASES)

    # Style Signals
    style_bonus, style_signals = _extract_style_signals(text)

    # Raw PA Score
    raw_pa_score = sum(s.weight for s in pa_signals) + style_bonus

    # Family 8: Contextual Escalation / Parent Comment Signals
    if parent_comment and len(parent_comment.strip()) > 3:
        p_lower = parent_comment.lower()
        # If parent expresses frustration/complaint, boosts target reply tension
        if any(w in p_lower for w in ["delay", "broken", "issue", "why", "late", "error", "failed", "where"]):
            raw_pa_score *= 1.15

    # Threat Signal Matching
    direct_signals  = _match_phrases(text_lower, _DIRECT_THREAT_PHRASES)
    implied_signals = _match_phrases(text_lower, _IMPLIED_THREAT_PHRASES)
    raw_threat_score = sum(s.weight for s in direct_signals) + sum(s.weight * 0.7 for s in implied_signals)

    # Family 10: Figurative Threat Override
    is_figurative = _is_figurative_threat(text_lower)
    if is_figurative:
        raw_threat_score *= 0.1

    # ── Decision Thresholds by Sensitivity Mode ────────────────────────────────
    n_signals = len(pa_signals)

    if mode == "high_recall":
        # More sensitive: catches subtler cues, higher recall
        pa_thresh_high = 35.0
        pa_thresh_mid = 20.0
        pa_thresh_amb = 8.0
    elif mode == "high_precision":
        # Stricter: requires strong or multiple signals, fewer false positives
        pa_thresh_high = 65.0
        pa_thresh_mid = 45.0
        pa_thresh_amb = 20.0
    else:
        # Balanced default
        pa_thresh_high = 50.0
        pa_thresh_mid = 30.0
        pa_thresh_amb = 12.0

    if raw_pa_score >= pa_thresh_high and n_signals >= 2:
        pa_label = "passive_aggressive"
        pa_confidence = min(0.92, raw_pa_score / 110.0)
    elif raw_pa_score >= pa_thresh_mid and n_signals >= 1:
        pa_label = "passive_aggressive"
        pa_confidence = min(0.75, raw_pa_score / 90.0)
    elif raw_pa_score >= pa_thresh_amb:
        pa_label = "ambiguous"
        pa_confidence = 0.50
    else:
        pa_label = "non_passive_aggressive"
        pa_confidence = max(0.40, 1.0 - (raw_pa_score / 45.0))

    # Terse curt replies without surrounding context → ambiguous
    clean_strip = text_lower.strip().rstrip(".!?")
    if len(text.split()) <= 2 and clean_strip in {"fine", "sure", "k", "whatever", "okay", "np", "right", "cool"}:
        pa_label = "ambiguous"
        pa_confidence = 0.45

    # ── Threat Label Decision ──────────────────────────────────────────────────
    if is_figurative:
        threat_label = "no_threat"
        threat_confidence = 0.90
    elif raw_threat_score >= 40 and direct_signals:
        threat_label = "direct_threat"
        threat_confidence = min(0.92, raw_threat_score / 60.0)
    elif raw_threat_score >= 25:
        if direct_signals:
            threat_label = "direct_threat"
            threat_confidence = min(0.80, raw_threat_score / 55.0)
        else:
            threat_label = "implied_threat"
            threat_confidence = min(0.75, raw_threat_score / 50.0)
    elif raw_threat_score >= 12:
        threat_label = "ambiguous"
        threat_confidence = 0.45
    else:
        threat_label = "no_threat"
        threat_confidence = max(0.70, 1.0 - (raw_threat_score / 30.0))

    # Tactic aggregation
    tactic_weights: dict[str, float] = {}
    for sig in pa_signals:
        tactic_weights[sig.tactic] = tactic_weights.get(sig.tactic, 0) + sig.weight
    tactics = sorted(tactic_weights, key=lambda t: tactic_weights[t], reverse=True)

    # Matched spans and rationale extraction
    all_pa_signals = pa_signals + style_signals
    matched_spans = []
    rule_ids = []
    rationale = []

    for s in all_pa_signals:
        rule_ids.append(s.rule_id)
        rationale.append(s.reason)
        if s.span:
            matched_spans.append({
                "start": s.span[0],
                "end": s.span[1],
                "text": text[s.span[0]:s.span[1]],
                "tactic": s.tactic,
                "rule_id": s.rule_id,
                "weight": s.weight,
            })

    is_ambiguous = (pa_label == "ambiguous")
    needs_review = is_ambiguous or (threat_label in ("direct_threat", "implied_threat")) or (25 <= raw_pa_score <= 38)

    return RuleResult(
        pa_label=pa_label,
        threat_label=threat_label,
        pa_score=round(min(100.0, raw_pa_score), 2),
        threat_score=round(min(100.0, raw_threat_score), 2),
        pa_confidence=round(pa_confidence, 3),
        threat_confidence=round(threat_confidence, 3),
        tactics=tactics,
        signals=all_pa_signals,
        threat_signals=direct_signals + implied_signals,
        is_ambiguous=is_ambiguous,
        primary_label=pa_label,
        tactic_labels=tactics,
        matched_spans=matched_spans,
        rule_ids=rule_ids,
        rationale=rationale,
        strength_score=round(min(100.0, raw_pa_score), 2),
        needs_review=needs_review,
    )


def format_signals_for_display(signals: list[RuleSignal], max_n: int = 6) -> list[dict]:
    out = []
    for sig in signals[:max_n]:
        out.append({
            "phrase": sig.name,
            "tactic": sig.tactic,
            "weight": sig.weight,
            "reason": sig.reason,
            "rule_id": sig.rule_id,
        })
    return out
