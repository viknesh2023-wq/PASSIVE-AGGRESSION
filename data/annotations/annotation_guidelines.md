# Annotation Guidelines — Passive Aggression Detector
## Version 1.0 | 2026-10-14

---

## Purpose

This guide defines the annotation schema for the **PassiveAggression Detector** gold-label dataset.
Annotators must read and understand this guide before labeling any examples.

The goal is to distinguish **passive-aggressive** language from:
- Direct disagreement (which is NOT passive aggression)
- Sarcasm without hostility (which is NOT automatically PA)
- Overt hostility (which is different from passive aggression)
- Genuinely neutral communication

---

## Labels

### 1. `neutral`
Communication that is **direct, clear, and non-hostile**, regardless of its emotional content.

**Key test:** Could a reasonable person read this and feel no hidden agenda or indirect attack?

Examples:
- "I disagree with your proposal — here are my concerns."
- "Could you send me the file by tomorrow?"
- "I don't think this approach will work."
- "Thanks for sending that."
- "That was hilarious 😂"
- "I'm frustrated by the delay but I understand the situation."

**Important neutrals that are NOT passive-aggressive:**
- Direct expressions of frustration: "I'm really annoyed by this."
- Negative opinions stated clearly: "I don't like this design."
- Requests stated plainly: "Please respond by end of day."

---

### 2. `passive_aggressive`
Communication that **indirectly expresses hostility, resentment, or criticism** through:
- Subtle digs disguised as neutral language
- Compliments that contain implicit criticism
- Polite framing that masks dismissal or blame
- Comments designed to make the recipient feel guilty
- Avoidance, silence, or curt responses as a form of protest

**Key test:** Is the hostility plausibly deniable? Does the writer appear cooperative while subtly undermining?

Examples:
- "As per my last email..." (blame-shifting with implicit criticism)
- "No worries! I'm sure you were just busy." (guilt-tripping + backhanded)
- "Fine. Do whatever you think is best." (curt dismissal disguised as agreement)
- "Thanks for finally getting that done." (backhanded compliment)
- "Not to be rude, but..." (mock politeness)
- "With all due respect, I think you're wrong." (passive-aggressive if context suggests contempt)

**Ambiguous cases — use `context_required = true`:**
- Short phrases like "OK." or "Sure." depend entirely on context.
- "Fine." as a curt reply is likely PA; "Fine, that works" is likely neutral.

---

### 3. `overt_hostile`
Communication that **directly attacks, insults, threatens, or demeans** the recipient.
The hostility is **not disguised** — it is explicit.

Examples:
- "You are completely useless."
- "Stop wasting everyone's time."
- "This is the worst work I've ever seen."
- "I can't believe how incompetent you are."

**Distinguish from passive-aggressive:**
- Overt hostility: "You never do anything right." (direct attack)
- Passive-aggressive: "I'm sure you tried your best." (backhanded)

---

## Tactics (Multi-label — apply all that are present)

Each labeled example should also receive one or more tactic labels:

| Tactic | Description | Example |
|--------|-------------|---------|
| `sarcasm` | Saying the opposite of what is meant, usually to mock | "Oh great, another brilliant idea." |
| `backhanded_compliment` | Apparent praise containing implicit criticism | "You actually finished on time for once." |
| `guilt_tripping` | Implying the reader has failed in an obligation | "I've been waiting two weeks, but no worries." |
| `mock_politeness` | Polite language masking condescension or dismissal | "With all due respect, you're wrong." |
| `blame_shifting` | Redirecting fault or responsibility to the reader | "As per my last email, which I sent three times..." |
| `curt_dismissal` | Abrupt disengagement disguised as agreement | "Fine. Whatever you think." |

---

## Edge Cases and Decision Rules

### Sarcasm WITHOUT passive aggression
Sarcasm is not automatically passive-aggressive.
- "That was hilarious 😂" — neutral humor
- "Because THAT makes total sense 🙄" — sarcasm but may be mild, context-dependent
Label as `sarcasm` tactic + `passive_aggressive` label ONLY if the sarcasm targets a specific person with hostile intent.

### Direct disagreement
"I disagree." / "I don't think that's correct." / "I think we should do X instead." = `neutral`
Only becomes PA if disguised as agreement: "Sure, if you think that's best." (while clearly believing otherwise)

### Genuine politeness
"Thank you so much for your help!" — genuine if tone and context support it.
"Thank you SO much" (when angry) — may be sarcastic. Use `context_required = true` when ambiguous.

### Emojis
Do not treat 😊 / 😂 / 👍 as PA signals by default.
An 🙄 or 😒 in a professional context may reinforce passive aggression — annotate tactfully.

### Context-dependent cases
If the label cannot be determined without knowing the prior conversation, speaker relationship, or situational context, mark `context_required = true`. The primary label should still be your best single-message assessment.

### Short ambiguous messages
- "OK." — context_required, default neutral
- "Sure." — context_required, default neutral
- "Fine." at end of argument — likely PA, mark context_required
- "Fine, that works." — likely neutral

---

## Annotation Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| sample_id | string | yes | Unique ID e.g., PA_001234 |
| text | string | yes | Original text (unmodified) |
| context | enum | yes | workplace / chat / social / unknown |
| speaker_role | string | no | e.g., manager, peer, subordinate, stranger |
| primary_label | enum | yes | neutral / passive_aggressive / overt_hostile |
| tactics | list[str] | yes | Zero or more tactic labels |
| context_required | bool | yes | True if label depends on prior context |
| annotator_id | string | yes | Anonymous annotator identifier |
| annotation_notes | string | no | Free-text reasoning (encouraged) |
| adjudicated_label | enum | no | Final label after disagreement resolution |
| source_dataset | string | yes | Source of the text |
| split_group | enum | no | train / val / test |

---

## Inter-Annotator Agreement

- At least **2 annotators** per example in the primary gold set.
- At least **3 annotators** per example in the evaluation gold set.
- Disagreements are resolved by a third annotator or adjudication session.
- Agreement measured by **Cohen's kappa** for 2 annotators.
- Report kappa per label class, not just overall.

**Target kappa:** >= 0.60 is acceptable; >= 0.70 is good; < 0.50 requires guideline revision.

---

## What NOT to do

- Do NOT label every negative or critical sentence as passive-aggressive.
- Do NOT label sarcasm as passive-aggression unless it specifically targets the recipient.
- Do NOT use the test set to tune labeling thresholds or fix guidelines retroactively.
- Do NOT include personally identifying information.
- Do NOT fabricate annotation agreement statistics.

---

## Privacy

- Do not annotate real private messages without explicit consent.
- All examples should be from public sources, synthetic data, or permissioned corpora.
- Remove or anonymize names, email addresses, phone numbers before annotation.

---

## Source Attribution

Every example must record its source dataset. Permitted sources:
1. Independently authored synthetic examples (clearly marked synthetic=True)
2. Public research corpora with research-use license
3. Redacted/anonymized examples contributed with consent

---

## Annotation Workflow

1. Read the complete guidelines.
2. Annotate a calibration batch of 50 examples.
3. Compare with second annotator — compute kappa.
4. Adjudicate disagreements.
5. Revise guidelines if kappa < 0.60.
6. Annotate main batch.
7. Final adjudication of contested examples.
8. Freeze test set — no further changes after ML model selection begins.
