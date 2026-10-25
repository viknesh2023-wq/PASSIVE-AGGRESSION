# PassiveAggression Detector 2.0 — Dataset Schema & Taxonomy Specification

**Version:** 2.0  
**Scope:** Strict English-Only Social Media Communication  
**Target Corpus:** 5,000 Quality-Controlled Synthetic Examples  

---

## 1. Primary Classification Taxonomy

PassiveAggression Detector 2.0 adopts a mathematically defensible, mutually exclusive 3-class primary label taxonomy:

### 1. `passive_aggressive`
- **Definition:** Indirectly communicates hostility, resentment, contempt, blame, mock politeness, or a similar negative interpersonal message. The literal wording may appear polite, neutral, or superficially positive while the contextual pragmatics convey hostility.
- **Key Indicators:**
  - Mock politeness (*"With all due respect, perhaps read the thread before commenting"*).
  - Backhanded compliments (*"You actually wrote something coherent for once"*).
  - Guilt tripping (*"Don't worry about replying, I'm used to being ignored"*).
  - Blame shifting (*"Per my previous tweet, which was clearly too complex for some"*).
  - Curt dismissals with resentment (*"Fine.", "Whatever you say.", "Cool story."*).
  - Indirect resentment / faux-cheerfulness masking frustration.

### 2. `non_passive_aggressive`
- **Definition:** Communication that does not convey covert hostility.
- **Includes:**
  - Sincere politeness and genuine gratitude (*"Thank you so much for clarifying that!"*).
  - Ordinary disagreement without hostility (*"I disagree with this conclusion because the data points in a different direction"*).
  - Direct criticism without passive aggression (*"That design is broken and needs to be completely redone"*).
  - Friendly sarcasm and non-hostile banter (*"Oh no, not more free pizza, what a terrible disaster 😂"*).
  - Sincere agreement (*"Sure, whatever you think is best, I trust your judgment"*).
  - Objective informational exchange and neutral chatter.

### 3. `ambiguous`
- **Definition:** The message and available context do not support a sufficiently reliable classification.
- **Includes:**
  - Context-deprived terse responses (*"Okay.", "Sure."* without prior tone).
  - Double-entendre without conversational history.
  - Expressions where tone could be either playful teasing or covert bite depending on unspoken personal relationship.
  - Understated reactions where evidence is insufficient to justify a definitive verdict.

> **CRITICAL RULE:** Do NOT automatically classify sarcasm, negative sentiment, anger, profanity, toxicity, punctuation, capitalization, or emojis as passive aggression.

---

## 2. Auxiliary Labels (Strictly Decoupled)

Auxiliary tasks are kept completely separate from the primary label:

1. **`tactic_labels`**: Semicolon-delimited tactics: `sarcasm`, `backhanded_compliment`, `guilt_tripping`, `mock_politeness`, `blame_shifting`, `curt_dismissal`, or `none`.
2. **`sarcasm_label`**: `sarcastic` | `non_sarcastic`.
3. **`sentiment_label`**: `positive` | `neutral` | `negative`.
4. **`hostility_label`**: `indirect_hostile` | `overt_hostile` | `non_hostile`.
5. **`context_dependence`**: `high` | `medium` | `low` (indicates whether context alters the primary interpretation).

---

## 3. Dataset Schema (18 Required Fields)

| Field Name | Type | Description | Nullable / Default |
|---|---|---|---|
| `id` | String | Unique sample ID (`synth_0001` .. `synth_5000`) | Non-null |
| `text` | String | Target comment / message text to classify | Non-null |
| `primary_label` | String | `passive_aggressive`, `non_passive_aggressive`, `ambiguous` | Non-null |
| `post_text` | String | Initiating root post text (if part of a thread) | Null for standalone |
| `conversation_id` | String | Thread ID grouping all related turns | Non-null |
| `turn_index` | Integer | Turn position in conversation (0 = post, 1 = comment, 2 = reply) | Non-null (0 for standalone) |
| `parent_comment_id` | String | ID of the parent comment being replied to | Null for root / standalone |
| `parent_comment` | String | Immediate parent text | Null for root / standalone |
| `thread_context` | String | Ordered preceding conversation history | Null for standalone |
| `target_message_id` | String | Identifier of the message to evaluate | Non-null |
| `tactic_labels` | String | Semicolon-delimited list of detected tactics or `none` | Non-null |
| `sarcasm_label` | String | Sarcasm indicator (`sarcastic`, `non_sarcastic`) | Non-null |
| `sentiment_label` | String | Surface sentiment (`positive`, `neutral`, `negative`) | Non-null |
| `generation_source` | String | `synthetic_social_engine_v2` | Non-null |
| `generator_model` | String | Engine identifier and parameter version | Non-null |
| `generation_batch` | String | Batch label (`pilot_500`, `full_5000`) | Non-null |
| `review_status` | String | Automated QC status (`automated_qc_passed`) | Non-null |
| `quality_flags` | String | Diagnostic flags (`clean`, `context_dependent`, etc.) | Non-null |

---

## 4. Linguistic & Social Media Coverage Requirements

The synthetic generator must sample across realistic social media archetypes:
- **Platforms:** Twitter/X micro-replies, Reddit debate threads, Instagram comment banter, LinkedIn corporate faux-politeness, Discord chat exchanges.
- **Linguistic Markers:** Slang, abbreviations (`tbh`, `ngl`, `imo`, `per my last`), emojis (`🙃`, `🙂`, `👏`, `💀`, `👀`), informal punctuation (`...`, `??`, `!?!`, all-lowercase, selective ALL CAPS).
- **Subtle Contrastive Pairs:** Identical or near-identical surface phrases with divergent labels depending on context:
  - *"Sure, whatever you think is best"* in response to a sincere design proposal -> `non_passive_aggressive`.
  - *"Sure, whatever you think is best"* in response to an overruled suggestion after an argument -> `passive_aggressive`.
  - *"Sure, whatever you think is best"* with zero prior context -> `ambiguous`.
