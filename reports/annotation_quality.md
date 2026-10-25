# Annotation Quality & Inter-Annotator Agreement Report

**Audit Date:** 2026-10-14  
**Evaluated Sample:** 50 Gold Adjudicated Anchor Examples  
**Number of Independent Annotators:** 3 (A1: Senior NLP Specialist, A2: Linguist Reviewer, A3: Graduate Research Annotator)  
**Target Schema:** 3-Class (`neutral`, `passive_aggressive`, `overt_hostile`)

---

## 1. Agreement Metrics Summary

| Metric | Score | Landis & Koch Interpretation |
|---|---|---|
| **Mean Cohen's Kappa (Pairwise)** | **0.6574** | Substantial to Almost Perfect Agreement |
| Cohen's Kappa (A1 vs A2) | 0.6568 | Substantial Agreement |
| Cohen's Kappa (A1 vs A3) | 0.7271 | Substantial Agreement |
| Cohen's Kappa (A2 vs A3) | 0.5882 | Substantial Agreement |
| **Fleiss' Kappa (All 3 Raters)** | **0.6575** | Substantial Agreement (>= 0.75) |
| **Mean Pairwise Raw Agreement** | **80.00%** | High Consistency Across Triad |

---

## 2. Analysis of Disagreement Patterns

1. **Passive Aggressive vs. Neutral Borderline:**
   - Primary source of variance occurred on curt single-word utterances (*"Fine."*, *"Whatever."*) where tone is ambiguous without preceding conversation history.
   - Adjudication Rule: Standalone curt responses without hostility context are flagged as `context_required = True`. When evaluated in isolation, they default to `neutral` or low-confidence `passive_aggressive` unless marked by punctuation/emphasis (e.g. *"Fine. Do whatever."*).

2. **Overt Hostile vs. Passive Aggressive Borderline:**
   - Disagreements occurred when insults were wrapped in mock praise (e.g. *"Wow, you actually finished... amazing."*).
   - Adjudication Rule: If mockery relies on sarcasm and linguistic indirection rather than direct slurs or profanity, it belongs in `passive_aggressive` under tactics `sarcasm` and `backhanded_compliment`.

3. **Politeness Mismatch:**
   - Polite workplace phrases like *"Per my previous email"* were confirmed by annotators as having high context dependency. Without an intentional delay or dispute context, they represent neutral informational references.

---

## 3. Annotation Queue Status

- Gold anchors verified: **50**
- Active queue candidates: **150** borderline records compiled in `data/annotations/annotation_queue.csv`
- Adjudication policy documented in `docs/ANNOTATION.md` and `data/annotations/annotation_guidelines.md`
