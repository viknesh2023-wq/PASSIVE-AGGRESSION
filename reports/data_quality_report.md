# Large-Scale Data Quality and Corpus Integrity Report

**Generated:** 2026-10-14T18:02:31.809070+00:00  
**Unified Corpus Records:** 10,489  
**Primary Task Records:** 9,489  
**Primary Split:** Train=6638, Val=1425, Held-out Test=1426

---

## 1. Primary Task Class Distribution

| Primary Label | Count | Percentage | Role |
|---|---|---|---|
| **Neutral** | 8,374 | 88.2% | Objective, polite, and non-hostile sarcasm hard-negatives |
| **Passive Aggressive** | 225 | 2.4% | Subtly hostile text across all 6 tactics |
| **Overt Hostile** | 890 | 9.4% | Explicit aggression, insults, and harassment |

---

## 2. Provenance and Label Origin Breakdown

| Origin | Count | Description |
|---|---|---|
| **Gold** | 50 | Manually verified gold-standard anchor data |
| **Verified Mapped** | 8,586 | Heuristically vetted mappings from TweetEval, GoEmotions & SARC |
| **Synthetic Curated** | 853 | Controlled tactic permutations & contrastive pairs |
| **Unlabelled / Auxiliary** | 1,000 | Multitask auxiliary emotion and conversational dialogue records |

---

## 3. Linguistic Length Statistics (Primary Corpus)

- **Mean Word Count:** 15.06 words
- **Median Word Count:** 15.0 words
- **Standard Deviation:** 6.70 words
- **Min / Max Word Count:** 1 / 53 words

---

## 4. Leakage and Quality Verifications

- [x] Zero null or empty text values.
- [x] Full Unicode NFC normalization applied across all sources.
- [x] Hash-based deduplication eliminated exact and near-exact duplicates.
- [x] Train, validation, and held-out test splits have 0 string overlap.
- [x] Gold-standard test partition strictly sequestered.
- [x] **Strict English-Only Enforcement:** 100% of ingested corpora verified as English. Non-English scripts are filtered and rejected at API boundary.

---

## 5. Cleanlab Label-Noise Audit Summary

- **Methodology:** Confident learning via 5-fold cross-validated out-of-fold predicted probabilities.
- **Identified Potential Label Issues:** **118 candidate records** (1.78% of the 6,638 training set).
- **Export Destination:** Flagged records with confidence deltas exported to [`reports/cleanlab_review_candidates.csv`](file:///d:/python_folder/reports/cleanlab_review_candidates.csv).
- **Adjudication Findings:** The majority of flagged items involved boundary sarcasm containing harsh language that was originally silver-mapped to neutral, as well as curt single-word dismissals. Human review protocol documented in `docs/ANNOTATION.md`.

