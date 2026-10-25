# Phase 0 Implementation Audit: PassiveAggression Detector 2.0

**Date:** 2026-10-25  
**Auditor:** Senior ML Engineer & Python Architect  
**Workspace Root:** `D:\PassiveAggression Detector`  
**Target Architecture:** Self-contained, data-independent dual-model passive aggression & threat detector with Streamlit frontend.

---

## 1. Discovered Project Assets & Components

### 1.1 Directory & File Inventory
| Path | Type | Status & Description |
|---|---|---|
| `streamlit_app.py` | Python Script | Streamlit main entrypoint (Home page, dark theme). |
| `pages/` | Directory | 5 multi-page views (`1_analyze_comment.py`, `2_analyze_conversation.py`, `3_history.py`, `4_insights.py`, `5_about.py`). |
| `inference/` | Directory | Standalone Python inference package (`pa_inference.py`, `rule_detector.py`, `preprocess.py`, `history_db.py`). Zero HTTP/FastAPI dependency for Streamlit. |
| `models/` | Directory | `pa_classifier_v2.pkl` (1.3 MB, LogReg 3-class), `threat_classifier_v1.pkl` (0.9 MB, 4-class), `pa_classifier.pkl` (7.87 MB, legacy LinearSVM). |
| `data/synthetic/` | Directory | 9,000 synthetic records in 4 batches (`batch1_SYNPA_00001_02500.csv` to `batch4_SYNPA_07501_10000.csv`), plus merged CSV and JSONL. All 19 canonical columns present. |
| `data/processed/` | Directory | `unified_corpus.csv` (10,489 records) and legacy `pa_dataset.csv`. |
| `data/annotations/` | Directory | `annotation_template.csv` (50 hand-annotated gold anchors), guidelines. |
| `backend/` | Directory | Preserved FastAPI backend, SQLite database (`backend/pa_detector.db`), and existing test suite. |
| `tests/` | Directory | `test_v2_inference.py` (9 tests for v2 inference, rule engine, history, and schema). |
| `reports/` | Directory | Evaluation reports, audits, and comparison JSON/MD files. |
| `requirements.txt` | Configuration | Core dependencies (`streamlit`, `scikit-learn`, `pandas`, `nltk`, etc.). |
| `.streamlit/config.toml` | Configuration | Pure black theme (`backgroundColor="#000000"`, `primaryColor="#E53E3E"`). |

---

## 2. Investigation of Dataset Sources & The 10,520 vs 10,489 Discrepancy

### 2.1 The 10 Dataset Sources
The historical 10 source datasets were identified across local project annotations and `D:/NLP_PROJECT/FINAL_DATASETS/`:
1. **TweetEval 3-Class Sentiment:** Extracted from `D:/NLP_PROJECT/FINAL_DATASETS/SENTIMENT/sentiment_combined.csv` (3,500 records).
2. **SARC Reddit Regular:** Extracted from `D:/NLP_PROJECT/FINAL_DATASETS/TEXT_SARCASM/text_sarcasm_combined.csv` (2,000 records).
3. **SARC Reddit Irony/Sarcasm:** Extracted from `D:/NLP_PROJECT/FINAL_DATASETS/TEXT_SARCASM/text_sarcasm_combined.csv` (2,000 records).
4. **GoEmotions Auxiliary:** Extracted from `D:/NLP_PROJECT/FINAL_DATASETS/EMOTION/emotion_combined.csv` (1,000 records).
5. **GoEmotions Anger & Hostility:** Extracted from `D:/NLP_PROJECT/FINAL_DATASETS/EMOTION/emotion_combined.csv` (883 records).
6. **GenZ-Sarc Digital Banter:** Extracted from `D:/NLP_PROJECT/GenZ-Sarc/*.csv` (800 records).
7. **SARC Authentic PA Mined:** Keyword-pattern mined from `TEXT_SARCASM` (143 records).
8. **Curated Tactic Syntactic Expansion:** Algorithmic template expansions across 6 tactics (53 actual records in `unified_corpus.csv`).
9. **Contrastive Hard Negatives:** Syntactic minimal pairs (60 records in `unified_corpus.csv`).
10. **Gold Passive Aggression Anchors:** Hand-annotated reference anchors in `data/annotations/annotation_template.csv` (50 actual records).

### 2.2 Cause of the 31-Record Discrepancy (10,520 vs 10,489)
* **Reported Theoretical Expectation:** 3,500 + 2,000 + 2,000 + 1,000 + 883 + 800 + 143 + 82 + 60 + 52 = **10,520 records**.
* **Actual Processed Unified Corpus:** Exactly **10,489 records**.
* **Variance Analysis:**
  1. Curated Tactic Syntactic Expansion: Originally planned as 82 rows. Deduplication and exact/near-duplicate pruning reduced this to exactly **53 unique rows** (Difference: **-29 rows**).
  2. Gold Passive Aggression Anchors: Originally drafted as 52 candidate rows. Final verified human annotations in `annotation_template.csv` yielded exactly **50 verified rows** (Difference: **-2 rows**).
  3. Total delta: -29 + -2 = **-31 records**.
  4. Formula: `10,520 - 31 = 10,489`.
* **Conclusion:** The discrepancy is completely accounted for by deduplication of template expansions (29 rows) and finalized human anchor verification (2 rows). No records were lost or unaccounted for.

---

## 3. Discovered Rewriting Implementation
The project previously implemented a deterministic rewrite engine in `backend/app/nlp/rewriter.py` paired with `backend/app/nlp/phrase_matcher.py`:
* Identified rewritable tactic matches (e.g., *"As per my last email"* -> *"As I mentioned earlier"*, *"Whatever you say"* -> *"I will go with your decision"*).
* Case-preserving string substitution.
* Returns 1–3 calmer alternatives without generative hallucinations.
* This logic will be migrated, enriched, and modernized into `inference/rewrite_engine.py` for direct use in Streamlit.

---

## 4. Identified Missing Items & Gap Analysis

1. **Self-Contained Raw Sources:** The 10 source datasets must be physically copied into `data/raw/existing_sources/` with SHA-256 checksums and manifests so runtime has zero external path dependencies.
2. **Reproducible Pipeline:** `training/audit_datasets.py` and `training/prepare_datasets.py` to generate `data/processed/` splits.
3. **Leakage-Resistant Splits in CSV:** `pa_train.csv`, `pa_validation.csv`, `pa_test.csv`, `threat_train.csv`, `threat_validation.csv`, `threat_test.csv`.
4. **Model Comparison with Legacy SVM:** Evaluate both `pa_classifier_v2.pkl` and `pa_classifier.pkl` on the identical test split.
5. **Rule Engine Enhancements:** Support 10 rule families, 3 sensitivity levels (`balanced`, `high_recall`, `high_precision`), and exact span tracking.
6. **Rewriting Assistant:** Create `inference/rewrite_engine.py` and dedicated Streamlit page `pages/4_rewrite_comment.py` (total 7 pages).
7. **Comprehensive Tests:** Add tests for rewrites, rules, data integrity, and edge cases.

---

## 5. Backup Manifest
Backups of all existing models, databases, and datasets have been secured in `backups/phase0_backup/` and indexed in `reports/backup_manifest.json` (23 files verified with SHA-256).
