# Dataset Engineering & Corpus Documentation

**Project:** PassiveAggression Detector — Large-Scale Unified Corpus  
**Version:** 2.0  
**Date:** 2026-10-14  

---

## 1. Corpus Hierarchy & Scale

The corpus is structured into two tiers to prevent task confusion while maximizing data scale:

1. **Unified Multi-Source Corpus (`data/processed/unified_corpus.csv`):**
   - **Total Records:** 10,489 unique records.
   - **Sources:** TweetEval Sentiment, SARC Reddit Corpus, GenZ-Sarc Synthetic Batches, GoEmotions Reddit Corpus, Curated Domain Scenarios, and Gold Annotation Anchors.
   - **Scope:** Includes primary 3-class records as well as auxiliary emotion, dialogue, and non-hostile sarcasm records for transfer learning and multi-task representation.

2. **Primary Classification Dataset (`data/processed/pa_dataset.csv`):**
   - **Total Records:** 9,489 unique, leakage-safe records.
   - **Class Distribution:**
     - `neutral`: 8,374 records (88.2%)
     - `overt_hostile`: 890 records (9.4%)
     - `passive_aggressive`: 225 records (2.4%)
   - **Partitioning:**
     - `train`: 6,638 records (70.0%)
     - `val`: 1,425 records (15.0%)
     - `test`: 1,426 records (15.0%)
   - **Leakage Status:** **0 string overlap** across Train, Val, and Held-out Test.

---

## 2. Source Ingestion Details

| Dataset Identifier | Source Description | Record Count | Target Role in Pipeline | License |
|---|---|---|---|---|
| `gold_annotation_template` | Curated gold-standard anchors | 50 | Ground truth anchor for evaluation & calibration | Project Specific |
| `curated_pa_expansion` | Tactic-specific syntactic templates | 82 unique | High-precision coverage across all 6 tactics | CC-BY-4.0 |
| `contrastive_hard_negatives`| Polite phrases with trigger words | 60 | Prevents false alarms on sincere messages | CC-BY-4.0 |
| `tweeteval_sentiment` | TweetEval 3-class sentiment | 3,500 | Clean neutral conversational baselines | Research Only |
| `sarc_reddit_regular` | SARC regular Reddit comments | 2,000 | Non-sarcastic everyday text baselines | Research Only |
| `sarc_reddit_sarcasm` | SARC sarcastic Reddit comments | 2,000 | Hard-negative non-hostile irony/sarcasm | Research Only |
| `sarc_mined_pa` | SARC comments with PA markers | 143 | Authentic real-world passive aggression | Research Only |
| `genz_synthetic_sarcasm` | GenZ-Sarc batches | 800 | Digital dialogue and modern slang | Project Specific |
| `goemotions_hostile` | GoEmotions anger/disapproval | 883 | Direct aggression, insults, and harassment | Apache-2.0 |
| `goemotions_auxiliary` | GoEmotions fine-grained emotions | 1,000 | Multi-task auxiliary emotion features | Apache-2.0 |

---

## 3. Data Cleansing & Leakage Prevention Protocol

- **Normalization:** Unicode NFC normalization, Windows newline standardization (`\r\n` -> `\n`), and horizontal whitespace collapsing.
- **Deduplication:** Lowercased text MD5 hashing eliminated 1,347 duplicate and near-duplicate records.
- **Gold Protection:** The 50-example gold anchor partition was sequestered with its predetermined train/val/test splits preserved untouched.
- **Stratification:** Stratified partitioning ensured class representation remained consistent across all splits with fixed seed 42.
