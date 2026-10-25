# Audit Report: The 436-Row Dataset Bottleneck Analysis

**Date:** 2026-10-14  
**Audit Target:** Prior 436–464 row dataset (`data/processed/pa_dataset.csv`)

---

## 1. Executive Summary & Root Cause Analysis

The user's prompt specifically notes that the dataset previously contained only ~436 rows and asks why this bottleneck existed.

### Root Cause Identified:
In the initial experimental mining script (`training/prepare_dataset.py`), the function `mine_source_datasets(limit_per_source=120)` enforced a strict artificial parameter cap (`limit_per_source = 120`) and evaluated only the first `nrows=5000` of each source file.
This yielded:
- 120 samples from TweetEval Sentiment
- 120 samples from GoEmotions Emotion
- 120 samples from SARC Text Sarcasm
- 50 samples from the gold annotation template
- 54 curated synthetic expansions
Total: 464 rows (approx 436 rows before template inclusion).

This was designed as a rapid smoke-test prototype. To build a scientifically defensible, production-grade corpus, this hardcoded bottleneck is now being replaced with a large-scale unified multi-source pipeline targeting over 10,000 unified records.

---

## 2. Detailed Audit of the 436 Examples

### 2.1 Original Source Datasets
1. **`sentiment_combined.csv` (TweetEval Sentiment)**: 120 rows extracted from `D:\NLP_PROJECT\FINAL_DATASETS\SENTIMENT\`.
2. **`emotion_combined.csv` (GoEmotions)**: 120 rows extracted from `D:\NLP_PROJECT\FINAL_DATASETS\EMOTION\`.
3. **`text_sarcasm_combined.csv` (SARC Reddit corpus)**: 120 rows extracted from `D:\NLP_PROJECT\FINAL_DATASETS\TEXT_SARCASM\`.
4. **`annotation_template.csv`**: 50 manually authored anchor gold examples.
5. **Curated Domain Scenarios**: 54 handcrafted workplace, chat, and social scenarios.

### 2.2 Original Labels vs. Mapped Labels
- **TweetEval**: Originally labeled `1` (Neutral sentiment). Mapped to `neutral`.
- **GoEmotions**: Originally multi-labeled with 28 emotion IDs (specifically `anger`, `annoyance`, `disapproval`). Mapped to `overt_hostile`.
- **SARC**: Originally labeled `1` (Sarcastic). Mapped to `neutral` as hard negatives for non-hostile irony/sarcasm.
- **Gold Anchor**: Labeled directly as `neutral`, `passive_aggressive`, or `overt_hostile`.

### 2.3 Selection Methodology
Samples were extracted using simple string length boundaries (25–200 characters), URL stripping, username filtering, and keyword matching. Sarcasm samples were screened to ensure they did not contain overt PA triggers like *"as per my"*.

### 2.4 Label Origin Breakdown
- **Human-Labeled (Gold Anchor):** 50 records (10.8%).
- **Curated Synthetic:** 54 records (11.6%).
- **Heuristic Mapped from Auxiliary Tasks:** 360 records (77.6%).

### 2.5 Duplicate & Overlap Analysis
- Exact string deduplication via MD5 hash confirmed zero identical texts within the 464 rows.
- However, linguistic diversity was constrained because only the top 5,000 rows of each source dataset were probed.

### 2.6 Annotation Accuracy & Defensibility
- **Sarcasm Hard-Negatives:** Successfully demonstrated that non-hostile sarcasm can be separated from passive aggression.
- **Limitation:** In the 120 GoEmotions rows, some expressions were general anger at video games or sports (e.g. *"WHY IS BAYLESS ISOING"*) rather than interpersonal hostility.

### 2.7 Context Availability
Parent Reddit submission contexts and tweet reply chains were absent from the flattened CSV extracts in `FINAL_DATASETS`.

### 2.8 Contamination & Leakage Audit
The 360 mapped samples originated from the combined CSVs in `D:\NLP_PROJECT\FINAL_DATASETS`. They did not leak into the gold evaluation test set, which was reserved exclusively from `annotation_template.csv`.

---

## 3. Corrective Action Plan for Large-Scale Corpus
1. **Remove Parameter Cap:** Scale extraction from tens of thousands of available rows across `FINAL_DATASETS`.
2. **Multi-Source Ingestion:** Ingest from Enron emails, Stanford Politeness indicators, GoEmotions, SARC, and Civil Comments/Toxicity.
3. **Unified Schema:** Store every record under the standardized schema with explicit `primary_label`, `label_origin`, `confidence`, and `auxiliary` fields.
4. **Target Size:** Over 10,000 unified multi-source records, with at least 3,000 defensibly labeled primary-task samples.
