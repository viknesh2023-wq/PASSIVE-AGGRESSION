# Phase 1 Audit Report: PassiveAggression Detector Codebase & Data Assets

**Date:** 2026-10-22  
**Auditor:** Senior NLP Researcher & Machine Learning Engineer  
**System Target:** PassiveAggression Detector 2.0 (English-Only Social Media Passive-Aggression Detection)

---

## 1. Executive Summary & Scope

An exhaustive audit of `d:\PassiveAggression Detector` and read-only inspection of auxiliary external sources was conducted. The audit examined all dataset files, record counts, filtering pipelines, model architectures, saved checkpoints, evaluation reports, backend routes, and frontend components.

### Key Audit Findings:
1. **The 10,489 Record Count Verified:** The file `data/processed/unified_corpus.csv` contains exactly **10,489 rows** (with 9,489 in `data/processed/pa_dataset.csv` after removing 1,000 unlabelled emotion samples).
2. **True Nature of the 10,489 Records:** Only **225 rows** (2.4%) represent actual passive-aggression samples (50 human-authored gold anchor samples, 53 curated synthetic expansions, 60 contrastive hard-negatives, and 143 weakly mined SARC forum comments). The remaining 8,374 samples are neutral sentiment tweets / forum comments, and 890 are anger/hostility comments from GoEmotions.
3. **The 436-Row Bottleneck Explained:** The initial prototype (`training/prepare_dataset.py`) applied an artificial cap of `limit_per_source = 120` rows from only the first 5,000 rows of each source dataset (120 TweetEval + 120 GoEmotions + 120 SARC Sarcasm + 50 Gold Anchors + 54 Synthetic expansions = 464 rows total, ~436 prior to anchor addition).
4. **Current Deployed Model:** The active FastAPI model is `pa-tfidf-svm-v1` (`models/pa_classifier.pkl`, 7.87 MB), a Platt-calibrated Linear SVM trained on `data/processed/pa_dataset.csv` using 3 legacy classes (`neutral`, `passive_aggressive`, `overt_hostile`).
5. **Data Policy Compliance:** The 10,489 auxiliary records are retained in full as separate benchmark/auxiliary data. They will **not** be merged into the primary synthetic training dataset of 5,000 records.

---

## 2. Inventory & Classification of the 10,489 Existing Records

Audit of `data/processed/unified_corpus.csv` breakdown across the 8 distinct categories required:

| Category | Source Dataset | Records | Original Label / Task | Role in Existing Project | Verified Status & Suitability for PA 2.0 |
|---|---|---:|---|---|---|
| **1. Direct Passive-Aggression** | `gold_annotation_template.csv` | 50 | 3-Class Ground Truth | Hand-authored gold anchors | Verified gold anchors. Kept as auxiliary gold reference. |
| **2. Sentiment-Only Data** | `tweeteval_sentiment` | 3,500 | Label `1` (Neutral) | Neutral conversational baseline | Sentiment task only; neutral tone != non-PA social reply. Preserved in auxiliary. |
| **3. Sarcasm & Irony Data** | `sarc_reddit_sarcasm` | 2,000 | Label `1` (Sarcastic) | Sarcasm hard-negatives | Non-hostile sarcasm/irony. Preserved in auxiliary. |
| **4. Emotion & Hostility Data** | `goemotions_hostile` | 883 | Anger, Annoyance | Overt hostility baseline | Direct anger/insults; overt, not indirect/PA. Preserved in auxiliary. |
| **5. Ordinary Conversational** | `sarc_reddit_regular` | 2,000 | Label `0` (Non-sarcastic) | Everyday forum dialogue | Unmoderated forum comments. Preserved in auxiliary. |
| **6. Synthetic Examples Present** | `genz_synthetic_sarcasm`<br>`curated_pa_expansion`<br>`contrastive_hard_negatives` | 800<br>53<br>60 | Modern banter<br>PA tactics<br>Minimal pairs | Digital banter<br>PA syntactic templates<br>Trigger word hard negatives | Synthetic expansions. Kept as reference/auxiliary. |
| **7. Weakly Labeled / Derived** | `sarc_mined_pa` | 143 | Keyword + Pattern mined | Silver PA examples | Silver heuristics from Reddit. Kept as auxiliary. |
| **8. Unlabelled Auxiliary Vectors** | `goemotions_auxiliary` | 1,000 | 28 Emotion categories | Feature extractor support | Preserved as auxiliary feature set. |
| **TOTAL** | | **10,489** | | | **Preserved intact; segregated from primary synthetic corpus.** |

---

## 3. Analysis of Existing Preprocessing & Filtering Logic

- **Text Normalization (`backend/app/nlp/preprocess.py`):**
  - Unicode NFC normalization.
  - Whitespace collapse (`\s+` -> `' '`).
  - URL stripping (`http\S+` -> ` `).
  - Handles emoji retaining (preserves sentiment-bearing emojis).
  - Sentence splitting with accurate character offset tracking for text highlights.
- **Phrase Matcher (`backend/app/nlp/phrase_matcher.py`):**
  - Span detection for 6 PA tactics: `sarcasm`, `backhanded_compliment`, `guilt_tripping`, `mock_politeness`, `blame_shifting`, `curt_dismissal`.
  - Calculates style bonuses: ellipsis bonus (`...`), excessive caps (`[A-Z]{3,}`), and rhetorical punctuation (`?!`).
- **Linguistic Features (`backend/app/nlp/features.py`):**
  - 28-dimensional linguistic feature vector (pronoun ratios, modal verbs, sentiment contrast, exclamation frequency, hedge words, etc.).
- **Rewriter (`backend/app/nlp/rewriter.py`):**
  - Rule-based calmer rewrite suggestions for common passive-aggressive triggers (e.g., *"Per my last email"* -> *"As mentioned earlier"*).

---

## 4. Model Artifacts & Evaluation Audit

| Model Artifact | Path | Size | Verified Status | Notes |
|---|---|---:|---|---|
| `pa_classifier.pkl` | `models/pa_classifier.pkl` | 7.87 MB | Active Deployed | TF-IDF (1, 2) + Calibrated Linear SVM (trained on `pa_dataset.csv` with legacy 3-class distribution). |
| `active_model.json` | `models/registry/active_model.json` | 30 KB | Active Metadata | Documents accuracy 0.9565, macro F1 0.8611 on held-out split of `pa_dataset.csv`. |
| `xlmr_pa_detector/` | `models/xlmr_pa_detector/` | 1.11 GB | External / Evaluated | XLM-RoBERTa fine-tuned weights (Multilingual; excluded from English-only 2.0). |

---

## 5. Inference & Frontend Evaluation

- **FastAPI Endpoints:**
  - `POST /api/analyze`: Single text prediction, highlights, tactics, sentence tension, and calm rewrites.
  - `POST /api/batch`: Multi-speaker message thread analysis.
  - `POST /api/feedback`: SQLite persistence for user feedback.
  - `GET /api/history`: SQLite history retrieval.
  - `GET /api/health`: Health status and active model version.
  - `GET /api/dashboard/*`: Dashboard metrics, architecture, dataset inventory, error analysis.
- **Frontend Capabilities:**
  - `src/pages/Analyzer.tsx`: Interactive single-text analyzer with sensitivity controls (Lenient, Balanced, Strict), tactic badges, sentence tension graph, and calm rewrites.
  - `src/pages/Thread.tsx`: Multi-speaker conversation thread viewer.
  - `src/pages/ResearchHub.tsx`: Dark-themed technical dashboard displaying architecture, model comparison, data inventory, and confusion matrices.
  - `src/pages/History.tsx`: Historical inspection and export.
  - `src/pages/Insights.tsx`: Tactic distribution and aggregate analytics.

---

## 6. Audit Verdict & Action Plan for 2.0

1. **Isolation:** Keep `data/processed/unified_corpus.csv` and existing raw files untouched. Document them as auxiliary datasets.
2. **New Primary Target:** Build an isolated synthetic dataset generation pipeline in `training/generate_synthetic_data.py` producing exactly **5,000 quality-controlled examples** in `data/synthetic/pa_synthetic_5000.csv`.
3. **Class Definition Update:** Transition primary classification to the three mandated classes:
   - `passive_aggressive` (1,667 rows)
   - `non_passive_aggressive` (1,667 rows)
   - `ambiguous` (1,666 rows)
4. **Conversation Threads:** Include conversational context (`post_text`, `parent_comment`, `turn_index`, `thread_context`) directly in the schema and model evaluation.
