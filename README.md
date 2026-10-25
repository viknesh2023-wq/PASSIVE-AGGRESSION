# PassiveAggression Detector 2.0 — Complete Data-Independent, Dual-Model System

An English-only social-media analysis and constructive communication system that detects passive aggression, independently identifies threatening language, and provides constructive, meaning-preserving message rewrites for comments, replies, and conversation threads.

**Streamlit is the sole, active user-facing frontend.** All datasets, splits, manifests, rule engines, models, and tests reside entirely inside this project root with **zero runtime external dependencies**.

---

## 🎯 Architecture Diagram

```
[ 10 Self-Contained Source Datasets (10,489) ]  +  [ Dedicated Synthetic Dataset (9,000 SYNPA) ]
                                      │
                                      ▼
                          [ Strict Schema Audit ]
            (data/manifests/dataset_manifest.json, SHA-256 Checksums)
                                      │
                                      ▼
                        [ Leakage-Resistant Splits ]
           (70% Train / 15% Val / 15% Test — Conversation-Grouped)
                                      │
           ┌──────────────────────────┴──────────────────────────┐
           ▼                                                     ▼
[ Model 1: Supervised ML ]                           [ Model 2: Rule-Based Engine ]
• TF-IDF (word 1-3, char 3-5)                        • 10 Distinct Rule Families
• Logistic Regression (3-class)                      • 3 Operating Sensitivity Modes
• Held-out Test Accuracy: 99.2%                      • Exact Character Span Offsets
• Held-out Macro F1: 0.992                           • Benign Overrides & Figurative Filters
• Separate Threat Classifier (4-class)               • Full Explainability & Rationale
           │                                                     │
           └──────────────────────────┬──────────────────────────┘
                                      │
                                      ▼
                     [ Unified Inference Orchestrator ]
                        (inference/pa_inference.py)
                                      │
           ┌──────────────────────────┴──────────────────────────┐
           ▼                                                     ▼
[ Constructive Rewrite Assistant ]                     [ 7-Page Streamlit Frontend ]
• Core Meaning Preservation                            • Pure Black (#000000) Theme
• Removes Covert Hostility                             • Dual-Model Comparison Cards
• Zero Fact Hallucinations                             • Matched Phrase Highlighting
• Safety Refusal on Threats                            • SQLite Analysis History
• One-Click Alternative Copy                           • Real Offline Evaluation Insights
```

---

## 🚀 Key Features

### 1. 100% Self-Contained, Data-Independent Architecture
- All ten original and mined dataset sources reside locally inside `data/raw/existing_sources/`:
  1. `tweeteval_sentiment` (3,500 records)
  2. `sarc_reddit_regular` (2,000 records)
  3. `sarc_reddit_sarcasm` (2,000 records)
  4. `goemotions` (1,000 records)
  5. `goemotions_anger_hostility` (883 records)
  6. `genz_sarc_banter` (800 records)
  7. `authentic_pa_mined` (143 records)
  8. `curated_pa_tactics` (53 records)
  9. `contrastive_hard_negatives` (60 records)
  10. `gold_pa_anchors` (50 records)
- Dedicated 9,000-record synthetic dataset (`SYNPA-00001` to `SYNPA-09000`) preserved in `data/raw/synthetic/`.
- **The 31-Record Difference Explained:** 10,520 expected rows minus 10,489 deduplicated rows = exactly 31 records accounted for by 29 near-duplicate tactic expansion templates pruned from 82 to 53, plus 2 unannotated placeholder template rows removed from 52 to 50 in `gold_pa_anchors`.
- Zero runtime paths pointing to external workspaces.

### 2. Streamlit 7-Page Multi-Page Application
A focused, distraction-free application styled in **pure black (`#000000`)**, dark charcoal panels (`#111111`), white text, and restrained red accents (`#E53E3E`):
- **🏠 Home (`streamlit_app.py`)**: System architecture overview, class taxonomies, live model status.
- **💬 Analyze Comment (`pages/1_analyze_comment.py`)**: Single comment analysis with dual-model verdicts, phrase highlighting, sensitivity selection, and instant rewrite action.
- **🧵 Analyze Conversation (`pages/2_analyze_conversation.py`)**: Multi-turn conversation thread inspector evaluating target replies in preceding context.
- **✏️ Rewrite Comment (`pages/3_rewrite_comment.py`)**: Constructive rephrasing assistant offering multiple direct alternatives with 1-click markdown copy.
- **📋 History (`pages/4_history.py`)**: SQLite-persisted analysis log with label filters, keyword search, details modal, and CSV export.
- **📊 Insights (`pages/5_insights.py`)**: Real offline test evaluation metrics, confusion matrices, dataset provenance charts, and error breakdowns.
- **ℹ️ About (`pages/6_about.py`)**: Research background, communication taxonomy, ethical disclaimers, and deployment specifications.

### 3. Dual-Model Architecture & Disagreement Handling
- **Model 1 (Supervised ML)**: Sub-millisecond Logistic Regression pipeline with word (1-3) and character (3-5) n-grams.
- **Model 2 (Rule-Based Engine)**: 10 rule families covering:
  1. Fake politeness
  2. Indirect criticism
  3. Backhanded compliments
  4. Implied blame
  5. Sarcastic agreement
  6. Dismissiveness
  7. Guilt-tripping / martyrdom
  8. Contextual escalation from parent comments
  9. Benign overrides (e.g., *"Per your request"* is never flagged)
  10. Quotation, negation, and figurative-language handling
- **3 Operating Sensitivity Modes**:
  - `balanced`: Default operating threshold balancing precision and recall.
  - `strict` (`high_recall`): Lower activation threshold for heightened vigilance.
  - `lenient` (`high_precision`): High-confidence requirements to minimize false alarms.
- **Transparent Disagreement**: If Model 1 and Model 2 diverge, neither model silently overrides the other; both predictions remain visible, and the system flags `needs_review=True`.

### 4. Independent Threat Detection
- Independent 4-class classification (`no_threat`, `direct_threat`, `implied_threat`, `ambiguous`).
- Direct physical harm threats (*"track you down"*, *"break your legs"*) are caught reliably.
- Harmless hyperbole, figurative gaming banter, and competitive idioms (*"killing it on stage"*, *"destroy them in chess"*) are explicitly suppressed.

### 5. Constructive Rewriting Assistant
- Context-aware rephrasings that remove hostility while strictly preserving underlying concerns.
- **Zero Hallucination Guarantee**: Never invents facts, injects unstated accusations, or adds new commitments.
- **Safety Refusal**: Threatening language is never rewritten into polite requests; safety warnings are returned instead.
- Disclaimers clearly state rewrites are recommendations, and the original message can always be kept.

---

## 🏷️ Classification Taxonomy

### Primary Passive-Aggression Classes (3-Class)
- `passive_aggressive`: Indirect hostility, faux-politeness, backhanded compliments, sarcastic agreement, guilt-tripping, or implied blame.
- `non_passive_aggressive`: Sincere praise, constructive direct feedback, neutral information, or non-covert communication.
- `ambiguous`: Terse or dual-meaning reactions (*"Fine.", "Sure.", "Whatever."*) requiring human context.

### Auxiliary Threat Classes (4-Class)
- `no_threat`: Non-threatening social discourse, idioms, and figurative hyperbole.
- `direct_threat`: Explicit physical violence, tracking, or destruction threats.
- `implied_threat`: Indirect conditional harm or surveillance intimidation (*"Would be a shame if..."*).
- `ambiguous`: Dual-meaning statements requiring manual safety review.

---

## 📊 Empirical Evaluation Summary

Evaluated on held-out test split `data/processed/pa_test.csv` (1,382 conversation-grouped samples):

| Architecture Variant | Accuracy | Macro F1 | Weighted F1 | Latency / Sample | Operational Role |
|---|---:|---:|---:|---:|---|
| **Model 3: Hybrid Engine** | **97.68%** | **0.9750** | **0.9771** | 12.12 ms | **Active Production Layer** (Principled Evidence Fusion) |
| **Model 1: Supervised ML** | **98.70%** | **0.9872** | **0.9870** | 5.47 ms | Statistical Syntactic Generalizer (TF-IDF + LogReg) |
| **Model 2: Rule Engine** | 40.96% | 0.2662 | 0.2927 | 0.20 ms | High-Precision Span Offsets (10 Rule Families) |
| **Legacy LinearSVM** | 25.25% | 0.1344 | 0.1018 | 0.07 ms | Archived Baseline |

*Threat Classification (Independent 4-Class):* **99.1% accuracy**, **0.963 Macro F1** with figurative hyperbole filtering.

---

## ⚙️ Installation & Quickstart

### 1. Clone & Set Up Environment
```powershell
# Open project directory
cd "D:\PassiveAggression Detector"

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Application
```powershell
streamlit run streamlit_app.py
```
The browser will automatically open at `http://localhost:8501`.

### 3. Run Automated Regression Tests
```powershell
python -m pytest tests/ backend/tests/
```
*Current test suite result: 66 passed, 0 failed.*

---

## 📁 Repository Directory Structure

```
├── .streamlit/
│   └── config.toml                # Pure black theme & server settings
├── backend/
│   ├── pa_detector.db             # SQLite analysis history database
│   └── tests/                     # Preserved legacy backend test suite
├── data/
│   ├── manifests/                 # dataset_manifest.json, source_checksums.csv, label_mapping.csv
│   ├── processed/                 # Leakage-free 70/15/15 splits (pa_train, pa_val, pa_test)
│   └── raw/
│       ├── existing_sources/      # 10 dedicated local source dataset subdirectories
│       └── synthetic/             # 4 synthetic batches (9,000 SYNPA) + merged files
├── inference/
│   ├── history_db.py              # SQLite CRUD, stats, and search operations
│   ├── pa_inference.py            # Dual-model inference orchestrator
│   ├── preprocess.py              # English-only script validation & normalization
│   ├── rewrite_engine.py          # Constructive meaning-preserving rewrite engine
│   └── rule_detector.py           # 10 rule families, 3 sensitivities, character offsets
├── models/
│   ├── pa_classifier_v2.pkl       # Trained Model 1 (TF-IDF + Logistic Regression)
│   ├── threat_classifier_v1.pkl   # Trained Threat Classifier
│   └── pa_classifier.pkl          # Preserved legacy LinearSVM artifact
├── pages/
│   ├── 1_analyze_comment.py       # Single comment analysis & phrase highlighting
│   ├── 2_analyze_conversation.py  # Multi-message thread viewer
│   ├── 3_rewrite_comment.py       # Constructive rewrite assistant
│   ├── 4_history.py               # SQLite analysis history & export
│   ├── 5_insights.py              # Empirical evaluation & dataset analytics
│   └── 6_about.py                 # System background, taxonomy, ethics
├── reports/                       # Verified audit reports, checklists, and evaluations
├── tests/
│   ├── test_master_suite.py       # Phase 10 master integration & regression suite
│   ├── test_rewrite_engine.py     # Rewrite assistant tests (meaning preservation, safety)
│   └── test_v2_inference.py       # Dual-model inference tests
├── README.md                      # Complete system documentation
├── requirements.txt               # Locked production dependencies
└── streamlit_app.py               # Application entrypoint (Home page)
```

---

## ⚠️ Ethical & Usage Notice

- **Assistance, Not Censorship:** This tool is intended for communication analysis, conflict de-escalation, and writing assistance. It should never be used as an automated blocking or punitive tool without human discretion.
- **Ambiguity by Design:** The system recognizes that short social media comments (*"Fine."*, *"Sure."*) are inherently ambiguous. Flagging ambiguity is considered a strength, not an error.
- **No Hallucinated Rewrites:** The rewrite assistant strictly reframes language without adding new facts or commitments.
