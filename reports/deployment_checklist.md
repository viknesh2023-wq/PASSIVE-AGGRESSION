# PassiveAggression Detector 2.0 — Production Deployment Checklist

**Deployment Target:** Streamlit Standalone (Single Active Frontend)  
**Workspace:** Independent Project Root (`D:\PassiveAggression Detector`)  
**Audit Date:** October 2026  
**Status:** ✅ **READY FOR DEPLOYMENT**

---

## 1. Data Independence & Self-Contained Isolation

| Requirement | Target Path | Verified Status | Details |
|---|---|:---:|---|
| Raw Source 1 | `data/raw/existing_sources/tweeteval_sentiment/` | ✅ PASS | 3,500 records verified |
| Raw Source 2 | `data/raw/existing_sources/sarc_reddit_regular/` | ✅ PASS | 2,000 records verified |
| Raw Source 3 | `data/raw/existing_sources/sarc_reddit_sarcasm/` | ✅ PASS | 2,000 records verified |
| Raw Source 4 | `data/raw/existing_sources/goemotions/` | ✅ PASS | 1,000 records verified |
| Raw Source 5 | `data/raw/existing_sources/goemotions_anger_hostility/` | ✅ PASS | 883 records verified |
| Raw Source 6 | `data/raw/existing_sources/genz_sarc_banter/` | ✅ PASS | 800 records verified |
| Raw Source 7 | `data/raw/existing_sources/authentic_pa_mined/` | ✅ PASS | 143 records verified |
| Raw Source 8 | `data/raw/existing_sources/curated_pa_tactics/` | ✅ PASS | 53 records verified (29 near-duplicates pruned) |
| Raw Source 9 | `data/raw/existing_sources/contrastive_hard_negatives/` | ✅ PASS | 60 records verified |
| Raw Source 10 | `data/raw/existing_sources/gold_pa_anchors/` | ✅ PASS | 50 records verified (2 placeholders pruned) |
| Synthetic Batches | `data/raw/synthetic/` | ✅ PASS | 9,000 records across 4 batches + merged |
| Manifests & Checksums | `data/manifests/` | ✅ PASS | SHA-256 checksums, label mapping, & manifests |
| Processed Splits | `data/processed/` | ✅ PASS | 70/15/15 conversation-grouped splits (0 leakage) |
| External Paths | Runtime Dependency | ✅ PASS | Zero runtime references to `D:\NLP_PROJECT` |

---

## 2. Model Artifacts & Performance

| Artifact | Location | Size | Classes | Test Accuracy | Macro F1 | Status |
|---|---|---:|:---:|---:|---:|:---:|
| **Model 1 (Supervised ML)** | `models/pa_classifier_v2.pkl` | 1.93 MB | 3 | 98.7% | 0.987 | ✅ Active |
| **Model 2 (Rule Detector)** | `inference/rule_detector.py` | 27 KB (code) | 3 | 41.0% | 0.266 | ✅ Active (High-Precision Spans) |
| **Model 3 (Hybrid Engine)** | `inference/pa_inference.py` | Core Layer | 3 | **97.7%** | **0.975** | ✅ **Active Production Layer** |
| **Threat Classifier** | `models/threat_classifier_v1.pkl` | 1.25 MB | 4 | 99.1% | 0.963 | ✅ Active |
| **Legacy Linear SVM** | `models/pa_classifier.pkl` | 7.87 MB | 3 | 25.3% | 0.134 | ✅ Archived / Comparable |

---

## 3. Streamlit Application & Navigation (7 Pages)

| Page Number | File Path | Route Name | Role | Theme Verified |
|:---:|---|---|---|:---:|
| 1 | `streamlit_app.py` | Home | Overview, system stats, quick links | Pure Black `#000000` |
| 2 | `pages/1_analyze_comment.py` | Analyze Comment | Single comment analysis with dual models & rewrite preview | Pure Black `#000000` |
| 3 | `pages/2_analyze_conversation.py` | Analyze Conversation | Multi-turn thread analysis & contextual escalation | Pure Black `#000000` |
| 4 | `pages/3_rewrite_comment.py` | Rewrite Comment | Constructive rewriting assistant with 1-click copy | Pure Black `#000000` |
| 5 | `pages/4_history.py` | History | SQLite query, filtering, deletion, & export | Pure Black `#000000` |
| 6 | `pages/5_insights.py` | Insights | Real offline metrics, dataset distributions, error analysis | Pure Black `#000000` |
| 7 | `pages/6_about.py` | About | System mission, taxonomy, disclaimers, & ethics | Pure Black `#000000` |

---

## 4. Software Architecture & Security

- **Frontend Isolation:** Streamlit is the sole frontend framework. No React, Vite, or Node server required.
- **Pure Black Theme:** Configured in `.streamlit/config.toml` (`#000000` background, `#111111` secondary, `#E53E3E` red accent, `#FFFFFF` text).
- **SQLite Database:** Preserved at `backend/pa_detector.db`, initialized and accessed via `inference/history_db.py`.
- **Secret Protection:** `.env` and `.env.local` are excluded via `.gitignore`. No private keys or tokens logged.
- **English-Only Filter:** Latin-character script validation rejects non-English input gracefully without crash.

---

## 5. Verification Commands

### 1. Run Complete Pytest Suite
```powershell
python -m pytest tests/ backend/tests/
```
*Expected Result:* 66 passed, 0 failed.

### 2. Launch Streamlit Application
```powershell
streamlit run streamlit_app.py
```
*Expected Result:* Application launches at `http://localhost:8501` with dark styling and all 7 pages visible in sidebar.
