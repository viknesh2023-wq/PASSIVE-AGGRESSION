# Initial Project and Environment Audit Report

**Audit Date:** 2026-10-14  
**Project:** PassiveAggression Detector  
**Scope:** `D:\PassiveAggression Detector`, `D:\python_folder`, and `D:\NLP_PROJECT`

---

## 1. Hardware and Software Environment

| Component | Specification |
|---|---|
| **Operating System** | Windows 11 (build 10.0.26200) |
| **GPU** | NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM) |
| **Base Python** | Python 3.12.10 (64-bit AMD64) |
| **Accelerated Python** | `D:\NLP_PROJECT\internvl_env\Scripts\python.exe` (PyTorch 2.1.2 + CUDA 12.1) |
| **Storage** | D: Drive: 161.0 GB Total, ~73.0 GB Free |
| **Installed Libraries** | `fastapi`, `uvicorn`, `scikit-learn`, `transformers`, `torch`, `pandas`, `numpy`, `lightgbm`, `pytest`, `httpx` |

---

## 2. Workspace & Directory Audit

- **`d:\python_folder` (Active Workspace):** Contains the operational React 18 frontend, FastAPI backend, ML models, test suite, and evaluation reports.
- **`D:\PassiveAggression Detector`:** Existing secondary project folder structure; synced with identical code, schemas, and reports.
- **`D:\NLP_PROJECT` (Source Repository — Read-Only):** Audited read-only. Contains over 1,000,000 sarcasm rows, 132,000 sentiment rows, 54,000 GoEmotions rows, and pretrained XLM-RoBERTa model checkpoints (`xlmr_text_sarcasm_full`, `xlmr_emotion_model`, `xlmr_sentiment_model`).

---

## 3. Data Ingestion & Audit Deliverables Created

1. [`reports/all_dataset_files.csv`](file:///d:/python_folder/reports/all_dataset_files.csv) — Catalog of 208 dataset files across both drives.
2. [`reports/dataset_label_audit.csv`](file:///d:/python_folder/reports/dataset_label_audit.csv) — Audits task definitions and label mappings.
3. [`reports/reusable_model_inventory.csv`](file:///d:/python_folder/reports/reusable_model_inventory.csv) — Detailed inventory of available pretrained transformers and scikit-learn models.
4. [`reports/initial_436_row_audit.md`](file:///d:/python_folder/reports/initial_436_row_audit.md) — Root-cause explanation of the previous 436-row bottleneck.
5. [`reports/frontend_audit.md`](file:///d:/python_folder/reports/frontend_audit.md) — Frontend component matrix and API contract conformance.
