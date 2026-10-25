# Phase 0 Implementation Plan: PassiveAggression Detector 2.0

## Master Execution Roadmap

### Phase 0: Project Discovery & Safe Workspace Setup [COMPLETED]
- [x] Inspect active workspace directory structure.
- [x] Create safe backup directory `backups/phase0_backup/` containing models, DB, and data.
- [x] Generate `reports/backup_manifest.json` with file sizes and SHA-256 checksums.
- [x] Create `reports/implementation_audit.md` documenting discovered assets and the 10,520 vs 10,489 discrepancy.
- [x] Create `reports/implementation_plan.md`.

### Phase 1: Make Datasets Fully Self-Contained
- [ ] Create directory structure: `data/raw/existing_sources/` (10 folders), `data/manifests/`, `data/interim/`, `data/processed/`, `data/audit/`.
- [ ] Perform verified one-time copy of each of the 10 source datasets into project-local folders.
- [ ] Generate working copies in CSV/JSONL format.
- [ ] Audit the 9,000-record synthetic dataset across all 19 canonical columns.
- [ ] Calculate SHA-256 checksums and build `data/manifests/source_checksums.csv` and `dataset_manifest.csv` / `.json`.

### Phase 2: Build Data Audit & Label Mapping Pipeline
- [ ] Implement `training/audit_datasets.py` to audit distributions, duplicates, and schemas.
- [ ] Implement `training/prepare_datasets.py` to map primary PA labels (`passive_aggressive`, `non_passive_aggressive`, `ambiguous`) and auxiliary tasks (`threat`, `sarcasm`, `sentiment`).
- [ ] Generate audit reports: `reports/dataset_audit.md` and `reports/label_mapping.md`.

### Phase 3: Create Leakage-Resistant Data Splits
- [ ] Implement group-aware splitting using `conversation_id` / thread grouping (70% train / 15% val / 15% test).
- [ ] Save splits: `data/processed/pa_train.csv`, `pa_validation.csv`, `pa_test.csv`, `threat_train.csv`, `threat_validation.csv`, `threat_test.csv`.
- [ ] Ensure strict segregation of gold reference anchors and contrastive pairs.
- [ ] Document split integrity in `reports/split_integrity_report.md`.

### Phase 4: Model 1 — Supervised Machine Learning
- [ ] Fit TF-IDF (word 1–3 + char 3–5) exclusively on training split.
- [ ] Train Logistic Regression classifier with balanced class weighting.
- [ ] Compare Logistic Regression with legacy LinearSVM (`models/pa_classifier.pkl`) on the same held-out test split.
- [ ] Evaluate probability calibration, accuracy, macro F1, confusion matrices, and latency.
- [ ] Save updated pipeline artifact: `models/pa_classifier_v2.pkl`.
- [ ] Document results in `reports/model_evaluation.md`.

### Phase 5: Model 2 — Explainable Rule-Based Detector
- [ ] Enhance `inference/rule_detector.py` with all 10 rule families:
  1. Fake politeness
  2. Indirect criticism
  3. Backhanded compliments
  4. Implied blame
  5. Sarcastic agreement
  6. Dismissiveness
  7. Other / unclear tactic signals
  8. Contextual escalation / repeated dismissiveness
  9. Benign overrides
  10. Quotation, negation, and figurative-language handling
- [ ] Support sensitivity modes: `balanced`, `high_recall`, `high_precision`.
- [ ] Output structured result with exact start/end character offsets, rule IDs, and rationale.
- [ ] Independently evaluate rule detector on held-out test split.

### Phase 6: Model Integration & Shared Orchestrator
- [ ] Update `inference/pa_inference.py` to orchestrate Model 1 and Model 2 side-by-side.
- [ ] Handle model agreement, divergence, and uncertainty flags.
- [ ] Ensure no blind score averaging or silent overrides.

### Phase 7: Independent Threat Detection
- [ ] Evaluate threat classifier (`models/threat_classifier_v1.pkl`) and rule-based threat engine.
- [ ] Implement unit tests for direct threats, implied threats, figurative expressions, idioms, and negated threats.

### Phase 8: Restore & Modernize Rewriting Assistant
- [ ] Create `inference/rewrite_engine.py` using curated contextual substitutions.
- [ ] Ensure preservation of core concern, removal of indirect hostility, and avoidance of hallucinated facts.
- [ ] Add unit tests for meaning preservation, respectful tone, and ambiguity handling.

### Phase 9: Streamlit Application (7 Complete Pages)
- [ ] Update page structure to 7 pages:
  1. `streamlit_app.py` (Home)
  2. `pages/1_analyze_comment.py` (Analyze Comment)
  3. `pages/2_analyze_conversation.py` (Analyze Conversation)
  4. `pages/3_rewrite_comment.py` (Rewrite Assistant)
  5. `pages/4_history.py` (History)
  6. `pages/5_insights.py` (Insights & Metrics)
  7. `pages/6_about.py` (About & Architecture)
- [ ] Maintain pure black visual style (`#000000`, charcoal `#111111`, red accent `#E53E3E`).

### Phase 10: Regression Testing & Verification
- [ ] Run complete test suite (`pytest tests/ backend/tests/`).
- [ ] Add tests for dataset splits, rewriting, rule engine, sensitivity modes, and threat filters.
- [ ] Ensure 100% test pass rate.

### Phase 11: Deployment Configuration & Documentation
- [ ] Update `README.md`, `requirements.txt`, `.streamlit/config.toml`, `.gitignore`.
- [ ] Generate `reports/deployment_checklist.md`.
- [ ] Verify local execution command: `streamlit run streamlit_app.py`.
