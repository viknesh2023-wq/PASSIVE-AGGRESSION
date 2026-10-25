# Controlled Experiments & Multi-Model Comparison Report

**Evaluated on:** Held-out test set (`data/processed/pa_test.csv`)
**Test Sample Count:** 1382 conversation-isolated examples
**Rule vs ML Disagreement Rate:** 59.55%

---

## 1. Aggregate Model Performance Comparison

| Architecture Variant | Accuracy | Macro F1 | Weighted F1 | Latency / Sample | Operational Role |
|---|---:|---:|---:|---:|---|
| **Legacy LinearSVM** | 25.25% | 0.1344 | 0.1018 | 0.07 ms | Archived Legacy Baseline |
| **Model 2: Rule Engine** | 40.96% | 0.2662 | 0.2927 | 0.20 ms | High-Precision Span Offsets |
| **Model 1: Supervised ML** | 98.70% | 0.9872 | 0.9870 | 5.47 ms | Statistical N-Gram Generalizer |
| **Model 3: Hybrid Engine** | **97.68%** | **0.9750** | **0.9771** | 12.12 ms | **Production Fused Decision Layer** |

---

## 2. Per-Class F1 Score Breakdown

| Architecture Variant | Passive-Aggressive | Non-Passive-Aggressive | Ambiguous |
|---|---:|---:|---:|
| **Legacy LinearSVM (3-class legacy)** | 0.0000 | 0.0000 | 0.4032 |
| **Model 2: Rule-Based Engine (10 Families)** | 0.1934 | 0.5722 | 0.0328 |
| **Model 1: Supervised ML (TF-IDF + LogReg)** | 0.9828 | 0.9901 | 0.9886 |
| **Model 3: Hybrid Decision Engine (ML + Rules)** | 0.9835 | 0.9841 | 0.9575 |

---

## 3. Hybrid Fusion Strategy Breakdown

How the Hybrid Intelligence Decision Layer resolved classifications on held-out test data:

| Fusion Policy Strategy | Trigger Count | Description |
|---|---:|---|
| `ml_statistical_generalization` | 450 (32.6%) | Specific arbitration mechanism |
| `concordant_consensus` | 552 (39.9%) | Specific arbitration mechanism |
| `conflict_uncertainty_arbitration` | 373 (27.0%) | Specific arbitration mechanism |
| `benign_override_shield` | 7 (0.5%) | Specific arbitration mechanism |

---

## 4. Performance by Dataset Provenance

| Dataset Provenance | Samples | Model 1 (ML) Acc | Model 2 (Rule) Acc | Hybrid Acc |
|---|---:|---:|---:|---:|
| **synthetic** | 1342 | 99.0% | 40.2% | **98.2%** |
| **gold_anchor** | 9 | 66.7% | 33.3% | **33.3%** |
| **authentic_mined** | 19 | 94.7% | 63.2% | **89.5%** |
| **contrastive_hard_negative** | 12 | 100.0% | 100.0% | **100.0%** |
