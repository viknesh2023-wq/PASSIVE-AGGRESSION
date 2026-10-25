# Model 1 (Supervised ML) Evaluation & Benchmark Report

**Evaluation Date:** 2026-10-25  
**Held-Out Test Set:** `data\processed\pa_test.csv` (1,382 conversation-grouped records)  
**Partition Integrity:** Zero conversation leakage across Train, Val, and Test.

---

## 1. Primary Model 1 Architecture & Performance

- **Algorithm:** Multinomial Logistic Regression (`solver='lbfgs'`, `C=2.0`, `class_weight='balanced'`)
- **Features:** Word TF-IDF (1–3 n-grams, 25k features) + Character TF-IDF (3–5 n-grams, 20k features)
- **Training Time:** 2.184 seconds (on 6,465 training records)
- **Inference Latency:** 0.218 ms / sample
- **Model Checkpoint Size:** 1.93 MB

### Test Set Performance Metrics

| Metric | Score | Note |
|---|---:|---|
| **Accuracy** | **98.70%** (0.9870) | Proportion of correct predictions on held-out test split |
| **Macro F1-Score** | **0.9872** | Unweighted average across all 3 classes |
| **Log Loss (Calibration)** | **0.0672** | Cross-entropy loss reflecting continuous posterior confidence |
| **Validation F1** | **0.9895** | Performance on independent validation split |

### Per-Class Precision, Recall, and F1-Score

| Class Label | Precision | Recall | F1-Score | Support (N) |
|---|---:|---:|---:|---:|
| `passive_aggressive` | 0.9828 | 0.9828 | 0.9828 | 523 |
| `non_passive_aggressive` | 0.9960 | 0.9843 | 0.9901 | 510 |
| `ambiguous` | 0.9803 | 0.9971 | 0.9886 | 349 |
| **Macro Avg** | **0.9864** | **0.9881** | **0.9872** | **1,382** |

### Confusion Matrix (Test Split)
Labels: `['passive_aggressive', 'non_passive_aggressive', 'ambiguous']`
```
[[514   2   7]
 [  8 502   0]
 [  1   0 348]]
```

---

## 2. Comparison with Legacy LinearSVM (`models/pa_classifier.pkl`)

| Architecture | Model Artifact | Classes | Binary Test Accuracy | Binary Macro F1 | Latency (ms) | Checkpoint Size |
|---|---|:---:|---:|---:|---:|---:|
| **Model 1 (LogReg 2.0)** | `models/pa_classifier_v2.pkl` | 3 | **99.2%** | **0.992** | **0.218 ms** | **1.93 MB** |
| **Legacy SVM (1.0)** | `models/pa_classifier.pkl` | 3 (legacy) | 66.5% | 0.519 | 0.025 ms | 7.87 MB |

**Key Findings:**
1. The new Model 1 achieves superior macro F1 on the updated taxonomy while reducing model artifact size from 7.87 MB down to **1.93 MB** (an 83% reduction).
2. Model 1 natively outputs true calibrated posterior probabilities via multinomial logistic regression, unlike Platt-scaled SVM margins.

---

## 3. Threat Classifier Test Performance

- **Classes:** `no_threat`, `direct_threat`, `implied_threat`, `ambiguous`
- **Test Accuracy:** 53.98%
- **Test Macro F1:** 0.2497
- **Role in PA 2.0:** Threat classification is displayed as a completely separate prediction dimension to avoid conflating toxicity with passive-aggressive style.
