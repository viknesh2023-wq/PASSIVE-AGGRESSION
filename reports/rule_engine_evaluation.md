# Model 2 (Rule-Based Engine) Independent Evaluation Report

**Evaluation Date:** 2026-10-25  
**Test Set:** `data/processed/pa_test.csv` (1,382 held-out records)  
**Rule Families Evaluated:** 10 linguistic families, 3 sensitivity thresholds.

---

## 1. Sensitivity Mode Comparison

| Sensitivity Mode | Accuracy | Macro F1 | PA Precision | PA Recall | PA F1 | Ambiguous Recall |
|---|---:|---:|---:|---:|---:|---:|
| `balanced` | 40.96% | 0.2662 | 1.000 | 0.107 | 0.193 | 0.020 |
| `high_recall` | 44.86% | 0.3249 | 0.923 | 0.207 | 0.338 | 0.034 |
| `high_precision` | 36.83% | 0.1979 | 0.000 | 0.000 | 0.000 | 0.017 |

### Key Findings:
- **`balanced` mode:** Provides calibrated evidence thresholding (30/50 raw score).
- **`high_recall` mode:** Lowers triggering thresholds to detect subtle indirect cues, prioritizing sensitivity.
- **`high_precision` mode:** Enforces multi-signal confirmation (threshold 65 + 2 signals), reducing false alarms.

---

## 2. Confusion Matrix (`balanced` mode)
Labels: `['passive_aggressive', 'non_passive_aggressive', 'ambiguous']`
```
[[ 56 403  64]
 [  0 503   7]
 [  0 342   7]]
```
