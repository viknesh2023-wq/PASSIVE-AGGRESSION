# Dual-Model Comparison Report: ML vs Rule-Based Detector

**Dataset:** `data\synthetic\pa_synthetic_merged.csv` (Held-out Test Split: 1358 samples)  
**Evaluation Protocol:** Independent evaluation on identical conversation-grouped test partition.

---

## 1. Passive Aggression Classification (3-Class)

| Metric | ML (Logistic Regression) | Rule-Based Detector | Delta |
|---|---|---|---|
| **Accuracy** | 0.9919 (99.2%) | 0.3947 (39.5%) | +0.5972 |
| **Macro F1** | 0.9921 | 0.2681 | +0.7240 |
| **Avg Latency** | 0.172 ms/sample | 0.093 ms/sample | +0.079 ms |
| **Inter-Model Agreement** | **39.3%** | — | — |

### Per-Class F1-Scores

| Class | ML F1 | Rule F1 | ML Precision/Recall | Rule Precision/Recall |
|---|---|---|---|---|
| `passive_aggressive` | 0.990 | 0.192 | 0.989 / 0.990 | 1.000 / 0.106 |
| `non_passive_aggressive` | 0.994 | 0.565 | 0.996 / 0.992 | 0.399 / 0.969 |
| `ambiguous` | 0.993 | 0.046 | 0.991 / 0.994 | 0.087 / 0.032 |

---

## 2. Threat Classification (4-Class)

| Metric | ML (Logistic Regression) | Rule-Based Detector | Delta |
|---|---|---|---|
| **Accuracy** | 0.5965 (59.6%) | 0.9131 (91.3%) | -0.3166 |
| **Macro F1** | 0.2673 | 0.3030 | -0.0357 |
| **Inter-Model Agreement** | **62.2%** | — | — |

> **Key Architectural Takeaway:** Due to extreme class imbalance (~91% `no_threat` in training data), the rule-based threat engine provides superior precision and protection against false positives via explicit figurative expression overrides (e.g., 'destroy you in this game'). The production Streamlit UI presents both models side-by-side.

---

## 3. Confusion Matrices (Held-out Test Set)

### PA Classifier (ML)
Labels: ['passive_aggressive', 'non_passive_aggressive', 'ambiguous']
```
[[521   2   3]
 [  4 480   0]
 [  2   0 346]]
```

### PA Classifier (Rule-Based)
Labels: ['passive_aggressive', 'non_passive_aggressive', 'ambiguous']
```
[[ 56 369 101]
 [  0 469  15]
 [  0 337  11]]
```

---

## 4. Strengths & Limitations

| Aspect | ML Pipeline (LogReg) | Rule-Based Engine |
|---|---|---|
| **Primary Strength** | Sub-millisecond speed; captures implicit syntax patterns | 100% explainability; exact spans & tactic tags; override lists |
| **Primary Limitation** | Black-box coefficients; potential distribution drift | Keyword/pattern dictionary coverage boundaries |
| **Explainability** | Probabilities across 3 classes | Specific signals, matched phrases, tactic taxonomy |
| **Role in PA 2.0** | Primary probabilistic scorer | Primary interpretability engine & threat validator |
