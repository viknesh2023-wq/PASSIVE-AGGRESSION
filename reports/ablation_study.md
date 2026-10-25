# Feature and Model Ablation Study

**Audit Date:** 2026-10-14T18:03:21.191598+00:00  
**Task:** 3-Class Passive Aggression Detection (Held-out Test Split)

---

## 1. Feature Representation Ablations

| Configuration | Macro F1 | Delta vs Full |
|---|---|---|
| Unigrams only (1, 1) | 0.7680 | +0.7680 |
| Full Pipeline (Unigrams + Bigrams + Sublinear TF) | 0.8287 | +0.0000 |
| Without Sublinear TF Scaling | 0.8227 | -0.0060 |

---

## 2. Key Insights

1. **Bigrams are Critical:** Incorporating bigrams provides a measurable boost because passive-aggressive cues often rely on multi-word constructs (e.g., *"as per"*, *"no worries"*, *"fine."*, *"my last"*).
2. **Sublinear TF Scaling:** Dampens the influence of repeated words in longer messages, preventing false positives on generic workplace phrases.
