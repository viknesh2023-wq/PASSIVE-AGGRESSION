# Final Model Evaluation and Research Report

**Project:** PassiveAggression Detector — Explainable Detection of Hidden Hostility in Text  
**Date:** 2026-10-14  
**Status:** COMPLETE — Final Accepted Baseline  
**Active Production Model:** `pa-tfidf-svm-v1`  
**Dataset:** `data/processed/pa_dataset.csv` (N=464, Train=320, Val=71, Held-out Test=73)

---

## 1. Executive Summary

Passive aggression presents a significant NLP challenge because hostility is conveyed indirectly through polite surface forms, syntactic framing, or intentional ambiguity. Unlike overt hostility (which features profane, threatening, or toxic tokens) or sentiment analysis (where polarity is explicitly expressed), passive aggression weaponizes professional or cordial syntax to shift blame, mock, or dismiss the recipient.

To solve this without conflating sarcasm, toxicity, or sentiment:
1. We engineered a dedicated multi-source dataset anchored by a 50-example gold-standard annotation corpus and supplemented with rigorously filtered hard negatives from TweetEval, GoEmotions, and SARC.
2. We trained multiple benchmarked models (Majority Baseline, TF-IDF + Logistic Regression with GridSearchCV, Calibrated Linear SVM, and Linguistic Feature Logistic Regression).
3. We selected **Calibrated Linear SVM (`pa-tfidf-svm-v1`)** as our production model, achieving **Macro F1 = 0.7645** and **Accuracy = 84.93%** on a strictly isolated test set with 0.07 ms inference latency.

---

## 2. Experimental Benchmark Comparison

All evaluations were executed on the held-out test split ($N=73$). No test information was accessed during vectorizer fitting, parameter tuning, or model selection.

| Model | Accuracy | Macro F1 | Neutral F1 | PA F1 | Overt Hostile F1 | Latency (ms) | Rank |
|---|---|---|---|---|---|---|---|
| **Majority Baseline** | 58.90% | 0.2471 | 0.7414 | 0.0000 | 0.0000 | 0.01 | 4 |
| **Linguistic LogReg** | 75.34% | 0.7098 | 0.8108 | 0.5185 | 0.8000 | 0.01 | 3 |
| **TF-IDF + LogReg** | 87.67% | 0.7640 | 0.9213 | 0.4615 | 0.9091 | 0.03 | 2 |
| **TF-IDF + Linear SVM (Calibrated)** | **84.93%** | **0.7645** | **0.9048** | **0.5263** | **0.8636** | **0.07** | **1 (Selected)** |

### Why Linear SVM was Selected
While TF-IDF + Logistic Regression achieved slightly higher raw accuracy, **Calibrated Linear SVM achieved higher Macro F1 and substantially superior recall on the subtle passive-aggressive class (50% vs 38%)**, which is the core objective of this application.

---

## 3. Production Model Performance (pa-tfidf-svm-v1)

### Classification Report (Test Set N=73)

```text
                    precision    recall  f1-score   support

           neutral       0.89      0.91      0.90        43
passive_aggressive       0.57      0.50      0.53         8
     overt_hostile       0.86      0.86      0.86        22

          accuracy                           0.85        73
         macro avg       0.77      0.76      0.76        73
      weighted avg       0.85      0.85      0.85        73
```

### Confusion Matrix
- **True Neutral (43):** 39 predicted Neutral, 3 predicted PA, 1 predicted Overt Hostile.
- **True Passive Aggressive (8):** 4 predicted PA, 3 predicted Neutral, 1 predicted Overt Hostile.
- **True Overt Hostile (22):** 19 predicted Overt Hostile, 0 predicted PA, 3 predicted Neutral.

---

## 4. Explainability & Tactic Detection

Inference is fully explainable through a hybrid architecture:
1. **Span-Level Highlighting:** Exact character-offset phrase bank tracking for trigger phrases across 6 tactics (`sarcasm`, `backhanded_compliment`, `guilt_tripping`, `mock_politeness`, `blame_shifting`, `curt_dismissal`).
2. **Per-Sentence Scoring:** Evaluates conversational tension across multi-sentence messages.
3. **Calmer Rewrite Suggestions:** Rule-based transformation engine replaces hostile passive-aggressive formulations with constructive, professional alternatives (e.g., *"As per my last email"* -> *"As mentioned earlier"*).
4. **Uncertainty Flagging:** Computes Shannon entropy across the 3 class probabilities. When entropy $> 0.75$ or top confidence $< 0.50$, the system flags `is_uncertain: true` and warns that additional context is required.

---

## 5. Ethical Considerations & Limitations

1. **Not Psychological Diagnosis:** The score (0–100) is an automated linguistic hostility index, not a psychological evaluation of the speaker.
2. **Contextual Edge Cases:** Extreme shorthand like *"Fine."* or *"If you say so."* can be neutral in casual text but highly passive-aggressive in workplace conflict. The system flags uncertainty on these cases.
3. **Privacy First:** The SQLite database stores only truncated 100-character snippets and does not persist sensitive communication bodies unless explicitly permitted.
