# Model v2 Evaluation Report

## Overview

| Component | Algorithm | Labels | Train | Val | Test |
|---|---|---|---|---|---|
| PA Classifier | TF-IDF (word 1-3, char 3-5) + LogisticRegression | 3 | 6298 | 1344 | 1358 |
| Threat Classifier | TF-IDF (word 1-3) + LogisticRegression (balanced) | 4 | 6298 | 1344 | 1358 |

Data source: `data\synthetic\pa_synthetic_merged.csv` (9000 records, synthetic)  
Split strategy: Conversation-grouped GroupShuffleSplit (70/15/15)

> **Important:** All evaluations are on synthetic held-out data.
> Real-world performance on genuine social-media content will differ
> and requires validation with human-annotated data.

---

## PA Classifier — Test Results

| Metric | Value |
|---|---|
| Accuracy | 0.9919 |
| Macro F1 | 0.9921 |
| Avg Latency | 0.757 ms/sample |
| Test N | 1358 |

### Per-Class Performance (Test)
- **passive_aggressive**: precision=0.991  recall=0.994  f1=0.993  support=348.0
- **non_passive_aggressive**: precision=0.996  recall=0.992  f1=0.994  support=484.0
- **ambiguous**: precision=0.989  recall=0.990  f1=0.990  support=526.0

### Confusion Matrix (Test)
Labels: ['passive_aggressive', 'non_passive_aggressive', 'ambiguous']

```
[
  [
    521,
    2,
    3
  ],
  [
    4,
    480,
    0
  ],
  [
    2,
    0,
    346
  ]
]
```

---

## Threat Classifier — Test Results

| Metric | Value |
|---|---|
| Accuracy | 0.5965 |
| Macro F1 | 0.2673 |
| Avg Latency | 0.158 ms/sample |
| Test N | 1358 |

### Per-Class Performance (Test)
- **no_threat**: precision=0.039  recall=0.190  f1=0.065  support=42.0
- **direct_threat**: precision=0.106  recall=0.390  f1=0.167  support=41.0
- **implied_threat**: precision=0.055  recall=0.220  f1=0.088  support=41.0
- **ambiguous**: precision=0.926  recall=0.630  f1=0.750  support=1234.0

### Confusion Matrix (Test)
Labels: ['no_threat', 'direct_threat', 'implied_threat', 'ambiguous']

```
[
  [
    777,
    127,
    148,
    182
  ],
  [
    16,
    16,
    4,
    5
  ],
  [
    20,
    3,
    9,
    9
  ],
  [
    26,
    5,
    3,
    8
  ]
]
```

---

## Data & Label Notes

- Dataset: 9000 synthetic records (`data\synthetic\pa_synthetic_merged.csv`)
- PA labels: `passive_aggressive=2382`  `non_passive_aggressive=2357`  `ambiguous=1559`
- Threat labels: `no_threat` dominates (~91%) — threat classifier uses balanced class weights
- Split strategy: conversation-grouped to prevent context leakage across splits

## Limitations

1. All training and evaluation data is synthetic — may not reflect real-world social media PA distribution
2. Template-based generation creates some lexical overlap between classes
3. The threat classifier has very limited positive-class support; real-world application requires human-curated threat examples
4. The `ambiguous` PA class has fewer templates and may underperform on genuinely ambiguous messages
5. Model performance on code-switched, very short, or image-caption texts is untested
