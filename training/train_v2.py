"""
PassiveAggression Detector 2.0 — Training Pipeline v2
training/train_v2.py

Trains:
  1. PA Classifier (3-class): passive_aggressive / non_passive_aggressive / ambiguous
     Using TF-IDF word + char n-grams + Logistic Regression
  2. Threat Classifier (4-class): no_threat / direct_threat / implied_threat / ambiguous
     Using TF-IDF + Logistic Regression

Input:  data/synthetic/pa_synthetic_merged.csv
Output: models/pa_classifier_v2.pkl
        models/threat_classifier_v1.pkl
        reports/model_v2_evaluation.json
        reports/model_v2_evaluation.md
"""

from __future__ import annotations
import json
import logging
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("train_v2")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH    = Path("data/synthetic/pa_synthetic_merged.csv")
MODELS_DIR   = Path("models")
REPORTS_DIR  = Path("reports")
PA_MODEL     = MODELS_DIR / "pa_classifier_v2.pkl"
THREAT_MODEL = MODELS_DIR / "threat_classifier_v1.pkl"
EVAL_JSON    = REPORTS_DIR / "model_v2_evaluation.json"
EVAL_MD      = REPORTS_DIR / "model_v2_evaluation.md"

PA_LABELS     = ["passive_aggressive", "non_passive_aggressive", "ambiguous"]
THREAT_LABELS = ["no_threat", "direct_threat", "implied_threat", "ambiguous"]


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """Lightweight preprocessing for model input."""
    import re
    import unicodedata
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"https?://\S+", "[URL]", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def enrich_text(row: pd.Series) -> str:
    """Optionally prepend context to the target message."""
    text = preprocess_text(str(row["text"]))
    if pd.notna(row.get("parent_comment")) and str(row.get("parent_comment", "")).strip():
        parent = preprocess_text(str(row["parent_comment"]))
        text = f"[CONTEXT]: {parent} [REPLY]: {text}"
    return text


# ── Data Loading & Splitting ───────────────────────────────────────────────────
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    logger.info("Loaded %d records from %s", len(df), path)

    # Validate labels
    df = df[df["primary_label"].isin(PA_LABELS)].copy()
    df = df[df["threat_label"].isin(THREAT_LABELS)].copy()
    logger.info("After label filtering: %d records", len(df))

    # Create enriched text column
    df["text_enriched"] = df.apply(enrich_text, axis=1)

    # Ensure conversation_id exists for grouped split
    df["conversation_id"] = df["conversation_id"].fillna(df["id"])

    return df


def make_grouped_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Leakage-resistant, conversation-grouped split.
    Ensures no conversation appears in both train and test.
    """
    groups = df["conversation_id"].values
    labels = df["primary_label"].values

    # First: split off test set
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    try:
        train_val_idx, test_idx = next(gss_test.split(df, labels, groups))
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df      = df.iloc[test_idx].reset_index(drop=True)
    except Exception as e:
        logger.warning("GroupShuffleSplit failed (%s) — falling back to random split", e)
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, stratify=labels, random_state=random_state
        )

    # Second: split train into train + val
    groups_tv = train_val_df["conversation_id"].values
    labels_tv = train_val_df["primary_label"].values
    val_frac   = val_size / (1 - test_size)

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_state + 1)
    try:
        train_idx, val_idx = next(gss_val.split(train_val_df, labels_tv, groups_tv))
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df   = train_val_df.iloc[val_idx].reset_index(drop=True)
    except Exception as e:
        logger.warning("Val GroupShuffleSplit failed (%s) — falling back to random split", e)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_frac, stratify=labels_tv, random_state=random_state + 1
        )

    logger.info("Split: train=%d | val=%d | test=%d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


# ── Model Building ─────────────────────────────────────────────────────────────
def build_pa_pipeline(class_weights: dict | None = None) -> Pipeline:
    """Build TF-IDF (word + char) + LogReg pipeline for PA classification."""
    from sklearn.pipeline import FeatureUnion

    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        max_features=40_000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.95,
        max_features=20_000,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    feature_union = FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf),
    ])

    clf = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight=class_weights,
        n_jobs=-1,
        random_state=42,
    )

    return Pipeline([
        ("features", feature_union),
        ("clf", clf),
    ])


def build_threat_pipeline() -> Pipeline:
    """Separate pipeline for threat detection."""
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        max_features=30_000,
        sublinear_tf=True,
    )
    clf = LogisticRegression(
        C=2.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("tfidf", word_tfidf), ("clf", clf)])


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate_model(
    pipeline: Pipeline,
    X_val: list[str],
    X_test: list[str],
    y_val,
    y_test,
    label_names: list[str],
    model_name: str,
) -> dict:
    results = {}

    for split_name, X, y in [("validation", X_val, y_val), ("test", X_test, y_test)]:
        t0 = time.perf_counter()
        y_pred = pipeline.predict(X)
        latency_ms = ((time.perf_counter() - t0) * 1000) / max(1, len(X))

        acc    = accuracy_score(y, y_pred)
        macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
        report = classification_report(
            y, y_pred, target_names=label_names, zero_division=0, output_dict=True
        )
        cm = confusion_matrix(y, y_pred, labels=label_names).tolist()

        results[split_name] = {
            "accuracy":        round(acc, 4),
            "macro_f1":        round(macro_f1, 4),
            "classification_report": report,
            "confusion_matrix": cm,
            "avg_latency_ms":  round(latency_ms, 4),
            "n_samples":       len(X),
        }

        logger.info("[%s] %s — accuracy=%.4f  macro_f1=%.4f  latency=%.3fms/sample",
                    model_name, split_name, acc, macro_f1, latency_ms)

        # Per-class breakdown
        for lbl in label_names:
            if lbl in report:
                r = report[lbl]
                logger.info("  %-28s precision=%.3f  recall=%.3f  f1=%.3f  support=%d",
                             lbl, r.get("precision",0), r.get("recall",0),
                             r.get("f1-score",0), r.get("support",0))

    return results


# ── Main Training Loop ─────────────────────────────────────────────────────────
def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Conversation-grouped split
    train_df, val_df, test_df = make_grouped_split(df, test_size=0.15, val_size=0.15)

    X_train = train_df["text_enriched"].tolist()
    X_val   = val_df["text_enriched"].tolist()
    X_test  = test_df["text_enriched"].tolist()

    # ── PA Classifier ──────────────────────────────────────────────────────────
    logger.info("=== Training PA Classifier (3-class) ===")
    y_train_pa = train_df["primary_label"].tolist()
    y_val_pa   = val_df["primary_label"].tolist()
    y_test_pa  = test_df["primary_label"].tolist()

    dist = dict(Counter(y_train_pa))
    logger.info("Train PA distribution: %s", dist)

    # Compute class weights for imbalanced ambiguous class
    classes = list(set(y_train_pa))
    cw_arr  = compute_class_weight("balanced", classes=np.array(sorted(classes)), y=np.array(y_train_pa))
    class_weights_pa = dict(zip(sorted(classes), cw_arr))
    logger.info("Class weights (PA): %s", {k: round(v, 3) for k, v in class_weights_pa.items()})

    pa_pipeline = build_pa_pipeline(class_weights=class_weights_pa)

    t0 = time.perf_counter()
    pa_pipeline.fit(X_train, y_train_pa)
    train_time = time.perf_counter() - t0
    logger.info("PA model trained in %.2fs", train_time)

    pa_eval = evaluate_model(
        pa_pipeline, X_val, X_test, y_val_pa, y_test_pa, PA_LABELS, "PA_Classifier"
    )

    joblib.dump(pa_pipeline, PA_MODEL)
    logger.info("Saved PA model → %s (%.2f MB)", PA_MODEL, PA_MODEL.stat().st_size / 1e6)

    # ── Threat Classifier ──────────────────────────────────────────────────────
    logger.info("=== Training Threat Classifier (4-class) ===")
    y_train_t = train_df["threat_label"].tolist()
    y_val_t   = val_df["threat_label"].tolist()
    y_test_t  = test_df["threat_label"].tolist()

    threat_dist = dict(Counter(y_train_t))
    logger.info("Train threat distribution: %s", threat_dist)

    threat_pipeline = build_threat_pipeline()
    t0 = time.perf_counter()
    threat_pipeline.fit(X_train, y_train_t)
    logger.info("Threat model trained in %.2fs", time.perf_counter() - t0)

    threat_eval = evaluate_model(
        threat_pipeline, X_val, X_test, y_val_t, y_test_t, THREAT_LABELS, "Threat_Classifier"
    )

    joblib.dump(threat_pipeline, THREAT_MODEL)
    logger.info("Saved threat model → %s (%.2f MB)", THREAT_MODEL, THREAT_MODEL.stat().st_size / 1e6)

    # ── Save Evaluation Report ─────────────────────────────────────────────────
    report = {
        "pa_classifier": {
            "model_file":     str(PA_MODEL),
            "algorithm":      "TF-IDF (word 1-3gram + char 3-5gram) + LogisticRegression",
            "labels":         PA_LABELS,
            "train_samples":  len(X_train),
            "val_samples":    len(X_val),
            "test_samples":   len(X_test),
            "class_weights":  {k: round(v, 4) for k, v in class_weights_pa.items()},
            "evaluation":     pa_eval,
            "note": "Evaluated on synthetic held-out data. Real-world performance may differ.",
        },
        "threat_classifier": {
            "model_file":     str(THREAT_MODEL),
            "algorithm":      "TF-IDF (word 1-3gram) + LogisticRegression (class_weight=balanced)",
            "labels":         THREAT_LABELS,
            "train_samples":  len(X_train),
            "val_samples":    len(X_val),
            "test_samples":   len(X_test),
            "evaluation":     threat_eval,
            "note": "Threat labels are independent of PA labels. ~91% of data is no_threat.",
        },
        "data_source":        str(DATA_PATH),
        "total_records":      len(df),
        "split": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "split_strategy":     "conversation-grouped GroupShuffleSplit",
    }

    with open(EVAL_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Saved evaluation JSON → %s", EVAL_JSON)

    # ── Markdown Report ────────────────────────────────────────────────────────
    pa_test  = pa_eval.get("test", {})
    thr_test = threat_eval.get("test", {})

    md = f"""# Model v2 Evaluation Report

## Overview

| Component | Algorithm | Labels | Train | Val | Test |
|---|---|---|---|---|---|
| PA Classifier | TF-IDF (word 1-3, char 3-5) + LogisticRegression | 3 | {len(train_df)} | {len(val_df)} | {len(test_df)} |
| Threat Classifier | TF-IDF (word 1-3) + LogisticRegression (balanced) | 4 | {len(train_df)} | {len(val_df)} | {len(test_df)} |

Data source: `{DATA_PATH}` ({len(df)} records, synthetic)  
Split strategy: Conversation-grouped GroupShuffleSplit (70/15/15)

> **Important:** All evaluations are on synthetic held-out data.
> Real-world performance on genuine social-media content will differ
> and requires validation with human-annotated data.

---

## PA Classifier — Test Results

| Metric | Value |
|---|---|
| Accuracy | {pa_test.get('accuracy', 'N/A')} |
| Macro F1 | {pa_test.get('macro_f1', 'N/A')} |
| Avg Latency | {pa_test.get('avg_latency_ms', 'N/A')} ms/sample |
| Test N | {pa_test.get('n_samples', 'N/A')} |

### Per-Class Performance (Test)
"""
    pa_cr = pa_test.get("classification_report", {})
    for lbl in PA_LABELS:
        if lbl in pa_cr:
            r = pa_cr[lbl]
            md += f"- **{lbl}**: precision={r.get('precision',0):.3f}  recall={r.get('recall',0):.3f}  f1={r.get('f1-score',0):.3f}  support={r.get('support',0)}\n"

    md += f"""
### Confusion Matrix (Test)
Labels: {PA_LABELS}

```
{json.dumps(pa_test.get('confusion_matrix', []), indent=2)}
```

---

## Threat Classifier — Test Results

| Metric | Value |
|---|---|
| Accuracy | {thr_test.get('accuracy', 'N/A')} |
| Macro F1 | {thr_test.get('macro_f1', 'N/A')} |
| Avg Latency | {thr_test.get('avg_latency_ms', 'N/A')} ms/sample |
| Test N | {thr_test.get('n_samples', 'N/A')} |

### Per-Class Performance (Test)
"""
    thr_cr = thr_test.get("classification_report", {})
    for lbl in THREAT_LABELS:
        if lbl in thr_cr:
            r = thr_cr[lbl]
            md += f"- **{lbl}**: precision={r.get('precision',0):.3f}  recall={r.get('recall',0):.3f}  f1={r.get('f1-score',0):.3f}  support={r.get('support',0)}\n"

    md += f"""
### Confusion Matrix (Test)
Labels: {THREAT_LABELS}

```
{json.dumps(thr_test.get('confusion_matrix', []), indent=2)}
```

---

## Data & Label Notes

- Dataset: {len(df)} synthetic records (`{DATA_PATH}`)
- PA labels: `passive_aggressive={dist.get('passive_aggressive',0)}`  `non_passive_aggressive={dist.get('non_passive_aggressive',0)}`  `ambiguous={dist.get('ambiguous',0)}`
- Threat labels: `no_threat` dominates (~91%) — threat classifier uses balanced class weights
- Split strategy: conversation-grouped to prevent context leakage across splits

## Limitations

1. All training and evaluation data is synthetic — may not reflect real-world social media PA distribution
2. Template-based generation creates some lexical overlap between classes
3. The threat classifier has very limited positive-class support; real-world application requires human-curated threat examples
4. The `ambiguous` PA class has fewer templates and may underperform on genuinely ambiguous messages
5. Model performance on code-switched, very short, or image-caption texts is untested
"""

    with open(EVAL_MD, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info("Saved evaluation markdown → %s", EVAL_MD)
    logger.info("=== Training complete ===")


if __name__ == "__main__":
    main()
