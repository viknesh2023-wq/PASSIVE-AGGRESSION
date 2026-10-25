"""
training/evaluate_controlled_experiments.py
Phase 12 Controlled Experiments & Systematic Model Comparison.
Evaluates 4 distinct model architectures on the exact same held-out test split
(data/processed/pa_test.csv, 1,382 conversation-isolated samples):
  1. Legacy SVM Classifier (models/pa_classifier.pkl)
  2. Model 2: Enhanced Rule-Based Engine (inference/rule_detector.py)
  3. Model 1: Supervised ML Logistic Regression Pipeline (models/pa_classifier_v2.pkl)
  4. True Hybrid Intelligence Decision Engine (inference/pa_inference.py)

Generates:
  - reports/controlled_experiments_comparison.json
  - reports/controlled_experiments_comparison.md
"""

from __future__ import annotations
import sys
import json
import time
import logging
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from inference.rule_detector import detect as rule_detect
from inference.pa_inference import analyze, _load_models, _pa_pipeline, _inference_mode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("controlled_experiments")

TEST_DATA_PATH = _ROOT / "data" / "processed" / "pa_test.csv"
REPORTS_DIR = _ROOT / "reports"
PA_CLASSES = ["passive_aggressive", "non_passive_aggressive", "ambiguous"]


def evaluate_variant(y_true: list[str], y_pred: list[str], name: str, latency_ms: float) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=PA_CLASSES, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=PA_CLASSES, average="weighted", zero_division=0))

    report = classification_report(y_true, y_pred, labels=PA_CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=PA_CLASSES).tolist()

    per_class = {}
    for c in PA_CLASSES:
        if c in report:
            per_class[c] = {
                "precision": round(float(report[c]["precision"]), 4),
                "recall": round(float(report[c]["recall"]), 4),
                "f1": round(float(report[c]["f1-score"]), 4),
                "support": int(report[c]["support"]),
            }

    return {
        "model_name": name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "latency_per_sample_ms": round(latency_ms, 3),
        "per_class": per_class,
        "confusion_matrix": {
            "classes": PA_CLASSES,
            "matrix": cm,
        },
    }


def main():
    logger.info("Loading held-out test data from: %s", TEST_DATA_PATH)
    if not TEST_DATA_PATH.exists():
        logger.error("Test data not found: %s", TEST_DATA_PATH)
        return

    test_df = pd.read_csv(TEST_DATA_PATH)
    logger.info("Loaded %d test samples.", len(test_df))
    y_true = test_df["primary_label"].astype(str).tolist()

    # Pre-format context text
    test_texts = test_df["text"].astype(str).tolist()
    parent_comments = test_df["parent_comment"].fillna("").astype(str).tolist()

    # Ensure inference models are loaded
    _load_models()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Variant 1: Legacy LinearSVM
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Evaluating Variant 1: Legacy LinearSVM...")
    legacy_path = _ROOT / "models" / "pa_classifier.pkl"
    legacy_preds = []
    legacy_latency = 0.0

    if legacy_path.exists():
        legacy_pipeline = joblib.load(legacy_path)
        t0 = time.perf_counter()
        raw_legacy = legacy_pipeline.predict(test_texts)
        legacy_latency = ((time.perf_counter() - t0) * 1000) / len(test_texts)

        # Legacy map: neutral -> non_passive_aggressive, overt_hostile -> non_passive_aggressive
        for p in raw_legacy:
            p_str = str(p).lower()
            if p_str == "passive_aggressive":
                legacy_preds.append("passive_aggressive")
            elif p_str in ("neutral", "non_passive_aggressive", "overt_hostile"):
                legacy_preds.append("non_passive_aggressive")
            else:
                legacy_preds.append("ambiguous")
    else:
        logger.warning("Legacy model artifact not found.")
        legacy_preds = ["ambiguous"] * len(test_df)

    v1_metrics = evaluate_variant(y_true, legacy_preds, "Legacy LinearSVM (3-class legacy)", legacy_latency)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Variant 2: Model 2 (Enhanced Rule-Based Engine)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Evaluating Variant 2: Enhanced Rule-Based Engine...")
    t0 = time.perf_counter()
    rule_preds = []
    for text, parent in zip(test_texts, parent_comments):
        res = rule_detect(text=text, parent_comment=parent if len(parent) > 3 else None, sensitivity="balanced")
        rule_preds.append(res.pa_label)
    rule_latency = ((time.perf_counter() - t0) * 1000) / len(test_texts)

    v2_metrics = evaluate_variant(y_true, rule_preds, "Model 2: Rule-Based Engine (10 Families)", rule_latency)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Variant 3: Model 1 (Supervised ML Logistic Regression)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Evaluating Variant 3: Model 1 (TF-IDF + Logistic Regression)...")
    v2_model_path = _ROOT / "models" / "pa_classifier_v2.pkl"
    ml_model = joblib.load(v2_model_path)
    t0 = time.perf_counter()
    ml_preds = []
    for text, parent in zip(test_texts, parent_comments):
        if len(parent) > 3:
            ctx_text = f"[context]: {parent.lower()} [reply]: {text.lower()}"
        else:
            ctx_text = text.lower()
        pred = ml_model.predict([ctx_text])[0]
        ml_preds.append(str(pred))
    ml_latency = ((time.perf_counter() - t0) * 1000) / len(test_texts)

    v3_metrics = evaluate_variant(y_true, ml_preds, "Model 1: Supervised ML (TF-IDF + LogReg)", ml_latency)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Variant 4: True Hybrid Intelligence Decision Engine
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Evaluating Variant 4: Hybrid Intelligence Decision Engine...")
    t0 = time.perf_counter()
    hybrid_preds = []
    fusion_strategies = {}
    agreements = 0

    for text, parent in zip(test_texts, parent_comments):
        res = analyze(
            text=text,
            parent_comment=parent if len(parent) > 3 else None,
            sensitivity="balanced",
        )
        hyb = res.get("hybrid_result", res.get("combined", {}))
        verdict = hyb.get("verdict", "ambiguous")
        hybrid_preds.append(verdict)

        strat = hyb.get("fusion_strategy", "concordant_consensus")
        fusion_strategies[strat] = fusion_strategies.get(strat, 0) + 1
        if hyb.get("agreement", False):
            agreements += 1

    hybrid_latency = ((time.perf_counter() - t0) * 1000) / len(test_texts)
    v4_metrics = evaluate_variant(y_true, hybrid_preds, "Model 3: Hybrid Decision Engine (ML + Rules)", hybrid_latency)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Provenance Segmentation (Performance by Data Source)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Segmenting evaluation by dataset provenance...")
    provenance_breakdown = {}
    for src in test_df["source"].unique():
        mask = (test_df["source"] == src)
        sub_true = [y for y, m in zip(y_true, mask) if m]
        sub_ml = [p for p, m in zip(ml_preds, mask) if m]
        sub_rule = [p for p, m in zip(rule_preds, mask) if m]
        sub_hyb = [p for p, m in zip(hybrid_preds, mask) if m]

        provenance_breakdown[str(src)] = {
            "samples": int(sum(mask)),
            "ml_accuracy": round(float(accuracy_score(sub_true, sub_ml)), 4),
            "rule_accuracy": round(float(accuracy_score(sub_true, sub_rule)), 4),
            "hybrid_accuracy": round(float(accuracy_score(sub_true, sub_hyb)), 4),
        }

    # Agreement / Disagreement Statistics
    disagreement_rate = round(float(1.0 - (agreements / len(test_df))), 4)

    # Output compilation
    full_comparison = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_dataset": str(TEST_DATA_PATH),
        "total_test_samples": len(test_df),
        "agreement_count": agreements,
        "disagreement_rate": disagreement_rate,
        "fusion_strategy_distribution": fusion_strategies,
        "variants": [
            v1_metrics,
            v2_metrics,
            v3_metrics,
            v4_metrics,
        ],
        "provenance_breakdown": provenance_breakdown,
    }

    # Save JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "controlled_experiments_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_comparison, f, indent=2)
    logger.info("Saved comparison JSON to: %s", json_path)

    # Save Markdown
    md_path = REPORTS_DIR / "controlled_experiments_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Controlled Experiments & Multi-Model Comparison Report\n\n")
        f.write(f"**Evaluated on:** Held-out test set (`data/processed/pa_test.csv`)\n")
        f.write(f"**Test Sample Count:** {len(test_df)} conversation-isolated examples\n")
        f.write(f"**Rule vs ML Disagreement Rate:** {disagreement_rate * 100:.2f}%\n\n")
        f.write("---\n\n")

        f.write("## 1. Aggregate Model Performance Comparison\n\n")
        f.write("| Architecture Variant | Accuracy | Macro F1 | Weighted F1 | Latency / Sample | Operational Role |\n")
        f.write("|---|---:|---:|---:|---:|---|\n")
        f.write(f"| **Legacy LinearSVM** | {v1_metrics['accuracy']*100:.2f}% | {v1_metrics['macro_f1']:.4f} | {v1_metrics['weighted_f1']:.4f} | {v1_metrics['latency_per_sample_ms']:.2f} ms | Archived Legacy Baseline |\n")
        f.write(f"| **Model 2: Rule Engine** | {v2_metrics['accuracy']*100:.2f}% | {v2_metrics['macro_f1']:.4f} | {v2_metrics['weighted_f1']:.4f} | {v2_metrics['latency_per_sample_ms']:.2f} ms | High-Precision Span Offsets |\n")
        f.write(f"| **Model 1: Supervised ML** | {v3_metrics['accuracy']*100:.2f}% | {v3_metrics['macro_f1']:.4f} | {v3_metrics['weighted_f1']:.4f} | {v3_metrics['latency_per_sample_ms']:.2f} ms | Statistical N-Gram Generalizer |\n")
        f.write(f"| **Model 3: Hybrid Engine** | **{v4_metrics['accuracy']*100:.2f}%** | **{v4_metrics['macro_f1']:.4f}** | **{v4_metrics['weighted_f1']:.4f}** | {v4_metrics['latency_per_sample_ms']:.2f} ms | **Production Fused Decision Layer** |\n\n")
        f.write("---\n\n")

        f.write("## 2. Per-Class F1 Score Breakdown\n\n")
        f.write("| Architecture Variant | Passive-Aggressive | Non-Passive-Aggressive | Ambiguous |\n")
        f.write("|---|---:|---:|---:|\n")
        for v in [v1_metrics, v2_metrics, v3_metrics, v4_metrics]:
            f.write(f"| **{v['model_name']}** | {v['per_class'].get('passive_aggressive', {}).get('f1', 0.0):.4f} | {v['per_class'].get('non_passive_aggressive', {}).get('f1', 0.0):.4f} | {v['per_class'].get('ambiguous', {}).get('f1', 0.0):.4f} |\n")
        f.write("\n---\n\n")

        f.write("## 3. Hybrid Fusion Strategy Breakdown\n\n")
        f.write("How the Hybrid Intelligence Decision Layer resolved classifications on held-out test data:\n\n")
        f.write("| Fusion Policy Strategy | Trigger Count | Description |\n")
        f.write("|---|---:|---|\n")
        for strat, count in fusion_strategies.items():
            f.write(f"| `{strat}` | {count} ({count/len(test_df)*100:.1f}%) | Specific arbitration mechanism |\n")
        f.write("\n---\n\n")

        f.write("## 4. Performance by Dataset Provenance\n\n")
        f.write("| Dataset Provenance | Samples | Model 1 (ML) Acc | Model 2 (Rule) Acc | Hybrid Acc |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for src, d in provenance_breakdown.items():
            f.write(f"| **{src}** | {d['samples']} | {d['ml_accuracy']*100:.1f}% | {d['rule_accuracy']*100:.1f}% | **{d['hybrid_accuracy']*100:.1f}%** |\n")

    logger.info("Saved comparison Markdown to: %s", md_path)
    print("Controlled experiments completed successfully.")


if __name__ == "__main__":
    main()
