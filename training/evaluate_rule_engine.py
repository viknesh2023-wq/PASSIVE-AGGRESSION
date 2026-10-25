"""
training/evaluate_rule_engine.py
Evaluates Model 2 (Rule-Based Detector) independently across:
- 3 sensitivity modes: balanced, high_recall, high_precision
- Against held-out test data (data/processed/pa_test.csv)
- Outputs reports/rule_engine_evaluation.md
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from inference.rule_detector import detect as rule_detect

PA_LABELS = ["passive_aggressive", "non_passive_aggressive", "ambiguous"]

def evaluate_rules():
    test_path = _ROOT / "data" / "processed" / "pa_test.csv"
    reports_dir = _ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df_test = pd.read_csv(test_path, low_memory=False)
    y_true = df_test["primary_label"].values

    results = {}
    modes = ["balanced", "high_recall", "high_precision"]

    for mode in modes:
        preds = []
        for _, row in df_test.iterrows():
            res = rule_detect(
                text=str(row["text"]),
                context=str(row.get("conversation_id", "")),
                parent_comment=str(row.get("parent_comment", "")) if pd.notna(row.get("parent_comment")) else None,
                sensitivity=mode,
            )
            preds.append(res.pa_label)

        acc = accuracy_score(y_true, preds)
        macro_f1 = f1_score(y_true, preds, average="macro")
        rep = classification_report(y_true, preds, labels=PA_LABELS, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, preds, labels=PA_LABELS).tolist()

        results[mode] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "classification_report": rep,
            "confusion_matrix": cm,
        }

    # Save Markdown report
    md_lines = [
        "# Model 2 (Rule-Based Engine) Independent Evaluation Report",
        "",
        "**Evaluation Date:** 2026-10-25  ",
        f"**Test Set:** `data/processed/pa_test.csv` ({len(df_test):,} held-out records)  ",
        "**Rule Families Evaluated:** 10 linguistic families, 3 sensitivity thresholds.",
        "",
        "---",
        "",
        "## 1. Sensitivity Mode Comparison",
        "",
        "| Sensitivity Mode | Accuracy | Macro F1 | PA Precision | PA Recall | PA F1 | Ambiguous Recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for mode in modes:
        r = results[mode]
        pa_r = r["classification_report"]["passive_aggressive"]
        amb_r = r["classification_report"]["ambiguous"]
        md_lines.append(
            f"| `{mode}` | {r['accuracy']*100:.2f}% | {r['macro_f1']:.4f} | "
            f"{pa_r['precision']:.3f} | {pa_r['recall']:.3f} | {pa_r['f1-score']:.3f} | {amb_r['recall']:.3f} |"
        )

    md_lines.extend([
        "",
        "### Key Findings:",
        "- **`balanced` mode:** Provides calibrated evidence thresholding (30/50 raw score).",
        "- **`high_recall` mode:** Lowers triggering thresholds to detect subtle indirect cues, prioritizing sensitivity.",
        "- **`high_precision` mode:** Enforces multi-signal confirmation (threshold 65 + 2 signals), reducing false alarms.",
        "",
        "---",
        "",
        "## 2. Confusion Matrix (`balanced` mode)",
        f"Labels: `{PA_LABELS}`",
        "```",
        str(np.array(results['balanced']['confusion_matrix'])),
        "```",
    ])

    report_path = reports_dir / "rule_engine_evaluation.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Rule engine evaluation complete! Report saved to {report_path}.")

if __name__ == "__main__":
    evaluate_rules()
