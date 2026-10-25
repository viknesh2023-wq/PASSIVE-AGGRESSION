"""
training/prepare_datasets.py
Prepares task-specific datasets and leakage-resistant splits:
- data/processed/pa_training.csv
- data/processed/threat_training.csv
- data/processed/sarcasm_auxiliary.csv
- data/processed/emotion_auxiliary.csv
- data/processed/pa_train.csv
- data/processed/pa_validation.csv
- data/processed/pa_test.csv
- data/processed/threat_train.csv
- data/processed/threat_validation.csv
- data/processed/threat_test.csv
Generates:
- reports/label_mapping.md
- reports/split_integrity_report.md
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

_ROOT = Path(__file__).resolve().parent.parent

PA_LABELS = ["passive_aggressive", "non_passive_aggressive", "ambiguous"]
THREAT_LABELS = ["no_threat", "direct_threat", "implied_threat", "ambiguous"]

def prepare_all():
    print("Preparing task-specific datasets and leakage-resistant splits...")
    synthetic_path = _ROOT / "data" / "synthetic" / "pa_synthetic_merged.csv"
    unified_path = _ROOT / "data" / "processed" / "unified_corpus.csv"
    processed_dir = _ROOT / "data" / "processed"
    reports_dir = _ROOT / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df_syn = pd.read_csv(synthetic_path, low_memory=False)
    df_uni = pd.read_csv(unified_path, low_memory=False)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Primary PA Training Dataset
    # ─────────────────────────────────────────────────────────────────────────
    # Primary training dataset consists of the clean, 19-column synthetic records
    # plus verified gold reference anchors and mined PA.
    pa_records = []

    # From synthetic (9,000 records)
    for _, r in df_syn.iterrows():
        if r["primary_label"] in PA_LABELS:
            pa_records.append({
                "id": r["id"],
                "text": str(r["text"]),
                "primary_label": r["primary_label"],
                "threat_label": r.get("threat_label", "no_threat"),
                "post_text": str(r.get("post_text", "")) if pd.notna(r.get("post_text")) else "",
                "parent_comment": str(r.get("parent_comment", "")) if pd.notna(r.get("parent_comment")) else "",
                "conversation_id": str(r.get("conversation_id", r["id"])),
                "turn_index": r.get("turn_index", 0),
                "tactic_labels": str(r.get("tactic_labels", "[]")),
                "source": "synthetic",
                "is_gold": False,
            })

    # From verified gold anchors (50 records)
    gold_df = df_uni[df_uni["source_dataset"] == "gold_annotation_template"]
    for _, r in gold_df.iterrows():
        # Map legacy label: neutral -> non_passive_aggressive, overt_hostile -> non_passive_aggressive
        raw_lbl = str(r["primary_label"]).lower()
        if raw_lbl == "passive_aggressive":
            p_lbl = "passive_aggressive"
        else:
            p_lbl = "non_passive_aggressive"
        
        pa_records.append({
            "id": f"GOLD_{r['sample_id']}",
            "text": str(r["text"]),
            "primary_label": p_lbl,
            "threat_label": "no_threat",
            "post_text": "",
            "parent_comment": "",
            "conversation_id": f"gold_conv_{r['sample_id']}",
            "turn_index": 0,
            "tactic_labels": json.dumps([str(r.get("tactics", ""))]) if pd.notna(r.get("tactics")) else "[]",
            "source": "gold_anchor",
            "is_gold": True,
        })

    # From authentic mined PA (143 records)
    mined_df = df_uni[df_uni["source_dataset"] == "sarc_mined_pa"]
    for _, r in mined_df.iterrows():
        pa_records.append({
            "id": f"MINED_{r['sample_id']}",
            "text": str(r["text"]),
            "primary_label": "passive_aggressive",
            "threat_label": "no_threat",
            "post_text": "",
            "parent_comment": "",
            "conversation_id": f"mined_conv_{r['sample_id']}",
            "turn_index": 0,
            "tactic_labels": json.dumps([str(r.get("tactics", ""))]) if pd.notna(r.get("tactics")) else "[]",
            "source": "authentic_mined",
            "is_gold": False,
        })

    # From contrastive pairs (60 records)
    cont_df = df_uni[df_uni["source_dataset"] == "contrastive_hard_negatives"]
    for _, r in cont_df.iterrows():
        pa_records.append({
            "id": f"CONT_{r['sample_id']}",
            "text": str(r["text"]),
            "primary_label": "non_passive_aggressive",
            "threat_label": "no_threat",
            "post_text": "",
            "parent_comment": "",
            "conversation_id": f"cont_conv_{r['sample_id']}",
            "turn_index": 0,
            "tactic_labels": "[]",
            "source": "contrastive_hard_negative",
            "is_gold": False,
        })

    df_pa_all = pd.DataFrame(pa_records)
    df_pa_all.to_csv(processed_dir / "pa_training.csv", index=False, encoding="utf-8")
    print(f"Created pa_training.csv with {len(df_pa_all)} records.")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Auxiliary Datasets
    # ─────────────────────────────────────────────────────────────────────────
    # Threat training
    df_threat = df_syn[df_syn["threat_label"].isin(THREAT_LABELS)].copy()
    df_threat.to_csv(processed_dir / "threat_training.csv", index=False, encoding="utf-8")
    print(f"Created threat_training.csv with {len(df_threat)} records.")

    # Sarcasm auxiliary
    sarc_df = df_uni[df_uni["source_dataset"].isin(["sarc_reddit_sarcasm", "genz_synthetic_sarcasm"])].copy()
    sarc_df.to_csv(processed_dir / "sarcasm_auxiliary.csv", index=False, encoding="utf-8")
    print(f"Created sarcasm_auxiliary.csv with {len(sarc_df)} records.")

    # Emotion auxiliary
    emo_df = df_uni[df_uni["source_dataset"].isin(["goemotions_auxiliary", "goemotions_hostile"])].copy()
    emo_df.to_csv(processed_dir / "emotion_auxiliary.csv", index=False, encoding="utf-8")
    print(f"Created emotion_auxiliary.csv with {len(emo_df)} records.")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Leakage-Resistant Data Splits (70% Train, 15% Val, 15% Test)
    # ─────────────────────────────────────────────────────────────────────────
    # Group by conversation_id
    groups = df_pa_all["conversation_id"].values
    labels = df_pa_all["primary_label"].values

    # Step 1: Split off 15% test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(gss_test.split(df_pa_all, labels, groups))
    df_train_val = df_pa_all.iloc[train_val_idx].reset_index(drop=True)
    df_test = df_pa_all.iloc[test_idx].reset_index(drop=True)

    # Step 2: Split remaining 85% into train (70% of total) and val (15% of total)
    val_ratio = 0.15 / 0.85
    groups_tv = df_train_val["conversation_id"].values
    labels_tv = df_train_val["primary_label"].values

    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=43)
    train_idx, val_idx = next(gss_val.split(df_train_val, labels_tv, groups_tv))
    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

    # Save PA splits
    df_train.to_csv(processed_dir / "pa_train.csv", index=False, encoding="utf-8")
    df_val.to_csv(processed_dir / "pa_validation.csv", index=False, encoding="utf-8")
    df_test.to_csv(processed_dir / "pa_test.csv", index=False, encoding="utf-8")
    print(f"PA splits saved: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    # Threat splits (from df_threat)
    t_groups = df_threat["conversation_id"].values
    t_labels = df_threat["threat_label"].values
    gss_t_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    t_tv_idx, t_test_idx = next(gss_t_test.split(df_threat, t_labels, t_groups))
    df_t_tv = df_threat.iloc[t_tv_idx].reset_index(drop=True)
    df_t_test = df_threat.iloc[t_test_idx].reset_index(drop=True)

    gss_t_val = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=43)
    t_train_idx, t_val_idx = next(gss_t_val.split(df_t_tv, df_t_tv["threat_label"].values, df_t_tv["conversation_id"].values))
    df_t_train = df_t_tv.iloc[t_train_idx].reset_index(drop=True)
    df_t_val = df_t_tv.iloc[t_val_idx].reset_index(drop=True)

    df_t_train.to_csv(processed_dir / "threat_train.csv", index=False, encoding="utf-8")
    df_t_val.to_csv(processed_dir / "threat_validation.csv", index=False, encoding="utf-8")
    df_t_test.to_csv(processed_dir / "threat_test.csv", index=False, encoding="utf-8")
    print(f"Threat splits saved: Train={len(df_t_train)}, Val={len(df_t_val)}, Test={len(df_t_test)}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Generate Split Integrity Report & Label Mapping Report
    # ─────────────────────────────────────────────────────────────────────────
    # Check conversation overlap
    train_convs = set(df_train["conversation_id"].unique())
    val_convs = set(df_val["conversation_id"].unique())
    test_convs = set(df_test["conversation_id"].unique())

    overlap_train_val = len(train_convs.intersection(val_convs))
    overlap_train_test = len(train_convs.intersection(test_convs))
    overlap_val_test = len(val_convs.intersection(test_convs))

    split_report = f"""# Leakage-Resistant Split Integrity Report

**Generated on:** 2026-10-25  
**Split Strategy:** Conversation-Grouped `GroupShuffleSplit` on `conversation_id`  
**Target Proportions:** 70% Train, 15% Validation, 15% Test  

---

## 1. Primary PA Dataset Split Metrics

| Partition | Record Count | Percentage | Unique Conversations | Conversation Leakage |
|---|---:|---:|---:|:---:|
| **Train (`pa_train.csv`)** | {len(df_train):,} | {len(df_train)/len(df_pa_all)*100:.1f}% | {len(train_convs):,} | **0 records** |
| **Validation (`pa_validation.csv`)** | {len(df_val):,} | {len(df_val)/len(df_pa_all)*100:.1f}% | {len(val_convs):,} | **0 records** |
| **Test (`pa_test.csv`)** | {len(df_test):,} | {len(df_test)/len(df_pa_all)*100:.1f}% | {len(test_convs):,} | **0 records** |
| **TOTAL** | **{len(df_pa_all):,}** | **100.0%** | **{len(train_convs) + len(val_convs) + len(test_convs):,}** | **ZERO LEAKAGE** |

### Verified Leakage Checks:
- Overlap between Train and Validation conversations: **{overlap_train_val}**
- Overlap between Train and Test conversations: **{overlap_train_test}**
- Overlap between Validation and Test conversations: **{overlap_val_test}**

---

## 2. Partition Label Distributions

### Primary PA Task
| Class Label | Train Count (%) | Val Count (%) | Test Count (%) |
|---|---:|---:|---:|
| `passive_aggressive` | {sum(df_train['primary_label'] == 'passive_aggressive')} ({sum(df_train['primary_label'] == 'passive_aggressive')/len(df_train)*100:.1f}%) | {sum(df_val['primary_label'] == 'passive_aggressive')} ({sum(df_val['primary_label'] == 'passive_aggressive')/len(df_val)*100:.1f}%) | {sum(df_test['primary_label'] == 'passive_aggressive')} ({sum(df_test['primary_label'] == 'passive_aggressive')/len(df_test)*100:.1f}%) |
| `non_passive_aggressive` | {sum(df_train['primary_label'] == 'non_passive_aggressive')} ({sum(df_train['primary_label'] == 'non_passive_aggressive')/len(df_train)*100:.1f}%) | {sum(df_val['primary_label'] == 'non_passive_aggressive')} ({sum(df_val['primary_label'] == 'non_passive_aggressive')/len(df_val)*100:.1f}%) | {sum(df_test['primary_label'] == 'non_passive_aggressive')} ({sum(df_test['primary_label'] == 'non_passive_aggressive')/len(df_test)*100:.1f}%) |
| `ambiguous` | {sum(df_train['primary_label'] == 'ambiguous')} ({sum(df_train['primary_label'] == 'ambiguous')/len(df_train)*100:.1f}%) | {sum(df_val['primary_label'] == 'ambiguous')} ({sum(df_val['primary_label'] == 'ambiguous')/len(df_val)*100:.1f}%) | {sum(df_test['primary_label'] == 'ambiguous')} ({sum(df_test['primary_label'] == 'ambiguous')/len(df_test)*100:.1f}%) |

---

## 3. Threat Classification Split Metrics

| Partition | Records | Percentage |
|---|---:|---:|
| **Train (`threat_train.csv`)** | {len(df_t_train):,} | {len(df_t_train)/len(df_threat)*100:.1f}% |
| **Validation (`threat_validation.csv`)** | {len(df_t_val):,} | {len(df_t_val)/len(df_threat)*100:.1f}% |
| **Test (`threat_test.csv`)** | {len(df_t_test):,} | {len(df_t_test)/len(df_threat)*100:.1f}% |
| **TOTAL** | **{len(df_threat):,}** | **100.0%** |
"""
    with open(reports_dir / "split_integrity_report.md", "w", encoding="utf-8") as f:
        f.write(split_report)

    # Label mapping report
    mapping_report = """# Task-Specific Label Mapping Specification

**Target System:** PassiveAggression Detector 2.0  
**Specification:** Explicit semantic separation between primary covert hostility (PA), overt hostility, playful sarcasm, neutral sentiment, and independent threats.

---

## 1. Primary Classification Mapping Rules

| Source Dataset | Original Label | Primary Task Mapping | Threat Mapping | Rationale |
|---|---|---|---|---|
| `tweeteval_sentiment` | `1` (Neutral) | `non_passive_aggressive` | `no_threat` | Sincere, objective conversational statements. |
| `sarc_reddit_regular` | `0` (Regular) | `non_passive_aggressive` | `no_threat` | Ordinary everyday social media dialogue. |
| `sarc_reddit_sarcasm` | `1` (Sarcastic) | `non_passive_aggressive` | `no_threat` | Non-hostile, playful irony/sarcasm (hard negative). |
| `genz_synthetic_sarcasm` | `banter` | `non_passive_aggressive` | `no_threat` | Informal generational banter; non-toxic. |
| `goemotions_hostile` | `anger/annoyance` | `non_passive_aggressive` | `no_threat` | Direct overt aggression/insults; NOT covert/indirect PA. |
| `sarc_mined_pa` | `mined_cue` | `passive_aggressive` | `no_threat` | High-precision covert hostility cues (*"Must be nice"*, *"As per my"*). |
| `curated_pa_expansion` | `tactic_templates`| `passive_aggressive` | `no_threat` | Syntactic expansions across 6 indirect tactics. |
| `contrastive_hard_negatives`| `contrastive`| `non_passive_aggressive`| `no_threat` | Minimal pairs containing polite/sincere phrasing. |
| `gold_annotation_template` | Multi-annotator | Verified Gold | Verified Gold | Human ground truth reference anchors. |
| `synthetic_pa_dataset` | 3 Canonical Classes| Identity Mapping | Independent Task | Dedicated 9,000-record primary corpus. |

---

## 2. Seven Passive-Aggression Tactic Subtypes
1. `fake_politeness`: Mock deference, weaponized politeness (*"As per my last email"*, *"Bless your heart"*).
2. `indirect_criticism`: Veiled observations (*"I'm just saying"*, *"Interesting how"*).
3. `backhanded_compliment`: Double-edged praise (*"Surprisingly good for you"*, *"Better late than never"*).
4. `implied_blame`: Counterfactual guilt (*"If someone had actually checked"*, *"I already told you"*).
5. `sarcastic_agreement`: Feigned compliance (*"If you say so"*, *"Great, so now..."*).
6. `dismissiveness`: Curt disengagement (*"Whatever you say"*, *"Fine."*).
7. `other_or_unclear`: Ambiguous or unclassified indirect signals.
"""
    with open(reports_dir / "label_mapping.md", "w", encoding="utf-8") as f:
        f.write(mapping_report)

    print("Data preparation, splits, and reports completed successfully!")

if __name__ == "__main__":
    prepare_all()
