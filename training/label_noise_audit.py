"""
training/label_noise_audit.py
Uses Cleanlab (Confident Learning) and 5-fold cross-validation out-of-fold probabilities
to audit label noise across mapped, silver, and synthetic dataset splits.
Exports flagged candidate review items to reports/cleanlab_review_candidates.csv.
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from cleanlab.filter import find_label_issues

def main():
    print("[INFO] Executing Cleanlab Label-Noise Audit on Primary Training Data...")
    
    # 1. Load dataset
    df_path = os.path.join("data", "processed", "pa_dataset.csv")
    df = pd.read_csv(df_path)
    
    train_mask = (df["split"] == "train")
    train_df = df[train_mask].copy().reset_index(drop=True)
    
    label_map = {"neutral": 0, "passive_aggressive": 1, "overt_hostile": 2}
    inv_label_map = {0: "neutral", 1: "passive_aggressive", 2: "overt_hostile"}
    
    labels = train_df["primary_label"].map(label_map).values
    texts = train_df["text"].fillna("").values
    
    # 2. Compute 5-fold out-of-fold predicted probabilities
    print("  [1/3] Computing 5-fold cross-validated out-of-sample probabilities...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(train_df), 3))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True)
        X_tr = vec.fit_transform(texts[train_idx])
        X_val = vec.transform(texts[val_idx])
        
        clf = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
        clf.fit(X_tr, labels[train_idx])
        
        oof_probs[val_idx] = clf.predict_proba(X_val)
        
    # 3. Cleanlab Confident Learning
    print("  [2/3] Running Cleanlab find_label_issues...")
    issue_indices = find_label_issues(
        labels=labels,
        pred_probs=oof_probs,
        return_indices_ranked_by="self_confidence"
    )
    
    print(f"  Identified {len(issue_indices)} candidate label issues out of {len(train_df)} training samples ({len(issue_indices)/len(train_df)*100:.2f}%)")
    
    # 4. Export Candidate Review Records
    candidates = []
    for idx in issue_indices:
        row = train_df.iloc[idx]
        pred_label_idx = np.argmax(oof_probs[idx])
        candidates.append({
            "sample_id": row.get("sample_id", f"TRAIN_{idx:05d}"),
            "text": row["text"],
            "current_label": row["primary_label"],
            "suggested_label": inv_label_map[pred_label_idx],
            "confidence_given_label": round(float(oof_probs[idx, labels[idx]]), 4),
            "confidence_suggested_label": round(float(oof_probs[idx, pred_label_idx]), 4),
            "source_dataset": row.get("source_dataset", "unknown"),
            "label_origin": row.get("label_origin", "silver"),
            "review_recommendation": "Verify if boundary text contains sarcastic subtext or genuine disagreement"
        })
        
    candidates_df = pd.DataFrame(candidates)
    review_path = os.path.join("reports", "cleanlab_review_candidates.csv")
    candidates_df.to_csv(review_path, index=False)
    print(f"[SUCCESS] Exported {len(candidates_df)} review records to {review_path}")
    
    # Print sample findings
    print("\nTop 5 Flagged Label Noise Candidates:")
    for i, c in enumerate(candidates[:5], 1):
        print(f"  {i}. [{c['current_label']} -> {c['suggested_label']}] {c['text'][:65]}... (P_curr={c['confidence_given_label']}, P_sugg={c['confidence_suggested_label']})")

if __name__ == "__main__":
    main()
