"""
Transformer Fine-Tuning Pipeline for PassiveAggression Detection.

Utilizes local XLM-RoBERTa pretrained weights from D:/NLP_PROJECT/models/xlmr_text_sarcasm_full
as an encoder backbone with a 3-class classification head on NVIDIA RTX 4050 GPU.

Outputs:
  - models/xlmr_pa_detector/ (fine-tuned checkpoint)
  - reports/transformer_results.json
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABELS = ["neutral", "passive_aggressive", "overt_hostile"]
LABEL_TO_INT = {l: i for i, l in enumerate(LABELS)}
BACKBONE_PATH = r"D:\NLP_PROJECT\models\xlmr_text_sarcasm_full"
OUTPUT_DIR = Path("models/xlmr_pa_detector")
SEED = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class PADataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def run_transformer_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s (VRAM: %.2f GB)", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1e9)

    # 1. Load Data
    data_path = Path("data/processed/pa_dataset.csv")
    df = pd.read_csv(data_path)
    df["label_int"] = df["primary_label"].map(LABEL_TO_INT)
    df = df.dropna(subset=["text", "label_int"])

    train_df = df[df["split_group"] == "train"]
    test_df  = df[df["split_group"] == "test"]

    # Stratified balance sample for fast efficient fine-tuning: 2500 train, 500 test
    train_sample = pd.concat([
        train_df[train_df["label_int"] == i].sample(min(len(train_df[train_df["label_int"] == i]), 1000), random_state=SEED)
        for i in range(3)
    ]).reset_index(drop=True)
    test_sample = pd.concat([
        test_df[test_df["label_int"] == i].sample(min(len(test_df[test_df["label_int"] == i]), 300), random_state=SEED)
        for i in range(3)
    ]).reset_index(drop=True)

    logger.info("Fine-tuning on %d train samples, evaluating on %d held-out test samples", len(train_sample), len(test_sample))

    # 2. Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        BACKBONE_PATH,
        num_labels=3,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # Class Weights for Imbalance
    class_counts = train_sample["label_int"].value_counts().sort_index().values
    total_samples = len(train_sample)
    weights = total_samples / (len(class_counts) * class_counts.astype(float))
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    train_ds = PADataset(train_sample["text"], train_sample["label_int"], tokenizer)
    test_ds  = PADataset(test_sample["text"], test_sample["label_int"], tokenizer)

    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 2
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    logger.info("Starting transformer fine-tuning for %d epochs...", epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=mask)
                loss = loss_fn(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            ep_loss += loss.item()
            if (step + 1) % 40 == 0 or (step + 1) == len(train_loader):
                logger.info("Epoch %d [%d/%d] - Loss: %.4f", ep + 1, step + 1, len(train_loader), ep_loss / (step + 1))

    # 3. Evaluation
    logger.info("Evaluating transformer on held-out test split...")
    model.eval()
    all_preds = []
    all_targets = []
    t0 = time.perf_counter()

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch["labels"].numpy())

    total_eval_time = (time.perf_counter() - t0) * 1000
    avg_latency = total_eval_time / len(test_sample)

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    report_str = classification_report(all_targets, all_preds, target_names=LABELS, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds).tolist()

    logger.info("=== Transformer XLM-R PA Detector ===")
    logger.info("Accuracy: %.4f | Macro F1: %.4f | Avg Latency: %.2f ms", acc, macro_f1, avg_latency)
    logger.info("\n%s", report_str)

    # 4. Save Artifacts & Reports
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("Saved fine-tuned transformer model to %s", OUTPUT_DIR)

    res = {
        "model_name": "XLM-RoBERTa-PA-FineTuned",
        "backbone": BACKBONE_PATH,
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "confusion_matrix": cm,
        "classification_report": report_str,
        "evaluation_samples": len(test_sample),
    }

    rep_path = Path("reports/transformer_results.json")
    with open(rep_path, "w") as f:
        json.dump(res, f, indent=2)
    logger.info("Saved transformer results: %s", rep_path)

    # Update reports/model_comparison.csv
    comp_path = Path("reports/model_comparison.csv")
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        new_row = pd.DataFrame([{
            "model": "XLM-RoBERTa-FineTuned",
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "latency_ms": round(avg_latency, 2),
        }])
        comp_df = pd.concat([comp_df, new_row], ignore_index=True)
        comp_df.to_csv(comp_path, index=False)
        logger.info("Updated model comparison table: %s", comp_path)

    print("\n=== Transformer Evaluation Complete ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Latency: {avg_latency:.2f} ms")


if __name__ == "__main__":
    run_transformer_pipeline()
