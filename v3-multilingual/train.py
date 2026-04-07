"""
train.py — Fine-tune XLM-RoBERTa for multilingual euphemism detection.

Supports two modes:
  1. Single CSV:    python train.py --data dataset.csv --output ./model
  2. Multilingual:  python train.py --data_dir ./data --output ./model_multilingual
                    (auto-discovers all *.csv files in the folder)

Warm-start mode (recommended for multilingual extension):
  python train.py --data_dir ./data --output ./model_multilingual --warm_start ./model

Input format: text column contains [PET_BOUNDARY] markers around the target phrase.
Label: 1 = euphemistic, 0 = literal
"""

import argparse
import glob
import os
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ── Config ────────────────────────────────────────────────────────────────────
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")


# ── Data loading ──────────────────────────────────────────────────────────────
def load_single(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    lang = os.path.basename(path).split("_pets_dat")[0]
    df["language"] = lang
    return df


def load_multilingual(data_dir: str) -> pd.DataFrame:
    paths = glob.glob(os.path.join(data_dir, "*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    dfs = []
    for p in sorted(paths):
        df = load_single(p)
        print(f"  Loaded {os.path.basename(p)}: {len(df)} rows (lang={df['language'].iloc[0]})")
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined dataset: {len(combined)} rows across {len(dfs)} languages")
    print("Language distribution:\n", combined["language"].value_counts().to_string())
    print("Label distribution:\n", combined["label"].value_counts().to_string())
    return combined


# ── Dataset ───────────────────────────────────────────────────────────────────
class EuphDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(weights=torch.tensor(weights, dtype=torch.float), num_samples=len(labels))


# ── Training / eval loops ─────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += outputs.loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader):
    model.eval()
    preds, trues = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
    f1 = f1_score(trues, preds, average="macro")
    return total_loss / len(loader), f1, preds, trues


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    if args.data_dir:
        df = load_multilingual(args.data_dir)
    else:
        df = load_single(args.data)
        print(f"Total samples: {len(df)}")

    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED)
    print(f"\nSplit -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    model_source = args.warm_start if args.warm_start else "xlm-roberta-base"
    print(f"\nLoading tokenizer from: {model_source}")
    tokenizer = AutoTokenizer.from_pretrained(model_source)

    existing_special = tokenizer.all_special_tokens
    if "[PET_BOUNDARY]" not in existing_special:
        tokenizer.add_special_tokens({"additional_special_tokens": ["[PET_BOUNDARY]"]})
        print("Added [PET_BOUNDARY] as special token.")
    else:
        print("[PET_BOUNDARY] already in tokenizer vocabulary.")

    print(f"Loading model from: {model_source}")
    model = AutoModelForSequenceClassification.from_pretrained(model_source, num_labels=2)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    train_labels = train_df["label"].tolist()
    train_ds = EuphDataset(train_df["text"].tolist(), train_labels, tokenizer, MAX_LEN)
    val_ds = EuphDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, MAX_LEN)
    test_ds = EuphDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, MAX_LEN)

    sampler = make_balanced_sampler(np.array(train_labels))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped, lr=LR)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_f1 = 0
    patience = 2
    no_improve = 0

    print("\n-- Training --")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        val_loss, val_f1, _, _ = eval_epoch(model, val_loader)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve = 0
            os.makedirs(args.output, exist_ok=True)
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            print(f"  Saved best model (Val F1: {best_val_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print("\n-- Overall Test Set Evaluation --")
    _, test_f1, test_preds, test_trues = eval_epoch(model, test_loader)
    print(classification_report(test_trues, test_preds, target_names=["Literal", "Euphemistic"]))

    if "language" in test_df.columns:
        print("-- Per-Language F1 --")
        test_df = test_df.copy()
        test_df["pred"] = test_preds
        for lang, group in test_df.groupby("language"):
            lang_f1 = f1_score(group["label"], group["pred"], average="macro")
            print(f"  {lang}: F1 = {lang_f1:.4f} (n={len(group)})")

    metrics = {
        "best_val_f1": best_val_f1,
        "test_f1": test_f1,
        "warm_start": args.warm_start or "none",
    }
    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {args.output}/metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to a single CSV dataset")
    parser.add_argument("--data_dir", type=str, default=None, help="Folder containing multiple language CSVs")
    parser.add_argument("--output", type=str, default="./model_multilingual", help="Directory to save fine-tuned model")
    parser.add_argument("--warm_start", type=str, default=None, help="Path to existing fine-tuned model to warm-start from")
    args = parser.parse_args()

    if not args.data and not args.data_dir:
        parser.error("Provide either --data (single file) or --data_dir (multilingual folder)")

    main(args)
