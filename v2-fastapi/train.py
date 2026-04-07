"""
train.py — Fine-tune XLM-RoBERTa for euphemism detection.

Input format: text column contains [PET_BOUNDARY] markers around the target phrase.
Label: 1 = euphemistic, 0 = literal

Usage:
    python train.py --data dataset.csv --output ./model
"""

import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import os
import json

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "xlm-roberta-base"
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


# ── Dataset ───────────────────────────────────────────────────────────────────
class EuphDataset(Dataset):
    """
    Feeds the model the full paragraph text WITH [PET_BOUNDARY] markers intact.
    XLM-R will learn to associate those markers with the classification target.
    Think of the markers as highlighting — you're telling the model 'pay attention here'.
    """

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


# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
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
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            trues.extend(labels.cpu().numpy())

    f1 = f1_score(trues, preds, average="macro")
    return total_loss / len(loader), f1, preds, trues


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    # Load data
    df = pd.read_csv(args.data)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    # Split: 80% train, 10% val, 10% test — stratified
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Tokenizer — add PET_BOUNDARY as special tokens so the model treats them as meaningful markers
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    special_tokens = {"additional_special_tokens": ["[PET_BOUNDARY]"]}
    tokenizer.add_special_tokens(special_tokens)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.resize_token_embeddings(len(tokenizer))  # account for new special token
    model = model.to(device)

    # Datasets & loaders
    train_ds = EuphDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, MAX_LEN)
    val_ds = EuphDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, MAX_LEN)
    test_ds = EuphDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Optimizer with weight decay (regularization — important for small datasets)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped, lr=LR)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training loop with early stopping
    best_val_f1 = 0
    patience = 2
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        val_loss, val_f1, _, _ = eval_epoch(model, val_loader)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve = 0
            # Save best model
            os.makedirs(args.output, exist_ok=True)
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            print(f"  ✓ Saved best model (Val F1: {best_val_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Final test evaluation
    print("\n── Test Set Evaluation ──")
    _, test_f1, test_preds, test_trues = eval_epoch(model, test_loader)
    print(classification_report(test_trues, test_preds, target_names=["Literal", "Euphemistic"]))

    # Save metrics
    metrics = {"best_val_f1": best_val_f1, "test_f1": test_f1}
    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {args.output}/metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset.csv", help="Path to CSV dataset")
    parser.add_argument("--output", type=str, default="./model", help="Directory to save fine-tuned model")
    args = parser.parse_args()
    main(args)
