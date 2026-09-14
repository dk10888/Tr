"""
BERT Food/Not-Food Classifier — Google Colab Training Script
=============================================================
Dataset  : final_dataset_augmented_v2.csv (22,030 rows)
Classes  : food (16136) | not_food (5894)  — imbalanced ~73/27
Model    : google/bert_uncased_L-2_H-128_A-2  (BERT-tiny, ~17MB)
           Best for on-device inference; handles OCR-noisy short text well.
Runtime  : Colab T4 GPU recommended (~15–20 min total)

Run cells top to bottom. After disconnect, re-run Cell 1→3, skip to Cell 7.
"""

# ============================================================
# CELL 1 — Mount Drive (run first, always)
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_DIR = "/content/drive/MyDrive/receipt_bert"
os.makedirs(DRIVE_DIR, exist_ok=True)
print("✅ Drive mounted →", DRIVE_DIR)


# ============================================================
# CELL 2 — Install packages
# ============================================================
!pip install -q transformers datasets scikit-learn accelerate

import torch
print("✅ PyTorch:", torch.__version__)
print("✅ CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("   GPU:", torch.cuda.get_device_name(0))


# ============================================================
# CELL 3 — Upload dataset to Drive (run once)
# ============================================================
# Option A: upload from your computer
from google.colab import files
uploaded = files.upload()   # select final_dataset_augmented_v2.csv

import shutil
for fname in uploaded:
    dest = os.path.join(DRIVE_DIR, fname)
    shutil.copy(fname, dest)
    print(f"✅ Saved to Drive: {dest}")

# Option B: if already in Drive, just set the path:
# CSV_PATH = "/content/drive/MyDrive/receipt_bert/final_dataset_augmented_v2.csv"


# ============================================================
# CELL 4 — Load & inspect dataset
# ============================================================
import pandas as pd
from collections import Counter

CSV_PATH = os.path.join(DRIVE_DIR, "final_dataset_augmented_v2.csv")
df = pd.read_csv(CSV_PATH)

print(f"Total rows : {len(df)}")
print(f"Label dist : {Counter(df['label'])}")
print(f"Text length: min={df['text'].str.len().min()}  "
      f"max={df['text'].str.len().max()}  "
      f"median={df['text'].str.len().median():.0f}")
print("\nSample food:")
print(df[df['label']=='food']['text'].sample(5).tolist())
print("\nSample not_food:")
print(df[df['label']=='not_food']['text'].sample(5).tolist())


# ============================================================
# CELL 5 — Preprocess: encode labels, stratified split
# ============================================================
import numpy as np
from sklearn.model_selection import train_test_split

# Map labels → integers
label2id = {"food": 1, "not_food": 0}
id2label = {v: k for k, v in label2id.items()}

df['label_id'] = df['label'].map(label2id)
df = df.dropna(subset=['text', 'label_id'])
df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'].str.len() >= 2].reset_index(drop=True)

# Stratified 80/10/10 split
train_df, temp_df = train_test_split(
    df, test_size=0.20, stratify=df['label_id'], random_state=42)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df['label_id'], random_state=42)

print(f"Train : {len(train_df)} | Val : {len(val_df)} | Test : {len(test_df)}")
print(f"Train label dist: {Counter(train_df['label'])}")


# ============================================================
# CELL 6 — Tokenize with HuggingFace Dataset
# ============================================================
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"  # BERT-tiny ~17MB
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

MAX_LEN = 32  # receipts are short; 32 tokens covers 99% of texts efficiently

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )

def make_dataset(df_):
    ds = Dataset.from_dict({
        "text":  df_["text"].tolist(),
        "label": df_["label_id"].astype(int).tolist(),
    })
    return ds.map(tokenize, batched=True, remove_columns=["text"])

train_ds = make_dataset(train_df)
val_ds   = make_dataset(val_df)
test_ds  = make_dataset(test_df)

train_ds.set_format("torch")
val_ds.set_format("torch")
test_ds.set_format("torch")

print("✅ Tokenization done")
print("  Train features:", train_ds.features)


# ============================================================
# CELL 7 — Compute class weights (handle 73/27 imbalance)
# ============================================================
import torch

label_counts = Counter(train_df['label_id'])
total        = sum(label_counts.values())

# Weight = total / (n_classes * count_of_class)
weights = torch.tensor([
    total / (2 * label_counts[0]),   # not_food (minority)
    total / (2 * label_counts[1]),   # food (majority)
], dtype=torch.float)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = weights.to(device)

print("Class weights:", weights)
print("  not_food weight:", weights[0].item())
print("  food weight    :", weights[1].item())


# ============================================================
# CELL 8 — Custom Trainer with weighted cross-entropy loss
# ============================================================
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch.nn as nn

class WeightedTrainer(Trainer):
    """Trainer that uses class-weighted loss to handle label imbalance."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        loss    = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ============================================================
# CELL 9 — Define metrics
# ============================================================
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report
)
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":          accuracy_score(labels, preds),
        "f1_macro":          f1_score(labels, preds, average="macro"),
        "f1_food":           f1_score(labels, preds, pos_label=1, average="binary"),
        "f1_not_food":       f1_score(labels, preds, pos_label=0, average="binary"),
        "precision_macro":   precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro":      recall_score(labels, preds, average="macro"),
    }


# ============================================================
# CELL 10 — Load model & set up training
# ============================================================
OUTPUT_DIR = os.path.join(DRIVE_DIR, "bert_food_classifier")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)
model = model.to(device)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # ── Training schedule ──────────────────────────────────
    num_train_epochs=8,           # 8 epochs — enough for tiny model on this size
    learning_rate=3e-5,           # slightly lower than default; better convergence with noise
    lr_scheduler_type="cosine",   # cosine decay — better than linear for this task
    warmup_ratio=0.1,             # 10% warmup steps

    # ── Batch sizes ────────────────────────────────────────
    per_device_train_batch_size=64,   # T4 can handle 64 for tiny model
    per_device_eval_batch_size=128,

    # ── Regularization ─────────────────────────────────────
    weight_decay=0.01,                # L2 regularization
    max_grad_norm=1.0,                # gradient clipping

    # ── Evaluation & saving ────────────────────────────────
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    save_total_limit=2,               # keep only 2 checkpoints (saves Drive space)

    # ── Logging ────────────────────────────────────────────
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=50,
    report_to="none",                 # disable wandb

    # ── Speed ──────────────────────────────────────────────
    fp16=torch.cuda.is_available(),   # mixed precision on GPU → faster
    dataloader_num_workers=2,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

print("✅ Trainer ready. Model parameters:", sum(p.numel() for p in model.parameters()))


# ============================================================
# CELL 11 — Train
# ============================================================
print("🚀 Starting training...")
trainer.train()
print("✅ Training complete!")


# ============================================================
# CELL 12 — Evaluate on test set
# ============================================================
from sklearn.metrics import classification_report
import numpy as np

print("\n📊 Test Set Evaluation:")
test_results = trainer.predict(test_ds)
preds  = np.argmax(test_results.predictions, axis=-1)
labels = test_results.label_ids

print(classification_report(
    labels, preds,
    target_names=["not_food", "food"],
    digits=4
))


# ============================================================
# CELL 13 — Save model & tokenizer to Drive
# ============================================================
SAVE_DIR = os.path.join(DRIVE_DIR, "bert_food_final")
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"✅ Model saved to Drive: {SAVE_DIR}")
print("Files:", os.listdir(SAVE_DIR))


# ============================================================
# CELL 14 — Quick inference test
# ============================================================
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model=SAVE_DIR,
    tokenizer=SAVE_DIR,
    device=0 if torch.cuda.is_available() else -1,
)

test_items = [
    "Organic Baby Spinach",         # food
    "Whole Milk 1 Gallon",          # food
    "Ziplock Sandwich Bags",        # not_food
    "127.20",                       # not_food (price)
    "Uncured Applewood Smoked Bacon", # food
    "Wt 1.89 lbs @ $19.50/lb",      # not_food
    "Beer",                          # food
    "Trash Bags 30pk",               # not_food
]

print("\n🧪 Inference test:")
results = classifier(test_items)
for item, res in zip(test_items, results):
    label = res['label']
    score = res['score']
    icon  = "🍎" if label == "food" else "📦"
    print(f"  {icon} [{label:8s}] {score:.3f}  →  {item}")


# ============================================================
# CELL 15 — Export model size info
# ============================================================
import os

total_size = 0
for root, dirs, files in os.walk(SAVE_DIR):
    for f in files:
        total_size += os.path.getsize(os.path.join(root, f))

print(f"\n📦 Saved model size: {total_size / 1024 / 1024:.1f} MB")
print("Files:")
for f in os.listdir(SAVE_DIR):
    fpath = os.path.join(SAVE_DIR, f)
    print(f"  {f}: {os.path.getsize(fpath)/1024:.1f} KB")
