"""
BERT Food/Not-Food Classifier — Google Colab Training Script
=============================================================
Dataset  : final_dataset_augmented_v2.csv (22,030 rows)
Classes  : food (16136) | not_food (5894)
Model    : google/bert_uncased_L-2_H-128_A-2  (BERT-tiny, ~17MB)
Runtime  : Colab T4 GPU (~15-20 min total)

Run cells top to bottom. After disconnect: re-run Cell 1-4, then resume from Cell 5.
"""

# ============================================================
# CELL 1 — Mount Google Drive (run first, always)
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
import subprocess
subprocess.run(["pip", "install", "-q", "-U", "transformers", "datasets",
                "scikit-learn", "accelerate", "packaging"], check=True)

import torch
import transformers
print("✅ PyTorch      :", torch.__version__)
print("✅ Transformers :", transformers.__version__)
print("✅ CUDA         :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("   GPU         :", torch.cuda.get_device_name(0))


# ============================================================
# CELL 3 — Upload dataset to Drive (run once)
# ============================================================
from google.colab import files
import shutil

uploaded = files.upload()   # select final_dataset_augmented_v2.csv
for fname in uploaded:
    dest = os.path.join(DRIVE_DIR, fname)
    shutil.copy(fname, dest)
    print(f"✅ Saved to Drive: {dest}")

# If already in Drive, skip this cell — CSV_PATH is set in Cell 4.


# ============================================================
# CELL 4 — Load & inspect dataset
# ============================================================
import pandas as pd
from collections import Counter

CSV_PATH = os.path.join(DRIVE_DIR, "final_dataset_augmented_v2.csv")
df = pd.read_csv(CSV_PATH)
df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'].str.len() >= 2].reset_index(drop=True)

print(f"Total rows  : {len(df)}")
print(f"Label dist  : {Counter(df['label'])}")
print(f"Text length : min={df['text'].str.len().min()}  "
      f"max={df['text'].str.len().max()}  "
      f"median={df['text'].str.len().median():.0f}")


# ============================================================
# CELL 5 — Stratified 80/10/10 split
# ============================================================
from sklearn.model_selection import train_test_split

label2id = {"food": 1, "not_food": 0}
id2label = {v: k for k, v in label2id.items()}

df['label_id'] = df['label'].map(label2id)

train_df, temp_df = train_test_split(
    df, test_size=0.20, stratify=df['label_id'], random_state=42)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df['label_id'], random_state=42)

print(f"Train : {len(train_df)} | Val : {len(val_df)} | Test : {len(test_df)}")
print(f"Train label dist: {Counter(train_df['label'])}")


# ============================================================
# CELL 6 — Tokenize
# ============================================================
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN    = 32

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

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
print("✅ Tokenization done. Features:", train_ds.features)


# ============================================================
# CELL 7 — Class weights (handle 73/27 imbalance)
# ============================================================
import torch

label_counts = Counter(train_df['label_id'])
total        = sum(label_counts.values())
weights = torch.tensor([
    total / (2 * label_counts[0]),   # not_food (minority → higher weight)
    total / (2 * label_counts[1]),   # food (majority → lower weight)
], dtype=torch.float)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = weights.to(device)
print(f"Device: {device}")
print(f"Class weights → not_food: {weights[0]:.3f} | food: {weights[1]:.3f}")


# ============================================================
# CELL 8 — Weighted loss Trainer
# ============================================================
from transformers import Trainer
import torch.nn as nn

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        loss    = nn.CrossEntropyLoss(weight=weights)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# ============================================================
# CELL 9 — Metrics
# ============================================================
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":        accuracy_score(labels, preds),
        "f1_macro":        f1_score(labels, preds, average="macro"),
        "f1_food":         f1_score(labels, preds, pos_label=1, average="binary"),
        "f1_not_food":     f1_score(labels, preds, pos_label=0, average="binary"),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro":    recall_score(labels, preds, average="macro"),
    }


# ============================================================
# CELL 10 — Load model + TrainingArguments (version-safe)
# ============================================================
from transformers import AutoModelForSequenceClassification, TrainingArguments
from packaging.version import Version

OUTPUT_DIR = os.path.join(DRIVE_DIR, "bert_food_classifier")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id
)
model = model.to(device)

# Auto-detect correct eval strategy key (renamed in transformers 4.46)
_new_api  = Version(transformers.__version__) >= Version("4.46.0")
_eval_key = "eval_strategy" if _new_api else "evaluation_strategy"
print(f"Transformers {transformers.__version__} → using '{_eval_key}'")

# Compute warmup steps (works on all transformers versions)
STEPS_PER_EPOCH = len(train_ds) // 64
TOTAL_STEPS     = STEPS_PER_EPOCH * 8
WARMUP_STEPS    = int(TOTAL_STEPS * 0.10)
print(f"Total steps: {TOTAL_STEPS} | Warmup steps: {WARMUP_STEPS}")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=8,
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    weight_decay=0.01,
    max_grad_norm=1.0,
    **{_eval_key: "epoch"},
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    save_total_limit=2,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=50,
    report_to="none",
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=2,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

print("✅ Trainer ready. Params:", f"{sum(p.numel() for p in model.parameters()):,}")


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

print("\n📊 Test Set Evaluation:")
test_results = trainer.predict(test_ds)
preds  = np.argmax(test_results.predictions, axis=-1)
labels = test_results.label_ids

print(classification_report(labels, preds, target_names=["not_food", "food"], digits=4))


# ============================================================
# CELL 13 — Save model to Drive
# ============================================================
SAVE_DIR = os.path.join(DRIVE_DIR, "bert_food_final")
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"✅ Model saved → {SAVE_DIR}")
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
    "Organic Baby Spinach",
    "Whole Milk 1 Gallon",
    "Ziplock Sandwich Bags",
    "127.20",
    "Uncured Applewood Smoked Bacon",
    "Wt 1.89 lbs @ $19.50/lb",
    "Beer",
    "Trash Bags 30pk",
]

print("\n🧪 Inference test:")
for item, res in zip(test_items, classifier(test_items)):
    icon = "🍎" if res['label'] == "food" else "📦"
    print(f"  {icon} [{res['label']:8s}] {res['score']:.3f}  →  {item}")


# ============================================================
# CELL 15 — Model size + download
# ============================================================
total_size = sum(
    os.path.getsize(os.path.join(r, f))
    for r, _, files in os.walk(SAVE_DIR)
    for f in files
)
print(f"\n📦 Model size: {total_size / 1024 / 1024:.1f} MB")

from google.colab import files as colab_files
colab_files.download(os.path.join(SAVE_DIR, "model.safetensors"))
colab_files.download(os.path.join(SAVE_DIR, "config.json"))
colab_files.download(os.path.join(SAVE_DIR, "tokenizer.json"))
colab_files.download(os.path.join(SAVE_DIR, "tokenizer_config.json"))
print("✅ Downloads started!")
