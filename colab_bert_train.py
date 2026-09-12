"""
BERT Food/Not-Food Classifier
Run this ENTIRE FILE in one Colab cell.
"""

# ── Step 1: Install ─────────────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "-q", "-U",
                "transformers", "datasets", "scikit-learn", "accelerate", "packaging"],
               check=True)

# ── Step 2: Mount Drive ─────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_DIR = "/content/drive/MyDrive/receipt_bert"
os.makedirs(DRIVE_DIR, exist_ok=True)
print("✅ Drive mounted →", DRIVE_DIR)

# ── Step 3: Import everything ───────────────────────────────
import torch
import transformers
import numpy as np
import pandas as pd
from collections import Counter
from packaging.version import Version
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, classification_report)
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
import torch.nn as nn

print("✅ PyTorch      :", torch.__version__)
print("✅ Transformers :", transformers.__version__)
print("✅ CUDA         :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("   GPU         :", torch.cuda.get_device_name(0))

# ── Step 4: Upload CSV (popup will appear) ──────────────────
from google.colab import files as colab_files
import shutil

CSV_PATH = os.path.join(DRIVE_DIR, "final_dataset_augmented_v2.csv")
if not os.path.exists(CSV_PATH):
    print("Upload final_dataset_augmented_v2.csv when prompted...")
    uploaded = colab_files.upload()
    for fname in uploaded:
        shutil.copy(fname, CSV_PATH)
    print(f"✅ Saved to: {CSV_PATH}")
else:
    print(f"✅ CSV already in Drive: {CSV_PATH}")

# ── Step 5: Load dataset ────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'].str.len() >= 2].reset_index(drop=True)
print(f"\nTotal rows : {len(df)}")
print(f"Label dist : {Counter(df['label'])}")

# ── Step 6: Split ───────────────────────────────────────────
label2id = {"food": 1, "not_food": 0}
id2label = {v: k for k, v in label2id.items()}
df['label_id'] = df['label'].map(label2id)

train_df, temp_df = train_test_split(df, test_size=0.20, stratify=df['label_id'], random_state=42)
val_df,   test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label_id'], random_state=42)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── Step 7: Tokenize ────────────────────────────────────────
MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN    = 32

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

def make_dataset(df_):
    ds = Dataset.from_dict({"text": df_["text"].tolist(), "label": df_["label_id"].astype(int).tolist()})
    return ds.map(tokenize, batched=True, remove_columns=["text"])

train_ds = make_dataset(train_df)
val_ds   = make_dataset(val_df)
test_ds  = make_dataset(test_df)
train_ds.set_format("torch")
val_ds.set_format("torch")
test_ds.set_format("torch")
print("✅ Tokenization done")

# ── Step 8: Class weights ───────────────────────────────────
label_counts = Counter(train_df['label_id'])
total        = sum(label_counts.values())
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = torch.tensor([
    total / (2 * label_counts[0]),
    total / (2 * label_counts[1]),
], dtype=torch.float).to(device)
print(f"Class weights → not_food: {weights[0]:.3f} | food: {weights[1]:.3f}")

# ── Step 9: Weighted Trainer ────────────────────────────────
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        loss    = nn.CrossEntropyLoss(weight=weights)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# ── Step 10: Metrics ────────────────────────────────────────
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

# ── Step 11: Model + TrainingArguments ──────────────────────
OUTPUT_DIR = os.path.join(DRIVE_DIR, "bert_food_classifier")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id
).to(device)

# Version-safe eval strategy (renamed in transformers 4.46)
_eval_key = "eval_strategy" if Version(transformers.__version__) >= Version("4.46.0") else "evaluation_strategy"
print(f"Using '{_eval_key}' for eval strategy")

STEPS_PER_EPOCH = len(train_ds) // 64
TOTAL_STEPS     = STEPS_PER_EPOCH * 8
WARMUP_STEPS    = int(TOTAL_STEPS * 0.10)

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

# ── Step 12: Train ──────────────────────────────────────────
print("\n🚀 Starting training...")
trainer.train()
print("✅ Training complete!")

# ── Step 13: Evaluate ───────────────────────────────────────
print("\n📊 Test Set Evaluation:")
test_results = trainer.predict(test_ds)
preds  = np.argmax(test_results.predictions, axis=-1)
labels = test_results.label_ids
print(classification_report(labels, preds, target_names=["not_food", "food"], digits=4))

# ── Step 14: Save model ─────────────────────────────────────
SAVE_DIR = os.path.join(DRIVE_DIR, "bert_food_final")
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"✅ Model saved → {SAVE_DIR}")

# ── Step 15: Quick inference test ───────────────────────────
from transformers import pipeline
classifier = pipeline("text-classification", model=SAVE_DIR, tokenizer=SAVE_DIR,
                      device=0 if torch.cuda.is_available() else -1)

test_items = ["Organic Baby Spinach", "Whole Milk 1 Gallon", "Ziplock Sandwich Bags",
              "127.20", "Uncured Applewood Smoked Bacon", "Wt 1.89 lbs @ $19.50/lb",
              "Beer", "Trash Bags 30pk"]

print("\n🧪 Inference test:")
for item, res in zip(test_items, classifier(test_items)):
    icon = "🍎" if res['label'] == "food" else "📦"
    print(f"  {icon} [{res['label']:8s}] {res['score']:.3f}  →  {item}")

# ── Step 16: Download model files ───────────────────────────
total_size = sum(os.path.getsize(os.path.join(r, f))
                 for r, _, fs in os.walk(SAVE_DIR) for f in fs)
print(f"\n📦 Model size: {total_size/1024/1024:.1f} MB")

colab_files.download(os.path.join(SAVE_DIR, "model.safetensors"))
colab_files.download(os.path.join(SAVE_DIR, "config.json"))
colab_files.download(os.path.join(SAVE_DIR, "tokenizer.json"))
colab_files.download(os.path.join(SAVE_DIR, "tokenizer_config.json"))
print("✅ Done! Downloads started.")
