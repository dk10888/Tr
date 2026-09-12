"""
BERT-mini Food/Not-Food Classifier — Optimized Colab Script
=============================================================
Dataset  : final_dataset_augmented_v5.csv (62,839 rows, 53/47 balanced)
Model    : google/bert_uncased_L-4_H-256_A-4 (BERT-mini, 11M params, ~45MB)
Upgrades : label smoothing, tuned LR, cosine schedule, balanced weights,
           longer token window for richer v5 text
Runtime  : Colab T4 GPU (~8-12 min)

Paste this ENTIRE script into ONE Colab cell and run.
"""

# ── Step 1: Install ─────────────────────────────────────────────────────────
import subprocess
subprocess.run(["pip", "install", "-q", "-U",
                "transformers", "datasets", "scikit-learn", "accelerate", "packaging"],
               check=True)

# ── Step 2: Imports ─────────────────────────────────────────────────────────
import os, torch, transformers, numpy as np, pandas as pd, shutil
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

# ── Step 3: Mount Drive ─────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

DRIVE_DIR = "/content/drive/MyDrive/receipt_bert"
os.makedirs(DRIVE_DIR, exist_ok=True)
print("✅ Drive mounted →", DRIVE_DIR)

# ── Step 4: Upload CSV (skipped if already in Drive) ────────────────────────
CSV_FILENAME = "final_dataset_augmented_v5.csv"
CSV_PATH     = os.path.join(DRIVE_DIR, CSV_FILENAME)

if not os.path.exists(CSV_PATH):
    print(f"'{CSV_FILENAME}' not found in Drive. Upload it now...")
    from google.colab import files as colab_files
    uploaded = colab_files.upload()
    for fname in uploaded:
        shutil.copy(fname, CSV_PATH)
    print(f"✅ Saved to: {CSV_PATH}")
else:
    print(f"✅ Found CSV in Drive: {CSV_PATH}")

# ── Step 5: Load & clean dataset ────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'].str.len() >= 2].reset_index(drop=True)
df = df.dropna(subset=['label'])

label_dist = Counter(df['label'])
print(f"\n📊 Dataset loaded:")
print(f"   Total rows : {len(df)}")
print(f"   food       : {label_dist['food']}  ({label_dist['food']/len(df)*100:.1f}%)")
print(f"   not_food   : {label_dist['not_food']}  ({label_dist['not_food']/len(df)*100:.1f}%)")

# ── Step 6: Stratified 80/10/10 split ───────────────────────────────────────
label2id = {"food": 1, "not_food": 0}
id2label = {v: k for k, v in label2id.items()}
df['label_id'] = df['label'].map(label2id)

train_df, temp_df = train_test_split(df, test_size=0.20, stratify=df['label_id'], random_state=42)
val_df,   test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label_id'], random_state=42)
print(f"\n   Train : {len(train_df)} | Val : {len(val_df)} | Test : {len(test_df)}")

# ── Step 7: Tokenize ─────────────────────────────────────────────────────────
# v5 has longer text (median 30 chars vs 15 in v2) → use MAX_LEN=64
MODEL_NAME = "google/bert_uncased_L-4_H-256_A-4"   # BERT-mini: 11M params, ~45MB
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN    = 64   # increased from 32 to handle richer v5 text

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LEN)

def make_ds(df_):
    ds = Dataset.from_dict({
        "text":  df_["text"].tolist(),
        "label": df_["label_id"].astype(int).tolist(),
    })
    return ds.map(tokenize, batched=True, remove_columns=["text"])

train_ds = make_ds(train_df); train_ds.set_format("torch")
val_ds   = make_ds(val_df);   val_ds.set_format("torch")
test_ds  = make_ds(test_df);  test_ds.set_format("torch")
print(f"✅ Tokenized | MAX_LEN={MAX_LEN} | Model={MODEL_NAME}")

# ── Step 8: Class weights (v5 is near-balanced; weights are mild) ────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lc  = Counter(train_df['label_id'])
tot = sum(lc.values())
weights = torch.tensor([
    tot / (2 * lc[0]),   # not_food
    tot / (2 * lc[1]),   # food
], dtype=torch.float).to(device)
print(f"   Class weights → not_food: {weights[0]:.3f} | food: {weights[1]:.3f}")
print(f"   (Close to 1.0 = near-balanced ✅)")

# ── Step 9: Weighted + Label-Smoothed Trainer ────────────────────────────────
# Label smoothing prevents overconfidence on noisy OCR labels
LABEL_SMOOTHING = 0.1   # softens hard labels: 1.0 → 0.9, 0.0 → 0.1

class WeightedSmoothTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        n_cls   = logits.size(-1)

        # One-hot encode for label smoothing
        smooth_labels = torch.full_like(logits, LABEL_SMOOTHING / (n_cls - 1))
        smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - LABEL_SMOOTHING)

        # Apply class weights per sample
        sample_weights = weights[labels]
        log_probs      = torch.nn.functional.log_softmax(logits, dim=-1)
        loss_per_sample = -(smooth_labels * log_probs).sum(dim=-1)
        loss = (loss_per_sample * sample_weights).mean()

        return (loss, outputs) if return_outputs else loss

# ── Step 10: Metrics ─────────────────────────────────────────────────────────
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

# ── Step 11: Load model ──────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(DRIVE_DIR, "bert_mini_food_classifier")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id
).to(device)
print(f"✅ Model loaded | Params: {sum(p.numel() for p in model.parameters()):,}")

# ── Step 12: TrainingArguments (version-safe) ────────────────────────────────
_eval_key = "eval_strategy" if Version(transformers.__version__) >= Version("4.46.0") else "evaluation_strategy"

# Larger dataset (62K) → fewer epochs needed; converges faster
BATCH_SIZE      = 64
STEPS_PER_EPOCH = len(train_ds) // BATCH_SIZE
TOTAL_STEPS     = STEPS_PER_EPOCH * 6   # 6 epochs (was 8, larger data converges faster)
WARMUP_STEPS    = int(TOTAL_STEPS * 0.06)  # 6% warmup (tighter for bigger dataset)

print(f"\n⚙️  Training config:")
print(f"   Steps/epoch : {STEPS_PER_EPOCH}")
print(f"   Total steps : {TOTAL_STEPS}")
print(f"   Warmup steps: {WARMUP_STEPS}")
print(f"   Eval key    : '{_eval_key}'")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # ── Schedule ── Tuned for BERT-mini on 62K rows
    num_train_epochs=6,
    learning_rate=2e-5,          # lower than 3e-5 → better generalisation for bigger model
    lr_scheduler_type="cosine",  # cosine decay: smooth LR reduction
    warmup_steps=WARMUP_STEPS,

    # ── Batches ──
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=128,

    # ── Regularisation ──
    weight_decay=0.01,           # L2 penalty on weights
    max_grad_norm=1.0,           # gradient clipping

    # ── Eval & saving ──
    **{_eval_key: "epoch"},
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    save_total_limit=2,

    # ── Logging ──
    logging_steps=100,
    report_to="none",

    # ── Speed ──
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=2,
)

trainer = WeightedSmoothTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)
print("✅ Trainer ready")

# ── Step 13: Train ───────────────────────────────────────────────────────────
print("\n🚀 Training started...")
trainer.train()
print("✅ Training complete!")

# ── Step 14: Evaluate on test set ────────────────────────────────────────────
print("\n📊 Test Set Evaluation:")
test_results = trainer.predict(test_ds)
preds  = np.argmax(test_results.predictions, axis=-1)
labels_arr = test_results.label_ids
print(classification_report(labels_arr, preds, target_names=["not_food", "food"], digits=4))

# ── Step 15: Save model to Drive ─────────────────────────────────────────────
SAVE_DIR = os.path.join(DRIVE_DIR, "bert_mini_food_final")
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"✅ Model saved → {SAVE_DIR}")

# ── Step 16: Quick inference test ────────────────────────────────────────────
from transformers import pipeline
classifier = pipeline(
    "text-classification", model=SAVE_DIR, tokenizer=SAVE_DIR,
    device=0 if torch.cuda.is_available() else -1
)

test_items = [
    "Organic Baby Spinach",           # food
    "Whole Milk 1 Gallon",            # food
    "Ziplock Sandwich Bags",          # not_food
    "127.20",                         # not_food (price)
    "Uncured Applewood Smoked Bacon", # food
    "Wt 1.89 lbs @ $19.50/lb",       # not_food
    "Beer",                           # food
    "Trash Bags 30pk",                # not_food
    "RevlonBlushHoneyBeige 0.1OZ",    # not_food (hard negative)
    "NatureWise Vitamin C 60CT",      # not_food (hard negative)
    "Pepperidge Farm Bagels 6CT",     # food
    "HST 13%",                        # not_food
]

print("\n🧪 Inference test:")
for item, res in zip(test_items, classifier(test_items)):
    icon = "🍎" if res['label'] == "food" else "📦"
    flag = "" if (("Vitamin" in item or "Revlon" in item or "Bags" in item
                   or "127" in item or "lbs" in item or "HST" in item)
                  and res['label'] == "not_food") or \
                 (("Spinach" in item or "Milk" in item or "Bacon" in item
                   or "Beer" in item or "Bagels" in item)
                  and res['label'] == "food") else " ⚠️ CHECK"
    print(f"  {icon} [{res['label']:8s}] {res['score']:.3f}  →  {item}{flag}")

# ── Step 17: Model size + download ───────────────────────────────────────────
total_size = sum(
    os.path.getsize(os.path.join(r, f))
    for r, _, fs in os.walk(SAVE_DIR) for f in fs
)
print(f"\n📦 Model size: {total_size/1024/1024:.1f} MB")
print(f"   Saved at  : {SAVE_DIR}")

from google.colab import files as cf
cf.download(os.path.join(SAVE_DIR, "model.safetensors"))
cf.download(os.path.join(SAVE_DIR, "config.json"))
cf.download(os.path.join(SAVE_DIR, "tokenizer.json"))
cf.download(os.path.join(SAVE_DIR, "tokenizer_config.json"))
print("✅ Downloads started!")
