"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  English Receipt Food Classifier — BERT-mini — v12 Dataset                  ║
║  Model   : google/bert_uncased_L-4_H-256_A-4  (11M params, ~42 MB FP32)    ║
║  INT8    : ~11 MB after quantization                                         ║
║  Dataset : final_dataset_augmented_v10.csv  (90,436 rows)                   ║
║  Runtime : Google Colab T4 GPU  (~10-12 min total)                          ║
║                                                                              ║
║  HOW TO USE:                                                                 ║
║    1. Colab → Runtime → Change runtime type → T4 GPU                        ║
║    2. Paste this ENTIRE script into one cell and run                         ║
║    3. Upload  final_dataset_augmented_v10.csv  when prompted                 ║
║    4. All outputs auto-saved to Google Drive                                 ║
║                                                                              ║
║  Outputs → Drive/receipt_bert/                                               ║
║    english_food_bert/           ← HuggingFace model (PyTorch)               ║
║    english_food_bert.onnx       ← FP32 ONNX (~42 MB) for Android            ║
║    english_food_bert_quant.onnx ← INT8 quantized (~11 MB) for Android       ║
║    english_vocab.txt            ← vocab for Android tokenizer                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Install
# ─────────────────────────────────────────────────────────────────────────────
import sys, subprocess, os, shutil, json, time, random

def pip_install(*pkgs):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-U"] + list(pkgs)
    )

pip_install("transformers", "datasets", "scikit-learn",
            "accelerate", "packaging", "onnx", "onnxruntime")
print("✅ Packages ready")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Imports
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from collections import Counter
from packaging.version import Version
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, classification_report)
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

print(f"✅ PyTorch      : {torch.__version__}")
print(f"✅ Transformers : {transformers.__version__}")
print(f"✅ CUDA         : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU         : {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Mount Drive
# ─────────────────────────────────────────────────────────────────────────────
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    from google.colab import drive, files as colab_files
    drive.mount("/content/drive")
    DRIVE_DIR = "/content/drive/MyDrive/receipt_bert"
else:
    DRIVE_DIR = "."

os.makedirs(DRIVE_DIR, exist_ok=True)
print(f"✅ Drive dir → {DRIVE_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Upload / find dataset
# ─────────────────────────────────────────────────────────────────────────────
CSV_FILENAME = "final_dataset_augmented_v10.csv"
CSV_PATH     = os.path.join(DRIVE_DIR, CSV_FILENAME)

if not os.path.exists(CSV_PATH):
    print(f"⬆  Upload  '{CSV_FILENAME}'  now...")
    if IN_COLAB:
        uploaded = colab_files.upload()
        for fname in uploaded:
            shutil.copy(fname, CSV_PATH)
            print(f"✅ Saved → {CSV_PATH}")
    else:
        raise FileNotFoundError(f"Place {CSV_FILENAME} at {CSV_PATH}")
else:
    print(f"✅ Found: {CSV_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Config
# ─────────────────────────────────────────────────────────────────────────────
# BERT-mini: 4 layers, 256 hidden, 4 heads — 11M params
# Much more capacity than BERT-tiny (2 layers, 128 hidden) for 90k dataset
MODEL_NAME  = "google/bert_uncased_L-4_H-256_A-4"
MAX_LEN     = 64
BATCH_SIZE  = 32     # smaller batch because model is bigger than tiny
EPOCHS      = 10
LR          = 3e-5
SEED        = 42

SAVE_DIR    = os.path.join(DRIVE_DIR, "english_food_bert")
ONNX_FP32   = os.path.join(DRIVE_DIR, "english_food_bert.onnx")
ONNX_QUANT  = os.path.join(DRIVE_DIR, "english_food_bert_quant.onnx")
VOCAB_OUT   = os.path.join(DRIVE_DIR, "english_vocab.txt")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

label2id = {"not_food": 0, "food": 1}
id2label = {0: "not_food", 1: "food"}
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Load & clean
# ─────────────────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(CSV_PATH)
except Exception:
    print("⚠️ Standard pandas C parser failed. Loading full dataset via csv parser...")
    with open(CSV_PATH, "r", encoding="utf-8", errors="replace") as f:
        df = pd.DataFrame(list(csv.DictReader(f)))
df["text"]  = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() >= 2].dropna(subset=["text", "label"]).reset_index(drop=True)
df["label_id"] = df["label"].map(label2id)

print(f"\n📊 Dataset")
print(f"   Total rows  : {len(df):,}")
print(f"   Label dist  : {Counter(df['label'])}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Stratified 90/10 split (No Test Data)
# ─────────────────────────────────────────────────────────────────────────────
train_df, val_df = train_test_split(df, test_size=0.10, stratify=df["label_id"], random_state=SEED)

print(f"   Train : {len(train_df):,}  |  Val : {len(val_df):,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Tokenize
# ─────────────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True,
                     padding="max_length", max_length=MAX_LEN)

def make_hf_dataset(df_):
    ds = Dataset.from_dict({
        "text":  df_["text"].tolist(),
        "label": df_["label_id"].astype(int).tolist(),
    })
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    return ds

train_ds = make_hf_dataset(train_df)
val_ds   = make_hf_dataset(val_df)
print("✅ Tokenization complete")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Class-weighted loss
# ─────────────────────────────────────────────────────────────────────────────
label_counts  = Counter(train_df["label_id"])
total         = sum(label_counts.values())
class_weights = torch.tensor([
    total / (2 * label_counts[0]),
    total / (2 * label_counts[1]),
], dtype=torch.float).to(device)

print(f"   Weights → not_food: {class_weights[0]:.3f} | food: {class_weights[1]:.3f}")

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        loss    = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=0.05
        )(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Metrics
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — Model + TrainingArguments
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR_TMP = "/content/bert_mini_checkpoints"
os.makedirs(OUTPUT_DIR_TMP, exist_ok=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id
).to(device)

# Count params
total_params = sum(p.numel() for p in model.parameters())
print(f"   Model params : {total_params/1e6:.1f}M")

_new_api  = Version(transformers.__version__) >= Version("4.46.0")
_eval_key = "eval_strategy" if _new_api else "evaluation_strategy"

STEPS_PER_EPOCH = len(train_ds) // BATCH_SIZE
WARMUP_STEPS    = int(STEPS_PER_EPOCH * EPOCHS * 0.06)

training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR_TMP,
    num_train_epochs            = EPOCHS,
    learning_rate               = LR,
    lr_scheduler_type           = "cosine",
    warmup_steps                = WARMUP_STEPS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = 64,
    weight_decay                = 0.01,
    max_grad_norm               = 1.0,
    **{_eval_key: "epoch"},
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "f1_macro",
    greater_is_better           = True,
    save_total_limit            = 2,
    logging_steps               = 200,
    report_to                   = "none",
    fp16                        = torch.cuda.is_available(),
    dataloader_num_workers      = 2,
)

trainer = WeightedTrainer(
    model=model, args=training_args,
    train_dataset=train_ds, eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 — Train
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n🚀 Training BERT-mini on {len(train_df):,} rows  ({EPOCHS} epochs) …")
t0 = time.time()
trainer.train()
print(f"✅ Training done in {(time.time()-t0)/60:.1f} min")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 13 — Validation-set evaluation
# ─────────────────────────────────────────────────────────────────────────────
print("\n📊 Validation Set Evaluation:")
val_results = trainer.predict(val_ds)
preds  = np.argmax(val_results.predictions, axis=-1)
labels = val_results.label_ids
print(classification_report(labels, preds,
      target_names=["not_food", "food"], digits=4))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 14 — Save HF model to Drive
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(SAVE_DIR, exist_ok=True)
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

vocab_src = os.path.join(SAVE_DIR, "vocab.txt")
if os.path.exists(vocab_src):
    shutil.copy(vocab_src, VOCAB_OUT)
print(f"✅ Model saved → {SAVE_DIR}")
print(f"✅ Vocab saved → {VOCAB_OUT}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 15 — ONNX FP32 export (legacy TorchScript exporter — Android compatible)
# ─────────────────────────────────────────────────────────────────────────────
print("\n📦 Exporting ONNX FP32 …")

# Move to CPU — required for reliable ONNX export
model_cpu = trainer.model.cpu().eval()

dummy_ids   = torch.zeros(1, MAX_LEN, dtype=torch.long)
dummy_mask  = torch.ones(1,  MAX_LEN, dtype=torch.long)
dummy_ttype = torch.zeros(1, MAX_LEN, dtype=torch.long)

with torch.no_grad():
    torch.onnx.export(
        model_cpu,
        (dummy_ids, dummy_mask, dummy_ttype),
        ONNX_FP32,
        input_names  = ["input_ids", "attention_mask", "token_type_ids"],
        output_names = ["logits"],
        dynamic_axes = {
            "input_ids":      {0: "batch"},
            "attention_mask": {0: "batch"},
            "token_type_ids": {0: "batch"},
            "logits":         {0: "batch"},
        },
        opset_version       = 17,
        do_constant_folding = True,
        dynamo              = False,   # ← force legacy TorchScript exporter
    )

size_fp32 = os.path.getsize(ONNX_FP32) / 1024 / 1024
print(f"✅ FP32 ONNX → {ONNX_FP32}  ({size_fp32:.1f} MB)")

if size_fp32 < 5:
    print("⚠  WARNING: ONNX file is suspiciously small — dynamo exporter may have been used.")
    print("   Re-run export cell or check torch version.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 16 — INT8 quantization
# ─────────────────────────────────────────────────────────────────────────────
print("\n📦 Quantizing to INT8 …")
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(ONNX_FP32, ONNX_QUANT, weight_type=QuantType.QInt8)
    size_q = os.path.getsize(ONNX_QUANT) / 1024 / 1024
    print(f"✅ INT8 ONNX → {ONNX_QUANT}  ({size_q:.1f} MB)")
except Exception as e:
    size_q = 0
    print(f"⚠  Quantization error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 17 — ONNX sanity check
# ─────────────────────────────────────────────────────────────────────────────
print("\n🧪 ONNX sanity check …")
import onnxruntime as ort

test_cases = [
    ("Chicken breast 500g",     "food"),
    ("Organic free range eggs",  "food"),
    ("Starbucks Cold Brew",      "food"),
    ("Walmart Supercenter",      "not_food"),
    ("Cherry Hill NJ",           "not_food"),
    ("Strawberry Shampoo",       "not_food"),
    ("Thank you for shopping",   "not_food"),
    ("Debit Card Total",         "not_food"),
]

ort_sess = ort.InferenceSession(ONNX_FP32, providers=["CPUExecutionProvider"])
all_ok = True
for text, expected in test_cases:
    enc = tokenizer(text, return_tensors="np", max_length=MAX_LEN,
                    truncation=True, padding="max_length")
    logits = ort_sess.run(["logits"], {
        "input_ids":      enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
        "token_type_ids": enc.get("token_type_ids",
                          np.zeros_like(enc["input_ids"])).astype(np.int64),
    })[0]
    pred = id2label[int(np.argmax(logits))]
    ok   = "✅" if pred == expected else "❌"
    if pred != expected: all_ok = False
    print(f"   {ok}  '{text}'  → {pred}")

print("   All passed! ✅" if all_ok else "   Some failed — review dataset or model.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 18 — Final summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING COMPLETE — BERT-mini English Food Classifier")
print("=" * 60)
print(f"  Dataset rows  : {len(df):,}")
print(f"  Train/Val/Test: {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")
print(f"  Model params  : {total_params/1e6:.1f}M  ({MODEL_NAME})")
print(f"  FP32 ONNX     : {size_fp32:.1f} MB  → {ONNX_FP32}")
try:
    print(f"  INT8 ONNX     : {size_q:.1f} MB   → {ONNX_QUANT}")
except: pass
print(f"  Vocab file    : {VOCAB_OUT}")
print("=" * 60)
print("\nCopy to Android app assets/:")
print("  english_food_bert_quant.onnx  (~11 MB, recommended)")
print("  english_vocab.txt")
