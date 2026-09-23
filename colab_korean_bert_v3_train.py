"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Korean Receipt Food Classifier — KoELECTRA — v3 Dataset                    ║
║  Model   : monologg/koelectra-small-v3-discriminator  (~14M params)         ║
║  Dataset : korean_receipt_dataset_full.csv  (49,061 rows)                   ║
║  Runtime : Google Colab T4 GPU  (~12-18 min total)                          ║
║                                                                              ║
║  HOW TO USE:                                                                 ║
║    1. Open Colab → Runtime → Change runtime type → T4 GPU                   ║
║    2. Paste this ENTIRE script into one cell and run                         ║
║    3. When prompted, upload  korean_receipt_dataset_full.csv                 ║
║    4. Wait for training + ONNX export to finish                              ║
║    5. All outputs saved to Google Drive                                      ║
║                                                                              ║
║  Outputs saved to Drive /receipt_bert/:                                      ║
║    korean_food_bert/               ← HuggingFace model (PyTorch)             ║
║    korean_food_classifier.onnx     ← FP32 ONNX (~55 MB)                     ║
║    korean_food_classifier_quant.onnx ← INT8 quantized (~14 MB)              ║
║    korean_vocab.txt                ← vocab for Android tokenizer             ║
║    korean_special_tokens.json      ← label map + tokenizer meta              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Install packages
# ─────────────────────────────────────────────────────────────────────────────
import sys, subprocess, os, shutil, json, time, random

def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U"] + list(pkgs))

pip_install("transformers", "datasets", "scikit-learn", "accelerate",
            "packaging", "onnx", "onnxruntime", "onnxscript", "evaluate")
print("✅ Packages ready")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Imports
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from packaging.version import Version
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, classification_report, confusion_matrix)
import transformers
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

print(f"✅ PyTorch      : {torch.__version__}")
print(f"✅ Transformers : {transformers.__version__}")
print(f"✅ CUDA         : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU         : {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Mount Google Drive
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
# STEP 4 — Load dataset (upload if not in Drive)
# ─────────────────────────────────────────────────────────────────────────────
CSV_FILENAME = "korean_receipt_dataset_full.csv"
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
MODEL_NAME  = "monologg/koelectra-small-v3-discriminator"
MAX_LEN     = 64
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 3e-5
SEED        = 42

SAVE_DIR    = os.path.join(DRIVE_DIR, "korean_food_bert")
ONNX_FP32   = os.path.join(DRIVE_DIR, "korean_food_classifier.onnx")
ONNX_QUANT  = os.path.join(DRIVE_DIR, "korean_food_classifier_quant.onnx")
VOCAB_OUT   = os.path.join(DRIVE_DIR, "korean_vocab.txt")
META_OUT    = os.path.join(DRIVE_DIR, "korean_special_tokens.json")
OUTPUT_DIR_TMP = "/content/korean_checkpoints"

label2id = {"not_food": 0, "food": 1}
id2label = {0: "not_food", 1: "food"}

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)
os.makedirs(OUTPUT_DIR_TMP, exist_ok=True)
os.makedirs(SAVE_DIR,       exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device : {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Load & clean dataset
# ─────────────────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(CSV_PATH)
except Exception:
    print("⚠️ Standard pandas C parser failed. Loading full dataset via csv parser...")
    with open(CSV_PATH, "r", encoding="utf-8", errors="replace") as f:
        df = pd.DataFrame(list(csv.DictReader(f)))

df["text"]  = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() >= 1].dropna(subset=["text", "label"]).reset_index(drop=True)
df["label_id"] = df["label"].map(label2id)

print(f"\n📊 Dataset loaded")
print(f"   Total rows  : {len(df):,}")
print(f"   Label dist  : {Counter(df['label'])}")
if "source" in df.columns:
    print(f"   Sources     : {df['source'].nunique()} unique sources")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Stratified 90/10 split (No test data)
# ─────────────────────────────────────────────────────────────────────────────
train_df, val_df = train_test_split(df, test_size=0.10, stratify=df["label_id"], random_state=SEED)

print(f"   Train : {len(train_df):,}  |  Val : {len(val_df):,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Tokenizer & PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class KoreanReceiptDataset(Dataset):
    def __init__(self, df_, tokenizer_, max_len):
        self.texts  = df_["text"].tolist()
        self.labels = df_["label_id"].tolist()
        self.tok    = tokenizer_
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids",
                              torch.zeros(self.max_len, dtype=torch.long)).squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

train_ds = KoreanReceiptDataset(train_df, tokenizer, MAX_LEN)
val_ds   = KoreanReceiptDataset(val_df,   tokenizer, MAX_LEN)

NUM_WORKERS = 2 if IN_COLAB else 0
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"✅ Tokenization complete  |  Train batches: {len(train_dl)}  |  Val batches: {len(val_dl)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Model + optimizer + scheduler
# ─────────────────────────────────────────────────────────────────────────────
label_counts  = Counter(train_df["label_id"])
total_samples = sum(label_counts.values())
class_weights = torch.tensor([
    total_samples / (2 * label_counts[0]),  # not_food
    total_samples / (2 * label_counts[1]),  # food
], dtype=torch.float).to(DEVICE)

print(f"   Class weights → not_food: {class_weights[0]:.3f} | food: {class_weights[1]:.3f}")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
).to(DEVICE)

criterion  = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps   = len(train_dl) * EPOCHS
warmup_steps  = int(total_steps * 0.06)
scheduler  = get_linear_schedule_with_warmup(optimizer,
             num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model_, loader):
    model_.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(DEVICE)
            mask  = batch["attention_mask"].to(DEVICE)
            ttype = batch["token_type_ids"].to(DEVICE)
            lbl   = batch["labels"].to(DEVICE)
            out   = model_(input_ids=ids, attention_mask=mask, token_type_ids=ttype)
            preds = out.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())
    acc      = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_food  = f1_score(all_labels, all_preds, pos_label=1, average="binary")
    f1_nf    = f1_score(all_labels, all_preds, pos_label=0, average="binary")
    return {"acc": acc, "f1_macro": f1_macro, "f1_food": f1_food, "f1_not_food": f1_nf}, all_preds, all_labels

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — Training loop
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n🚀 Training KoELECTRA on {len(train_df):,} rows  ({EPOCHS} epochs) …\n")

best_val_f1  = 0.0
best_ckpt    = os.path.join(OUTPUT_DIR_TMP, "best_checkpoint")
history      = []
t0           = time.time()

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for step, batch in enumerate(train_dl):
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        ttype = batch["token_type_ids"].to(DEVICE)
        lbl   = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out  = model(input_ids=ids, attention_mask=mask, token_type_ids=ttype)
            loss = criterion(out.logits, lbl)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        scheduler.step()

        preds = out.logits.argmax(dim=-1)
        train_correct += (preds == lbl).sum().item()
        train_total   += lbl.size(0)
        train_loss    += loss.item()

        if (step + 1) % 100 == 0 or (step + 1) == len(train_dl):
            print(f"   Epoch {epoch+1:02d}/{EPOCHS} | Step {step+1:04d}/{len(train_dl)} "
                  f"| Loss {train_loss/(step+1):.4f} | TrainAcc {train_correct/train_total*100:.2f}%",
                  flush=True)

    val_metrics, _, _ = evaluate(model, val_dl)
    history.append({
        "epoch":        epoch + 1,
        "train_loss":   train_loss / len(train_dl),
        "val_acc":      val_metrics["acc"],
        "val_f1_macro": val_metrics["f1_macro"],
        "val_f1_food":  val_metrics["f1_food"],
    })
    print(f"\n✅ Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Val Acc: {val_metrics['acc']*100:.2f}% | "
          f"Val F1: {val_metrics['f1_macro']*100:.2f}% "
          f"(food: {val_metrics['f1_food']*100:.1f}% | "
          f"not_food: {val_metrics['f1_not_food']*100:.1f}%)\n")

    if val_metrics["f1_macro"] > best_val_f1:
        best_val_f1 = val_metrics["f1_macro"]
        model.save_pretrained(best_ckpt)
        tokenizer.save_pretrained(best_ckpt)
        print(f"   💾 New best checkpoint saved (F1: {best_val_f1*100:.2f}%)")

print(f"\n✅ Training done in {(time.time()-t0)/60:.1f} min")
print(f"   Best Val F1 Macro: {best_val_f1*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 — Load best checkpoint + validation-set evaluation
# ─────────────────────────────────────────────────────────────────────────────
print("\n📊 Loading best checkpoint for final evaluation …")
model = AutoModelForSequenceClassification.from_pretrained(best_ckpt).to(DEVICE)
val_metrics, val_preds, val_labels = evaluate(model, val_dl)

print("\n" + "=" * 60)
print("VALIDATION SET RESULTS")
print("=" * 60)
print(classification_report(val_labels, val_preds,
      target_names=["not_food", "food"], digits=4))

print("Confusion Matrix:")
cm = confusion_matrix(val_labels, val_preds)
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 13 — Save HuggingFace model to Drive
# ─────────────────────────────────────────────────────────────────────────────
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

# Copy vocab
vocab_src = os.path.join(best_ckpt, "vocab.txt")
if os.path.exists(vocab_src):
    shutil.copy(vocab_src, VOCAB_OUT)

# Save metadata JSON for Android
meta = {
    "model_name":   MODEL_NAME,
    "max_len":      MAX_LEN,
    "label2id":     label2id,
    "id2label":     id2label,
    "dataset_rows": len(df),
    "train_rows":   len(train_df),
    "val_rows":     len(val_df),
    "best_val_f1":  round(best_val_f1, 4),
    "val_acc":      round(val_metrics["acc"], 4),
    "val_f1_macro": round(val_metrics["f1_macro"], 4),
}
with open(META_OUT, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\n✅ Model saved → {SAVE_DIR}")
print(f"✅ Vocab saved → {VOCAB_OUT}")
print(f"✅ Meta  saved → {META_OUT}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 14 — Export FP32 ONNX (legacy TorchScript exporter — Android compatible)
# ─────────────────────────────────────────────────────────────────────────────
print("\n📦 Exporting FP32 ONNX …")

# Move to CPU — required for reliable cross-platform ONNX export
model_cpu = model.cpu().eval()

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

if size_fp32 < 10:
    print("⚠  WARNING: ONNX file is suspiciously small — dynamo exporter may have fired.")
    print("   Expected ~55 MB for KoELECTRA-small. Check torch version.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 15 — INT8 quantization
# ─────────────────────────────────────────────────────────────────────────────
print("\n📦 Quantizing to INT8 …")
try:
    quantize_dynamic(ONNX_FP32, ONNX_QUANT, weight_type=QuantType.QInt8)
    size_q = os.path.getsize(ONNX_QUANT) / 1024 / 1024
    print(f"✅ INT8 ONNX → {ONNX_QUANT}  ({size_q:.1f} MB)")
except Exception as e:
    print(f"⚠  Quantization skipped: {e}")
    size_q = 0

# ─────────────────────────────────────────────────────────────────────────────
# STEP 16 — ONNX sanity check on a few real Korean examples
# ─────────────────────────────────────────────────────────────────────────────
print("\n🧪 ONNX sanity check …")
ort_sess = ort.InferenceSession(ONNX_FP32, providers=["CPUExecutionProvider"])

test_cases = [
    ("삼겹살",          "food"),
    ("김치찌개",        "food"),
    ("불닭볶음면",      "food"),
    ("이마트",          "not_food"),
    ("서울특별시 강남구", "not_food"),
    ("배달의민족",      "not_food"),
    ("딸기향샴푸",      "not_food"),
    ("밀양시",          "not_food"),
]

all_pass = True
for text, expected in test_cases:
    enc = tokenizer(text, return_tensors="np",
                    max_length=MAX_LEN, truncation=True, padding="max_length")
    logits = ort_sess.run(["logits"], {
        "input_ids":      enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
        "token_type_ids": enc.get("token_type_ids",
                          np.zeros_like(enc["input_ids"])).astype(np.int64),
    })[0]
    pred = id2label[int(np.argmax(logits))]
    ok   = "✅" if pred == expected else "❌"
    if pred != expected:
        all_pass = False
    print(f"   {ok}  '{text}'  → {pred}  (expected: {expected})")

if all_pass:
    print("   All sanity checks passed!")
else:
    print("   Some checks failed — review the model or dataset.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 17 — Training history
# ─────────────────────────────────────────────────────────────────────────────
print("\n📈 Training history:")
print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val Acc':>8} | {'Val F1':>8} | {'F1 Food':>9}")
print("-" * 55)
for h in history:
    print(f"  {h['epoch']:>4d} | {h['train_loss']:>10.4f} | "
          f"{h['val_acc']*100:>7.2f}% | {h['val_f1_macro']*100:>7.2f}% | "
          f"{h['val_f1_food']*100:>8.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 18 — Final summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING COMPLETE — SUMMARY")
print("=" * 60)
print(f"  Dataset rows      : {len(df):,}")
print(f"  Train / Val       : {len(train_df):,} / {len(val_df):,}")
print(f"  Best Val F1 Macro : {best_val_f1*100:.2f}%")
print(f"  Val Accuracy      : {val_metrics['acc']*100:.2f}%")
print(f"  Val F1 Macro      : {val_metrics['f1_macro']*100:.2f}%")
print(f"  Val F1 Food       : {val_metrics['f1_food']*100:.2f}%")
print(f"  Val F1 Not-Food   : {val_metrics['f1_not_food']*100:.2f}%")
print(f"\n  FP32 ONNX  : {size_fp32:.1f} MB  → {ONNX_FP32}")
try:
    print(f"  INT8 ONNX  : {size_q:.1f} MB   → {ONNX_QUANT}")
except: pass
print(f"  Vocab      : {VOCAB_OUT}")
print(f"  Meta JSON  : {META_OUT}")
print("=" * 60)
print("\nCopy these to Android app assets/:")
print("  korean_food_classifier_quant.onnx  (recommended, ~14 MB)")
print("  korean_vocab.txt")
print("  korean_special_tokens.json")
