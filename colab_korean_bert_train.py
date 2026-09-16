"""
╔══════════════════════════════════════════════════════════════════════╗
║  Korean Receipt Food Classifier — Colab Training & Quantization      ║
║  Model : monologg/koelectra-small-v3-discriminator (~14M params)     ║
║  Dataset: korean_receipt_dataset_full.csv (40,011 rows)              ║
║  Output: Korean ELECTRA/BERT ONNX + INT8 Quantized for Android       ║
╚══════════════════════════════════════════════════════════════════════╝

HOW TO USE IN GOOGLE COLAB:
  1. Open Google Colab (https://colab.research.google.com)
  2. Runtime -> Change runtime type -> T4 GPU
  3. Upload `korean_receipt_dataset_full.csv` or let CELL 3 prompt you
  4. Run all cells top-to-bottom
  5. Outputs generated & automatically downloaded:
       - korean_food_classifier_quant.onnx  (~14 MB INT8 quantized model)
       - korean_food_classifier.onnx        (~55 MB FP32 baseline)
       - korean_vocab.txt                   (vocab file for Android tokenizer)
       - korean_special_tokens.json         (tokenizer metadata & label map)
       - korean_food_android_assets.zip     (bundled zip ready for Android)
"""

# ─────────────────────────────────────────────────────────────────────
# CELL 1 — Install dependencies
# ─────────────────────────────────────────────────────────────────────
# %%
# !pip install -q transformers datasets torch onnx onnxruntime onnxruntime-tools onnxscript scikit-learn evaluate accelerate pandas

import sys, subprocess, os
def auto_install():
    try:
        import onnx, onnxruntime, transformers, pandas, onnxscript
    except ImportError:
        print("⏬ Installing required packages (including onnxscript for PyTorch ONNX exporter)...")
        pkgs = ["transformers", "datasets", "torch", "onnx", "onnxruntime", "onnxruntime-tools", "onnxscript", "scikit-learn", "evaluate", "accelerate", "pandas"]
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)
        print("✅ Installation complete.")

auto_install()

# ─────────────────────────────────────────────────────────────────────
# CELL 2 — Mount Google Drive & Environment Setup
# ─────────────────────────────────────────────────────────────────────
# %%
import json, random, time, shutil
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# Check if running in Google Colab
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    from google.colab import drive, files
    drive.mount('/content/drive')
    DRIVE_DIR = "/content/drive/MyDrive/receipt_bert"
    os.makedirs(DRIVE_DIR, exist_ok=True)
    CSV_PATH = os.path.join(DRIVE_DIR, "korean_receipt_dataset_full.csv")
    print(f"✅ Drive mounted -> {DRIVE_DIR}")
else:
    DRIVE_DIR = "."
    CSV_PATH = "./korean_receipt_dataset_full.csv"

# ─────────────────────────────────────────────────────────────────────
# CELL 3 — Upload Dataset to Drive (Optional - run if CSV not in Drive)
# ─────────────────────────────────────────────────────────────────────
# %%
if IN_COLAB and not os.path.exists(CSV_PATH):
    print("📤 Upload `korean_receipt_dataset_full.csv` from your computer:")
    from google.colab import files
    uploaded = files.upload()
    for fname in uploaded:
        dest = os.path.join(DRIVE_DIR, fname)
        shutil.copy(fname, dest)
        print(f"✅ Saved to Drive: {dest}")

# ─────────────────────────────────────────────────────────────────────
# CELL 4 — Configuration & Random Seeds
# ─────────────────────────────────────────────────────────────────────
# %%
MODEL_NAME    = "monologg/koelectra-small-v3-discriminator"
MAX_LEN       = 64       # Receipt lines are short
BATCH_SIZE    = 32
EPOCHS        = 10       # 10 Epochs for full fine-tuning
LR            = 3e-5
SEED          = 42
OUTPUT_DIR    = "./korean_food_model_output"
ONNX_PATH     = "./korean_food_classifier.onnx"
ONNX_QUANT    = "./korean_food_classifier_quant.onnx"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✅ Using Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU Name   : {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────────────────
# CELL 5 — Load & Inspect CSV Dataset
# ─────────────────────────────────────────────────────────────────────
# %%
if not os.path.exists(CSV_PATH):
    if os.path.exists("./korean_receipt_dataset_full.csv"):
        CSV_PATH = "./korean_receipt_dataset_full.csv"
    elif IN_COLAB:
        print("📤 CSV file not found! Please upload `korean_receipt_dataset_full.csv` now:")
        from google.colab import files
        uploaded = files.upload()
        for fname in uploaded:
            dest = os.path.join(DRIVE_DIR, fname)
            shutil.copy(fname, dest)
            CSV_PATH = dest
            print(f"✅ Saved and using: {CSV_PATH}")
    else:
        raise FileNotFoundError(
            f"❌ Could not find {CSV_PATH}. Please place korean_receipt_dataset_full.csv "
            f"in {DRIVE_DIR} or working directory."
        )

print(f"📂 Loading dataset from: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

print(f"✅ Dataset shape: {df.shape}")
print(f"   Columns      : {df.columns.tolist()}")

# Clean data
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() >= 1].reset_index(drop=True)

# Map string labels to numeric IDs
label_map = {"not_food": 0, "food": 1}
id_map = {0: "not_food", 1: "food"}
df["label_id"] = df["label"].map(label_map)

print(f"✅ Cleaned dataset: {len(df)} rows")
print("   Label distribution:")
print(df["label"].value_counts().to_string())

# ─────────────────────────────────────────────────────────────────────
# CELL 6 — Stratified Train/Val/Test Split (80% / 10% / 10%)
# ─────────────────────────────────────────────────────────────────────
# %%
train_df, temp_df = train_test_split(
    df, test_size=0.20, stratify=df["label_id"], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=SEED
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"✅ Data Splits -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ─────────────────────────────────────────────────────────────────────
# CELL 7 — PyTorch Dataset & DataLoaders
# ─────────────────────────────────────────────────────────────────────
# %%
print(f"⏬ Loading Tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class KoreanReceiptDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.texts = df["text"].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get(
                "token_type_ids", torch.zeros(self.max_len, dtype=torch.long)
            ).squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

train_ds = KoreanReceiptDataset(train_df, tokenizer, MAX_LEN)
val_ds   = KoreanReceiptDataset(val_df,   tokenizer, MAX_LEN)
test_ds  = KoreanReceiptDataset(test_df,  tokenizer, MAX_LEN)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print("✅ DataLoaders initialized successfully.")

# ─────────────────────────────────────────────────────────────────────
# CELL 8 — Load Model Architecture
# ─────────────────────────────────────────────────────────────────────
# %%
print(f"⏬ Loading Pre-trained Model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id_map,
    label2id=label_map,
)
model = model.to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Loaded {MODEL_NAME} — {total_params / 1e6:.2f}M parameters")

# ─────────────────────────────────────────────────────────────────────
# CELL 9 — Training & Validation Loop with BERT-tiny Metrics
# ─────────────────────────────────────────────────────────────────────
# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_dl) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)

def evaluate_pytorch(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            total_val_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_val_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_food = f1_score(all_labels, all_preds, pos_label=1, average="binary")
    f1_not_food = f1_score(all_labels, all_preds, pos_label=0, average="binary")
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro")

    metrics = {
        "val_loss": val_loss,
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_food": f1_food,
        "f1_not_food": f1_not_food,
        "precision": precision,
        "recall": recall,
    }
    return metrics, all_preds, all_labels

print(f"\n🚀 Starting Model Fine-Tuning ({EPOCHS} Epochs)...")
best_val_f1 = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0
    train_correct = 0
    train_total = 0
    t0 = time.time()

    for step, batch in enumerate(train_dl):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch["token_type_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

        if (step + 1) % 200 == 0 or (step + 1) == len(train_dl):
            current_train_acc = (train_correct / train_total) * 100
            print(
                f"   Epoch {epoch+1:02d}/{EPOCHS:02d} | Step {step+1:04d}/{len(train_dl):04d} | "
                f"Train Loss: {total_train_loss / (step+1):.4f} | Train Acc: {current_train_acc:.2f}%",
                flush=True,
            )

    train_loss = total_train_loss / len(train_dl)
    train_acc = (train_correct / train_total) * 100
    val_metrics, _, _ = evaluate_pytorch(model, val_dl)
    elapsed = time.time() - t0

    print(
        f"✅ Epoch {epoch+1:02d}/{EPOCHS:02d} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_metrics['val_loss']:.4f} | "
        f"Val Acc: {val_metrics['acc'] * 100:.2f}% | "
        f"Val F1: {val_metrics['f1_macro'] * 100:.2f}% "
        f"(Food: {val_metrics['f1_food']*100:.1f}%, Non-Food: {val_metrics['f1_not_food']*100:.1f}%) | "
        f"Time: {elapsed:.1f}s"
    )

    if val_metrics["f1_macro"] > best_val_f1:
        best_val_f1 = val_metrics["f1_macro"]
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"   ⭐ New best model saved! (Val F1: {val_metrics['f1_macro'] * 100:.2f}% | Acc: {val_metrics['acc'] * 100:.2f}%)")

print(f"\n🏆 Best Validation F1 Score: {best_val_f1 * 100:.2f}%")

# ─────────────────────────────────────────────────────────────────────
# CELL 10 — Test Set Final Evaluation & Classification Report
# ─────────────────────────────────────────────────────────────────────
# %%
best_model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR).to(DEVICE)
test_metrics, test_preds, test_labels = evaluate_pytorch(best_model, test_dl)

print()
print(f"📊 Test Set Accuracy : {test_metrics['acc'] * 100:.2f}%")
print(f"📊 Test Set F1 Score : {test_metrics['f1_macro'] * 100:.2f}%")
print(f"📊 Test Precision    : {test_metrics['precision'] * 100:.2f}%")
print(f"📊 Test Recall       : {test_metrics['recall'] * 100:.2f}%")
print("\nDetailed Classification Report on Test Set:")
print(classification_report(test_labels, test_preds, target_names=["not_food", "food"]))

print("Confusion Matrix:")
cm = confusion_matrix(test_labels, test_preds)
print(f"   [[TN={cm[0][0]:5d}, FP={cm[0][1]:5d}]\n    [FN={cm[1][0]:5d}, TP={cm[1][1]:5d}]]")

# ─────────────────────────────────────────────────────────────────────
# CELL 11 — Export Model to ONNX (FP32)
# ─────────────────────────────────────────────────────────────────────
# %%
print("\n⚙️  Exporting PyTorch model to ONNX FP32 format...")

# Ensure onnxscript is installed for PyTorch ONNX exporter
try:
    import onnxscript
except ImportError:
    print("⏬ Installing onnxscript required by PyTorch ONNX exporter...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnxscript"])
    import onnxscript

best_model.eval().cpu()

dummy_input_ids = torch.ones(1, MAX_LEN, dtype=torch.long)
dummy_attn_mask = torch.ones(1, MAX_LEN, dtype=torch.long)
dummy_token_type = torch.zeros(1, MAX_LEN, dtype=torch.long)

torch.onnx.export(
    best_model,
    (dummy_input_ids, dummy_attn_mask, dummy_token_type),
    ONNX_PATH,
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "token_type_ids": {0: "batch", 1: "seq"},
        "logits": {0: "batch"},
    },
    opset_version=14,
    do_constant_folding=True,
)

size_fp32 = os.path.getsize(ONNX_PATH) / (1024 * 1024)
print(f"✅ ONNX FP32 Model exported successfully! Size: {size_fp32:.2f} MB")
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
print("✅ ONNX model validity check passed.")

# ─────────────────────────────────────────────────────────────────────
# CELL 12 — Dynamic INT8 Quantization (~14 MB for Android Mobile ONNX)
# ─────────────────────────────────────────────────────────────────────
# %%
print("\n⚡ Quantizing ONNX model using Dynamic INT8 Quantization...")
quantize_dynamic(
    model_input=ONNX_PATH,
    model_output=ONNX_QUANT,
    weight_type=QuantType.QInt8,
    extra_options={"MatMulConstBOnly": True, "EnableSubgraph": True},
)

size_quant = os.path.getsize(ONNX_QUANT) / (1024 * 1024)
compression_ratio = size_fp32 / size_quant
print(f"✅ Quantized ONNX Model created: {ONNX_QUANT}")
print(f"   FP32 Size : {size_fp32:.2f} MB")
print(f"   INT8 Size : {size_quant:.2f} MB  ({compression_ratio:.1f}x reduction!)")

# ─────────────────────────────────────────────────────────────────────
# CELL 13 — Verify Accuracy: FP32 vs Quantized ONNX
# ─────────────────────────────────────────────────────────────────────
# %%
print("\n🧪 Running ONNX Runtime accuracy verification on Test set...")

def evaluate_onnx(onnx_file_path, df_sub):
    session = ort.InferenceSession(onnx_file_path, providers=["CPUExecutionProvider"])
    texts = df_sub["text"].tolist()
    labels = df_sub["label_id"].tolist()
    preds = []

    for text in texts:
        enc = tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        inputs = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
            "token_type_ids": enc.get(
                "token_type_ids", np.zeros((1, MAX_LEN), dtype=np.int64)
            ).astype(np.int64),
        }
        logits = session.run(["logits"], inputs)[0]
        pred_id = int(np.argmax(logits, axis=-1)[0])
        preds.append(pred_id)

    return accuracy_score(labels, preds)

# Test on 1,000 random samples from test set for fast evaluation
test_sample = test_df.sample(n=min(1000, len(test_df)), random_state=SEED)
acc_fp32_onnx = evaluate_onnx(ONNX_PATH, test_sample)
acc_quant_onnx = evaluate_onnx(ONNX_QUANT, test_sample)

print(f"   ONNX FP32 Accuracy : {acc_fp32_onnx * 100:.2f}%")
print(f"   ONNX INT8 Accuracy : {acc_quant_onnx * 100:.2f}%")
print(f"   Accuracy Change    : {(acc_quant_onnx - acc_fp32_onnx) * 100:+.2f}%")

# ─────────────────────────────────────────────────────────────────────
# CELL 14 — Export Vocabulary & Tokenizer Config for Android
# ─────────────────────────────────────────────────────────────────────
# %%
print("\n📦 Generating Android tokenizer assets (korean_vocab.txt & metadata)...")
vocab_src = os.path.join(OUTPUT_DIR, "vocab.txt")
vocab_dst = "./korean_vocab.txt"

if os.path.exists(vocab_src):
    shutil.copy(vocab_src, vocab_dst)
else:
    with open(vocab_dst, "w", encoding="utf-8") as f:
        for token, _ in sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]):
            f.write(token + "\n")

special_tokens_meta = {
    "cls_token_id": tokenizer.cls_token_id,
    "sep_token_id": tokenizer.sep_token_id,
    "pad_token_id": tokenizer.pad_token_id,
    "unk_token_id": tokenizer.unk_token_id,
    "vocab_size":   tokenizer.vocab_size,
    "max_length":   MAX_LEN,
    "model_name":   MODEL_NAME,
    "labels":       {0: "not_food", 1: "food"},
}

with open("./korean_special_tokens.json", "w", encoding="utf-8") as f:
    json.dump(special_tokens_meta, f, ensure_ascii=False, indent=2)

print("✅ Saved ./korean_vocab.txt")
print("✅ Saved ./korean_special_tokens.json")

# ─────────────────────────────────────────────────────────────────────
# CELL 15 — Bundle Assets into ZIP & Auto Download
# ─────────────────────────────────────────────────────────────────────
# %%
import zipfile

zip_filename = "korean_food_android_assets.zip"
zip_filepath = os.path.join(".", zip_filename)

asset_files = [
    (ONNX_QUANT,                    "korean_food_classifier_quant.onnx"),
    (ONNX_PATH,                     "korean_food_classifier.onnx"),
    ("./korean_vocab.txt",          "korean_vocab.txt"),
    ("./korean_special_tokens.json","korean_special_tokens.json"),
]

print(f"\n📦 Packaging Android artifacts into ZIP: {zip_filename}...")
with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as zf:
    for src, arcname in asset_files:
        if os.path.exists(src):
            size_mb = os.path.getsize(src) / (1024 * 1024)
            zf.write(src, arcname)
            print(f"   + Added {arcname:35s} ({size_mb:.2f} MB)")

zip_size_mb = os.path.getsize(zip_filepath) / (1024 * 1024)
print(f"✅ Bundled ZIP file complete: {zip_filename} ({zip_size_mb:.2f} MB)")

# Copy to Google Drive if in Colab & trigger automatic download prompt
if IN_COLAB:
    drive_dest = os.path.join(DRIVE_DIR, zip_filename)
    shutil.copy(zip_filepath, drive_dest)
    print(f"✅ Saved ZIP copy to Google Drive -> {drive_dest}")
    try:
        print("⏬ Triggering automatic browser ZIP download...")
        from google.colab import files
        files.download(zip_filepath)
        print("✅ Download started automatically!")
    except Exception as e:
        print(f"ℹ️ Auto-download prompt skipped: {e}. ZIP is available in Drive at {drive_dest}")

# ─────────────────────────────────────────────────────────────────────
# CELL 16 — Live Sample Predictions (Validation Test)
# ─────────────────────────────────────────────────────────────────────
# %%
print("\n🔍 Running test predictions using Quantized ONNX Model:")

def predict_korean_receipt_items(sample_texts, onnx_model_path=ONNX_QUANT):
    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    
    enc = tokenizer(
        sample_texts,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    
    inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
        "token_type_ids": enc.get(
            "token_type_ids", np.zeros((len(sample_texts), MAX_LEN), dtype=np.int64)
        ).astype(np.int64),
    }

    logits = session.run(["logits"], inputs)[0]
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    preds = np.argmax(probs, axis=-1)

    print("-" * 65)
    print(f"{'Text Line':30} | {'Prediction':10} | Confidence")
    print("-" * 65)
    for text, pred, prob in zip(sample_texts, preds, probs):
        label_str = id_map[pred]
        confidence = prob[pred] * 100
        icon = "🍜 FOOD    " if pred == 1 else "🧾 NOT_FOOD"
        print(f"{text:30} | {icon:10} | {confidence:6.2f}%")
    print("-" * 65)

test_samples = [
    "김치찌개 1개",
    "삼겹살 2인분",
    "아메리카노 (ICE)",
    "종량제봉투 20L",
    "과세물품가액 15,000원",
    "신용카드 승인번호 928104",
    "대표자: 김철수",
    "후라이드치킨 반반",
    "부가세 10%",
    "공급가액 15,000원",
]

predict_korean_receipt_items(test_samples)

print("\n🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
