"""
Export English BERT-tiny v7 (bert_tiny_v7_food_final) to ONNX for Android.
Run from receipt_scanner/ directory:
    python3 export_bert_to_onnx_android.py

Outputs (into android_app/app/src/main/assets/):
  - english_food_bert.onnx        (~17 MB FP32)
  - english_food_bert_quant.onnx  (~5 MB INT8 dynamic)
  - english_vocab.txt             (30522 tokens, standard BERT-uncased)
"""
import os, shutil, json
import torch
from transformers import BertConfig, BertForSequenceClassification

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
# ── Use v7 model (trained on 76k samples, 94.18% accuracy) ───────────
MODEL_DIR  = os.path.join(BASE_DIR, "models", "bert_tiny_v7_food_final")
ASSETS_DIR = os.path.join(BASE_DIR, "android_app", "app", "src", "main", "assets")
ONNX_FP32  = os.path.join(ASSETS_DIR, "english_food_bert.onnx")
ONNX_QUANT = os.path.join(ASSETS_DIR, "english_food_bert_quant.onnx")
MAX_LEN    = 64

os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Load model (safetensors format) ──────────────────────────────────
print(f"Loading BERT-tiny v7 from {MODEL_DIR}...")
config = BertConfig.from_pretrained(MODEL_DIR, local_files_only=True)
model  = BertForSequenceClassification(config)

# v7 uses safetensors
safetensors_path = os.path.join(MODEL_DIR, "model.safetensors")
if os.path.exists(safetensors_path):
    from safetensors.torch import load_file
    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict, strict=True)
    print("  Loaded from safetensors")
else:
    # fallback to pytorch_model.bin
    bin_path = os.path.join(MODEL_DIR, "pytorch_model.bin")
    state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    print("  Loaded from pytorch_model.bin")

model.eval()
params = sum(p.numel() for p in model.parameters())
print(f"✅ Model loaded: {params/1e6:.1f}M params")
print(f"   Labels: {config.id2label}")

# ── Export ONNX FP32 ──────────────────────────────────────────────────
print(f"\nExporting ONNX FP32 → {ONNX_FP32}")
dummy = torch.ones(1, MAX_LEN, dtype=torch.long)
torch.onnx.export(
    model,
    (dummy, dummy, torch.zeros(1, MAX_LEN, dtype=torch.long)),
    ONNX_FP32,
    input_names  = ["input_ids", "attention_mask", "token_type_ids"],
    output_names = ["logits"],
    dynamic_axes = {
        "input_ids":      {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "token_type_ids": {0: "batch", 1: "seq"},
        "logits":         {0: "batch"},
    },
    opset_version = 14,
    do_constant_folding = True,
)
size_fp32 = os.path.getsize(ONNX_FP32) / 1024 / 1024
print(f"✅ FP32: {size_fp32:.1f} MB")

# ── Verify ONNX model ─────────────────────────────────────────────────
import onnx
onnx.checker.check_model(onnx.load(ONNX_FP32))
print("✅ ONNX model check passed")

# ── Dynamic INT8 Quantization (→ ~5 MB for Android) ──────────────────
print(f"\nQuantizing to INT8 → {ONNX_QUANT}")
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input   = ONNX_FP32,
        model_output  = ONNX_QUANT,
        weight_type   = QuantType.QInt8,
        extra_options = {"MatMulConstBOnly": True},
    )
    size_quant = os.path.getsize(ONNX_QUANT) / 1024 / 1024
    print(f"✅ INT8: {size_quant:.1f} MB ({size_fp32/size_quant:.1f}x smaller)")
except ImportError:
    print("⚠️  onnxruntime not installed, skipping quantization")
    print("    pip install onnxruntime onnxruntime-tools")

# ── Extract vocab.txt for Android WordPiece tokenizer ─────────────────
print("\nExtracting vocab.txt from tokenizer.json...")
tok_json_path = os.path.join(MODEL_DIR, "tokenizer.json")
vocab_dst     = os.path.join(ASSETS_DIR, "english_vocab.txt")

if os.path.exists(tok_json_path):
    with open(tok_json_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)
    vocab = tok_data["model"]["vocab"]  # token → id
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    with open(vocab_dst, "w", encoding="utf-8") as f:
        for token, _ in sorted_vocab:
            f.write(token + "\n")
    cls_id = vocab.get("[CLS]", 101)
    sep_id = vocab.get("[SEP]", 102)
    pad_id = vocab.get("[PAD]", 0)
    unk_id = vocab.get("[UNK]", 100)
    print(f"✅ vocab.txt: {len(sorted_vocab)} tokens")
    print(f"   [CLS]={cls_id}, [SEP]={sep_id}, [PAD]={pad_id}, [UNK]={unk_id}")
elif os.path.exists(os.path.join(MODEL_DIR, "vocab.txt")):
    shutil.copy(os.path.join(MODEL_DIR, "vocab.txt"), vocab_dst)
    print(f"✅ Copied vocab.txt")
else:
    print("⚠️  No vocab file found! Android tokenizer won't work.")

print(f"\n✅ All files ready in: {ASSETS_DIR}")
print("   english_food_bert.onnx      →", size_fp32, "MB (FP32)")
try:
    print("   english_food_bert_quant.onnx →", size_quant, "MB (INT8)")
except:
    pass
print("   english_vocab.txt")
print("\nNow add Korean model files from Colab:")
print("   korean_food_classifier_quant.onnx")
print("   korean_vocab.txt")
