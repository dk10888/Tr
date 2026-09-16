"""
Export English BERT (bert_tiny_v5_food_final) to ONNX for Android.
Run from receipt_scanner/ directory:
    python3 export_bert_to_onnx_android.py

Outputs (into android_app/app/src/main/assets/):
  - english_food_bert.onnx     (~55 MB FP32, for reference)
  - english_food_bert_quant.onnx  (~14 MB INT8, for Android)
  - english_vocab.txt
"""
import os, shutil
import torch
from transformers import BertConfig, BertForSequenceClassification

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models", "bert_tiny_v5_food_final")
ASSETS_DIR = os.path.join(BASE_DIR, "android_app", "app", "src", "main", "assets")
ONNX_FP32  = os.path.join(ASSETS_DIR, "english_food_bert.onnx")
ONNX_QUANT = os.path.join(ASSETS_DIR, "english_food_bert_quant.onnx")
MAX_LEN    = 64

os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────
print("Loading BERT model...")
config = BertConfig.from_pretrained(MODEL_DIR, local_files_only=True)
model  = BertForSequenceClassification(config)

bin_path = os.path.join(MODEL_DIR, "pytorch_model.bin")
if os.path.exists(bin_path):
    state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
else:
    from safetensors.torch import load_file
    state_dict = load_file(os.path.join(MODEL_DIR, "model.safetensors"))

model.load_state_dict(state_dict, strict=True)
model.eval()
print(f"✅ Loaded model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

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

# ── Dynamic INT8 Quantization ─────────────────────────────────────────
print(f"\nQuantizing to INT8 → {ONNX_QUANT}")
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input  = ONNX_FP32,
        model_output = ONNX_QUANT,
        weight_type  = QuantType.QInt8,
        extra_options = {"MatMulConstBOnly": True},
    )
    size_quant = os.path.getsize(ONNX_QUANT) / 1024 / 1024
    print(f"✅ INT8: {size_quant:.1f} MB ({size_fp32/size_quant:.1f}x smaller)")
except ImportError:
    print("⚠️  onnxruntime-tools not installed, skipping quantization")
    print("    Run: pip install onnxruntime-tools")

# ── Copy vocab.txt ────────────────────────────────────────────────────
vocab_src = os.path.join(MODEL_DIR, "vocab.txt")
vocab_dst = os.path.join(ASSETS_DIR, "english_vocab.txt")
if os.path.exists(vocab_src):
    shutil.copy(vocab_src, vocab_dst)
    print(f"\n✅ Copied vocab: {vocab_dst} ({os.path.getsize(vocab_dst)//1024} KB)")
else:
    print(f"\n⚠️  vocab.txt not found in {MODEL_DIR}")
    print("    Download tokenizer files from HuggingFace or copy from training output")

print("\n✅ Done! Files ready in:", ASSETS_DIR)
print("   Now place korean_food_classifier_quant.onnx + korean_vocab.txt there too")
print("   (Download from Colab → korean_food_android_assets.zip)")
