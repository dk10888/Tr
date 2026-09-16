"""
Deadlock-safe ONNX export using subprocess spawning.
macOS OpenMP mutex deadlock fix: torch is imported in a FRESH spawned process,
never in the parent process that already has an event loop or mutex.

Run: python3 export_bert_to_onnx_android_safe.py
"""
import os
import sys
import multiprocessing as mp

def export_worker(model_dir, assets_dir, max_len, result_queue):
    """Runs in a fresh spawned process — no inherited mutexes."""
    import os
    os.environ["OMP_NUM_THREADS"]        = "1"
    os.environ["MKL_NUM_THREADS"]        = "1"
    os.environ["OPENBLAS_NUM_THREADS"]   = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["KMP_DUPLICATE_LIB_OK"]  = "TRUE"

    try:
        import json, shutil
        import torch
        torch.set_num_threads(1)

        from transformers import BertConfig, BertForSequenceClassification

        onnx_fp32  = os.path.join(assets_dir, "english_food_bert.onnx")
        onnx_quant = os.path.join(assets_dir, "english_food_bert_quant.onnx")
        vocab_dst  = os.path.join(assets_dir, "english_vocab.txt")

        os.makedirs(assets_dir, exist_ok=True)

        # ── Load model ────────────────────────────────────────────────
        print("  [worker] Loading BERT-tiny v7...", flush=True)
        config = BertConfig.from_pretrained(model_dir, local_files_only=True)
        model  = BertForSequenceClassification(config)

        sf_path  = os.path.join(model_dir, "model.safetensors")
        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(sf_path):
            from safetensors.torch import load_file
            state_dict = load_file(sf_path)
        else:
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        print(f"  [worker] Loaded: {params/1e6:.1f}M params", flush=True)

        # ── Export ONNX FP32 ──────────────────────────────────────────
        print(f"  [worker] Exporting ONNX...", flush=True)
        dummy = torch.ones(1, max_len, dtype=torch.long)
        torch.onnx.export(
            model,
            (dummy, dummy, torch.zeros(1, max_len, dtype=torch.long)),
            onnx_fp32,
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
        size_fp32 = os.path.getsize(onnx_fp32) / 1024 / 1024
        print(f"  [worker] FP32: {size_fp32:.1f} MB", flush=True)

        # ── Dynamic INT8 Quantization ─────────────────────────────────
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            model_input   = onnx_fp32,
            model_output  = onnx_quant,
            weight_type   = QuantType.QInt8,
            extra_options = {"MatMulConstBOnly": True},
        )
        size_quant = os.path.getsize(onnx_quant) / 1024 / 1024
        print(f"  [worker] INT8: {size_quant:.1f} MB ({size_fp32/size_quant:.1f}x smaller)", flush=True)

        # ── Extract vocab.txt ─────────────────────────────────────────
        tok_json = os.path.join(model_dir, "tokenizer.json")
        if os.path.exists(tok_json):
            with open(tok_json, "r", encoding="utf-8") as f:
                tok_data = json.load(f)
            vocab = tok_data["model"]["vocab"]
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
            with open(vocab_dst, "w", encoding="utf-8") as f:
                for token, _ in sorted_vocab:
                    f.write(token + "\n")
            print(f"  [worker] vocab.txt: {len(sorted_vocab)} tokens", flush=True)
        elif os.path.exists(os.path.join(model_dir, "vocab.txt")):
            shutil.copy(os.path.join(model_dir, "vocab.txt"), vocab_dst)
            print(f"  [worker] Copied vocab.txt", flush=True)

        result_queue.put(("ok", size_fp32, size_quant))

    except Exception as e:
        import traceback
        result_queue.put(("error", str(e), traceback.format_exc()))


if __name__ == "__main__":
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR  = os.path.join(BASE_DIR, "models", "bert_tiny_v7_food_final")
    ASSETS_DIR = os.path.join(BASE_DIR, "android_app", "app", "src", "main", "assets")
    MAX_LEN    = 64

    print("🚀 Starting ONNX export in isolated subprocess (macOS deadlock fix)...")
    print(f"   Model  : {MODEL_DIR}")
    print(f"   Output : {ASSETS_DIR}")

    # MUST use 'spawn' — 'fork' inherits the parent's mutex state = deadlock
    ctx = mp.get_context("spawn")
    q   = ctx.Queue()
    p   = ctx.Process(target=export_worker, args=(MODEL_DIR, ASSETS_DIR, MAX_LEN, q))
    p.start()
    p.join(timeout=300)  # 5-minute timeout

    if p.exitcode is None:
        p.terminate()
        print("❌ Export timed out after 5 minutes")
        sys.exit(1)

    result = q.get()
    if result[0] == "ok":
        _, size_fp32, size_quant = result
        print(f"\n✅ Export complete!")
        print(f"   english_food_bert.onnx       → {size_fp32:.1f} MB (FP32)")
        print(f"   english_food_bert_quant.onnx → {size_quant:.1f} MB (INT8)")
        print(f"   english_vocab.txt")
        print(f"\n   Files saved to: {ASSETS_DIR}")
        print("\nNext: add korean_food_classifier_quant.onnx + korean_vocab.txt from Colab")
    else:
        _, err, tb = result
        print(f"❌ Export failed: {err}")
        print(tb)
        sys.exit(1)
