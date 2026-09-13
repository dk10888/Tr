import os
import sys
import time
import argparse
import pandas as pd
from typing import List, Dict, Any

"""
BERT-tiny Model Gap Audit & Active Learning Pipeline
---------------------------------------------------
Runs fine-tuned BERT-tiny (v5) inference over a batch of receipt images.
Categorizes predictions based on an 80% confidence threshold to isolate:
  1. High confidence predictions (≥ 80%) -> Ready for auto-labeling
  2. Low confidence predictions (< 80%)  -> Model gaps / untrained food items
"""

def setup_env():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

def load_ocr_and_model(model_folder: str = "bert_tiny_v5_food_final"):
    setup_env()
    import cv2
    import torch
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
        
    from rapidocr_onnxruntime import RapidOCR
    from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. OCR Engine
    models_dir = os.path.join(base_dir, "models", "onnx")
    det_path = os.path.join(models_dir, "v6_medium_det.onnx")
    rec_path = os.path.join(models_dir, "v6_medium_rec.onnx")
    rec_keys_path = os.path.join(models_dir, "rec_keys.txt")
    if os.path.exists(det_path) and os.path.exists(rec_path):
        ocr = RapidOCR(det_model_path=det_path, rec_model_path=rec_path, rec_keys_path=rec_keys_path)
    else:
        ocr = RapidOCR()
        
    # 2. BERT-tiny Model
    model_dir = os.path.join(base_dir, "models", model_folder)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    config = BertConfig.from_pretrained(model_dir, local_files_only=True)
    model = BertForSequenceClassification(config)
    
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=True)
    else:
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
        model.load_state_dict(state_dict)
        
    model.eval()
    return ocr, tokenizer, model

def run_batch_audit(receipts_dir: str, confidence_threshold: float = 0.80, output_dir: str = "audit_output"):
    import cv2
    import torch
    
    print(f"🚀 Initializing BERT-tiny Audit Pipeline (Confidence Threshold: {confidence_threshold*100:.0f}%)...")
    ocr, tokenizer, model = load_ocr_and_model("bert_tiny_v5_food_final")
    
    os.makedirs(output_dir, exist_ok=True)
    
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = [
        os.path.join(receipts_dir, f)
        for f in os.listdir(receipts_dir)
        if f.lower().endswith(valid_exts)
    ]
    
    if not image_files:
        print(f"❌ No images found in '{receipts_dir}'!")
        return
        
    print(f"📁 Processing {len(image_files)} receipt images from '{receipts_dir}'...")
    
    high_conf_food = []
    low_conf_food = []
    high_conf_not_food = []
    low_conf_not_food = []
    
    start_time = time.time()
    
    for idx, img_path in enumerate(image_files, 1):
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        ocr_result, _ = ocr(img)
        if not ocr_result:
            continue
            
        lines = [r[1].strip() for r in ocr_result if len(r[1].strip()) >= 2]
        if not lines:
            continue
            
        # Batch inference
        inputs = tokenizer(lines, return_tensors="pt", padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
        id2label = model.config.id2label
        
        for text, pred, prob in zip(lines, preds, probs):
            label = id2label[pred.item()]
            conf = float(prob[pred.item()])
            
            record = {
                "receipt_image": img_name,
                "text": text,
                "predicted_label": label,
                "confidence": round(conf, 4)
            }
            
            if label == "food":
                if conf >= confidence_threshold:
                    high_conf_food.append(record)
                else:
                    low_conf_food.append(record)
            else:
                if conf >= confidence_threshold:
                    high_conf_not_food.append(record)
                else:
                    low_conf_not_food.append(record)
                    
        if idx % 10 == 0 or idx == len(image_files):
            print(f"  [Progress] Processed {idx}/{len(image_files)} receipts...")

    elapsed = round(time.time() - start_time, 2)
    
    # Save CSV Audit Reports
    df_high_food = pd.DataFrame(high_conf_food)
    df_low_food = pd.DataFrame(low_conf_food)
    df_high_not_food = pd.DataFrame(high_conf_not_food)
    df_low_not_food = pd.DataFrame(low_conf_not_food)
    
    path_high_food = os.path.join(output_dir, "high_confidence_food.csv")
    path_low_food = os.path.join(output_dir, "low_confidence_food_audit.csv")
    path_high_not_food = os.path.join(output_dir, "high_confidence_not_food.csv")
    path_low_not_food = os.path.join(output_dir, "low_confidence_not_food_audit.csv")
    
    df_high_food.to_csv(path_high_food, index=False)
    df_low_food.to_csv(path_low_food, index=False)
    df_high_not_food.to_csv(path_high_not_food, index=False)
    df_low_not_food.to_csv(path_low_not_food, index=False)
    
    # Print Executive Summary
    print("\n=======================================================")
    print(f"📊 BERT-TINY AUDIT REPORT (Completed in {elapsed}s)")
    print("=======================================================")
    print(f"🍏 High Confidence Food (≥ {confidence_threshold*100:.0f}%):     {len(high_conf_food):,}")
    print(f"⚠️  Low Confidence Food (< {confidence_threshold*100:.0f}% AUDIT): {len(low_conf_food):,}")
    print(f"📦 High Confidence Non-Food (≥ {confidence_threshold*100:.0f}%): {len(high_conf_not_food):,}")
    print(f"⚠️  Low Confidence Non-Food (< {confidence_threshold*100:.0f}% AUDIT): {len(low_conf_not_food):,}")
    print("-------------------------------------------------------")
    print(f"📁 Audit CSV files saved to '{output_dir}/':")
    print(f"  • {path_low_food}  <-- Inspect for untrained food items!")
    print(f"  • {path_low_not_food}  <-- Inspect for ambiguous receipt lines!")
    print("=======================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit BERT-tiny predictions on new receipts.")
    parser.add_argument("--receipts_dir", type=str, default="Receipts/test", help="Directory containing receipt images")
    parser.add_argument("--threshold", type=float, default=0.80, help="Confidence threshold (default: 0.80)")
    parser.add_argument("--output_dir", type=str, default="audit_output", help="Directory to save audit CSVs")
    
    args = parser.parse_args()
    run_batch_audit(args.receipts_dir, args.threshold, args.output_dir)
