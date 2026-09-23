#!/usr/bin/env python3
"""
QQ Receipts Model Processor — Generates Exactly 3 Model-Specific JSON Files
================================================────────────────────────────
Scans all images in 'qq/' using EasyOCR + Smart Line Clustering, and produces:
  1. qq_json_results/korean_bert_onnx_results.json
  2. qq_json_results/fasttext_v25_results.json
  3. qq_json_results/fasttext_v23_results.json
"""

import os
import sys
import re
import time
import json
import shutil
import numpy as np
from typing import List, Dict, Any

# Ensure user site-packages are loaded
user_site = os.path.expanduser('~/Library/Python/3.9/lib/python/site-packages')
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QQ_DIR = os.path.join(BASE_DIR, "qq")
OUTPUT_DIR = os.path.join(BASE_DIR, "qq_json_results")

# Model paths
BERT_ONNX_PATH = os.path.join(BASE_DIR, "korean_new", "korean_food_classifier_quant.onnx")
FASTTEXT_V25_PATH = os.path.join(BASE_DIR, "fastText_v25", "fasttext_korean_food.ftz")
FASTTEXT_V23_PATH = os.path.join(BASE_DIR, "fastText_v23", "fasttext_korean_food.ftz")

# Regex unicode ranges
HANGUL_REGEX = re.compile(r'[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]')
JAPANESE_REGEX = re.compile(r'[\u3040-\u309f\u30a0-\u30ff]') # Hiragana & Katakana

# Price & Currency Regex Pattern
PRICE_PATTERN = re.compile(
    r'^([₩$￥€]?\s*\d{1,3}(,\d{3})*(\.\d{1,2})?\s*원?|\d+\s*원|\d+\.\d{2})$',
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL SUITE LOADERS
# ─────────────────────────────────────────────────────────────────────────────
class DualModelSuite:
    def __init__(self):
        print("📦 Initializing Models...")
        
        # 1. Korean BERT ONNX
        self.bert_session = None
        self.bert_tokenizer = None
        if os.path.exists(BERT_ONNX_PATH):
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer
                self.bert_session = ort.InferenceSession(BERT_ONNX_PATH, providers=["CPUExecutionProvider"])
                self.bert_tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
                print(f"  ✅ Loaded Korean BERT ONNX")
            except Exception as e:
                print(f"  ⚠️ Failed to load BERT ONNX: {e}")

        # 2. FastText v25
        self.ft_v25 = None
        if os.path.exists(FASTTEXT_V25_PATH):
            try:
                import fasttext
                self.ft_v25 = fasttext.load_model(FASTTEXT_V25_PATH)
                print(f"  ✅ Loaded FastText v25")
            except Exception as e:
                print(f"  ⚠️ Failed to load FastText v25: {e}")

        # 3. FastText v23
        self.ft_v23 = None
        if os.path.exists(FASTTEXT_V23_PATH):
            try:
                import fasttext
                self.ft_v23 = fasttext.load_model(FASTTEXT_V23_PATH)
                print(f"  ✅ Loaded FastText v23")
            except Exception as e:
                print(f"  ⚠️ Failed to load FastText v23: {e}")

    def predict_bert(self, text: str) -> Dict[str, Any]:
        if not self.bert_session or not self.bert_tokenizer:
            return {"label": "unknown", "confidence": 0.0, "time_ms": 0.0}

        t0 = time.time()
        inputs = self.bert_tokenizer(text, return_tensors="np", truncation=True, padding="max_length", max_length=64)
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in inputs:
            ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

        outputs = self.bert_session.run(None, ort_inputs)
        logits = outputs[0][0]
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)

        pred_id = int(np.argmax(probs))
        label = "food" if pred_id == 1 else "not_food"
        conf = float(probs[pred_id])
        t_ms = (time.time() - t0) * 1000

        return {"label": label, "confidence": round(conf, 4), "time_ms": round(t_ms, 2)}

    def predict_fasttext(self, model, text: str) -> Dict[str, Any]:
        if not model:
            return {"label": "unknown", "confidence": 0.0, "time_ms": 0.0}

        t0 = time.time()
        labels, probs = model.predict(text.replace("\n", " "))
        label = labels[0].replace("__label__", "") if labels else "not_food"
        conf = float(probs[0]) if probs else 0.0
        t_ms = (time.time() - t0) * 1000

        return {"label": label, "confidence": round(conf, 4), "time_ms": round(t_ms, 3)}

# ─────────────────────────────────────────────────────────────────────────────
# 2. SMART LINE EXTRACTION WITH PRICE COLUMN SPLITTING
# ─────────────────────────────────────────────────────────────────────────────
def extract_smart_lines(ocr_res: List[Any]) -> List[str]:
    if not ocr_res:
        return []

    items = []
    for bbox, text, score in ocr_res:
        txt = text.strip()
        if not txt or score < 0.15:
            continue
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        center_y = (top + bottom) / 2.0
        items.append({
            'text': txt,
            'left': left,
            'right': right,
            'top': top,
            'bottom': bottom,
            'center_y': center_y
        })

    if not items:
        return []

    # Group into vertical row clusters
    sorted_items = sorted(items, key=lambda x: x['center_y'])
    row_clusters = []
    for item in sorted_items:
        added = False
        for cluster in row_clusters:
            avg_y = sum(x['center_y'] for x in cluster) / float(len(cluster))
            if abs(item['center_y'] - avg_y) <= 15.0:
                cluster.append(item)
                added = True
                break
        if not added:
            row_clusters.append([item])

    final_lines = []

    # Split horizontal gaps > 70px or price patterns
    for cluster in row_clusters:
        left_to_right = sorted(cluster, key=lambda x: x['left'])
        current_words = []
        last_right = -1

        for info in left_to_right:
            txt = info['text']
            is_price = bool(PRICE_PATTERN.match(txt))
            x_gap = (info['left'] - last_right) if last_right != -1 else 0

            if current_words and (x_gap > 70 or is_price):
                line_str = " ".join(current_words).strip()
                if line_str:
                    final_lines.append(line_str)
                current_words = [txt]
            else:
                current_words.append(txt)

            last_right = info['right']

        if current_words:
            line_str = " ".join(current_words).strip()
            if line_str:
                final_lines.append(line_str)

    return final_lines

# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN EXECUTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline():
    # Clear directory so ONLY 3 JSON files will exist
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    models = DualModelSuite()

    print("\n🔍 Initializing EasyOCR Reader (Korean + English)...")
    import easyocr
    reader = easyocr.Reader(['ko', 'en'], gpu=False)

    image_files = sorted([
        f for f in os.listdir(QQ_DIR)
        if f.lower().endswith(('.jpeg', '.jpg', '.png'))
    ])
    print(f"📸 Found {len(image_files)} receipt image files in 'qq/'.")

    # Data structures for 3 separate model JSON files
    bert_json_receipts = []
    ft_v25_json_receipts = []
    ft_v23_json_receipts = []

    total_lines = 0

    for idx, img_name in enumerate(image_files, start=1):
        img_path = os.path.join(QQ_DIR, img_name)
        print(f"[{idx}/{len(image_files)}] Processing: {img_name}")

        try:
            ocr_res = reader.readtext(img_path)
        except Exception as e:
            print(f"  ❌ OCR error on {img_name}: {e}")
            continue

        raw_full_text = " ".join([text for _, text, _ in ocr_res])
        if JAPANESE_REGEX.search(raw_full_text) and not HANGUL_REGEX.search(raw_full_text):
            print(f"   administrative skipping non-Korean receipt.")
            continue

        extracted_lines = extract_smart_lines(ocr_res)

        bert_items = []
        ft_v25_items = []
        ft_v23_items = []

        food_bert = 0
        food_ft_v25 = 0
        food_ft_v23 = 0

        for line_no, line_text in enumerate(extracted_lines, start=1):
            total_lines += 1

            # 1. BERT
            res_bert = models.predict_bert(line_text)
            if res_bert["label"] == "food": food_bert += 1
            bert_items.append({
                "line_no": line_no,
                "text": line_text,
                "label": res_bert["label"],
                "confidence": res_bert["confidence"],
                "time_ms": res_bert["time_ms"]
            })

            # 2. FastText v25
            res_v25 = models.predict_fasttext(models.ft_v25, line_text)
            if res_v25["label"] == "food": food_ft_v25 += 1
            ft_v25_items.append({
                "line_no": line_no,
                "text": line_text,
                "label": res_v25["label"],
                "confidence": res_v25["confidence"],
                "time_ms": res_v25["time_ms"]
            })

            # 3. FastText v23
            res_v23 = models.predict_fasttext(models.ft_v23, line_text)
            if res_v23["label"] == "food": food_ft_v23 += 1
            ft_v23_items.append({
                "line_no": line_no,
                "text": line_text,
                "label": res_v23["label"],
                "confidence": res_v23["confidence"],
                "time_ms": res_v23["time_ms"]
            })

        # Append to respective model datasets
        bert_json_receipts.append({
            "receipt_id": idx,
            "receipt_name": img_name,
            "line_count": len(extracted_lines),
            "food_count": food_bert,
            "not_food_count": len(extracted_lines) - food_bert,
            "items": bert_items
        })

        ft_v25_json_receipts.append({
            "receipt_id": idx,
            "receipt_name": img_name,
            "line_count": len(extracted_lines),
            "food_count": food_ft_v25,
            "not_food_count": len(extracted_lines) - food_ft_v25,
            "items": ft_v25_items
        })

        ft_v23_json_receipts.append({
            "receipt_id": idx,
            "receipt_name": img_name,
            "line_count": len(extracted_lines),
            "food_count": food_ft_v23,
            "not_food_count": len(extracted_lines) - food_ft_v23,
            "items": ft_v23_items
        })

    # Save EXACTLY 3 JSON files
    path_bert = os.path.join(OUTPUT_DIR, "korean_bert_onnx_results.json")
    path_v25 = os.path.join(OUTPUT_DIR, "fasttext_v25_results.json")
    path_v23 = os.path.join(OUTPUT_DIR, "fasttext_v23_results.json")

    with open(path_bert, "w", encoding="utf-8") as f:
        json.dump({"model": "Korean BERT ONNX (KoELECTRA)", "total_receipts": len(bert_json_receipts), "total_lines": total_lines, "receipts": bert_json_receipts}, f, ensure_ascii=False, indent=2)

    with open(path_v25, "w", encoding="utf-8") as f:
        json.dump({"model": "FastText v25", "total_receipts": len(ft_v25_json_receipts), "total_lines": total_lines, "receipts": ft_v25_json_receipts}, f, ensure_ascii=False, indent=2)

    with open(path_v23, "w", encoding="utf-8") as f:
        json.dump({"model": "FastText v23", "total_receipts": len(ft_v23_json_receipts), "total_lines": total_lines, "receipts": ft_v23_json_receipts}, f, ensure_ascii=False, indent=2)

    print("\n" + "="*70)
    print("🏆 GENERATED EXACTLY 3 MODEL JSON FILES IN 'qq_json_results/'")
    print("="*70)
    print(f"  1. {path_bert}")
    print(f"  2. {path_v25}")
    print(f"  3. {path_v23}")
    print("="*70)

if __name__ == "__main__":
    run_pipeline()
