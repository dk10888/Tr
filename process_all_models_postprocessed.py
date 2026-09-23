#!/usr/bin/env python3
"""
Post-Processed Model Benchmark Pipeline
=========================================
Runs EasyOCR + 3 Post-Processing Rules (Single-Char Filter, Price Digit Repair, BBox Merging)
across all 31 receipt images for ALL FIVE MODELS:
  1. Korean BERT ONNX (v25)
  2. Korean BERT ONNX (new_aug_v1 — 54k)
  3. FastText v25
  4. FastText v23
  5. FastText new_aug_v1 (54k)

Exports respective model JSON files into 'qq_json_results/'.
"""

import os
import sys
import re
import time
import json
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
BERT_V25_ONNX_PATH = os.path.join(BASE_DIR, "korean_new", "korean_food_classifier_quant.onnx")
BERT_NEW_AUG_ONNX_PATH = os.path.join(BASE_DIR, "korean_new_augv1", "korean_food_classifier_quant.onnx")

FASTTEXT_V25_PATH = os.path.join(BASE_DIR, "fastText_v25", "fasttext_korean_food.ftz")
FASTTEXT_V23_PATH = os.path.join(BASE_DIR, "fastText_v23", "fasttext_korean_food.ftz")
FASTTEXT_NEW_AUG_PATH = os.path.join(BASE_DIR, "fastText_new_aug_v1", "fasttext_korean_food.ftz")

# Single-char whitelist
VALID_SINGLE_CHAR_FOOD_WHITELIST = {
    "밥", "죽", "국", "탕", "면", "차", "술", "떡", "회", "전", "잼", "밀", "파", "김", "소", "닭", "돈", "양"
}

HANGUL_REGEX = re.compile(r'[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]')
JAPANESE_REGEX = re.compile(r'[\u3040-\u309f\u30a0-\u30ff]')

PRICE_PATTERN = re.compile(
    r'^([₩$￥€]?\s*\d{1,3}(,\d{3})*(\.\d{1,2})?\s*원?|\d+\s*원|\d+\.\d{2})$',
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING RULES
# ─────────────────────────────────────────────────────────────────────────────
def is_valid_line(text: str) -> bool:
    clean_t = text.strip()
    if not clean_t:
        return False

    # Standalone single character check
    if len(clean_t) == 1:
        if clean_t not in VALID_SINGLE_CHAR_FOOD_WHITELIST:
            return False  # Drop single-char noise like 꽃, 빠, 움, 합, 8, L

    # Filter out pure symbol noise
    if len(clean_t) <= 2 and not any(c.isalnum() for c in clean_t):
        return False

    return True

def repair_ocr_price_digits(text: str) -> str:
    repaired = text
    repaired = re.sub(r'(\d+)[,\.]?([UOo]{2})', r'\1,000', repaired)
    repaired = re.sub(r'0[OOo]', '000', repaired)
    repaired = re.sub(r'(\d+)[oO]', r'\1 0', repaired)
    return repaired

def extract_smart_lines_postprocessed(ocr_res: List[Any]) -> List[str]:
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

    cleaned_lines = []
    for line in final_lines:
        repaired_line = repair_ocr_price_digits(line)
        if is_valid_line(repaired_line):
            cleaned_lines.append(repaired_line)

    return cleaned_lines

# ─────────────────────────────────────────────────────────────────────────────
# MODEL SUITE LOADERS
# ─────────────────────────────────────────────────────────────────────────────
class AllModelSuite:
    def __init__(self):
        print("📦 Initializing All 5 Models...")
        import fasttext
        import onnxruntime as ort
        from transformers import AutoTokenizer

        # 1. BERT ONNX v25
        self.bert_v25 = ort.InferenceSession(BERT_V25_ONNX_PATH, providers=["CPUExecutionProvider"])
        # 2. BERT ONNX new_aug_v1
        self.bert_new_aug = ort.InferenceSession(BERT_NEW_AUG_ONNX_PATH, providers=["CPUExecutionProvider"])
        self.bert_tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
        print("  ✅ Loaded Korean BERT ONNX v25 & new_aug_v1 (54k)")

        # 3. FastText v25
        self.ft_v25 = fasttext.load_model(FASTTEXT_V25_PATH)
        print("  ✅ Loaded FastText v25")

        # 4. FastText v23
        self.ft_v23 = fasttext.load_model(FASTTEXT_V23_PATH)
        print("  ✅ Loaded FastText v23")

        # 5. FastText new_aug_v1
        self.ft_new_aug = fasttext.load_model(FASTTEXT_NEW_AUG_PATH)
        print("  ✅ Loaded FastText new_aug_v1 (Hangul Visual + Multi-Space 54k)")

    def predict_bert(self, session, text: str) -> Dict[str, Any]:
        t0 = time.time()
        inputs = self.bert_tokenizer(text, return_tensors="np", truncation=True, padding="max_length", max_length=64)
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in inputs:
            ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

        outputs = session.run(None, ort_inputs)
        logits = outputs[0][0]
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)

        pred_id = int(np.argmax(probs))
        label = "food" if pred_id == 1 else "not_food"
        conf = float(probs[pred_id])
        t_ms = (time.time() - t0) * 1000

        return {"label": label, "confidence": round(conf, 4), "time_ms": round(t_ms, 2)}

    def predict_ft(self, model, text: str) -> Dict[str, Any]:
        t0 = time.time()
        labels, probs = model.predict(text.replace("\n", " "))
        label = labels[0].replace("__label__", "") if labels else "not_food"
        conf = float(probs[0]) if probs else 0.0
        t_ms = (time.time() - t0) * 1000
        return {"label": label, "confidence": round(conf, 4), "time_ms": round(t_ms, 3)}

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def run_all_models_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    models = AllModelSuite()

    import easyocr
    reader = easyocr.Reader(['ko', 'en'], gpu=False)

    image_files = sorted([f for f in os.listdir(QQ_DIR) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
    print(f"📸 Running Post-Processed Pipeline across all 5 models on {len(image_files)} receipt images...")

    bert_v25_receipts = []
    bert_new_aug_receipts = []
    ft_v25_receipts = []
    ft_v23_receipts = []
    ft_new_aug_receipts = []

    total_lines = 0

    for idx, img_name in enumerate(image_files, start=1):
        img_path = os.path.join(QQ_DIR, img_name)
        try:
            ocr_res = reader.readtext(img_path)
        except Exception as e:
            continue

        raw_full_text = " ".join([text for _, text, _ in ocr_res])
        if JAPANESE_REGEX.search(raw_full_text) and not HANGUL_REGEX.search(raw_full_text):
            continue

        lines = extract_smart_lines_postprocessed(ocr_res)

        items_b_v25, items_b_new, items_v25, items_v23, items_new_aug = [], [], [], [], []
        food_b_v25 = food_b_new = food_v25 = food_v23 = food_new_aug = 0

        for line_no, line_text in enumerate(lines, start=1):
            total_lines += 1

            # 1. BERT v25
            res_b_v25 = models.predict_bert(models.bert_v25, line_text)
            if res_b_v25["label"] == "food": food_b_v25 += 1
            items_b_v25.append({"line_no": line_no, "text": line_text, "label": res_b_v25["label"], "confidence": res_b_v25["confidence"], "time_ms": res_b_v25["time_ms"]})

            # 2. BERT new_aug_v1 (54k)
            res_b_new = models.predict_bert(models.bert_new_aug, line_text)
            if res_b_new["label"] == "food": food_b_new += 1
            items_b_new.append({"line_no": line_no, "text": line_text, "label": res_b_new["label"], "confidence": res_b_new["confidence"], "time_ms": res_b_new["time_ms"]})

            # 3. FastText v25
            res_v25 = models.predict_ft(models.ft_v25, line_text)
            if res_v25["label"] == "food": food_v25 += 1
            items_v25.append({"line_no": line_no, "text": line_text, "label": res_v25["label"], "confidence": res_v25["confidence"], "time_ms": res_v25["time_ms"]})

            # 4. FastText v23
            res_v23 = models.predict_ft(models.ft_v23, line_text)
            if res_v23["label"] == "food": food_v23 += 1
            items_v23.append({"line_no": line_no, "text": line_text, "label": res_v23["label"], "confidence": res_v23["confidence"], "time_ms": res_v23["time_ms"]})

            # 5. FastText new_aug_v1 (54k)
            res_new = models.predict_ft(models.ft_new_aug, line_text)
            if res_new["label"] == "food": food_new_aug += 1
            items_new_aug.append({"line_no": line_no, "text": line_text, "label": res_new["label"], "confidence": res_new["confidence"], "time_ms": res_new["time_ms"]})

        bert_v25_receipts.append({"receipt_id": idx, "receipt_name": img_name, "line_count": len(lines), "food_count": food_b_v25, "not_food_count": len(lines) - food_b_v25, "items": items_b_v25})
        bert_new_aug_receipts.append({"receipt_id": idx, "receipt_name": img_name, "line_count": len(lines), "food_count": food_b_new, "not_food_count": len(lines) - food_b_new, "items": items_b_new})
        ft_v25_receipts.append({"receipt_id": idx, "receipt_name": img_name, "line_count": len(lines), "food_count": food_v25, "not_food_count": len(lines) - food_v25, "items": items_v25})
        ft_v23_receipts.append({"receipt_id": idx, "receipt_name": img_name, "line_count": len(lines), "food_count": food_v23, "not_food_count": len(lines) - food_v23, "items": items_v23})
        ft_new_aug_receipts.append({"receipt_id": idx, "receipt_name": img_name, "line_count": len(lines), "food_count": food_new_aug, "not_food_count": len(lines) - food_new_aug, "items": items_new_aug})

    # Save all 5 post-processed JSON files
    p_b_v25 = os.path.join(OUTPUT_DIR, "korean_bert_onnx_results.json")
    p_b_new = os.path.join(OUTPUT_DIR, "korean_bert_new_aug_v1_results.json")
    p_v25 = os.path.join(OUTPUT_DIR, "fasttext_v25_results.json")
    p_v23 = os.path.join(OUTPUT_DIR, "fasttext_v23_results.json")
    p_new = os.path.join(OUTPUT_DIR, "fasttext_new_aug_v1_results.json")

    with open(p_b_v25, "w", encoding="utf-8") as f:
        json.dump({"model": "Korean BERT ONNX (v25)", "postprocessed": True, "total_receipts": len(bert_v25_receipts), "total_lines": total_lines, "receipts": bert_v25_receipts}, f, ensure_ascii=False, indent=2)

    with open(p_b_new, "w", encoding="utf-8") as f:
        json.dump({"model": "Korean BERT ONNX (new_aug_v1 — 54k dataset)", "postprocessed": True, "total_receipts": len(bert_new_aug_receipts), "total_lines": total_lines, "receipts": bert_new_aug_receipts}, f, ensure_ascii=False, indent=2)

    with open(p_v25, "w", encoding="utf-8") as f:
        json.dump({"model": "FastText v25", "postprocessed": True, "total_receipts": len(ft_v25_receipts), "total_lines": total_lines, "receipts": ft_v25_receipts}, f, ensure_ascii=False, indent=2)

    with open(p_v23, "w", encoding="utf-8") as f:
        json.dump({"model": "FastText v23", "postprocessed": True, "total_receipts": len(ft_v23_receipts), "total_lines": total_lines, "receipts": ft_v23_receipts}, f, ensure_ascii=False, indent=2)

    with open(p_new, "w", encoding="utf-8") as f:
        json.dump({"model": "FastText new_aug_v1 (Hangul Visual 54k)", "postprocessed": True, "total_receipts": len(ft_new_aug_receipts), "total_lines": total_lines, "receipts": ft_new_aug_receipts}, f, ensure_ascii=False, indent=2)

    print("\n" + "="*80)
    print("🏆 POST-PROCESSED EVALUATION COMPLETE ACROSS ALL 5 MODELS!")
    print("="*80)
    print(f"  1. {p_b_v25}")
    print(f"  2. {p_b_new}")
    print(f"  3. {p_v25}")
    print(f"  4. {p_v23}")
    print(f"  5. {p_new}")
    print("="*80)

if __name__ == "__main__":
    run_all_models_pipeline()
