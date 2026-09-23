#!/usr/bin/env python3
"""
QQ Receipts Korean Filter & Single JSON Generator (Smart Line Extraction)
========================================================================
1. Scans images in 'qq/' directory using EasyOCR (Korean + English).
2. Uses Smart Line Clustering:
   - Merges multi-word item names on the same horizontal row (e.g. "Pan Fried Gyoza" + "Pork" -> "Pan Fried Gyoza Pork").
   - SEPARATES price columns on the far right if horizontal gap > 70px or if text matches price pattern.
3. Filters out non-Korean receipts (Japanese Hiragana/Katakana or pure Chinese without Hangul) and deletes them.
4. Classifies text lines using Korean BERT model in 'korean_new/'.
5. Saves ALL processed receipt data into 1 single combined JSON file ('combined_korean_receipts_dataset.json').
"""

import os
import sys
import re
import json
import shutil
from typing import List, Dict, Any, Tuple

# Insert user site-packages for Mac system python
user_site = os.path.expanduser('~/Library/Python/3.9/lib/python/site-packages')
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QQ_DIR = os.path.join(BASE_DIR, "qq")
KOREAN_NEW_DIR = os.path.join(BASE_DIR, "korean_new")
OUTPUT_JSON_PATH = os.path.join(BASE_DIR, "combined_korean_receipts_dataset.json")

# Regex unicode ranges
HANGUL_REGEX = re.compile(r'[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]')
JAPANESE_REGEX = re.compile(r'[\u3040-\u309f\u30a0-\u30ff]') # Hiragana & Katakana
KANJI_HANZI_REGEX = re.compile(r'[\u4e00-\u9fff]')

# Price & Currency Regex Pattern
PRICE_PATTERN = re.compile(
    r'^([₩$￥€]?\s*\d{1,3}(,\d{3})*(\.\d{1,2})?\s*원?|\d+\s*원|\d+\.\d{2})$',
    re.IGNORECASE
)


def extract_smart_receipt_lines(easy_res: List[Any]) -> List[str]:
    """
    Smart Line Extractor:
    - Merges word fragments on the same horizontal row (within 15px height tolerance).
    - SEPARATES the price column on the far right if horizontal gap > 70px or if fragment matches a price pattern.
    """
    if not easy_res:
        return []

    items = []
    for bbox, text, score in easy_res:
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
            'center_y': center_y,
            'score': score
        })

    if not items:
        return []

    # 1. Group into vertical row clusters (same horizontal line within 15px)
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

    # 2. For each row cluster, merge left-to-right BUT split large horizontal gaps (>70px) or price patterns
    for cluster in row_clusters:
        left_to_right = sorted(cluster, key=lambda x: x['left'])
        current_words = []
        last_right = -1

        for info in left_to_right:
            txt = info['text']
            is_price = bool(PRICE_PATTERN.match(txt))
            x_gap = (info['left'] - last_right) if last_right != -1 else 0

            # If gap > 70px OR this fragment is a standalone price -> Split into separate line item!
            if current_words and (x_gap > 70 or is_price):
                final_lines.append(" ".join(current_words))
                current_words = []

            current_words.append(txt)
            last_right = info['right']

        if current_words:
            final_lines.append(" ".join(current_words))

    return final_lines


def is_korean_receipt(lines: List[str]) -> Tuple[bool, str]:
    """
    Analyzes OCR lines to determine if receipt is Korean.
    Returns (is_korean: bool, reason: str).
    """
    full_text = " ".join(lines)
    if not full_text.strip():
        return True, "empty_text"

    hangul_count = len(HANGUL_REGEX.findall(full_text))
    japanese_count = len(JAPANESE_REGEX.findall(full_text))
    kanji_count = len(KANJI_HANZI_REGEX.findall(full_text))

    if japanese_count > 0:
        return False, f"Japanese characters detected (Hiragana/Katakana count: {japanese_count})"

    if kanji_count > 2 and hangul_count == 0:
        return False, f"Chinese/Kanji characters detected without any Korean Hangul (Kanji count: {kanji_count})"

    return True, f"Valid Korean receipt (Hangul character count: {hangul_count})"


class KoreanBERTInference:
    """Loads Korean BERT model from korean_new directory."""
    
    def __init__(self, model_dir: str):
        self.session = None
        self.tokenizer = None
        self.label_map = {0: "not_food", 1: "food"}
        self._init_model(model_dir)

    def _init_model(self, model_dir: str):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        quant_model = os.path.join(model_dir, "korean_food_classifier_quant.onnx")
        fp32_model = os.path.join(model_dir, "korean_food_classifier.onnx")

        target_onnx = quant_model if os.path.exists(quant_model) else fp32_model
        if not os.path.exists(target_onnx):
            raise FileNotFoundError(f"No ONNX model found in {model_dir}")

        print(f"📦 Loading ONNX model from: {target_onnx}")
        self.session = ort.InferenceSession(target_onnx, providers=["CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
        print("✅ Korean BERT ONNX model initialized.")

    def predict(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if not text:
            return {"label": "not_food", "confidence": 0.0}

        import numpy as np
        enc = self.tokenizer(text, return_tensors="np", max_length=64, truncation=True, padding="max_length")
        inputs = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in enc:
            inputs["token_type_ids"] = enc["token_type_ids"].astype(np.int64)
        elif "token_type_ids" in [inp.name for inp in self.session.get_inputs()]:
            inputs["token_type_ids"] = np.zeros_like(enc["input_ids"], dtype=np.int64)

        logits = self.session.run(["logits"], inputs)[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        pred_id = int(np.argmax(probs))
        return {
            "label": self.label_map.get(pred_id, "not_food"),
            "confidence": round(float(probs[pred_id]), 4)
        }


def main():
    print("=" * 60)
    print("QQ RECEIPTS KOREAN FILTER & SINGLE JSON GENERATOR (SMART LINE MERGE)")
    print("=" * 60)

    if not os.path.exists(QQ_DIR):
        print(f"❌ Error: Folder '{QQ_DIR}' does not exist.")
        return

    # 1. Initialize EasyOCR with Korean + English
    import easyocr
    print("📷 Initializing EasyOCR Engine (Korean + English)...")
    ocr_reader = easyocr.Reader(['ko', 'en'], gpu=False)
    print("✅ EasyOCR Engine initialized.")

    # 2. Initialize Korean BERT Model
    classifier = KoreanBERTInference(KOREAN_NEW_DIR)

    # 3. Find files in qq/
    valid_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    all_files = [f for f in sorted(os.listdir(QQ_DIR)) if f.lower().endswith(valid_exts)]
    print(f"📁 Total receipt images found in 'qq/': {len(all_files)}")

    deleted_files = []
    processed_receipts = []
    all_combined_dataset_rows = []

    total_lines_count = 0
    total_food_count = 0
    total_not_food_count = 0

    for idx, fname in enumerate(all_files, start=1):
        fpath = os.path.join(QQ_DIR, fname)
        print(f"\n[{idx}/{len(all_files)}] Processing: {fname}")

        # Smart OCR extraction
        easy_res = ocr_reader.readtext(fpath)
        raw_lines = extract_smart_receipt_lines(easy_res)

        # Check language
        is_kor, reason = is_korean_receipt(raw_lines)
        if not is_kor:
            print(f"  ❌ DELETING NON-KOREAN RECEIPT: {fname} ({reason})")
            os.remove(fpath)
            deleted_files.append({"file_name": fname, "reason": reason})
            continue

        print(f"  ✅ Korean receipt verified ({len(raw_lines)} smart lines extracted)")

        # Run BERT Classification
        items = []
        r_food = 0
        r_not_food = 0

        for line_no, line_text in enumerate(raw_lines, start=1):
            pred = classifier.predict(line_text)
            lbl = pred["label"]
            conf = pred["confidence"]

            if lbl == "food":
                r_food += 1
            else:
                r_not_food += 1

            item_obj = {
                "line_no": line_no,
                "text": line_text,
                "label": lbl,
                "confidence": conf,
                "source": "korean_new"
            }
            items.append(item_obj)

            clean_txt = f'"{line_text}"' if ',' in line_text else line_text
            all_combined_dataset_rows.append(f"{clean_txt},{lbl},korean_new")

        total_lines_count += len(items)
        total_food_count += r_food
        total_not_food_count += r_not_food

        receipt_obj = {
            "receipt_id": len(processed_receipts) + 1,
            "receipt_name": fname,
            "line_count": len(items),
            "food_count": r_food,
            "not_food_count": r_not_food,
            "items": items
        }
        processed_receipts.append(receipt_obj)

    # Compile final single JSON
    final_output = {
        "summary": {
            "total_initial_images": len(all_files),
            "total_korean_receipts": len(processed_receipts),
            "total_non_korean_deleted": len(deleted_files),
            "deleted_files": deleted_files,
            "total_extracted_lines": total_lines_count,
            "total_food_items": total_food_count,
            "total_not_food_items": total_not_food_count
        },
        "receipts": processed_receipts,
        "combined_dataset_rows": all_combined_dataset_rows
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY & RESULT")
    print("=" * 60)
    print(f"  Total Images Processed : {len(all_files)}")
    print(f"  Valid Korean Receipts  : {len(processed_receipts)}")
    print(f"  Non-Korean Deleted     : {len(deleted_files)}")
    print(f"  Total Lines Extracted  : {total_lines_count:,}")
    print(f"  Food Items Labeled     : {total_food_count:,}")
    print(f"  Not-Food Items Labeled : {total_not_food_count:,}")
    print(f"  Single Output JSON     : {OUTPUT_JSON_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
