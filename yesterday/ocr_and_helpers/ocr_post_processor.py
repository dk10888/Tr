#!/usr/bin/env python3
"""
OCR Post-Processing Pipeline (Lightweight Cleanup Module)
==========================================================
Applies 3 post-processing rules right after OCR before classification:

1. Filter Out Isolated Single-Character Residue:
   - Drops standalone 1-character fragments (e.g. 꽃, 빠, 움, 합, 8, L)
     UNLESS the character is in a whitelist of valid single-character food words (밥, 죽, 국, 탕, 면, 차, 술, 떡, 회, 전, 잼, 밀, 파, 김).

2. Common OCR Character Repair (Price & Numeric Repair):
   - Repairs digit/letter confusion in price columns (e.g., 13,5UU -> 13,500, 0OO -> 000).

3. Word-Level Bounding Box Merging (Gap-Detection Clustering):
   - Merges same-row word fragments (e.g. 포 + 테이 + 토 -> 포테이토) while splitting off far-right price columns.
"""

import os
import sys
import re
import json
import fasttext
from typing import List, Dict, Any

# Valid single-character Korean food words whitelist
VALID_SINGLE_CHAR_FOOD_WHITELIST = {
    "밥", "죽", "국", "탕", "면", "차", "술", "떡", "회", "전", "잼", "밀", "파", "김", "소", "닭", "돈", "양"
}

# Regex unicode ranges
HANGUL_REGEX = re.compile(r'[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]')
JAPANESE_REGEX = re.compile(r'[\u3040-\u309f\u30a0-\u30ff]') # Hiragana & Katakana

PRICE_PATTERN = re.compile(
    r'^([₩$￥€]?\s*\d{1,3}(,\d{3})*(\.\d{1,2})?\s*원?|\d+\s*원|\d+\.\d{2})$',
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────────────────────
# RULE 1: FILTER OUT ISOLATED SINGLE-CHARACTER RESIDUE
# ─────────────────────────────────────────────────────────────────────────────
def is_valid_line(text: str) -> bool:
    """
    Filters out unrecoverable 1-character OCR noise unless it's in the food whitelist.
    """
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

# ─────────────────────────────────────────────────────────────────────────────
# RULE 2: COMMON OCR CHARACTER REPAIR (DIGIT / PRICE CLEANER)
# ─────────────────────────────────────────────────────────────────────────────
def repair_ocr_price_digits(text: str) -> str:
    """
    Fixes common OCR digit-letter confusion in numeric and price patterns:
    13,5UU -> 13,500
    0OO -> 000
    4,3oU -> 4,300
    """
    # Fix price trailing 'UU', 'OO', 'oU', 'OU'
    repaired = text
    repaired = re.sub(r'(\d+)[,\.]?([UOo]{2})', r'\1,000', repaired)
    repaired = re.sub(r'0[OOo]', '000', repaired)
    repaired = re.sub(r'(\d+)[oO]', r'\1 0', repaired)

    return repaired

# ─────────────────────────────────────────────────────────────────────────────
# RULE 3: WORD-LEVEL BOUNDING BOX MERGING (GAP Splitting)
# ─────────────────────────────────────────────────────────────────────────────
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

    # Same-row vertical height tolerance <= 15px
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

    # Apply Post-Processing Rule 1 & 2
    cleaned_lines = []
    for line in final_lines:
        repaired_line = repair_ocr_price_digits(line)
        if is_valid_line(repaired_line):
            cleaned_lines.append(repaired_line)

    return cleaned_lines

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE EXECUTION ON REAL RECEIPTS
# ─────────────────────────────────────────────────────────────────────────────
def run_postprocessed_pipeline():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    QQ_DIR = os.path.join(BASE_DIR, "qq")
    MODEL_PATH = os.path.join(BASE_DIR, "fastText_new_aug_v1", "fasttext_korean_food.ftz")

    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = os.path.join(BASE_DIR, "fastText_v25", "fasttext_korean_food.ftz")

    print(f"📦 Loading FastText Model from: {MODEL_PATH}")
    model = fasttext.load_model(MODEL_PATH)

    import easyocr
    reader = easyocr.Reader(['ko', 'en'], gpu=False)

    image_files = sorted([f for f in os.listdir(QQ_DIR) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
    print(f"📸 Running Post-Processing Pipeline on {len(image_files)} receipt images...")

    receipt_results = []
    total_lines = 0
    total_food = 0

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
        food_cnt = 0
        line_items = []

        for line_no, line_text in enumerate(lines, start=1):
            total_lines += 1
            lbls, prbs = model.predict(line_text.replace("\n", " "))
            pred = lbls[0].replace("__label__", "") if lbls else "not_food"
            conf = float(prbs[0]) if prbs else 0.0

            if pred == "food":
                food_cnt += 1
                total_food += 1

            line_items.append({
                "line_no": line_no,
                "text": line_text,
                "label": pred,
                "confidence": round(conf, 4)
            })

        receipt_results.append({
            "receipt_id": idx,
            "receipt_name": img_name,
            "line_count": len(lines),
            "food_count": food_cnt,
            "not_food_count": len(lines) - food_cnt,
            "items": line_items
        })

    out_json = os.path.join(BASE_DIR, "qq_json_results", "postprocessed_fasttext_new_aug_v1_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "pipeline": "OCR Post-Processing (Single-Char Filter + Price Digit Repair + BBox Merging)",
            "total_receipts": len(receipt_results),
            "total_lines": total_lines,
            "total_food_detected": total_food,
            "receipts": receipt_results
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "="*70)
    print("🏆 OCR POST-PROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"  Total Lines Evaluated after Post-Processing: {total_lines}")
    print(f"  Total Food Items Detected                  : {total_food}")
    print(f"  📄 Saved Output JSON                        : {out_json}")
    print("="*70)

if __name__ == "__main__":
    run_postprocessed_pipeline()
