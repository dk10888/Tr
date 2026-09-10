"""
extract_training_data.py

Reads the COCO annotations from train/valid/test splits.
For each annotated bounding box:
  1. Crops the region from the receipt image
  2. Runs PP-OCRv6 on the crop to get text
  3. Labels it: "food" (category=Item) or "not_food" (everything else)

Also grabs unannotated image regions and adds them as extra "not_food" examples
so the model sees real leftover junk (barcodes, payment lines, footers).

Output: data/train.jsonl, data/valid.jsonl, data/test.jsonl
Each line: {"text": "...", "label": "food"/"not_food", "source_label": "Item"/"Tax"/...}

Then builds the 3-line sliding-window context around each line.
"""

import json
import os
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

# ── Config ─────────────────────────────────────────────────────────────
RECEIPTS_DIR = "Receipts"
OUTPUT_DIR   = "data"
SPLITS       = ["train", "valid", "test"]

# How much padding to add around each bounding box crop (pixels)
# Helps OCR not get cut-off characters at edges
BOX_PADDING  = 4

# Minimum crop size to bother OCR-ing (skip tiny stray boxes)
MIN_CROP_W   = 20
MIN_CROP_H   = 10

# Category that maps to "food" — everything else → "not_food"
FOOD_CATEGORY = "Item"

# ── Load OCR ────────────────────────────────────────────────────────────
script_dir  = os.path.dirname(os.path.abspath(__file__))
models_dir  = os.path.join(script_dir, "models", "onnx")
det_path    = os.path.join(models_dir, "v6_medium_det.onnx")
rec_path    = os.path.join(models_dir, "v6_medium_rec.onnx")
rec_keys    = os.path.join(models_dir, "rec_keys.txt")

print("Loading OCR model...")
if os.path.exists(det_path) and os.path.exists(rec_path):
    ocr = RapidOCR(det_model_path=det_path, rec_model_path=rec_path, rec_keys_path=rec_keys)
    print("  Using PP-OCRv6 medium")
else:
    ocr = RapidOCR()
    print("  Falling back to bundled PP-OCRv4")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def ocr_crop(img, x, y, w, h):
    """Crop + pad a region from the image, run OCR, return joined text."""
    H, W = img.shape[:2]
    x1 = max(0, int(x) - BOX_PADDING)
    y1 = max(0, int(y) - BOX_PADDING)
    x2 = min(W, int(x + w) + BOX_PADDING)
    y2 = min(H, int(y + h) + BOX_PADDING)

    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w < MIN_CROP_W or crop_h < MIN_CROP_H:
        return None

    crop = img[y1:y2, x1:x2]
    result, _ = ocr(crop)
    if not result:
        return None

    # Join all OCR lines in this crop into one string
    return " ".join(item[1] for item in result).strip()


def get_unannotated_strips(img, annotations, num_strips=5):
    """
    Sample horizontal strips from areas of the image NOT covered by any
    bounding box. These are the true leftover junk lines (barcodes, 
    payment method lines, footers, etc.) — add as extra not_food examples.
    """
    H, W = img.shape[:2]

    # Build a mask of annotated Y-ranges
    annotated_ys = set()
    for ann in annotations:
        bx, by, bw, bh = ann["bbox"]
        for y in range(int(by), int(by + bh)):
            annotated_ys.add(y)

    # Find unannotated row runs
    unannotated_rows = [y for y in range(H) if y not in annotated_ys]
    if not unannotated_rows:
        return []

    # Sample a few horizontal strips from unannotated regions
    strips = []
    step = max(1, len(unannotated_rows) // (num_strips + 1))
    for i in range(1, num_strips + 1):
        center_y = unannotated_rows[min(i * step, len(unannotated_rows) - 1)]
        y1 = max(0, center_y - 10)
        y2 = min(H, center_y + 10)
        strip = img[y1:y2, 0:W]
        if strip.shape[0] < MIN_CROP_H:
            continue
        result, _ = ocr(strip)
        if result:
            text = " ".join(item[1] for item in result).strip()
            if text:
                strips.append(text)
    return strips


def build_sliding_window(records):
    """
    Wrap each record with its prev/next line as context.
    Input:  list of {"text": ..., "label": ..., "source_label": ...}
    Output: same list with "context" added:
              "[prev text] [SEP] [current text] [SEP] [next text]"
    """
    windowed = []
    for i, rec in enumerate(records):
        prev_text = records[i - 1]["text"] if i > 0          else ""
        next_text = records[i + 1]["text"] if i < len(records) - 1 else ""
        context = f"{prev_text} [SEP] {rec['text']} [SEP] {next_text}".strip()
        windowed.append({**rec, "context": context})
    return windowed


def process_split(split):
    split_dir   = os.path.join(RECEIPTS_DIR, split)
    coco_path   = os.path.join(split_dir, "_annotations.coco.json")
    output_path = os.path.join(OUTPUT_DIR, f"{split}.jsonl")

    with open(coco_path, "r") as f:
        coco = json.load(f)

    # Build lookups
    cat_map   = {c["id"]: c["name"] for c in coco["categories"]}
    img_map   = {img["id"]: img      for img in coco["images"]}
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    records  = []
    total    = len(coco["images"])
    skipped  = 0
    no_text  = 0

    for idx, img_info in enumerate(coco["images"]):
        img_path = os.path.join(split_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            skipped += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            skipped += 1
            continue

        image_records = []
        anns = ann_by_img.get(img_info["id"], [])

        # Sort annotations top-to-bottom so sliding window order is correct
        anns_sorted = sorted(anns, key=lambda a: a["bbox"][1])

        for ann in anns_sorted:
            cat_name = cat_map.get(ann["category_id"], "Unknown")
            label    = "food" if cat_name == FOOD_CATEGORY else "not_food"
            x, y, w, h = ann["bbox"]

            text = ocr_crop(img, x, y, w, h)
            if not text:
                no_text += 1
                continue

            image_records.append({
                "text":         text,
                "label":        label,
                "source_label": cat_name,
                "image":        img_info["file_name"],
            })

        # Add unannotated strips as extra not_food examples
        junk_texts = get_unannotated_strips(img, anns)
        for jt in junk_texts:
            image_records.append({
                "text":         jt,
                "label":        "not_food",
                "source_label": "Unannotated",
                "image":        img_info["file_name"],
            })

        # Apply 3-line sliding-window context within this receipt
        image_records = build_sliding_window(image_records)
        records.extend(image_records)

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            food_count     = sum(1 for r in records if r["label"] == "food")
            not_food_count = sum(1 for r in records if r["label"] == "not_food")
            print(f"  [{split}] {idx+1}/{total} images | "
                  f"{len(records)} records | "
                  f"food={food_count} not_food={not_food_count} | "
                  f"no_text_skipped={no_text}")

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    food_count     = sum(1 for r in records if r["label"] == "food")
    not_food_count = sum(1 for r in records if r["label"] == "not_food")
    print(f"\n[{split}] Done → {output_path}")
    print(f"  Total records : {len(records)}")
    print(f"  food          : {food_count}")
    print(f"  not_food      : {not_food_count}")
    print(f"  images skipped: {skipped}")
    print(f"  boxes no text : {no_text}\n")
    return records


# ── Main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nExtracting training data from {RECEIPTS_DIR}/\n{'='*50}")
    all_counts = {}
    for split in SPLITS:
        print(f"\nProcessing {split}...")
        recs = process_split(split)
        all_counts[split] = len(recs)

    print("\n" + "="*50)
    print("Summary:")
    for split, count in all_counts.items():
        print(f"  {split:6s}: {count} records → data/{split}.jsonl")
    print("\nNext step: run train_bert.py to fine-tune BERT-tiny on this data.")
