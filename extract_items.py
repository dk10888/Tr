"""
extract_items.py

Reads all COCO annotations across train/valid/test.
For every bounding box with category = "Item":
  1. Crops that region from the image
  2. Runs PP-OCRv6 on the crop to get the text
  3. Deduplicates (case-insensitive)

Output: data/all_items.txt — one item text per line, deduplicated
        data/all_items_with_source.jsonl — full detail with image/split info
"""

import json
import os
import cv2
from rapidocr_onnxruntime import RapidOCR

# ── Config ─────────────────────────────────────────────────────────────
RECEIPTS_DIR = "Receipts"
OUTPUT_DIR   = "data"
SPLITS       = ["train", "valid", "test"]
FOOD_CATEGORY = "Item"
BOX_PADDING  = 4
MIN_CROP_W   = 20
MIN_CROP_H   = 10

# ── Load OCR ────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "models", "onnx")
det_path   = os.path.join(models_dir, "v6_medium_det.onnx")
rec_path   = os.path.join(models_dir, "v6_medium_rec.onnx")
rec_keys   = os.path.join(models_dir, "rec_keys.txt")

import sys
print("Loading PP-OCRv6 medium...", flush=True)
if os.path.exists(det_path) and os.path.exists(rec_path):
    ocr = RapidOCR(det_model_path=det_path, rec_model_path=rec_path, rec_keys_path=rec_keys)
    print("  Using PP-OCRv6 medium", flush=True)
else:
    ocr = RapidOCR()
    print("  Warning: v6 medium not found, using bundled v4", flush=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def ocr_crop(img, x, y, w, h):
    H, W = img.shape[:2]
    x1 = max(0, int(x) - BOX_PADDING)
    y1 = max(0, int(y) - BOX_PADDING)
    x2 = min(W, int(x + w) + BOX_PADDING)
    y2 = min(H, int(y + h) + BOX_PADDING)
    if (x2 - x1) < MIN_CROP_W or (y2 - y1) < MIN_CROP_H:
        return None
    crop = img[y1:y2, x1:x2]
    result, _ = ocr(crop)
    if not result:
        return None
    return " ".join(r[1] for r in result).strip()


# ── Checkpoint / Resume Support ─────────────────────────────────────────
ckpt_path = os.path.join(OUTPUT_DIR, "extraction_checkpoint.jsonl")
all_records = []
seen_texts = set()
processed_ann_ids = set()

if os.path.exists(ckpt_path):
    print(f"Found checkpoint at {ckpt_path}, loading previous progress...", flush=True)
    with open(ckpt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                all_records.append(rec)
                seen_texts.add(rec["text"].lower().strip())
                processed_ann_ids.add(rec.get("ann_id"))
    print(f"  Loaded {len(all_records)} records ({len(seen_texts)} unique) from checkpoint.", flush=True)

total_boxes = len(all_records)
no_text     = 0

ckpt_file = open(ckpt_path, "a", encoding="utf-8")

for split in SPLITS:
    split_dir  = os.path.join(RECEIPTS_DIR, split)
    coco_path  = os.path.join(split_dir, "_annotations.coco.json")

    with open(coco_path) as f:
        coco = json.load(f)

    cat_map  = {c["id"]: c["name"] for c in coco["categories"]}
    img_map  = {img["id"]: img      for img in coco["images"]}

    # Keep only Item annotations
    item_anns = [a for a in coco["annotations"]
                 if cat_map.get(a["category_id"]) == FOOD_CATEGORY]

    print(f"\n[{split}] {len(item_anns)} Item boxes across "
          f"{len(coco['images'])} images", flush=True)

    for i, ann in enumerate(item_anns):
        ann_key = f"{split}_{ann.get('id', i)}"
        if ann_key in processed_ann_ids:
            continue

        total_boxes += 1
        img_info = img_map[ann["image_id"]]
        img_path = os.path.join(split_dir, img_info["file_name"])

        if not os.path.exists(img_path):
            no_text += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            no_text += 1
            continue

        x, y, w, h = ann["bbox"]
        text = ocr_crop(img, x, y, w, h)

        if not text:
            no_text += 1
            continue

        key = text.lower().strip()
        is_duplicate = key in seen_texts
        seen_texts.add(key)

        record = {
            "ann_id":    ann_key,
            "text":      text,
            "split":     split,
            "image":     img_info["file_name"],
            "bbox":      [x, y, w, h],
            "duplicate": is_duplicate,
            "label":     ""    # to be filled by you / Gemini
        }
        all_records.append(record)
        ckpt_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        ckpt_file.flush()

        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(item_anns)} | "
                  f"unique so far: {len(seen_texts)} | no_text: {no_text}", flush=True)

ckpt_file.close()

# ── Write outputs ────────────────────────────────────────────────────────
# 1. Plain text — unique items only, one per line (for pasting into Gemini)
txt_path = os.path.join(OUTPUT_DIR, "all_items.txt")
unique_texts = []
seen_for_txt = set()
for rec in all_records:
    key = rec["text"].lower().strip()
    if key not in seen_for_txt:
        seen_for_txt.add(key)
        unique_texts.append(rec["text"])

with open(txt_path, "w", encoding="utf-8") as f:
    for t in unique_texts:
        f.write(t + "\n")

# 2. Full JSONL — all records including duplicates, with empty label field
jsonl_path = os.path.join(OUTPUT_DIR, "all_items_with_source.jsonl")
with open(jsonl_path, "w", encoding="utf-8") as f:
    for rec in all_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"Total Item boxes processed : {total_boxes}")
print(f"Boxes with no OCR text     : {no_text}")
print(f"Total records written      : {len(all_records)}")
print(f"Unique item texts          : {len(unique_texts)}")
print(f"\nOutput files:")
print(f"  {txt_path}")
print(f"    → Paste this into Gemini/ChatGPT to label food/not_food")
print(f"  {jsonl_path}")
print(f"    → Full records (for building training data after labeling)")
