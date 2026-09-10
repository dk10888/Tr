"""
extract_items.py

Reads all COCO annotations across train/valid/test.
For every bounding box with category = "Item":
  1. Crops that region from the image
  2. Runs PP-OCRv6 on the crop to get ALL detected text lines
  3. Stores EACH line as a separate record (not joined into one string)
  4. Deduplicates per-line (case-insensitive)

Output:
  data/all_items.txt              – one OCR line per row, deduplicated
  data/all_items_with_source.jsonl – full detail per OCR line (image/split/bbox/line_index)
  data/extraction_checkpoint.jsonl – incremental checkpoint (auto-resume on restart)
"""

import json
import os
import sys
import cv2
from rapidocr_onnxruntime import RapidOCR

# ── Config ─────────────────────────────────────────────────────────────
RECEIPTS_DIR  = "Receipts"
OUTPUT_DIR    = "data"
SPLITS        = ["train", "valid", "test"]
FOOD_CATEGORY = "Item"
BOX_PADDING   = 4
MIN_CROP_W    = 20
MIN_CROP_H    = 10

# ── Load OCR ────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "models", "onnx")
det_path   = os.path.join(models_dir, "v6_medium_det.onnx")
rec_path   = os.path.join(models_dir, "v6_medium_rec.onnx")
rec_keys   = os.path.join(models_dir, "rec_keys.txt")

print("Loading PP-OCRv6 medium...", flush=True)
if os.path.exists(det_path) and os.path.exists(rec_path):
    ocr = RapidOCR(det_model_path=det_path, rec_model_path=rec_path, rec_keys_path=rec_keys)
    print("  Using PP-OCRv6 medium", flush=True)
else:
    ocr = RapidOCR()
    print("  Warning: v6 medium not found, using bundled default", flush=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def ocr_crop_lines(img, x, y, w, h):
    """
    Crops the bounding box from the image and runs OCR.
    Returns a LIST of (text, confidence, line_index) tuples — one per detected line.
    Lines are sorted top-to-bottom by their vertical center position.
    Returns an empty list if the crop is too small or OCR finds nothing.
    """
    H, W = img.shape[:2]
    x1 = max(0, int(x) - BOX_PADDING)
    y1 = max(0, int(y) - BOX_PADDING)
    x2 = min(W, int(x + w) + BOX_PADDING)
    y2 = min(H, int(y + h) + BOX_PADDING)

    if (x2 - x1) < MIN_CROP_W or (y2 - y1) < MIN_CROP_H:
        return []

    crop = img[y1:y2, x1:x2]
    result, _ = ocr(crop)
    if not result:
        return []

    # result: list of (box, text, confidence)
    # Sort by vertical center of bounding box (top-to-bottom reading order)
    def y_center(r):
        box = r[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        ys = [pt[1] for pt in box]
        return (min(ys) + max(ys)) / 2

    sorted_result = sorted(result, key=y_center)

    lines = []
    for line_idx, r in enumerate(sorted_result):
        text = r[1].strip()
        conf = round(float(r[2]), 4) if r[2] is not None else None
        if text:
            lines.append((text, conf, line_idx))

    return lines


# ── Checkpoint / Resume Support ──────────────────────────────────────────
ckpt_path = os.path.join(OUTPUT_DIR, "extraction_checkpoint.jsonl")
all_records        = []
seen_texts         = set()   # for deduplication (lowercased lines)
processed_ann_ids  = set()   # already-done annotation IDs

if os.path.exists(ckpt_path):
    print(f"Found checkpoint at {ckpt_path}, loading previous progress...", flush=True)
    with open(ckpt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                all_records.append(rec)
                seen_texts.add(rec["text"].lower().strip())
                processed_ann_ids.add(rec.get("ann_id"))
    print(f"  Loaded {len(all_records)} line-records from checkpoint "
          f"({len(seen_texts)} unique texts).", flush=True)

total_boxes  = 0   # annotation boxes attempted
no_text      = 0   # boxes with zero OCR output
total_lines  = len(all_records)  # individual line records (includes resumed)

ckpt_file = open(ckpt_path, "a", encoding="utf-8")

# ── Main Loop ────────────────────────────────────────────────────────────
for split in SPLITS:
    split_dir = os.path.join(RECEIPTS_DIR, split)
    coco_path = os.path.join(split_dir, "_annotations.coco.json")

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
            continue  # already processed in a previous run

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

        # Get each OCR line separately (NOT joined)
        lines = ocr_crop_lines(img, x, y, w, h)

        if not lines:
            no_text += 1
            continue

        for text, conf, line_idx in lines:
            key        = text.lower().strip()
            is_dup     = key in seen_texts
            seen_texts.add(key)
            total_lines += 1

            record = {
                "ann_id":     ann_key,          # unique annotation ID
                "line_index": line_idx,          # position within the box (0 = top)
                "text":       text,              # the OCR text for this single line
                "confidence": conf,
                "split":      split,
                "image":      img_info["file_name"],
                "bbox":       [x, y, w, h],      # the annotation bounding box
                "duplicate":  is_dup,
                "label":      ""                 # to be filled by Gemini / manually
            }
            all_records.append(record)
            ckpt_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        ckpt_file.flush()

        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(item_anns)} boxes | "
                  f"total lines so far: {total_lines} | "
                  f"unique: {len(seen_texts)} | no_text: {no_text}", flush=True)

ckpt_file.close()

# ── Write outputs ─────────────────────────────────────────────────────────
# 1. Plain text — unique lines only, one per row (paste into Gemini to label)
txt_path     = os.path.join(OUTPUT_DIR, "all_items.txt")
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

# 2. Full JSONL — every line-record, with metadata and empty label field
jsonl_path = os.path.join(OUTPUT_DIR, "all_items_with_source.jsonl")
with open(jsonl_path, "w", encoding="utf-8") as f:
    for rec in all_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"Annotation boxes attempted : {total_boxes}")
print(f"Boxes with no OCR text     : {no_text}")
print(f"Total LINE records written : {len(all_records)}")
print(f"Unique item line texts     : {len(unique_texts)}")
print(f"\nOutput files:")
print(f"  {txt_path}")
print(f"    → Each row is ONE OCR line. Paste into Gemini to label food/not_food.")
print(f"  {jsonl_path}")
print(f"    → Full records with ann_id, line_index, bbox, confidence.")
