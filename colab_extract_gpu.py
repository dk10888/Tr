"""
Receipt Item Extractor — Google Colab (GPU)
============================================
Run this notebook on Colab with GPU:
  Runtime → Change runtime type → T4 GPU

This uses PaddlePaddle + PaddleOCR which natively supports CUDA GPU acceleration.
PaddleOCR v4 (PP-OCRv4 server model) gives equivalent or better accuracy vs PP-OCRv6 on GPU.

Steps:
  1. Install dependencies
  2. Clone your GitHub repo
  3. Run OCR over all Item bounding boxes (each OCR line → one record)
  4. Download results
"""

# ============================================================
# CELL 1: Install Dependencies (Run once)
# ============================================================
!pip install -q paddlepaddle-gpu paddleocr opencv-python-headless
# Note: paddlepaddle-gpu automatically uses CUDA if available.
# On Colab T4, this gives ~5-10x speedup over CPU.

import subprocess
result = subprocess.run(["python", "-c", "import paddle; print('CUDA available:', paddle.is_compiled_with_cuda()); print('Device:', paddle.get_device())"], capture_output=True, text=True)
print(result.stdout)


# ============================================================
# CELL 2: Clone your GitHub repository
# ============================================================
import os

REPO_URL = "https://github.com/dk10888/Tr.git"
REPO_DIR = "/content/Tr"

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
else:
    print("Repo already cloned, pulling latest...")
    !cd {REPO_DIR} && git pull

os.chdir(REPO_DIR)
print("Working directory:", os.getcwd())
print("Files:", os.listdir("."))


# ============================================================
# CELL 3: Initialize PaddleOCR with GPU
# ============================================================
from paddleocr import PaddleOCR
import paddle

print("GPU available:", paddle.is_compiled_with_cuda())
print("Current device:", paddle.get_device())

# use_gpu=True uses CUDA automatically on Colab T4
# lang='en' for English receipts
# use_angle_cls=True handles rotated/tilted text on receipts
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    use_gpu=True,
    show_log=False,
    # Server-grade accurate models (larger but much better)
    det_model_dir=None,  # auto-downloads PP-OCRv4 server det
    rec_model_dir=None,  # auto-downloads PP-OCRv4 server rec
)
print("PaddleOCR loaded successfully with GPU!")


# ============================================================
# CELL 4: Core OCR Function — returns EACH line separately
# ============================================================
import cv2
import json

RECEIPTS_DIR  = "Receipts"
OUTPUT_DIR    = "data"
SPLITS        = ["train", "valid", "test"]
FOOD_CATEGORY = "Item"
BOX_PADDING   = 4
MIN_CROP_W    = 20
MIN_CROP_H    = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)


def ocr_crop_lines(img, x, y, w, h):
    """
    Crops the bounding box from the image and runs GPU OCR.
    Returns a LIST of (text, confidence, line_index) tuples.
    One entry per detected line, sorted top-to-bottom.
    """
    H, W = img.shape[:2]
    x1 = max(0, int(x) - BOX_PADDING)
    y1 = max(0, int(y) - BOX_PADDING)
    x2 = min(W, int(x + w) + BOX_PADDING)
    y2 = min(H, int(y + h) + BOX_PADDING)

    if (x2 - x1) < MIN_CROP_W or (y2 - y1) < MIN_CROP_H:
        return []

    crop = img[y1:y2, x1:x2]

    # PaddleOCR returns: [line, line, ...]
    # Each line: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)]
    result = ocr.ocr(crop, cls=True)

    if not result or not result[0]:
        return []

    # Sort lines top-to-bottom by vertical center of their bounding box
    def y_center(r):
        box = r[0]
        ys = [pt[1] for pt in box]
        return (min(ys) + max(ys)) / 2

    sorted_lines = sorted(result[0], key=y_center)

    lines = []
    for line_idx, r in enumerate(sorted_lines):
        text = r[1][0].strip()        # r[1][0] = text string
        conf = round(float(r[1][1]), 4)  # r[1][1] = confidence score
        if text:
            lines.append((text, conf, line_idx))

    return lines


# ============================================================
# CELL 5: Main Extraction Loop (with checkpoint/resume)
# ============================================================
ckpt_path = os.path.join(OUTPUT_DIR, "extraction_checkpoint.jsonl")
all_records       = []
seen_texts        = set()
processed_ann_ids = set()

# Load checkpoint if resuming
if os.path.exists(ckpt_path):
    print(f"Found checkpoint, loading previous progress...")
    with open(ckpt_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                all_records.append(rec)
                seen_texts.add(rec["text"].lower().strip())
                processed_ann_ids.add(rec.get("ann_id"))
    print(f"  Resumed: {len(all_records)} line-records, {len(seen_texts)} unique texts")

total_boxes = 0
no_text     = 0
total_lines = len(all_records)

ckpt_file = open(ckpt_path, "a", encoding="utf-8")

for split in SPLITS:
    split_dir = os.path.join(RECEIPTS_DIR, split)
    coco_path = os.path.join(split_dir, "_annotations.coco.json")

    with open(coco_path) as f:
        coco = json.load(f)

    cat_map  = {c["id"]: c["name"] for c in coco["categories"]}
    img_map  = {img["id"]: img      for img in coco["images"]}

    item_anns = [a for a in coco["annotations"]
                 if cat_map.get(a["category_id"]) == FOOD_CATEGORY]

    print(f"\n[{split}] {len(item_anns)} Item boxes across {len(coco['images'])} images")

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
        lines = ocr_crop_lines(img, x, y, w, h)

        if not lines:
            no_text += 1
            continue

        # Store EACH OCR line as a separate record
        for text, conf, line_idx in lines:
            key    = text.lower().strip()
            is_dup = key in seen_texts
            seen_texts.add(key)
            total_lines += 1

            record = {
                "ann_id":     ann_key,
                "line_index": line_idx,      # 0 = first/top line in the box
                "text":       text,          # THIS SINGLE LINE only
                "confidence": conf,
                "split":      split,
                "image":      img_info["file_name"],
                "bbox":       [x, y, w, h],
                "duplicate":  is_dup,
                "label":      ""             # fill later via Gemini
            }
            all_records.append(record)
            ckpt_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        ckpt_file.flush()

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(item_anns)} boxes | lines: {total_lines} | "
                  f"unique: {len(seen_texts)} | no_text: {no_text}")

ckpt_file.close()
print(f"\nDone! Total line records: {total_lines} | Unique: {len(seen_texts)}")


# ============================================================
# CELL 6: Write Output Files
# ============================================================
# 1. Plain text — unique OCR lines only (one per row → paste into Gemini)
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

# 2. Full JSONL — all line records with metadata
jsonl_path = os.path.join(OUTPUT_DIR, "all_items_with_source.jsonl")
with open(jsonl_path, "w", encoding="utf-8") as f:
    for rec in all_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Wrote {len(unique_texts)} unique lines → {txt_path}")
print(f"Wrote {len(all_records)} full records  → {jsonl_path}")


# ============================================================
# CELL 7: Download Results to your computer
# ============================================================
from google.colab import files

files.download(txt_path)           # all_items.txt (paste into Gemini)
files.download(jsonl_path)         # all_items_with_source.jsonl (full dataset)
files.download(ckpt_path)          # extraction_checkpoint.jsonl (for resuming)

print("Downloads started!")
