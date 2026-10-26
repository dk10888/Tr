import os, sys, json
BASE_DIR = '/Users/dikshit/Desktop/untitled folder 5/receipt_scanner'
sys.path.insert(0, BASE_DIR)
from rapidocr_onnxruntime import RapidOCR
from receipt_structurer_v2 import cluster_ocr_boxes_into_lines, pre_fuse_multiline_items, parse_receipt_line_advanced, clean_item_name_for_bert

IMG_PATH = '/Users/dikshit/.gemini/antigravity-ide/brain/6f7c6445-21e3-451b-a9b3-b4e9eeb2bcc0/.user_uploaded/media_1792951675324.png'
ocr_engine = RapidOCR()

ocr_res, _ = ocr_engine(IMG_PATH)
clustered = cluster_ocr_boxes_into_lines(ocr_res)
raw_lines = [c['line_text'] for c in clustered]
fused_lines = pre_fuse_multiline_items(raw_lines)

print("=" * 70)
print(f"RAW OCR CLUSTERED LINES ({len(raw_lines)} lines):")
print("=" * 70)
for idx, l in enumerate(raw_lines, 1):
    print(f"L{idx:2d}: {l}")

print("\n" + "=" * 70)
print(f"MULTI-LINE FUSED LINES ({len(fused_lines)} lines):")
print("=" * 70)
for idx, l in enumerate(fused_lines, 1):
    print(f"F{idx:2d}: {l}")

print("\n" + "=" * 70)
print("REGEX PARSING (PRICE & QUANTITY EVALUATION):")
print("=" * 70)
for idx, l in enumerate(fused_lines, 1):
    parsed = parse_receipt_line_advanced(l)
    if parsed:
        clean = clean_item_name_for_bert(parsed['item_name'])
        print(f"ITEM {idx:2d}: \"{parsed['item_name']}\" (Clean: \"{clean}\")")
        print(f"         Qty: {parsed['quantity']} {parsed['unit']} | UnitPrice: {parsed['unit_price']} | TotalPrice: {parsed['total_price']}")
        print(f"         From: \"{l}\"\n")
    else:
        print(f"SKIP {idx:2d}: \"{l}\"\n")
