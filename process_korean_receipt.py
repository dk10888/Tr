#!/usr/bin/env python3
"""
Korean Receipt OCR & BERT Dataset Expansion Pipeline
====================================================
Processes Korean receipt images (or OCR text files), runs line-by-line OCR,
classifies items using the trained Korean BERT model (food vs not_food),
and outputs structured JSON files formatted for dataset expansion.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


class KoreanBERTClassifier:
    """Supports inference via ONNX Runtime or HuggingFace Transformers."""
    
    def __init__(self, model_path: Optional[str] = None, vocab_path: Optional[str] = None):
        self.mode = None  # 'onnx' or 'transformers'
        self.session = None
        self.tokenizer = None
        self.model = None
        self.label_map = {0: "not_food", 1: "food"}
        self._init_model(model_path, vocab_path)

    def _init_model(self, model_path: Optional[str], vocab_path: Optional[str]):
        # Default search paths
        candidate_paths = []
        if model_path:
            candidate_paths.append(model_path)
        
        candidate_paths.extend([
            os.path.join(BASE_DIR, "korean_food_classifier_quant.onnx"),
            os.path.join(BASE_DIR, "korean_food_classifier.onnx"),
            os.path.join(BASE_DIR, "models", "korean_food_bert"),
            os.path.join(BASE_DIR, "models", "onnx", "korean_food_classifier_quant.onnx"),
        ])

        target_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                target_path = p
                break

        if not target_path:
            print("⚠️ Warning: No trained Korean BERT model found. Running in OCR-only mode.")
            return

        print(f"📦 Loading Korean BERT Model from: {target_path}")

        # Case A: ONNX Model (.onnx)
        if target_path.endswith(".onnx"):
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer

                self.session = ort.InferenceSession(target_path, providers=["CPUExecutionProvider"])
                
                # Load tokenizer from local files or default KoELECTRA/monologg tokenizer
                tok_dir = os.path.dirname(target_path)
                if os.path.exists(os.path.join(tok_dir, "vocab.txt")):
                    self.tokenizer = AutoTokenizer.from_pretrained(tok_dir)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
                
                self.mode = "onnx"
                print("✅ Successfully initialized ONNX Runtime model engine.")
                return
            except Exception as e:
                print(f"⚠️ Failed to load ONNX model ({e}). Trying Transformers fallback...")

        # Case B: PyTorch / HuggingFace Model directory
        if os.path.isdir(target_path) or not self.mode:
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification

                self.tokenizer = AutoTokenizer.from_pretrained(target_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(target_path)
                self.model.eval()
                self.mode = "transformers"
                print("✅ Successfully initialized PyTorch Transformers model engine.")
            except Exception as e:
                print(f"⚠️ Failed to load PyTorch model: {e}")

    def predict(self, text: str) -> Dict[str, Any]:
        """Classify a text string into 'food' or 'not_food' with confidence score."""
        text = text.strip()
        if not text:
            return {"label": "not_food", "confidence": 0.0}

        if not self.mode:
            # Fallback heuristic if model is unavailable
            return {"label": "unknown", "confidence": 0.0}

        try:
            if self.mode == "onnx":
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

            elif self.mode == "transformers":
                import torch
                import torch.nn.functional as F
                enc = self.tokenizer(text, return_tensors="pt", max_length=64, truncation=True, padding="max_length")
                with torch.no_grad():
                    out = self.model(**enc)
                    probs = F.softmax(out.logits, dim=-1)[0]
                    pred_id = int(torch.argmax(probs))
                    return {
                        "label": self.label_map.get(pred_id, "not_food"),
                        "confidence": round(float(probs[pred_id].item()), 4)
                    }
        except Exception as e:
            print(f"⚠️ Error classifying '{text}': {e}")
            return {"label": "not_food", "confidence": 0.0}


class ReceiptOCREngine:
    """Extracts text line-by-line from Korean receipt images."""

    def __init__(self):
        self.ocr = None
        self._init_ocr()

    def _init_ocr(self):
        try:
            import easyocr
            self.easy_reader = easyocr.Reader(['ko', 'en'], gpu=False)
            print("📷 EasyOCR Engine (Korean + English) loaded successfully.")
            return
        except Exception as e:
            print(f"⚠️ EasyOCR load failed ({e}). Trying RapidOCR fallback...")
        
        try:
            from rapidocr_onnxruntime import RapidOCR
            self.ocr = RapidOCR()
            print("📷 RapidOCR Engine loaded successfully.")
        except Exception as e:
            print(f"⚠️ RapidOCR not loaded ({e}).")

    def process_image(self, image_path: str) -> List[str]:
        lines = []
        if hasattr(self, 'easy_reader') and self.easy_reader:
            try:
                res = self.easy_reader.readtext(image_path)
                if res:
                    sorted_res = sorted(res, key=lambda item: item[0][0][1])
                    for item in sorted_res:
                        txt = item[1].strip()
                        score = float(item[2])
                        if txt and score > 0.15:
                            lines.append(txt)
                return lines
            except Exception as e:
                print(f"⚠️ EasyOCR processing failed: {e}")

        if hasattr(self, 'ocr') and self.ocr:
            try:
                result, _ = self.ocr(image_path)
                if result:
                    sorted_results = sorted(result, key=lambda item: item[0][0][1])
                    for item in sorted_results:
                        txt = item[1].strip()
                        score = item[2]
                        if txt and score > 0.3:
                            lines.append(txt)
                return lines
            except Exception as e:
                print(f"❌ OCR processing failed for {image_path}: {e}")

        return lines


def process_single_receipt(
    input_path: str,
    ocr_engine: ReceiptOCREngine,
    classifier: KoreanBERTClassifier,
    output_dir: str,
    source_tag: str = "korean_ocr_new",
    append_csv_path: Optional[str] = None
) -> Dict[str, Any]:
    """Processes an image or text file, classifies lines, and writes JSON."""
    
    file_name = os.path.basename(input_path)
    print(f"\n🧾 Processing receipt: {file_name}")

    raw_lines = []
    if input_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
        raw_lines = ocr_engine.process_image(input_path)
    elif input_path.lower().endswith((".txt", ".csv")):
        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            raw_lines = [line.strip() for line in f if line.strip()]
    else:
        print(f"⚠️ Unsupported file type: {input_path}")
        return {}

    items = []
    dataset_rows = []

    for idx, line in enumerate(raw_lines, start=1):
        prediction = classifier.predict(line)
        item_data = {
            "line_no": idx,
            "text": line,
            "label": prediction["label"],
            "confidence": prediction["confidence"],
            "source": source_tag
        }
        items.append(item_data)
        
        # Formatted CSV row: text,label,source
        # Wrap text in quotes if it contains commas
        clean_text = f'"{line}"' if ',' in line else line
        dataset_rows.append(f"{clean_text},{prediction['label']},{source_tag}")

    food_count = sum(1 for item in items if item["label"] == "food")
    not_food_count = sum(1 for item in items if item["label"] == "not_food")

    result_json = {
        "receipt_name": file_name,
        "processed_at": datetime.now().isoformat(),
        "summary": {
            "total_lines": len(items),
            "food_items": food_count,
            "not_food_items": not_food_count,
        },
        "items": items,
        "dataset_rows": dataset_rows
    }

    # Save output JSON
    os.makedirs(output_dir, exist_ok=True)
    json_filename = os.path.splitext(file_name)[0] + "_result.json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    print(f"✅ Results saved → {json_path}")
    print(f"   Total Lines: {len(items)} | Food: {food_count} | Not Food: {not_food_count}")

    # Append to dataset CSV if requested
    if append_csv_path:
        append_to_dataset(dataset_rows, append_csv_path)

    return result_json


def append_to_dataset(dataset_rows: List[str], csv_path: str):
    """Appends formatted dataset rows to korean_receipt_dataset_full.csv safely."""
    try:
        exists = os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8") as f:
            if not exists:
                f.write("text,label,source\n")
            for row in dataset_rows:
                f.write(row + "\n")
        print(f"➕ Appended {len(dataset_rows)} rows to dataset CSV: {csv_path}")
    except Exception as e:
        print(f"❌ Failed to append to dataset CSV ({csv_path}): {e}")


def main():
    parser = argparse.ArgumentParser(description="Korean Receipt OCR & BERT Dataset Expansion Pipeline")
    parser.add_argument("--image", "-i", type=str, help="Path to single Korean receipt image or txt file")
    parser.add_argument("--dir", "-d", type=str, help="Path to directory containing receipt images")
    parser.add_argument("--model", "-m", type=str, help="Path to ONNX model file or PyTorch model directory")
    parser.add_argument("--output-dir", "-o", type=str, default=os.path.join(BASE_DIR, "parsed_receipts_json"), help="Directory to save JSON output")
    parser.add_argument("--append-csv", "-a", type=str, help="Path to CSV dataset to append result lines (e.g. korean_receipt_dataset_full.csv)")
    parser.add_argument("--source-tag", "-s", type=str, default="korean_ocr_new", help="Source tag for dataset expansion")
    
    args = parser.parse_args()

    if not args.image and not args.dir:
        print("💡 Usage examples:")
        print("   python process_korean_receipt.py --image receipt_01.jpg")
        print("   python process_korean_receipt.py --dir ./receipt_folder --append-csv korean_receipt_dataset_full.csv")
        return

    ocr_engine = ReceiptOCREngine()
    classifier = KoreanBERTClassifier(model_path=args.model)

    target_files = []
    if args.image and os.path.exists(args.image):
        target_files.append(args.image)
    elif args.dir and os.path.exists(args.dir):
        valid_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".txt")
        for root, _, files in os.walk(args.dir):
            for f in files:
                if f.lower().endswith(valid_exts):
                    target_files.append(os.path.join(root, f))

    if not target_files:
        print("❌ No input files found to process.")
        return

    print(f"🚀 Processing {len(target_files)} file(s)...")
    for fpath in target_files:
        process_single_receipt(
            input_path=fpath,
            ocr_engine=ocr_engine,
            classifier=classifier,
            output_dir=args.output_dir,
            source_tag=args.source_tag,
            append_csv_path=args.append_csv
        )

    print("\n✨ Processing complete!")


if __name__ == "__main__":
    main()
