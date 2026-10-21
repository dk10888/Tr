#!/usr/bin/env python3
"""
Receipt Structurer & MobileBERT INT8 Demo Server
================================================
Combines receipt_structurer.py with the 10-Class MobileBERT INT8 ONNX model
to parse receipts into clean structured items and classify them on-device.
"""

import os
import sys

# Ensure immediate unbuffered console logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import json
import time
import re
import base64
import cv2
import numpy as np
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn

# Add parent dir to sys.path so we can import receipt_structurer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from receipt_structurer_v2 import (
    parse_receipt_line_advanced,
    clean_item_name_for_bert,
    is_noise_or_header_line,
    cluster_ocr_boxes_into_lines,
    pre_fuse_multiline_items,
)

import onnxruntime as ort
from tokenizers import Tokenizer
from rapidocr_onnxruntime import RapidOCR

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(PARENT_DIR, "receipt_mobilebert_android")
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "mobilebert_11class_int8.onnx")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")
LABELS_PATH = os.path.join(MODEL_DIR, "labels_11class.json")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# ── Initialize ONNX Runtime & Tokenizer ───────────────────────────────────────
print(f"[INIT] Loading MobileBERT INT8 ONNX session from {os.path.basename(ONNX_MODEL_PATH)}...")
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 4
session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options)

print("[INIT] Loading tokenizer...")
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
tokenizer.enable_padding(length=48)
tokenizer.enable_truncation(max_length=48)

print(f"[INIT] Loading label mappings from {os.path.basename(LABELS_PATH)}...")
with open(LABELS_PATH, "r") as f:
    label_data = json.load(f)
    id2label = {int(k): v for k, v in label_data["id2label"].items()}

print("[INIT] Loading RapidOCR ONNX engine...")
ocr_engine = RapidOCR()

print(f"[INIT] Loaded {len(id2label)} classes: {list(id2label.values())}")
print("[INIT] Ready.")

# ── Inference Helper ──────────────────────────────────────────────────────────
def classify_text(text: str):
    t0 = time.perf_counter()
    enc = tokenizer.encode(text)
    input_ids = np.array([enc.ids], dtype=np.int64)
    attention_mask = np.array([enc.attention_mask], dtype=np.int64)
    token_type_ids = np.array([enc.type_ids], dtype=np.int64)

    outputs = session.run(["logits"], {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    })[0][0]

    latency_ms = (time.perf_counter() - t0) * 1000.0

    exp_l = np.exp(outputs - np.max(outputs))
    probs = exp_l / np.sum(exp_l)
    top_id = int(np.argmax(probs))
    confidence = float(probs[top_id])
    category = id2label[top_id]

    # Top 3 predictions
    top_indices = np.argsort(probs)[::-1][:3]
    top_3 = [
        {"category": id2label[int(i)], "probability": round(float(probs[i]) * 100.0, 1)}
        for i in top_indices
    ]

    return {
        "category": category,
        "confidence": round(confidence * 100.0, 1),
        "latency_ms": round(latency_ms, 2),
        "top_3": top_3,
    }


# ── Receipt Presets ───────────────────────────────────────────────────────────
PRESETS = {
    "audit_test": {
        "title": "Audit Test (Wild Turkey & Sub-Rows)",
        "icon": "🦃",
        "text": """SUPERMART STORE #9012
TEL: 555-0199
CASHIER: JANE 04
DATE: 10/22/2026 15:30:00
--------------------------------
WILD TURKEY STRAIGHT BOURBON      28.99
BREAD & BUTTER PINOT NOIR 750ML   14.99
ORGANIC HONEYCRISP APPLES 2LB      5.99
Qty 2 each @ 1.99                  3.98
PROGRESSO CHICKEN BROTH 32OZ       2.49
FRESCHETTA BBQ CHICKEN PIZZA       7.89
BOAR'S HEAD ROAST TURKEY DELI      6.99
FELS NAPTHA LAUNDRY SOAP           2.29
BEN & JERRY'S HALF BAKED 14OZ      5.49
--------------------------------
SUBTOTAL                          79.10
TAX 6.5%                           5.14
TOTAL PURCHASE                    84.24
VISA BALANCE TO PAY $84.24
THANK YOU FOR SHOPPING WITH US!"""
    },
    "walmart": {
        "title": "Walmart Supercenter",
        "icon": "🛒",
        "text": """WALMART SUPERCENTER #3214
STORE MGR: JOHN DOE
TEL: (555) 321-4567
DATE: 10/20/2026 14:22:05
================================
4011 BANANAS 1.42 LB @ 0.58       0.82
GREAT VALUE WHOLE MILK 1 GAL      3.78
2 x TYSON CRISPY NUGGETS 32OZ    13.98
DOVE BODY WASH DEEP MOISTURE      7.44
BARILLA SPAGHETTI 16OZ            1.88
HEINEKEN LAGER BEER 6PK           9.99
WONDER CLASSIC WHITE BREAD        2.48
OREO DOUBLE STUF COOKIES          4.29
USDA CHOICE RIBEYE STEAK         14.50
BOUNTY PAPER TOWELS 6 ROLL       11.99
--------------------------------
SUBTOTAL                         71.15
SALES TAX                         5.69
TOTAL                            76.84
VISA CARD **** **** **** 1234
AUTH: 098124 REF: 882194
THANK YOU FOR SHOPPING WITH US!"""
    },
    "target": {
        "title": "Target Grocery & Home",
        "icon": "🎯",
        "text": """TARGET STORE #1842
MINNEAPOLIS MN
03/15/2026 18:45:11
--------------------------------
CHOBANI GREEK YOGURT 4PK          4.49
TIDE PODS SPRING MEADOW 42CT     13.99
3 * LAY'S CLASSIC POTATO CHIPS    8.97
COLGATE TOTAL TOOTHPASTE 4.8OZ    3.99
ORGANIC HONEYCRISP APPLES 2LB     5.99
KRAFT MACARONI & CHEESE 4PK       4.89
STOUFFER'S LASAGNA FROZEN 14OZ    5.29
SARA LEE HONEY WHEAT BREAD        2.99
COCA COLA ZERO SUGAR 12PK         7.99
--------------------------------
SUBTOTAL                         58.60
TAX 6.5%                          3.81
TOTAL                            62.41
MASTERCARD ENDING IN 9012
APPROVED - THANK YOU"""
    },
    "costco": {
        "title": "Costco Wholesale",
        "icon": "🥩",
        "text": """COSTCO WHOLESALE #482
MEMBER 1184920491
--------------------------------
KIRKLAND SIGNATURE BACON 4PK     17.99
KIRKLAND ORGANIC EGGS 24CT        6.89
AVOCADOS BAG OF 6                 7.99
DIGIORNO RISING CRUST PIZZA 3PK  15.49
KIRKLAND SPRING WATER 40PK        4.99
JIM BEAM BOURBON 1.75L           24.99
NATURE VALLEY GRANOLA BARS 49CT  12.49
CHARMIN ULTRA SOFT 30 ROLLS      26.99
THAI JASMINE RICE 25LB           21.99
--------------------------------
TOTAL NUMBER OF ITEMS SOLD = 9
TOTAL TAX                         4.20
**** TOTAL                     143.93
CREDIT CARD CHASE VISA 5510"""
    },
    "korean": {
        "title": "H-Mart / Korean Mart",
        "icon": "🏪",
        "text": """H-MART SUPERMARKET
TEL: 02-310-5038
매출 영수증 (RECEIPT)
--------------------------------
신라면 멀티팩 5입                4,850
CJ 비비고 왕교자 만두 1.05KG     9,900
서울우유 1L                      2,980
참이슬 후레쉬 소주 360ML         1,650
제주 삼다수 2L x 6개             5,800
국내산 삼겹살 600G              16,800
오리온 초코파이 12개입           3,980
파리바게뜨 식빵                  2,800
크리넥스 3겹 화장지 24롤        15,900
신선 양파 1.5KG 1망              3,490
--------------------------------
소계 (SUBTOTAL)                 68,150
부가세 (VAT)                     4,250
합계 금액                       72,400
국민카드 승인 9481-****"""
    }
}


# ── Full Receipt Processing Pipeline ──────────────────────────────────────────
def process_receipt_text(text: str):
    t_start = time.perf_counter()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    lines = pre_fuse_multiline_items(lines)

    structured_items = []
    filtered_lines = []

    for idx, line in enumerate(lines):
        res = parse_receipt_line_advanced(line)
        if res:
            raw_name = res["item_name"]
            cleaned_for_bert = clean_item_name_for_bert(raw_name)

            # Classify with INT8 MobileBERT
            pred = classify_text(cleaned_for_bert if cleaned_for_bert else raw_name)

            if pred["category"] == "non_item":
                filtered_lines.append({
                    "line_no": idx + 1,
                    "text": line,
                    "reason": f"Classified as non_item (Receipt Metadata/Noise) by MobileBERT ({pred['confidence']}%)"
                })
                continue

            structured_items.append({
                "line_no": idx + 1,
                "raw_line": line,
                "item_name": raw_name,
                "bert_query": cleaned_for_bert,
                "quantity": res["quantity"],
                "unit": res["unit"],
                "unit_price": res["unit_price"],
                "total_price": res["total_price"],
                "category": pred["category"],
                "confidence": pred["confidence"],
                "latency_ms": pred["latency_ms"],
                "top_3": pred["top_3"],
            })
        else:
            # Noise or metadata line
            is_noise = is_noise_or_header_line(line)
            reason = "Header / Noise / Financial Metadata" if is_noise else "No Valid Price / Unparsable"
            filtered_lines.append({
                "line_no": idx + 1,
                "text": line,
                "reason": reason
            })

    # Summary Statistics
    total_spend = sum(it["total_price"] for it in structured_items if it["total_price"] is not None)
    avg_conf = (
        sum(it["confidence"] for it in structured_items) / len(structured_items)
        if structured_items else 0.0
    )
    avg_latency = (
        sum(it["latency_ms"] for it in structured_items) / len(structured_items)
        if structured_items else 0.0
    )

    # Category Spend Breakdown
    category_breakdown = {}
    for it in structured_items:
        cat = it["category"]
        price = it["total_price"] or 0.0
        if cat not in category_breakdown:
            category_breakdown[cat] = {"count": 0, "total_spend": 0.0}
        category_breakdown[cat]["count"] += 1
        category_breakdown[cat]["total_spend"] += price

    cat_list = []
    for cat, data in sorted(category_breakdown.items(), key=lambda x: x[1]["total_spend"], reverse=True):
        pct = (data["total_spend"] / total_spend * 100.0) if total_spend > 0 else 0.0
        cat_list.append({
            "category": cat,
            "count": data["count"],
            "total_spend": round(data["total_spend"], 2),
            "percentage": round(pct, 1)
        })

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    return {
        "items": structured_items,
        "filtered_lines": filtered_lines,
        "summary": {
            "total_items": len(structured_items),
            "filtered_lines_count": len(filtered_lines),
            "total_spend": round(total_spend, 2),
            "average_confidence": round(avg_conf, 1),
            "average_latency_ms": round(avg_latency, 2),
            "total_pipeline_ms": round(elapsed_ms, 1),
        },
        "category_breakdown": cat_list,
    }


def process_receipt_image(img_bytes: bytes):
    """Processes an uploaded receipt image with RapidOCR + ReceiptStructurer + MobileBERT."""
    t_start = time.perf_counter()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image format.")

    t_ocr_start = time.perf_counter()
    ocr_result, _ = ocr_engine(img)
    ocr_ms = (time.perf_counter() - t_ocr_start) * 1000.0

    if not ocr_result:
        return {
            "items": [],
            "filtered_lines": [],
            "raw_text": "",
            "summary": {
                "total_items": 0,
                "filtered_lines_count": 0,
                "total_spend": 0.0,
                "average_confidence": 0.0,
                "average_latency_ms": 0.0,
                "ocr_latency_ms": round(ocr_ms, 1),
                "total_pipeline_ms": round(ocr_ms, 1),
            },
            "category_breakdown": [],
            "ocr_fragments_count": 0,
        }

    # Cluster OCR fragments spatially into rows
    clustered_lines = cluster_ocr_boxes_into_lines(ocr_result)
    raw_lines_text = [l["line_text"] for l in clustered_lines]
    joined_text = "\n".join(raw_lines_text)

    # Process structured lines
    res = process_receipt_text(joined_text)
    res["raw_text"] = joined_text
    res["summary"]["ocr_latency_ms"] = round(ocr_ms, 1)
    res["summary"]["total_pipeline_ms"] = round((time.perf_counter() - t_start) * 1000.0, 1)
    res["ocr_fragments_count"] = len(ocr_result)
    return res


# ── Sample Images ─────────────────────────────────────────────────────────────
SAMPLE_IMAGES = [
    {
        "id": "sample1",
        "title": "Central Park Store",
        "url": "/samples/sample1.jpg",
        "desc": "Asian Mart grocery receipt with item lines"
    },
    {
        "id": "sample2",
        "title": "Korean Supermarket",
        "url": "/samples/sample2.jpg",
        "desc": "Convenience store bill with Hangul items"
    },
    {
        "id": "sample3",
        "title": "Lotteria Food Order",
        "url": "/samples/sample3.jpg",
        "desc": "Quick service restaurant food order receipt"
    }
]


# ── HTTP Server Request Handler ───────────────────────────────────────────────
import socket

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self):
        # Override to prevent macOS socket.getfqdn reverse DNS hang
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)
        self.server_name = "localhost"
        self.server_port = self.server_address[1]

class ReceiptDemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def address_string(self):
        # Avoid reverse DNS on incoming client requests
        return self.client_address[0]

    def send_json(self, data, status=200):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.path = "/index.html"
            return super().do_GET()
        elif self.path == "/api/presets":
            return self.send_json(PRESETS)
        elif self.path == "/api/sample_images":
            return self.send_json(SAMPLE_IMAGES)
        elif self.path == "/api/info":
            info = {
                "model": "MobileBERT INT8 ONNX",
                "size_mb": round(os.path.getsize(ONNX_MODEL_PATH) / 1e6, 1),
                "num_classes": len(id2label),
                "classes": list(id2label.values()),
                "status": "ready"
            }
            return self.send_json(info)

        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/process_receipt":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                text = data.get("text", "")
                result = process_receipt_text(text)
                return self.send_json(result)
            except Exception as e:
                return self.send_json({"error": str(e)}, status=500)

        elif self.path == "/api/process_image":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                b64_str = data.get("image_base64", "")
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_str)
                result = process_receipt_image(img_bytes)
                return self.send_json(result)
            except Exception as e:
                return self.send_json({"error": str(e)}, status=500)

        elif self.path == "/api/classify_single":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                line = (data.get("line") or data.get("text") or "").strip()

                struct = parse_receipt_line_advanced(line)
                if struct:
                    clean_name = clean_item_name_for_bert(struct["item_name"])
                    pred = classify_text(clean_name if clean_name else struct["item_name"])
                    resp = {
                        "is_item": True,
                        "structured": struct,
                        "cleaned_name": clean_name,
                        "prediction": pred,
                    }
                else:
                    is_noise = is_noise_or_header_line(line)
                    resp = {
                        "is_item": False,
                        "is_noise": is_noise,
                        "reason": "Header / Noise / Financial Metadata" if is_noise else "No Valid Price Found"
                    }

                return self.send_json(resp)
            except Exception as e:
                return self.send_json({"error": str(e)}, status=500)

        self.send_response(404)
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_server(port=8088):
    os.makedirs(STATIC_DIR, exist_ok=True)
    server_address = ("0.0.0.0", port)
    httpd = ThreadedHTTPServer(server_address, ReceiptDemoHandler)
    print(f"\n🚀 Receipt Demo Server running at: http://localhost:{port}")
    print(f"   Model: {ONNX_MODEL_PATH} ({os.path.getsize(ONNX_MODEL_PATH)/1e6:.1f} MB)")
    print(f"   Open in browser: http://localhost:{port}\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8088
    run_server(port)
