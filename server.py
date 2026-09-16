import os
import sys
import cv2
import numpy as np
import json
import base64
import time
import cgi
import webbrowser
from threading import Timer
from http.server import HTTPServer, SimpleHTTPRequestHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# ── Global Model Cache ────────────────────────────────────────────────────────
_OCR_ENGINE = None
_MODEL_CACHE = {}

def setup_thread_env():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

def get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        setup_thread_env()
        from rapidocr_onnxruntime import RapidOCR
        models_dir = os.path.join(BASE_DIR, "models", "onnx")
        det_path = os.path.join(models_dir, "v6_medium_det.onnx")
        rec_path = os.path.join(models_dir, "v6_medium_rec.onnx")
        rec_keys_path = os.path.join(models_dir, "rec_keys.txt")
        if os.path.exists(det_path) and os.path.exists(rec_path):
            _OCR_ENGINE = RapidOCR(det_model_path=det_path, rec_model_path=rec_path, rec_keys_path=rec_keys_path)
        else:
            _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE

def get_bert_model(model_folder_name):
    global _MODEL_CACHE
    if model_folder_name not in _MODEL_CACHE:
        setup_thread_env()
        import torch
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

        model_dir = os.path.join(BASE_DIR, "models", model_folder_name)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

        config = BertConfig.from_pretrained(model_dir, local_files_only=True)
        model = BertForSequenceClassification(config)

        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict, strict=True)
        else:
            from safetensors.torch import load_file
            state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
            model.load_state_dict(state_dict)

        model.eval()
        _MODEL_CACHE[model_folder_name] = (tokenizer, model)
    return _MODEL_CACHE[model_folder_name]

def run_bert_inference(texts, tokenizer, model):
    if not texts:
        return []
    import torch
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

    id2label = model.config.id2label
    results = []
    for i, pred in enumerate(preds):
        results.append({
            "label": id2label[pred.item()],
            "confidence": float(probs[i, pred.item()])
        })
    return results

# ── HTTP Server Request Handler ───────────────────────────────────────────────
class ReceiptAppRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def log_message(self, format, *args):
        # Clean terminal logging
        sys.stdout.write(f"[{time.strftime('%H:%M:%S')}] HTTP Request: {format % args}\n")
        sys.stdout.flush()

    def do_POST(self):
        if self.path == "/api/process":
            self.handle_process_api()
        else:
            self.send_error(404, "Endpoint not found")

    def handle_process_api(self):
        start_time = time.time()
        try:
            content_type = self.headers.get('Content-Type')
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': content_type}
            )

            if 'file' not in form:
                self.send_json({"error": "No file uploaded"}, status=400)
                return

            file_item = form['file']
            image_bytes = file_item.file.read()
            model_choice = form.getvalue('model_choice', 'bert_mini')
            conf_threshold = float(form.getvalue('conf_threshold', '0.60'))

            # Decode OpenCV Image
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img_cv is None:
                self.send_json({"error": "Failed to decode image"}, status=400)
                return

            # 1. OCR Extraction
            ocr_engine = get_ocr_engine()
            ocr_result, _ = ocr_engine(img_cv)

            if not ocr_result:
                self.send_json({
                    "total_lines": 0,
                    "food_items": [],
                    "discarded_items": [],
                    "comparison": [],
                    "annotated_image_base64": "",
                    "elapsed_ms": int((time.time() - start_time) * 1000)
                })
                return

            items = []
            for box, text, conf in ocr_result:
                text_str = text.strip()
                if len(text_str) >= 2:
                    xs = [pt[0] for pt in box]
                    ys = [pt[1] for pt in box]
                    items.append({
                        "box": box,
                        "text": text_str,
                        "ocr_conf": round(float(conf), 3) if conf else 1.0,
                        "x1": int(min(xs)), "y1": int(min(ys)),
                        "x2": int(max(xs)), "y2": int(max(ys)),
                    })

            raw_texts = [it["text"] for it in items]

            # 2. Model Inference
            comparison_list = []
            if model_choice == "compare":
                tok_tiny, mod_tiny = get_bert_model("bert_tiny_v7_food_final")
                tok_mini, mod_mini = get_bert_model("bert_mini_food_final")

                preds_tiny = run_bert_inference(raw_texts, tok_tiny, mod_tiny)
                preds_mini = run_bert_inference(raw_texts, tok_mini, mod_mini)

                # Primary choice uses BERT-mini
                classifications = preds_mini

                for text, pt, pm in zip(raw_texts, preds_tiny, preds_mini):
                    comparison_list.append({
                        "text": text,
                        "tiny_pred": pt["label"],
                        "tiny_conf": round(pt["confidence"], 4),
                        "mini_pred": pm["label"],
                        "mini_conf": round(pm["confidence"], 4),
                        "agree": (pt["label"] == pm["label"])
                    })
            else:
                # v7 = default (76k samples, 94.18% accuracy)
                if model_choice == "bert_tiny":
                    folder_name = "bert_tiny_v7_food_final"
                elif model_choice == "bert_mini":
                    folder_name = "bert_mini_food_final"
                else:
                    folder_name = "bert_tiny_v7_food_final"  # default
                tokenizer, bert_model = get_bert_model(folder_name)
                classifications = run_bert_inference(raw_texts, tokenizer, bert_model)

            # 3. Separate Food vs Discarded & Annotate Image
            annotated_img = img_cv.copy()
            food_items = []
            discarded_items = []

            for item, cls in zip(items, classifications):
                is_food = (cls["label"] == "food") and (cls["confidence"] >= conf_threshold)
                rec = {
                    "text": item["text"],
                    "confidence": round(cls["confidence"], 4),
                    "ocr_confidence": item["ocr_conf"],
                }

                if is_food:
                    food_items.append(rec)
                    cv2.rectangle(annotated_img, (item["x1"], item["y1"]), (item["x2"], item["y2"]), (0, 200, 0), 2)
                else:
                    discarded_items.append(rec)
                    cv2.rectangle(annotated_img, (item["x1"], item["y1"]), (item["x2"], item["y2"]), (180, 180, 180), 1)

            # Encode annotated image to JPEG Base64
            _, buffer = cv2.imencode('.jpg', annotated_img)
            img_b64 = base64.b64encode(buffer).decode('utf-8')

            elapsed_ms = int((time.time() - start_time) * 1000)

            response_payload = {
                "total_lines": len(items),
                "food_items": food_items,
                "discarded_items": discarded_items,
                "comparison": comparison_list,
                "annotated_image_base64": img_b64,
                "elapsed_ms": elapsed_ms
            }

            print(f"[{time.strftime('%H:%M:%S')}] ✅ Successfully processed receipt ({len(items)} lines, {len(food_items)} food items) in {elapsed_ms}ms", flush=True)
            self.send_json(response_payload)

        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ❌ Processing error: {str(e)}", flush=True)
            self.send_json({"error": str(e)}, status=500)

    def send_json(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

def open_browser(url):
    time.sleep(1.0)
    webbrowser.open(url)

def run_server(port=8000):
    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, ReceiptAppRequestHandler)
    url = f"http://localhost:{port}"

    print(f"\n=======================================================", flush=True)
    print(f"🚀 AI Receipt Food Extractor Web App is ACTIVE!", flush=True)
    print(f"🌐 Server URL: {url}", flush=True)
    print(f"📁 Web UI Directory: {WEB_DIR}", flush=True)
    print(f"💡 Opening {url} in your browser now...", flush=True)
    print(f"Press Ctrl+C in this terminal anytime to stop the server.", flush=True)
    print(f"=======================================================\n", flush=True)

    # Auto-open browser after 1 second
    Timer(1.0, open_browser, [url]).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web server...", flush=True)
        httpd.server_close()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)
