import streamlit as st
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="Receipt Scanner", layout="wide")

@st.cache_resource
def load_ocr():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models", "onnx")
    det_path = os.path.join(models_dir, "v6_medium_det.onnx")
    rec_path = os.path.join(models_dir, "v6_medium_rec.onnx")
    rec_keys_path = os.path.join(models_dir, "rec_keys.txt")
    if os.path.exists(det_path) and os.path.exists(rec_path):
        return RapidOCR(det_model_path=det_path, rec_model_path=rec_path, rec_keys_path=rec_keys_path)
    return RapidOCR()

ocr = load_ocr()


def reconstruct_receipt(result, img_width, char_width=9):
    """
    Use bounding box X/Y coordinates to reconstruct the receipt layout.
    Groups detected text into rows by Y position, then sorts each row by X.
    Maps X pixel position → character column so left/right layout is preserved.
    """
    if not result:
        return ""

    # Each item: (box, text, conf)
    # box: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    items = []
    for box, text, conf in result:
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        x_left  = min(xs)
        x_right = max(xs)
        y_center = (min(ys) + max(ys)) / 2
        items.append({
            "text": text,
            "x_left": x_left,
            "x_right": x_right,
            "y_center": y_center,
            "width_px": x_right - x_left,
        })

    # Sort by Y so we process top→bottom
    items.sort(key=lambda i: i["y_center"])

    # Cluster items into rows: if two boxes' Y centers are within threshold, same row
    row_threshold = 12  # pixels
    rows = []
    current_row = [items[0]]
    for item in items[1:]:
        if abs(item["y_center"] - current_row[-1]["y_center"]) <= row_threshold:
            current_row.append(item)
        else:
            rows.append(current_row)
            current_row = [item]
    rows.append(current_row)

    # How many character columns fit in the image width?
    num_cols = max(40, img_width // char_width)

    lines = []
    for row in rows:
        # Sort items in row left→right
        row.sort(key=lambda i: i["x_left"])

        # Build a character buffer for this line
        buf = [" "] * num_cols

        for item in row:
            # Map pixel X → character column
            col = int((item["x_left"] / img_width) * num_cols)
            col = max(0, min(col, num_cols - len(item["text"]) - 1))
            for ci, ch in enumerate(item["text"]):
                if col + ci < num_cols:
                    buf[col + ci] = ch

        lines.append("".join(buf).rstrip())

    return "\n".join(lines)


# ── UI ──────────────────────────────────────────────────────────────────────
st.title("🧾 Receipt Scanner")

uploaded_file = st.file_uploader("Upload a receipt image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_h, img_w = img_cv.shape[:2]

    col_img, col_text = st.columns([1, 1])

    with col_img:
        st.subheader("Image")
        st.image(img_cv, channels="BGR", use_container_width=True)

    with st.spinner("Running OCR..."):
        result, elapsed = ocr(img_cv)

    with col_text:
        st.subheader("Receipt Text")
        if result:
            receipt_text = reconstruct_receipt(result, img_w)

            # Monospace pre block so spacing is exactly preserved
            st.code(receipt_text, language=None)

            st.download_button(
                label="⬇ Download as TXT",
                data=receipt_text,
                file_name="receipt.txt",
                mime="text/plain",
            )

            json_data = [
                {
                    "text": item[1],
                    "box": item[0],
                    "confidence": round(item[2], 3)
                }
                for item in result
            ]
            st.download_button(
                label="⬇ Download as JSON",
                data=json.dumps(json_data, indent=2, ensure_ascii=False),
                file_name="receipt.json",
                mime="application/json",
            )

            if elapsed:
                total_elapsed = sum(elapsed) if isinstance(elapsed, (list, tuple)) else elapsed
                st.caption(f"OCR elapsed: {total_elapsed:.3f}s")
        else:
            st.error("OCR returned no text. Try a clearer photo.")
