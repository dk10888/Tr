import os
import csv
import fasttext
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

def run_clean_core_experiment():
    v25_csv = "korean_receipt_dataset_v25.csv"

    # Extract ONLY clean, unaugmented sources
    clean_texts = []
    clean_labels = []

    with open(v25_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row.get("source", "")
            # Filter out any source containing '_ocr_aug'
            if "_ocr_aug" not in src:
                t = row.get("text", "").strip()
                l = row.get("label", "").strip()
                if t and l in ["food", "not_food"]:
                    clean_texts.append(t)
                    clean_labels.append(l)

    print(f"📊 Extracted Clean Unaugmented Core Dataset: {len(clean_texts)} rows")
    print(f"   Food: {clean_labels.count('food')}, Not-Food: {clean_labels.count('not_food')}")

    # Write clean fasttext train file
    train_file = "fasttext_clean_core.tmp"
    with open(train_file, "w", encoding="utf-8") as f:
        for t, l in zip(clean_texts, clean_labels):
            f.write(f"__label__{l} {t}\n")

    # Train FastText on CLEAN DATASET ONLY
    print("\n🚀 Training FastText on Clean Core Dataset (No noisy augmentation)...")
    model = fasttext.train_supervised(
        input=train_file,
        lr=0.5,
        epoch=25,
        wordNgrams=2,
        minn=2,
        maxn=5,
        dim=25,
        loss='ova'
    )
    
    # Save model
    clean_model_path = "fasttext_clean_core.ftz"
    model.quantize(input=train_file, retrain=True, epoch=5, lr=0.1)
    model.save_model(clean_model_path)
    print(f"✅ Clean Model Saved: {clean_model_path}")

    # Evaluate on the 31 qq Receipts
    from process_qq_dual_model import extract_smart_lines, QQ_DIR, JAPANESE_REGEX, HANGUL_REGEX
    import easyocr
    reader = easyocr.Reader(['ko', 'en'], gpu=False)

    image_files = sorted([f for f in os.listdir(QQ_DIR) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])

    clean_model_items = []
    total_lines = 0
    total_food = 0

    for idx, img_name in enumerate(image_files, start=1):
        img_path = os.path.join(QQ_DIR, img_name)
        try:
            ocr_res = reader.readtext(img_path)
        except:
            continue

        raw_full_text = " ".join([text for _, text, _ in ocr_res])
        if JAPANESE_REGEX.search(raw_full_text) and not HANGUL_REGEX.search(raw_full_text):
            continue

        extracted_lines = extract_smart_lines(ocr_res)
        
        food_in_receipt = 0
        receipt_items = []
        for line_no, line_text in enumerate(extracted_lines, start=1):
            total_lines += 1
            labels, probs = model.predict(line_text.replace("\n", " "))
            pred = labels[0].replace("__label__", "") if labels else "not_food"
            conf = probs[0] if probs else 0.0

            if pred == "food":
                food_in_receipt += 1
                total_food += 1

            receipt_items.append({"line_no": line_no, "text": line_text, "label": pred, "confidence": round(float(conf), 4)})

        clean_model_items.append({
            "receipt_id": idx,
            "receipt_name": img_name,
            "line_count": len(extracted_lines),
            "food_count": food_in_receipt,
            "not_food_count": len(extracted_lines) - food_in_receipt,
            "items": receipt_items
        })

    print("\n" + "="*70)
    print("🏆 CLEAN CORE MODEL RESULTS ON 31 REAL RECEIPTS")
    print("="*70)
    print(f"  Clean Training Set Size  : {len(clean_texts)} rows")
    print(f"  Total Lines Evaluated    : {total_lines}")
    print(f"  Total Food Items Detected: {total_food}")
    print("="*70)

    # Save Clean Model Results JSON
    import json
    with open("qq_clean_core_model_results.json", "w", encoding="utf-8") as f:
        json.dump({"model": "FastText Clean Core (~9.3k rows)", "total_receipts": len(clean_model_items), "total_lines": total_lines, "receipts": clean_model_items}, f, ensure_ascii=False, indent=2)

    if os.path.exists(train_file): os.remove(train_file)

if __name__ == "__main__":
    run_clean_core_experiment()
