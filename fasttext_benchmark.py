import os
import sys
import time
import random
import csv
import fasttext
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# ─────────────────────────────────────────────────────────────────────────────
# 1. OCR-ROBUST AUGMENTATION FOR KOREAN & ENGLISH RECEIPT TEXT
# ─────────────────────────────────────────────────────────────────────────────
def augment_text(text: str) -> list:
    """
    Generates realistic OCR variants of text:
    - Original text
    - Space-removed version ('신 라면' -> '신라면')
    - Space-scattered version ('신라면' -> '신 라 면')
    - Extra space version ('신  라면')
    """
    variants = [text]
    
    # 1. Remove all spaces
    no_space = text.replace(" ", "")
    if no_space != text and len(no_space) >= 1:
        variants.append(no_space)
        
    # 2. Add spaces between Hangul characters randomly if short
    if len(text) <= 6 and " " not in text:
        spaced = " ".join(list(text))
        variants.append(spaced)
        
    # 3. Double spacing
    if " " in text:
        variants.append(text.replace(" ", "  "))

    return list(set(variants))

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATASET & PREPARE FASTTEXT FORMAT
# ─────────────────────────────────────────────────────────────────────────────
def prepare_fasttext_data(csv_path: str, augment: bool = True):
    print(f"📂 Loading dataset from: {csv_path}")
    
    texts, labels = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("text", "").strip()
            l = row.get("label", "").strip()
            if t and l in ["food", "not_food"]:
                texts.append(t)
                labels.append(l)

    print(f"  Total raw samples: {len(texts)} (Food: {labels.count('food')}, Not-Food: {labels.count('not_food')})")

    # Train-test split (80/20) before augmentation so no leak between train & test
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )

    # Augment Training Set
    train_lines = []
    for text, label in zip(X_train_raw, y_train_raw):
        fasttext_label = f"__label__{label}"
        if augment:
            variants = augment_text(text)
            for var in variants:
                train_lines.append(f"{fasttext_label} {var}")
        else:
            train_lines.append(f"{fasttext_label} {text}")

    # Format Test Set (Unaugmented raw test set for realistic evaluation)
    test_lines = [f"__label__{l} {t}" for t, l in zip(X_test, y_test)]

    print(f"  Training samples after augmentation: {len(train_lines)}")
    print(f"  Test samples: {len(test_lines)}")

    # Write temporary fasttext files
    train_file = "fasttext_train_tmp.txt"
    test_file = "fasttext_test_tmp.txt"

    with open(train_file, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines))

    with open(test_file, "w", encoding="utf-8") as f:
        f.write("\n".join(test_lines))

    return train_file, test_file, X_test, y_test

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAIN & EVALUATE FASTTEXT
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark(dataset_csv: str):
    train_file, test_file, X_test, y_test = prepare_fasttext_data(dataset_csv, augment=True)

    print("\n🚀 Training FastText Model (minn=2, maxn=5, epoch=25, lr=0.5)...")
    start_time = time.time()
    
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
    train_time = time.time() - start_time
    print(f"✅ Training completed in {train_time:.2f} seconds.")

    # Save raw model
    raw_model_path = "fasttext_korean_food.bin"
    model.save_model(raw_model_path)
    raw_size_mb = os.path.getsize(raw_model_path) / (1024 * 1024)
    print(f"  Raw FastText Model Size: {raw_size_mb:.2f} MB")

    # Quantize model
    print("\n⚡ Quantizing FastText Model (-quantize)...")
    try:
        model.quantize(input=train_file, retrain=True, epoch=5, lr=0.1)
        quant_model_path = "fasttext_korean_food.ftz"
        model.save_model(quant_model_path)
        quant_size_mb = os.path.getsize(quant_model_path) / (1024 * 1024)
        print(f"✅ Quantized FastText Model Size: {quant_size_mb:.2f} MB (Compression: {raw_size_mb / quant_size_mb:.1f}x smaller)")
    except Exception as e:
        print(f"⚠️ Quantization warning: {e}. Using raw model size: {raw_size_mb:.2f} MB")
        quant_model_path = raw_model_path
        quant_size_mb = raw_size_mb

    # Inference Benchmark on Test Set
    print("\n📊 Evaluating Inference Speed & Accuracy on Test Set...")
    start_inf = time.time()
    
    preds = []
    confidences = []
    for text in X_test:
        labels, probs = model.predict(text.replace("\n", " "))
        pred_label = labels[0].replace("__label__", "") if labels else "not_food"
        preds.append(pred_label)
        confidences.append(probs[0] if probs else 0.0)

    total_inf_time_ms = (time.time() - start_inf) * 1000
    avg_latency_ms = total_inf_time_ms / len(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='binary', pos_label='food')

    print("\n" + "="*70)
    print(f"🏆 FASTTEXT BENCHMARK RESULTS ({os.path.basename(dataset_csv)})")
    print("="*70)
    print(f"  Accuracy                : {acc * 100:.2f}%")
    print(f"  Food Precision           : {precision * 100:.2f}%")
    print(f"  Food Recall              : {recall * 100:.2f}%")
    print(f"  Food F1-Score            : {f1 * 100:.2f}%")
    print(f"  Quantized Model File Size: {quant_size_mb:.2f} MB")
    print(f"  Inference Latency        : {avg_latency_ms * 1000:.2f} microseconds / item")
    print(f"  Throughput               : {len(X_test) / (total_inf_time_ms / 1000):.0f} items / sec")
    print("="*70)

    # Detailed Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, preds, digits=4))

    # Test on Specific Tricky / OCR-corrupted Korean Receipts
    print("\n🧪 Testing on Difficult / Ambiguous Receipt Lines:")
    test_cases = [
        "신라면",             # Food
        "신 라면",            # Food (space corrupted)
        "신라 면",            # Food (bad spacing)
        "코카콜라 500ml",     # Food
        "소금 1kg",           # Food (ambiguous single character '금')
        "포장주문",           # Not food (metadata keyword)
        "포장 아메리카노",     # Food (packaging + coffee)
        "합계금액 15,000원",  # Not food
        "부가세",             # Not food
        "CJ비비고왕교자",     # Food
        "영수증출력",         # Not food
    ]
    
    print(f"{'Text':<25} | {'Predicted':<10} | {'Confidence':<10}")
    print("-" * 52)
    for tc in test_cases:
        lbls, prbs = model.predict(tc)
        l = lbls[0].replace("__label__", "")
        p = prbs[0]
        print(f"{tc:<25} | {l:<10} | {p:.4f}")

    # Cleanup temp files
    if os.path.exists(train_file): os.remove(train_file)
    if os.path.exists(test_file): os.remove(test_file)

if __name__ == "__main__":
    dataset_file = "korean_receipt_dataset_v25.csv" if os.path.exists("korean_receipt_dataset_v25.csv") else "korean_receipt_dataset_v23.csv"
    run_benchmark(dataset_file)
