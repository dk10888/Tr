import os
import sys
import time
import random
import csv
import json
import fasttext
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONTROLLED PRECISION AUGMENTATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

# Hard negative single-character / 2-character OCR residue fragments (Highest Priority Fix)
ISOLATED_HARD_NEGATIVES = [
    "꽃", "빠", "움", "철", "매", "꿀", "합", "용", "무", "올", "품", "들",
    "림", "삼", "개", "보", "자", "권", "학", "습", "가", "리", "레", "나",
    "트", "래", "말", "다", "8 용", "L 무", "히 올", "물 품", "필들", "궁림",
    "삼접삼", "김퇴미개", "우사라보", "진계포자", "{신이", "오권", "틈백학",
    "PST)", "~세로습가 골리", "부레 |찰t|1z05", "나드로 1B0g (트래", "논 무; 말다리"
]

DIGIT_MAP = {
    '0': ['O', 'o', 'U'],
    '1': ['I', 'l', '|', '1'],
    '8': ['B', '8'],
    '5': ['S', 's', '5'],
    '6': ['G', '6'],
    '2': ['Z', 'z', '2']
}

PREFIX_SYMBOLS = ['#', '~', '-', '*', "'", '`', '+']

def apply_digit_letter_confusion(text: str) -> str:
    res = []
    for ch in text:
        if ch in DIGIT_MAP and random.random() < 0.35:
            res.append(random.choice(DIGIT_MAP[ch]))
        else:
            res.append(ch)
    return "".join(res)

def apply_space_splitting(text: str) -> list:
    variants = []
    # If Hangul or English, insert spaces between characters
    if len(text) >= 3:
        chars = list(text)
        # Random space insertion
        spaced = ""
        for i, c in enumerate(chars):
            spaced += c
            if i < len(chars) - 1 and random.random() < 0.4 and c != ' ':
                spaced += " "
        variants.append(spaced)
    return variants

def apply_symbol_prefix(text: str) -> str:
    sym = random.choice(PREFIX_SYMBOLS)
    return f"{sym}{text}"

def generate_controlled_augmentations(csv_path: str):
    print(f"📂 Reading Clean Core Dataset from: {csv_path}")

    raw_food_texts = []
    raw_not_food_texts = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row.get("source", "")
            # Keep ONLY clean unaugmented sources
            if "_ocr_aug" not in src:
                t = row.get("text", "").strip()
                l = row.get("label", "").strip()
                if t and l == "food":
                    raw_food_texts.append(t)
                elif t and l == "not_food":
                    raw_not_food_texts.append(t)

    print(f"  Clean Base Samples: {len(raw_food_texts)} food, {len(raw_not_food_texts)} not_food")

    augmented_training_lines = []

    # 1. Base Clean Samples
    for t in raw_food_texts:
        augmented_training_lines.append(f"__label__food {t}")
    for t in raw_not_food_texts:
        augmented_training_lines.append(f"__label__not_food {t}")

    # 2. Pattern 6 [HIGHEST PRIORITY]: Isolated 1-2 character residue HARD NEGATIVES
    print("  ✨ Pattern 6: Mining isolated 1-2 char residue as NOT_FOOD hard negatives...")
    for hard_neg in ISOLATED_HARD_NEGATIVES:
        # Repeat hard negatives multiple times so FastText strongly penalizes isolated fragments
        for _ in range(15):
            augmented_training_lines.append(f"__label__not_food {hard_neg}")

    # 3. Pattern 2 & 7: Word space splitting & Latin OCR noise (labeled food)
    print("  ✨ Pattern 2 & 7: Generating space-split and Latin OCR noise variants...")
    for t in raw_food_texts:
        if random.random() < 0.5:
            for var in apply_space_splitting(t):
                augmented_training_lines.append(f"__label__food {var}")

    # 4. Pattern 4: Leading symbol noise (#돈까스, ~콜라)
    print("  ✨ Pattern 4: Prepending POS printer symbol noise (#, ~, -, *)...")
    for t in random.sample(raw_food_texts, min(1500, len(raw_food_texts))):
        augmented_training_lines.append(f"__label__food {apply_symbol_prefix(t)}")

    # 5. Pattern 1: Digit-letter confusion on price and quantity lines (labeled not_food)
    print("  ✨ Pattern 1: Applying digit-letter confusion to price and numeric lines...")
    for t in random.sample(raw_not_food_texts, min(2000, len(raw_not_food_texts))):
        if any(c.isdigit() for c in t):
            augmented_training_lines.append(f"__label__not_food {apply_digit_letter_confusion(t)}")

    # 6. Pattern 3: Adjacent item merging (Concatenating 2 food items into 1 line)
    print("  ✨ Pattern 3: Generating merged adjacent menu items...")
    for _ in range(1000):
        item1 = random.choice(raw_food_texts)
        item2 = random.choice(raw_food_texts)
        merged_line = f"{item1} {item2}"
        augmented_training_lines.append(f"__label__food {merged_line}")

    random.shuffle(augmented_training_lines)
    print(f"✅ Total Precision-Augmented Training Samples: {len(augmented_training_lines)}")

    return augmented_training_lines

# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAIN & EVALUATE FASTTEXT PRECISION MODEL
# ─────────────────────────────────────────────────────────────────────────────
def run_precision_training():
    output_dir = "fastText_precision_augmented"
    os.makedirs(output_dir, exist_ok=True)

    csv_path = "korean_receipt_dataset_v25.csv"
    train_lines = generate_controlled_augmentations(csv_path)

    tmp_train_file = "fasttext_train_precision.tmp"
    with open(tmp_train_file, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines))

    print("\n🚀 Training FastText Precision-Augmented Model (dim=25, minn=2, maxn=5)...")
    model = fasttext.train_supervised(
        input=tmp_train_file,
        lr=0.5,
        epoch=25,
        wordNgrams=2,
        minn=2,
        maxn=5,
        dim=25,
        loss='ova'
    )

    raw_path = os.path.join(output_dir, "fasttext_korean_food.bin")
    quant_path = os.path.join(output_dir, "fasttext_korean_food.ftz")

    model.save_model(raw_path)
    model.quantize(input=tmp_train_file, retrain=True, epoch=5, lr=0.1)
    model.save_model(quant_path)

    print(f"✅ FastText Precision Model Saved to: {quant_path} ({os.path.getsize(quant_path)/(1024*1024):.2f} MB)")

    # Test predictions on 31 qq Receipts
    from process_qq_dual_model import extract_smart_lines, QQ_DIR, JAPANESE_REGEX, HANGUL_REGEX
    import easyocr
    reader = easyocr.Reader(['ko', 'en'], gpu=False)

    image_files = sorted([f for f in os.listdir(QQ_DIR) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])

    receipt_json_results = []
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
            conf = float(probs[0]) if probs else 0.0

            if pred == "food":
                food_in_receipt += 1
                total_food += 1

            receipt_items.append({
                "line_no": line_no,
                "text": line_text,
                "label": pred,
                "confidence": round(conf, 4)
            })

        receipt_json_results.append({
            "receipt_id": idx,
            "receipt_name": img_name,
            "line_count": len(extracted_lines),
            "food_count": food_in_receipt,
            "not_food_count": len(extracted_lines) - food_in_receipt,
            "items": receipt_items
        })

    # Save exact JSON result file
    json_path = "qq_json_results/fasttext_precision_augmented_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "FastText Precision Augmented (Controlled 6-Pattern Noise Fix)",
            "total_receipts": len(receipt_json_results),
            "total_lines": total_lines,
            "total_food_detected": total_food,
            "receipts": receipt_json_results
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "="*70)
    print("🏆 PRECISION-AUGMENTED MODEL RESULTS ON 31 REAL RECEIPTS")
    print("="*70)
    print(f"  Total Lines Evaluated    : {total_lines}")
    print(f"  Total Food Items Detected: {total_food}")
    print(f"  📄 JSON Results Saved to  : {json_path}")
    print("="*70)

    if os.path.exists(tmp_train_file): os.remove(tmp_train_file)

if __name__ == "__main__":
    run_precision_training()
