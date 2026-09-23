import os
import sys
import csv
import json
import random
import time
import fasttext

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
V25_CSV = os.path.join(BASE_DIR, "korean_receipt_dataset_v25.csv")

OUT_CSV = os.path.join(BASE_DIR, "korean_receipt_dataset_new_aug_v1.csv")
OUT_TSV = os.path.join(BASE_DIR, "korean_receipt_dataset_new_aug_v1.tsv")

# ─────────────────────────────────────────────────────────────────────────────
# 1. HANGUL VISUAL CONFUSION DICTIONARY
# ─────────────────────────────────────────────────────────────────────────────
HANGUL_VISUAL_MAP = {
    # 1. ㅇ vs ㅁ
    '아': '마', '마': '아', '오': '모', '모': '오', '우': '무', '무': '우',
    # 2. ㄷ vs ㄹ vs ㅌ
    '다': '라', '라': '타', '타': '다', '도': '로', '로': '토', '토': '도',
    # 3. ㄱ vs ㄴ
    '가': '나', '나': '가', '고': '노', '노': '고', '구': '누', '누': '구',
    # 4. ㅏ vs ㅓ
    '거': '가', '서': '사', '너': '나', '더': '다', '러': '라',
    # 5. ㅗ vs ㅜ
    '구': '고', '두': '도', '루': '로', '무': '모', '수': '소',
    # 6. ㅐ vs ㅔ
    '개': '게', '게': '개', '대': '데', '데': '대', '래': '레', '레': '래'
}

DIGIT_MAP = {
    '0': ['O', 'o', 'U'],
    '1': ['I', 'l', '|', '1'],
    '8': ['B', '8'],
    '5': ['S', 's', '5'],
    '6': ['G', '6'],
    '2': ['Z', 'z', '2']
}

PREFIX_SYMBOLS = ['#', '~', '-', '*', "'", '`', '+']

ISOLATED_HARD_NEGATIVES = [
    "꽃", "빠", "움", "철", "매", "꿀", "합", "용", "무", "올", "품", "들",
    "림", "삼", "개", "보", "자", "권", "학", "습", "가", "리", "레", "나",
    "트", "래", "말", "다", "8 용", "L 무", "히 올", "물 품", "필들", "궁림",
    "삼접삼", "김퇴미개", "우사라보", "진계포자", "{신이", "오권", "틈백학",
    "PST)", "~세로습가 골리", "부레 |찰t|1z05", "나드로 1B0g (트래", "논 무; 말다리",
    "포", "테", "이", "토", "도", "차", "슈", "유", "자", "청", "돈", "까", "스",
    "짜", "장", "면", "짬", "뽕", "볶", "음", "밥", "비", "빔", "냉", "탕", "찌", "개"
]

def apply_hangul_visual_confusion(text: str) -> str:
    res = []
    for ch in text:
        if ch in HANGUL_VISUAL_MAP and random.random() < 0.30:
            res.append(HANGUL_VISUAL_MAP[ch])
        else:
            res.append(ch)
    return "".join(res)

def apply_digit_letter_confusion(text: str) -> str:
    res = []
    for ch in text:
        if ch in DIGIT_MAP and random.random() < 0.40:
            res.append(random.choice(DIGIT_MAP[ch]))
        else:
            res.append(ch)
    return "".join(res)

def generate_rich_space_split_variants(text: str, n_variants: int = 5) -> list:
    variants = []
    chars = list(text)
    if len(chars) < 2:
        return []

    for _ in range(n_variants):
        spaced = ""
        for i, c in enumerate(chars):
            spaced += c
            if i < len(chars) - 1 and c != ' ' and random.random() < 0.50:
                spaced += " "
        if spaced != text and spaced.strip():
            variants.append(spaced.strip())

    return list(set(variants))

def apply_symbol_prefix(text: str) -> str:
    sym = random.choice(PREFIX_SYMBOLS)
    return f"{sym}{text}"

def build_hangul_visual_dataset():
    print(f"📂 Reading Clean Base Samples from: {V25_CSV}")

    raw_food_texts = []
    raw_not_food_texts = []
    rows_out = []

    with open(V25_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row.get("source", "")
            if "_ocr_aug" not in src:
                t = row.get("text", "").strip()
                l = row.get("label", "").strip()
                if t and l == "food":
                    raw_food_texts.append(t)
                    rows_out.append({"text": t, "label": "food", "source": src})
                elif t and l == "not_food":
                    raw_not_food_texts.append(t)
                    rows_out.append({"text": t, "label": "not_food", "source": src})

    print(f"  Clean Base Rows: {len(rows_out)} (Food: {len(raw_food_texts)}, Not-Food: {len(raw_not_food_texts)})")

    # 1. Hangul Visual Confusion Augmentation (NEW)
    print("  ✨ Pattern 8 (NEW): Generating Hangul Visual Confusion (ㅇ/ㅁ, ㄷ/ㄹ/ㅌ, ㄱ/ㄴ, ㅏ/ㅓ, ㅗ/ㅜ, ㅐ/ㅔ)...")
    for t in raw_food_texts:
        v = apply_hangul_visual_confusion(t)
        if v != t:
            rows_out.append({"text": v, "label": "food", "source": "pattern8_hangul_visual_confusion"})

    for t in raw_not_food_texts:
        v = apply_hangul_visual_confusion(t)
        if v != t:
            rows_out.append({"text": v, "label": "not_food", "source": "pattern8_hangul_visual_confusion"})

    # 2. Rich Multi-Space Splitting (5 variants per food item)
    print("  ✨ Pattern 2 & 7: Generating 5 space-split variants per food item...")
    for t in raw_food_texts:
        for var in generate_rich_space_split_variants(t, n_variants=5):
            rows_out.append({"text": var, "label": "food", "source": "pattern2_word_space_split"})

    for t in raw_not_food_texts:
        if len(t) >= 3:
            for var in generate_rich_space_split_variants(t, n_variants=2):
                rows_out.append({"text": var, "label": "not_food", "source": "pattern2_not_food_space_split"})

    # 3. Symbol Noise (#, ~, -, *)
    print("  ✨ Pattern 4: Adding POS printer symbol noise to all food items...")
    for t in raw_food_texts:
        rows_out.append({"text": apply_symbol_prefix(t), "label": "food", "source": "pattern4_leading_symbol"})

    # 4. Digit-letter confusion (0<->O, 1<->I)
    print("  ✨ Pattern 1: Applying digit-letter confusion to all numeric/price lines...")
    for t in raw_not_food_texts:
        if any(c.isdigit() for c in t):
            rows_out.append({"text": apply_digit_letter_confusion(t), "label": "not_food", "source": "pattern1_digit_letter_confusion"})

    # 5. Adjacent item merging (8,000 merged food pairs)
    print("  ✨ Pattern 3: Generating 8,000 merged adjacent menu item pairs...")
    for _ in range(8000):
        item1 = random.choice(raw_food_texts)
        item2 = random.choice(raw_food_texts)
        rows_out.append({"text": f"{item1} {item2}", "label": "food", "source": "pattern3_adjacent_merged"})

    # 6. Hard Negatives for isolated 1-2 char residue
    print("  ✨ Pattern 6: Adding 2,500 hard negative single/double char residue samples...")
    for hard_neg in ISOLATED_HARD_NEGATIVES:
        for _ in range(35):
            rows_out.append({"text": hard_neg, "label": "not_food", "source": "pattern6_isolated_residue_hardneg"})

    random.shuffle(rows_out)
    print(f"\n🚀 Total Rows in fully expanded dataset: {len(rows_out)}")

    # Write CSV
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["text", "label", "source"], quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows_out)

    # Write TSV
    with open(OUT_TSV, "w", encoding="utf-8", newline="") as f_tsv:
        writer = csv.writer(f_tsv, delimiter="\t")
        writer.writerow(["text", "label", "source"])
        for r in rows_out:
            writer.writerow([r["text"], r["label"], r["source"]])

    print(f"📄 Saved CSV Dataset : {OUT_CSV}")
    print(f"📄 Saved TSV Dataset : {OUT_TSV}")

    # Train FastText
    output_dir = os.path.join(BASE_DIR, "fastText_new_aug_v1")
    os.makedirs(output_dir, exist_ok=True)

    tmp_train = "fasttext_train_hangul_aug.tmp"
    with open(tmp_train, "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(f"__label__{r['label']} {r['text']}\n")

    print("\n🚀 Training FastText Model on Hangul Visual & Multi-Space Dataset...")
    model = fasttext.train_supervised(
        input=tmp_train,
        lr=0.5,
        epoch=25,
        wordNgrams=2,
        minn=2,
        maxn=5,
        dim=25,
        loss='ova'
    )

    raw_model_path = os.path.join(output_dir, "fasttext_korean_food.bin")
    quant_model_path = os.path.join(output_dir, "fasttext_korean_food.ftz")

    model.save_model(raw_model_path)
    model.quantize(input=tmp_train, retrain=True, epoch=5, lr=0.1)
    model.save_model(quant_model_path)

    print(f"✅ Saved FastText Model to: {quant_model_path} ({os.path.getsize(quant_model_path)/(1024*1024):.2f} MB)")

    # Run predictions on 31 qq Receipts
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

    json_path = os.path.join(BASE_DIR, "qq_json_results", "fasttext_new_aug_v1_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "FastText korean_receipt_dataset_new_aug_v1 (Hangul Visual + Multi-Space Expanded)",
            "total_dataset_rows": len(rows_out),
            "total_receipts": len(receipt_json_results),
            "total_lines": total_lines,
            "total_food_detected": total_food,
            "receipts": receipt_json_results
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "="*70)
    print("🏆 HANGUL VISUAL & MULTI-SPACE AUGMENTATION COMPLETED!")
    print("="*70)
    print(f"  Total Dataset Rows      : {len(rows_out)}")
    print(f"  Dataset Files Created   : korean_receipt_dataset_new_aug_v1.csv & .tsv")
    print(f"  Trained Model Directory : {output_dir}")
    print(f"  Total Lines Evaluated   : {total_lines}")
    print(f"  Total Food Items        : {total_food}")
    print(f"  📄 JSON Results Saved to : {json_path}")
    print("="*70)

    if os.path.exists(tmp_train): os.remove(tmp_train)

if __name__ == "__main__":
    build_hangul_visual_dataset()
