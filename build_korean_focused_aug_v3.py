import os
import sys
import csv
import json
import random
import time
import fasttext
from jamo_visual_corruption import corrupt_mid_word_syllables, SYLLABLE_VISUAL_MAP

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_AUG_V1_CSV = os.path.join(BASE_DIR, "korean_receipt_dataset_new_aug_v1.csv")

OUT_CSV = os.path.join(BASE_DIR, "korean_receipt_dataset_v30_korean_focused.csv")
OUT_TSV = os.path.join(BASE_DIR, "korean_receipt_dataset_v30_korean_focused.tsv")

PREFIX_SYMBOLS = ['#', '~', '-', '*', "'", '`', '+']

# ─────────────────────────────────────────────────────────────────────────────
# 1. REAL HARD EXAMPLES VAULT (BBQ & Real Receipt Ground Truth)
# Upweighted 50x to guarantee zero dilution regressions
# ─────────────────────────────────────────────────────────────────────────────
REAL_HARD_FOOD_VAULT = [
    "궁림", "소감비삼", "삼접삼", "김퇴미개", "우사라보", "진계포자", "{신이", "오권", "틈백학", "~세로습가 골리",
    "동서 코코별", "조청쌀엿", "표고버섯", "돈까스*디진다", "#양", "#~곰 라(R)", "~칠리", "~컬리", "~포데이도", "-포테이토",
    "베이컨에그S", "'강정(순한)"
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. MASTER CLOSED-VOCABULARY HARD NEGATIVE FAMILIES (not_food)
# ─────────────────────────────────────────────────────────────────────────────
HEADER_METADATA_FAMILY = [
    "상품코드", "상품번호", "상품명", "상품명칭", "제품코드", "제품번호", "품목코드", "품목번호", "품명", "물품명",
    "메뉴코드", "메뉴명", "상품정보", "상품내역", "판매상품", "구매내역", "주문목록", "단가", "단가 gy", "수량",
    "수량계", "수량합계", "규격", "단위", "판매단가", "정상단가", "할인가", "판매가", "가격", "금액", "중량", "무게",
    "와이파이", "화장실비번", "외장실비번", "비밀번호", "제품이", "제품을", "재품이", "제둥이 제품흘", "디처트럭인"
]

PACKAGING_STATUS_FAMILY = [
    "포장유무", "포장여부", "포장상태", "포장구분", "포장방법", "포장종류", "포장가능", "포장불가", "포장비", "포장비용",
    "포장주문", "포장주문서", "포장번호", "포장시간", "포장여부확인", "포장유무확인", "포장가능여부", "개별포장", "진공포장", "밀봉포장",
    "매장식사", "매장 식사", "[ 매장 식사 ]", "[매장식사]", "포장/매장", "포장구분"
]

ORDER_STATUS_CUSTOMER_FAMILY = [
    "고객님", "고객님이", "고객님께서", "고객님의", "주문하신", "주문하신 메뉴", "주문하신 상품", "주문하신 제품",
    "고객님이 주문한 상품", "고객님이 주문한 메뉴", "고객님이 주문하신 제품을 준비하고 있습니다",
    "주문하신 메뉴를 준비하고 있습니다", "주문하신 메뉴가 준비되었습니다", "주문하신 상품이 나왔습니다",
    "주문하신 메뉴를 픽업해주세요", "준비중", "준비완료", "픽업대기", "픽업완료", "대기번호", "대기 번호", "대기순번",
    "대기 순번", "대기순서", "주문번호", "픽업번호", "픽업대에서", "영수증을 버리지 마세요", "교환권", "대기순번표"
]

LOYALTY_MILEAGE_FAMILY = [
    "EATZ마일", "EATZ 마일", "EATZ마일리지", "EATZ마일 적립", "EATZ마일 사용", "L.POINT", "L포인트", "LPOINT", "엘포인트",
    "엘 포인트", "롯데포인트", "롯데 L포인트", "CJ ONE", "해피포인트", "신세계포인트", "현대M포인트", "OK캐쉬백", "마일리지",
    "마일리지 적립", "마일리지 사용", "포인트", "포인트 적립", "포인트 사용", "포인트 잔액", "적립금", "POS번호", "승인번호", "카카오페이"
]

PAYMENT_TRANSACTION_FAMILY = [
    "결제", "결제금액", "최종결제금액", "실결제금액", "결제완료", "결제취소", "결제승인", "결제수단", "카드", "카드결제",
    "카드승인", "신용카드", "체크카드", "현금", "현금결제", "현금영수증", "현금영수증발급", "현금영수증승인", "무서명거래",
    "무서명 결제", "무서명", "서명생략", "서명불필요", "거래", "거래번호", "거래일시", "거래일자", "거래금액", "거래후",
    "거스름돈", "거스름", "잔액", "잔돈", "받은금액", "현금거스름돈", "영수증", "영수증번호", "영수증 발행", "교환", "환불", "반품",
    "부가세", "부가가치세", "과세", "과세금액", "과세물품", "면세", "면세금액", "면세물품", "사업자번호", "사업자등록번호",
    "대표자", "가맹점명", "대표전화", "전화번호", "경남양산시덕게로 78 대표 목승민", "단체주문은 역시 롯데리아 제일점",
    "시물시 순구 소공로 63", "주소", "도로명주소", "지번주소", "외상", "저리결과 : 싱싱 저리", "수수묘: *0원", "Central", "Rcpti",
    "오직 맛' 오직 정량! 오직 정성 ! 청결한매장", "오직 맛", "오직 정량", "오직 정성", "청결한매장", "KB 몰래티늄", "거래번드 오 POS:"
]

DISCOUNT_COUPON_FAMILY = [
    "할인", "할인금액", "할인율", "할인적용", "행사할인", "즉시할인", "카드할인", "제휴카드할인", "회원할인", "멤버십할인",
    "쿠폰", "쿠폰할인", "쿠폰적용", "쿠폰사용", "상품권", "모바일상품권", "교환권", "제품교환권"
]

NUMERIC_HARD_NEGATIVES = [
    "500", "200", "300", "30", "000", "0000", "13,500", "3,830", "15,000", "4,380", "2,680",
    "500g", "1kg", "100g", "1개", "2개", "0A N", "PDS:30", "B0U", "0000원", "0원", "1,000원", "1 #모", "[무 인 ]",
    "[아이스]", "[[아이스]]", "[[아이스] (R)]", "(R)", "[매장]", "[포장]", "(HOT)", "(ICE)", "테이"
]

ISOLATED_HARD_NEGATIVES = [
    "꽃", "빠", "움", "철", "매", "꿀", "합", "용", "무", "올", "품", "들", "림", "삼", "개", "보", "자", "권", "학", "습",
    "가", "리", "레", "나", "트", "래", "말", "다", "8 용", "L 무", "히 올", "물 품", "필들", "PST)", "부레 |찰t|1z05"
] + NUMERIC_HARD_NEGATIVES

# ─────────────────────────────────────────────────────────────────────────────
# 3. CONTEXTUAL CONTRAST PAIRS ("Same Root, Opposite Label")
# ─────────────────────────────────────────────────────────────────────────────
CONTRAST_FOOD_POSITIVES = [
    # '금' Contrast
    "소금", "맛소금", "천일염", "꽃소금", "죽염", "금귤청", "금귤차", "금귤",
    # '포장' Contrast
    "포장 아메리카노", "포장 아이스아메리카노", "포장 핫아메리카노", "포장 카페라떼", "포장 헤이즐넛아메리카노",
    "포장 김치찌개", "포장 된장찌개", "포장 부대찌개", "포장 라면", "포장 우동", "포장 소바", "포장 김밥",
    "포장 떡볶이", "포장 순대", "포장 돈까스", "포장 햄버거", "포장 치킨", "포장 피자", "포장 샌드위치", "포장 도시락", "포장 초밥",
    # '상품' Contrast
    "신라면", "진라면", "삼양라면", "신라면 5개입", "안성탕면", "짜파게티", "너구리",
    # '주문' Contrast
    "주문: 김치찌개", "주문: 돈까스", "주문: 제육볶음",
    # '식' Contrast
    "정식", "한식", "식사", "식혜", "제육정식", "불고기정식", "돈까스정식", "회정식", "생선구이정식",
    # '장' Contrast
    "간장", "진간장", "양조간장", "국간장", "고추장", "된장", "쌈장", "춘장",
    # '김' / '파' / '배' / '밤' Contrast
    "김밥", "김치", "김가루", "김자반", "대파", "쪽파", "파채", "파김치", "배추김치", "배즙", "밤양갱", "군밤", "맛밤"
]

KIOSK_TAG_PREFIXES = ["[포장]", "[매장]", "(HOT)", "(ICE)", "[배달]", "[픽업]", "(1인분)", "(2인분)", "[옵션]"]

def apply_digit_price_fusion(text: str) -> list:
    fused = []
    prices = ["3,830원", "13,500원", "15,000원", "4,380원", "2,680원", "4,500원", "3,500원", "7,900원"]
    weights_qty = ["300g", "1.2kg", "2kg", "500g", "1L", "1.5L", "2개", "5개입", "10개"]

    for p in random.sample(prices, min(2, len(prices))):
        fused.append(f"{text} {p}")

    for w in random.sample(weights_qty, min(2, len(weights_qty))):
        p = random.choice(prices)
        fused.append(f"{text} {w} {p}")

    return fused

def apply_space_splits(text: str, n_variants: int = 2) -> list:
    variants = []
    chars = list(text)
    if len(chars) < 2:
        return []
    for _ in range(n_variants):
        spaced = ""
        for i, c in enumerate(chars):
            spaced += c
            if i < len(chars) - 1 and c != ' ' and random.random() < 0.40:
                spaced += " "
        if spaced != text and spaced.strip():
            variants.append(spaced.strip())
    return list(set(variants))

# ─────────────────────────────────────────────────────────────────────────────
# MASTER DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_v30_dataset():
    print(f"📂 Reading Entire Base Dataset from: {NEW_AUG_V1_CSV} (54k dataset)")

    raw_food_texts = set()
    raw_not_food_texts = set()
    rows_out = []

    with open(NEW_AUG_V1_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("text", "").strip()
            l = row.get("label", "").strip()
            src = row.get("source", "new_aug_v1_base")
            if t and l == "food":
                raw_food_texts.add(t)
                rows_out.append({"text": t, "label": "food", "source": src})
            elif t and l == "not_food":
                raw_not_food_texts.add(t)
                rows_out.append({"text": t, "label": "not_food", "source": src})

    for cf in CONTRAST_FOOD_POSITIVES:
        raw_food_texts.add(cf)
        rows_out.append({"text": cf, "label": "food", "source": "contrast_food_positive"})

    print(f"  Base new_aug_v1 Rows Loaded: {len(rows_out)} (Food Terms: {len(raw_food_texts)}, Not-Food Terms: {len(raw_not_food_texts)})")

    # 1. UPWEIGHT REAL HARD EXAMPLES VAULT (50x Repetition)
    print("  🔥 Sub-group 1: Upweighting Real Hard Examples Vault (50x repetition)...")
    for item in REAL_HARD_FOOD_VAULT:
        for _ in range(50):
            rows_out.append({"text": item, "label": "food", "source": "subgroup1_real_hard_food_vault"})

    # 2. LAYER NEW VISUAL JAMO BLUR & PRICE FUSION ONTO ALL FOOD TERMS
    print("  ✨ Sub-groups 2-6: Layering new visual Jamo blur & price/quantity fusions onto food terms...")
    food_list = list(raw_food_texts)
    
    # Sample 15,000 food terms to layer new Jamo blur and price fusion
    sampled_food = random.sample(food_list, min(15000, len(food_list)))
    for t in sampled_food:
        corrupt_jamo = corrupt_mid_word_syllables(t, corruption_prob=0.30)
        if corrupt_jamo != t:
            rows_out.append({"text": corrupt_jamo, "label": "food", "source": "subgroup2_jamo_visual_corruption"})

        for fused in apply_digit_price_fusion(t):
            rows_out.append({"text": fused, "label": "food", "source": "subgroup3_price_quantity_fusion"})

        prefix_tag = random.choice(KIOSK_TAG_PREFIXES)
        rows_out.append({"text": f"{prefix_tag} {t}", "label": "food", "source": "subgroup6_kiosk_tag_prefix"})

    # 3. MASTER CLOSED-VOCABULARY HARD NEGATIVE FAMILIES (not_food)
    print("  🛡️ Sub-groups 7-11: Layering Corrupted Families for closed-vocabulary hard negatives...")

    all_neg_families = [
        (HEADER_METADATA_FAMILY, "subgroup7_receipt_header_metadata"),
        (PACKAGING_STATUS_FAMILY, "subgroup2_packaging_status_family"),
        (ORDER_STATUS_CUSTOMER_FAMILY, "subgroup8_order_status_customer"),
        (LOYALTY_MILEAGE_FAMILY, "subgroup9_loyalty_mileage"),
        (PAYMENT_TRANSACTION_FAMILY, "subgroup10_payment_transaction"),
        (DISCOUNT_COUPON_FAMILY, "subgroup11_discount_coupon")
    ]

    for family, src_tag in all_neg_families:
        for term in family:
            rows_out.append({"text": term, "label": "not_food", "source": src_tag})

            corrupt_t = corrupt_mid_word_syllables(term, corruption_prob=0.35)
            rows_out.append({"text": corrupt_t, "label": "not_food", "source": f"{src_tag}_jamo_blur"})

            for sp in apply_space_splits(term, n_variants=2):
                rows_out.append({"text": sp, "label": "not_food", "source": f"{src_tag}_space_split"})

            sym = random.choice(PREFIX_SYMBOLS)
            rows_out.append({"text": f"{sym}{term}", "label": "not_food", "source": f"{src_tag}_symbol_prefix"})

    # 4. NUMERIC & ISOLATED SINGLE-CHARACTER HARD NEGATIVES (Upweighted 80x)
    print("  🛡️ Adding numeric & single-character isolated hard negatives (80x upweight)...")
    for hard_neg in ISOLATED_HARD_NEGATIVES:
        for _ in range(80):
            rows_out.append({"text": hard_neg, "label": "not_food", "source": "subgroup12_isolated_hard_neg"})

    # Shuffle
    random.shuffle(rows_out)
    
    food_rows = [r for r in rows_out if r["label"] == "food"]
    not_food_rows = [r for r in rows_out if r["label"] == "not_food"]

    print(f"\n📊 Pre-balance row count: Food={len(food_rows)}, Not-Food={len(not_food_rows)}")

    # Balance
    max_len = max(len(food_rows), len(not_food_rows))
    while len(food_rows) < max_len:
        food_rows.append(random.choice(food_rows))
    while len(not_food_rows) < max_len:
        not_food_rows.append(random.choice(not_food_rows))

    final_rows = food_rows + not_food_rows
    random.shuffle(final_rows)

    print(f"🚀 Final Balanced Rows in Dataset v3.0: {len(final_rows)} (Food: {len(food_rows)}, Not-Food: {len(not_food_rows)})")

    # Save CSV
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["text", "label", "source"], quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(final_rows)

    # Save TSV
    with open(OUT_TSV, "w", encoding="utf-8", newline="") as f_tsv:
        writer = csv.writer(f_tsv, delimiter="\t")
        writer.writerow(["text", "label", "source"])
        for r in final_rows:
            writer.writerow([r["text"], r["label"], r["source"]])

    print(f"📄 Saved CSV Dataset : {OUT_CSV}")
    print(f"📄 Saved TSV Dataset : {OUT_TSV}")

    # Train FastText v3.0
    output_dir = os.path.join(BASE_DIR, "fastText_v30_korean")
    os.makedirs(output_dir, exist_ok=True)

    tmp_train = "fasttext_train_v30.tmp"
    with open(tmp_train, "w", encoding="utf-8") as f:
        for r in final_rows:
            f.write(f"__label__{r['label']} {r['text']}\n")

    print("\n🚀 Training FastText v3.0 Model on Full new_aug_v1 (54k) + v3.0 Family Augmentations...")
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

    bin_path = os.path.join(output_dir, "fasttext_korean_food.bin")
    ftz_path = os.path.join(output_dir, "fasttext_korean_food.ftz")

    model.save_model(bin_path)
    print(f"💾 Saved .bin model: {bin_path}")

    model.quantize(input=tmp_train, retrain=True)
    model.save_model(ftz_path)
    print(f"💾 Saved .ftz quantized model: {ftz_path}")

    if os.path.exists(tmp_train):
        os.remove(tmp_train)

if __name__ == "__main__":
    build_v30_dataset()
