import random

# ─────────────────────────────────────────────────────────────────────────────
# KOREAN VISUAL CORRUPTION & JAMO STROKE SIMILARITY MATRIX
# Based on optical degradation in thermal printer scans & low-DPI mobile OCR
# ─────────────────────────────────────────────────────────────────────────────

# 1. Direct Syllable Visual Substitutions (High-frequency OCR misreads)
SYLLABLE_VISUAL_MAP = {
    # Consonant / Vowel Blur Pairs
    '새': ['사', '서', '게', '새'],
    '화': ['외', '와', '과', '화'],
    '휘': ['취', '회', '귀', '휘'],
    '세': ['시', '세', '소', '데'],
    '한': ['논', '한', '학', '함'],
    '우': ['무', '우', '오', '유'],
    '버': ['니', '너', '버', '배'],
    '거': ['가', '거', '고', '계'],
    '핑': ['평', '풍', '핑'],
    '다': ['라', '타', '다', '더'],
    '리': ['디', '피', '리', '티'],
    '김': ['겸', '감', '김'],
    '밥': ['밤', '방', '밥'],
    '국': ['궁', '굴', '국'],
    '면': ['년', '명', '면'],
    '갈': ['감', '강', '갈'],
    '비': ['바', '피', '비'],
    '삼': ['상', '서', '삼'],
    '겹': ['겸', '경', '겹'],
    '살': ['설', '삭', '살'],
    '돈': ['돌', '동', '돈'],
    '까': ['가', '싸', '까'],
    '스': ['으', '소', '스'],
    '아': ['마', '야', '아'],
    '메': ['마', '모', '메'],
    '카': ['차', '타', '카'],
    '노': ['도', '모', '노'],
    '라': ['더', '다', '라'],
    '떼': ['따', '데', '떼'],
    '상': ['삼', '상', '서'],
    '품': ['풍', '풀', '품'],
    '코': ['쿼', '코', '크'],
    '드': ['더', '트', '드'],
    '포': ['모', '푸', '포'],
    '장': ['자', '정', '장'],
    '유': ['우', '유', '요'],
    '무': ['모', '무', '문'],
    '마': ['아', '모', '마'],
    '일': ['인', '일', '입'],
    '점': ['정', '접', '점'],
    '고': ['도', '조', '고'],
    '객': ['각', '객', '책'],
    '님': ['남', '님', '림'],
    '준': ['중', '줄', '준'],
    '비': ['피', '비', '바'],
}

def corrupt_mid_word_syllables(text: str, corruption_prob: float = 0.35) -> str:
    """
    Applies deep visual corruption to syllables anywhere inside the word.
    Handles mid-word substitution, character deletion, and character duplication.
    """
    if len(text) < 2:
        return text

    chars = list(text)
    res = []

    for i, ch in enumerate(chars):
        r = random.random()
        if ch in SYLLABLE_VISUAL_MAP and r < corruption_prob:
            # 1. Syllable Visual Substitution
            res.append(random.choice(SYLLABLE_VISUAL_MAP[ch]))
        elif r < (corruption_prob * 0.15) and len(chars) > 3 and i > 0 and i < len(chars) - 1:
            # 2. Random Character Duplication (OCR ghosting)
            res.append(ch)
            res.append(ch)
        elif r < (corruption_prob * 0.25) and len(chars) > 4 and i > 0 and i < len(chars) - 1 and ch != ' ':
            # 3. Random Character Deletion (OCR stroke drop)
            continue
        else:
            res.append(ch)

    corrupted = "".join(res)
    return corrupted if corrupted.strip() else text

if __name__ == "__main__":
    # Test cases
    test_terms = ["새우버거세트", "화이어윙4", "휘핑", "한우앞다리", "상품코드", "포장유무", "EATZ마일", "조청쌀엿"]
    print("--- Jamo Visual Corruption Test ---")
    for t in test_terms:
        print(f"Original: {t:12s} -> Corrupted: {corrupt_mid_word_syllables(t, corruption_prob=0.5)}")
