#!/usr/bin/env python3
"""
receipt_patterns_additions.py
=============================
Drop-in additions for the Receipt Structurer. Every pattern here is either
(a) based on a receipt format found while researching, or (b) a fix for a gap
in the original (marked FIX). Patterns marked [UNVERIFIED] come from general
knowledge, not from a source I could open - test them on real receipts.

Integration order inside parse_receipt_line_advanced():
  1. parse_savings_modifier(line)   -> attach to previous item, stop
  2. parse_tax_line / parse_total_line -> store for reconciliation, stop
  3. classify_fee_line(line)        -> keep as fee/deposit item, skip noise
  4. is_added_noise(line)           -> drop
  5. existing pipeline
"""

import re
import unicodedata
from typing import Optional, Dict, Any, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# 1. ADDED NOISE PATTERNS (all word-bounded; products protected below)
# ─────────────────────────────────────────────────────────────────────────────
ADDED_NOISE_KEYWORDS = [
    # --- EBT / SNAP / FSA (US). Source: Xenial, Square, usa.gov, CDSS ---
    r'\bebt\b',
    r'\bsnap\s*(?:eligible|amount\s*due|balance|total|tend(?:er)?|payment|benefit)s?\b',
    r'\bfood\s*stamps?\b',
    r'\bfsa\s*(?:eligible|tend(?:er)?|total|card)\b',
    r'\bpaid\s*with\s*(?:ebt|additional\s*tenders?|cash|card)\b',
    r'^\s*(?:snap\s*)?(?:non[\s\-]*)?eligible\s*items?\s*:?\s*$',
    r'\b(?:remaining|previous|new|ending|available)\s*(?:snap\s*|ebt\s*|food\s*|cash\s*)?balance\b',
    r'\b(?:food|cash)\s*(?:benefit\s*)?balance\b',

    # --- Loyalty / rewards footers (Kroger fuel points, Clubcard, Nectar) ---
    r'\bfuel\s*points?\b',
    r'\b(?:rewards?|points?)\s*(?:balance|earned|redeemed|this\s*trip|summary|number|total)\b',
    r'\bclub\s*card\s*(?:number|no|#|savings\s*total)\b',
    r'\bclubcard\s*(?:points?|balance|number|no|voucher|summary)\b',
    r'\bnectar\s*(?:points?|card|balance)\b',
    r'\bboost\s*member', r'\bpoints\s*to\s*(?:next|go)\b',
    r'\b(?:take|complete)\s*(?:our|the)\s*survey\b', r'\bsurvey\b', r'\bfeedback\b',
    r'\bspend\s*\$?\d+.*\bget\s*\$?\d+',            # SPEND $50 GET $5

    # --- URLs / emails ---
    r'\bwww\.', r'https?://', r'\b[\w.\-]+@[\w\-]+\.[a-z]{2,}\b',

    # --- Card-terminal slips (CHIP / SIGNATURE deliberately NOT bare words) ---
    r'\bapproval\s*(?:code|no|#)?\b', r'\bapproved\b',
    r'\bauth(?:orization)?\s*(?:code|#|no)\b',
    r'\bcontactless\b', r'\bpin\s*verified\b', r'\bsignature\s*(?:required|verified)\b',
    r'\bno\s*signature\s*required\b', r'^\s*x\s*_{3,}',
    r'\baid\s*:\s*[a-f0-9]{8,}', r'\bapplication\s*(?:label|id)\b',

    # --- Indian GST / retail invoice scaffolding. Source: BUSY, Scribd bills ---
    r'\binvoice\s*(?:no|number|#|date)\b', r'\bbill\s*(?:no|number|#|date)\b',
    r'\bplace\s*of\s*supply\b', r'\breverse\s*charge\b', r'\bhsn(?:/sac)?\b',
    r'\bitems?\s*purchased\s*[:=]', r'\bgst\s*receipt\s*summary\b',
    r'\btotal\s*amount\s*before\s*tax\b', r'\btotal\s*(?:in\s*)?words\b',
    r'\b(?:cgst|sgst|utgst|igst|cess)\s*(?:@|-)?\s*\d',
    r'\bsavings?\s*(?:on\s*mrp|today)\b', r'\byou\s*have\s*saved\b',
    r'\brupees\b.*\bonly\b',

    # --- German (Kassenbon). Source: reversesalestaxcalc.org/de, press ---
    r'\bzwischensumme\b', r'\bsumme\b', r'\bgesamt(?:betrag|summe)?\b',
    r'\bmwst\b', r'\bust[\s\-]*id', r'\bsteuer(?:nummer|[\s\-]*nr)\b',
    r'\bgegeben\b', r'\br[üu]ckgeld\b', r'\bec[\s\-]*karte\b', r'\bgirocard\b',
    r'\bkartenzahlung\b', r'\bvielen\s*dank\b', r'\bkassenbon\b', r'\bpfandbon\b',
    r'\bkundenbeleg\b', r'\btse[\s\-]*(?:seriennummer|signatur|start|stop)\b',
    r'^\s*bar\s*:?\s*[\d.,]+\s*(?:eur|€)?\s*$',      # anchored: not "GRANOLA BAR"

    # --- Czech / Slovak extras (existing list has celkem, DPH, platba) ---
    r'\bi[čc]o\b', r'\bdi[čc]\b', r'\bzaplaceno\b', r'\bk\s*[uú]hrad[eě]\b',
    r'\bplatba\s*kartou\b', r'\bhotov[eě]\b', r'\bzákaznick[aá]\s*karta\b',

    # --- Korean extras [UNVERIFIED wording; terms are standard receipt labels] ---
    r'현금영수증', r'신용카드', r'체크카드', r'승인번호', r'카드번호', r'공급가액',
    r'사업자등록', r'과세물품', r'면세물품', r'할부', r'일시불', r'결제금액',
    r'받은금액', r'적립', r'포인트', r'영수증번호', r'거래일시', r'단말기', r'주문번호',
]
ADDED_NOISE_REGEX = re.compile('(?:' + '|'.join(ADDED_NOISE_KEYWORDS) + ')', re.IGNORECASE)

# Real products that would otherwise trip the patterns above or the ORIGINAL list.
ADDED_PRODUCT_PROTECTION = re.compile(
    r'\b(?:snap\s*peas?|sugar\s*snap|snap\s*pea|table\s*salt|tablets?|regular|reggiano|'
    r'chicken\s*tenders?|tender\s*loin|phone\s*charger|chk?n\s*(?:breast|thigh)|'
    r'granola\s*bar|protein\s*bar|soap\s*bar|nectar|agave|clubcard\s*price|'
    r'kirkland\s*signature|signature\s*(?:select|farms|kitchens)|choc(?:olate)?\s*chip|'
    r'chip\s*ahoy|potato\s*chips?|tortilla\s*chips?|cashew|cashier\s*pick)\b',
    re.IGNORECASE,
)


def is_added_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if ADDED_PRODUCT_PROTECTION.search(s):
        return False
    return bool(ADDED_NOISE_REGEX.search(s))


# ─────────────────────────────────────────────────────────────────────────────
# 2. FIX: DROP-IN REPLACEMENTS FOR OVER-BROAD ORIGINAL NOISE PATTERNS
#    Remove the originals from NOISE_KEYWORDS and add these instead.
# ─────────────────────────────────────────────────────────────────────────────
FIXED_NOISE_REPLACEMENTS = [
    # was: (?:cashier|cshr|reg|register)\s*[:#\s]?\s*[A-Za-z0-9]+   (ate REGULAR, REGGIANO)
    r'\b(?:cashier|cshr|reg(?:ister)?)\s*[:#]\s*\w+',
    # was: ^\s*(?:table|guest|server|station|order|chk|check)\s*[:#\s]?\s*[A-Za-z0-9/.\-]+
    #      (ate TABLE SALT, CHKN BREAST)
    r'^\s*(?:table|guest|server|station|order|chk|check)\s*[:#]\s*[\w/.\-]+',
    # was: \b(?:amt\s*)?tend(?:er(?:ed)?)?\b   (ate CHICKEN TENDER)
    r'^\s*[*\-\s]*(?:amt\s*|cash\s*|debit\s*|card\s*)?tender(?:ed)?\s*[:=]?\s*[\$€£]?\s*\d',
    r'^\s*[*\-\s]*tend\s*[:=]?\s*[\$€£]?\s*\d',
    # was: \bphone[:.\s]  (ate PHONE CHARGER)
    r'\b(?:phone|tel)\s*[:.]\s*[\d(+]',
    # was: \bvoid(?:ed)?\b  (keep, but anchor to POS wording)
    r'^\s*(?:\*+\s*)?void(?:ed)?\b',
]


# ─────────────────────────────────────────────────────────────────────────────
# 3. SAVINGS / DISCOUNT CHILD ROWS (attach to previous item, do not drop)
#    Sources: Safeway two-line "regular price + card savings", Whole Foods
#    "Prime Extra 10%" / "Prime Buy 1 Get 1", Kroger digital coupons, Tesco
#    Clubcard prices.
# ─────────────────────────────────────────────────────────────────────────────
SAVINGS_LABEL_RE = re.compile(
    r'^\s*(?P<label>'
    r'(?:(?:card|club|clubcard|member|prime|loyalty|digital|store|mfr|manufacturer|instant|promo|nectar)\s*)?'
    r'(?:coupon|savings?|discount|price\s*savings?|rebate|deal)'
    r'|prime\s*extra\s*\d+\s*%'
    r'|prime\s*buy\s*\d+\s*get\s*\d+(?:\s*(?:free|\d+\s*%\s*off))?'
    r'|buy\s*\d+\s*get\s*\d+(?:\s*(?:free|\d+\s*%\s*off))?'
    r'|bogo'
    r'|you\s*saved'
    r')(?:\b|(?<=%))',
    re.IGNORECASE,
)
REGULAR_PRICE_RE = re.compile(
    r'^\s*(?:reg(?:ular)?|orig(?:inal)?|was)\.?\s*(?:price)?\s*[:$]?\s*[\$€£]?\s*(?P<amt>\d+[.,]\d{2})\s*$',
    re.IGNORECASE,
)
_AMOUNT_RE = re.compile(r'(?<![\w.])(?P<neg1>-)?\s*[\$€£₹₩]?\s*(?P<amt>\d{1,3}(?:[.,]\d{3})*[.,]\d{2}|\d+[.,]\d{2})\s*(?P<neg2>-)?(?![\w])')


def _to_float(txt: str) -> Optional[float]:
    t = txt.replace(' ', '')
    if re.search(r'^-?\d{1,3}(?:\.\d{3})+,\d{2}$', t):
        t = t.replace('.', '').replace(',', '.')
    elif re.search(r'^-?\d+,\d{2}$', t):
        t = t.replace(',', '.')
    else:
        t = t.replace(',', '')
    try:
        return float(t)
    except ValueError:
        return None


def parse_savings_modifier(line: str) -> Optional[Dict[str, Any]]:
    """Return {'kind','label','amount'} for per-item savings rows, else None.
    amount is negative for discounts; None when the row only names a deal
    (e.g. 'Prime Extra 10%') and the dollar value is printed elsewhere."""
    s = line.strip()
    m = REGULAR_PRICE_RE.match(s)
    if m:
        return {"kind": "regular_price", "label": "regular price", "amount": _to_float(m.group("amt"))}
    m = SAVINGS_LABEL_RE.match(s)
    if not m:
        return None
    rest = s[m.end():]
    amt = None
    am = None
    for am in _AMOUNT_RE.finditer(rest):
        pass
    if am:
        val = _to_float(am.group("amt"))
        if val is not None:
            amt = -abs(val)          # savings rows are always a reduction
    return {"kind": "savings", "label": m.group("label").strip().lower(), "amount": amt}


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEES, BOTTLE DEPOSITS, REFUNDS (keep as items; tag; allow negative)
#    Sources: Wikipedia CRV, beancount.io bottle-bill guide, Daily Dot (TJ's bag
#    fee), AWS Textract Leergut thread (negative-price lines get dropped).
# ─────────────────────────────────────────────────────────────────────────────
FEE_LINE_RE = re.compile(
    r'^\s*[*#]?\s*'
    r'(?:(?:ca|ny|ct|ia|me|ma|mi|vt|hi|or)\s+)?'
    r'(?P<kind>'
    r'crv|redemp(?:tion)?\s*val(?:ue)?|(?:bottle|container|beverage)\s*dep(?:osit)?|deposit|'
    r'bag\s*(?:fee|charge)|carrier\s*bag|bag\s*for\s*life|(?:paper|plastic|checkout|reusable)\s*bag|'
    r'env(?:ironmental)?\s*fee|handling\s*fee|delivery\s*fee|'
    r'pfand|einwegpfand|mehrwegpfand|leergut(?:r[üu]ckgabe)?|z[aá]loha|'
    r'obd|meb'
    r')\b',
    re.IGNORECASE,
)
DEPOSIT_REFUND_HINT = re.compile(r'\b(?:refund|return(?:ed)?|leergut|r[üu]ckgabe|redeem(?:ed)?)\b', re.IGNORECASE)
EQUALS_MULT_RE = re.compile(
    r'(?P<q>\d+)\s*[xX×@]\s*[\$€£]?\s*(?P<u>\d*[.,]\d{2})\s*=\s*[\$€£]?\s*(?P<t>\d+[.,]\d{2})')


def classify_fee_line(line: str) -> Optional[Dict[str, Any]]:
    """Return {'kind': 'deposit'|'fee'|'deposit_refund', 'quantity', 'unit_price'} or None.
    Callers should keep the line as an item and allow a negative total."""
    s = line.strip()
    m = FEE_LINE_RE.match(s)
    if not m:
        return None
    raw = m.group('kind').lower()
    is_refund = bool(DEPOSIT_REFUND_HINT.search(s)) or bool(re.search(r'\d[.,]\d{2}\s*-\s*$|-\s*[\$€£]?\d', s))
    if re.search(r'bag|env|handling|delivery', raw):
        kind = 'fee'
    else:
        kind = 'deposit_refund' if is_refund else 'deposit'
    out: Dict[str, Any] = {"kind": kind, "quantity": 1.0, "unit_price": None}
    em = EQUALS_MULT_RE.search(s)
    if em:
        out["quantity"] = float(em.group('q'))
        out["unit_price"] = _to_float(em.group('u'))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. TAX / TOTAL CAPTURE (stop throwing these away - use them to reconcile)
# ─────────────────────────────────────────────────────────────────────────────
TAX_LABEL_RE = re.compile(
    r'^\s*(?:[*\-_#\s]*)(?P<label>(?:sales\s*|state\s*|local\s*|city\s*|service\s*)?tax(?:\s*\d)?|'
    r'vat|gst|hst|pst|qst|cgst|sgst|igst|utgst|mwst(?:\s*[ab12])?|ust|dph|부가세|세액)\b',
    re.IGNORECASE,
)
_PRICE_ANY = re.compile(r'(?<![\w.])-?\d{1,3}(?:[.,]\d{3})*[.,]\d{2}(?![\w])|(?<![\w.])-?\d+[.,]\d{2}(?![\w])')
_RATE_RE = re.compile(r'(\d{1,2}(?:[.,]\d{1,3})?)\s*%')


def parse_tax_line(line: str) -> Optional[Dict[str, Any]]:
    """'TAX 1 7.250 % 1.23' / 'MwSt A 19% 1,32' / 'CGST @ 9% 18.00' / 'Tax (9.50%) 1.90'."""
    m = TAX_LABEL_RE.match(line)
    if not m:
        return None
    rate_m = _RATE_RE.search(line)
    rate = _to_float(rate_m.group(1)) if rate_m else None
    tail = line[rate_m.end():] if rate_m else line[m.end():]
    amts = [a for a in _PRICE_ANY.finditer(tail)]
    amount = _to_float(amts[-1].group(0)) if amts else None
    return {"kind": "tax", "label": m.group('label').lower(), "rate_pct": rate, "amount": amount}


SUBTOTAL_RE = re.compile(r'^\s*[*\-_#\s]*(?:sub\s*-?\s*total|zwischensumme|mezisou[cč]et|소계)\b', re.IGNORECASE)
TOTAL_RE = re.compile(
    r'^\s*[*\-_#\s]*(?:grand\s*|order\s*|net\s*)?(?:total|summe|gesamt(?:betrag)?|celkem|합계|총액)'
    r'(?!\s*(?:savings?|items?|discount|points?|tax))\b', re.IGNORECASE)


def parse_total_line(line: str) -> Optional[Dict[str, Any]]:
    kind = 'subtotal' if SUBTOTAL_RE.match(line) else 'total' if TOTAL_RE.match(line) else None
    if not kind:
        return None
    amts = list(_PRICE_ANY.finditer(line))
    if not amts:
        # Korean / integer totals e.g. '합계 23,400'
        ints = list(re.finditer(r'\d{1,3}(?:,\d{3})+|\d{3,}', line))
        if not ints:
            return None
        return {"kind": kind, "amount": _to_float(ints[-1].group(0))}
    return {"kind": kind, "amount": _to_float(amts[-1].group(0))}


def reconcile_items(items: List[Dict[str, Any]], subtotal: Optional[float] = None,
                    total: Optional[float] = None, tax: Optional[float] = None,
                    tolerance: float = 0.05) -> Dict[str, Any]:
    """Sum item totals (incl. negative discount/deposit-refund rows) and compare.
    Prices are tax-exclusive in US receipts (compare to subtotal) and tax-inclusive
    in EU/IN/KR receipts (compare to total). Both are checked."""
    s = round(sum((i.get('total_price') or 0.0) for i in items), 2)
    res = {"items_sum": s, "ok": None, "diff_vs_subtotal": None, "diff_vs_total": None}
    if subtotal is not None:
        res["diff_vs_subtotal"] = round(s - subtotal, 2)
    if total is not None:
        cand = [round(s - total, 2)]
        if tax is not None:
            cand.append(round(s + tax - total, 2))   # US style: items + tax = total
        res["diff_vs_total"] = min(cand, key=abs)
    diffs = [d for d in (res["diff_vs_subtotal"], res["diff_vs_total"]) if d is not None]
    res["ok"] = any(abs(d) <= tolerance for d in diffs) if diffs else None
    return res


# ─────────────────────────────────────────────────────────────────────────────
# 6. TAX FLAGS: capture instead of discard
#    Seen: Walmart A,B,P,R,S + N,X,O; grocery E/A/H/F; German A=19% B=7%.
#    Flag meanings vary by chain and state - keep the raw letter.
# ─────────────────────────────────────────────────────────────────────────────
_UNIT_TOKENS = {"EA", "LB", "KG", "OZ", "GM", "PK", "CT", "ML", "L", "G", "KC", "KE"}
_TRAILING_FLAG_RE = re.compile(r'(?P<price>-?\d+[.,]\d{2})\s*(?P<flag>[A-Z]{1,2})\s*$')


def split_trailing_flag(line: str) -> Tuple[str, Optional[str]]:
    """'MILK 3.49 N' -> ('MILK 3.49', 'N'); 'BANANAS 0.79 EA' -> unchanged."""
    m = _TRAILING_FLAG_RE.search(line.rstrip())
    if not m or m.group('flag') in _UNIT_TOKENS:
        return line, None
    return line[:m.start('flag')].rstrip(), m.group('flag')


# ─────────────────────────────────────────────────────────────────────────────
# 7. FIX: NEGATIVE-PRICE DETECTION (hyphen in COCA-COLA is not a minus)
# ─────────────────────────────────────────────────────────────────────────────
def price_sign(text: str, start: int, end: int) -> int:
    """-1 only for a leading minus directly attached to the number (not preceded
    by a word char) or a trailing minus directly after it (optionally + 1 flag letter)."""
    before = text[:start].rstrip(' $€£₹₩')
    if before.endswith('-') and (len(before) == 1 or not (before[-2].isalnum())):
        return -1
    if re.match(r'\s{0,1}-\s{0,1}[A-Za-z]?(?![\w])', text[end:]):
        return -1
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# 8. FIX: PACK SIZE vs PURCHASE QUANTITY
# ─────────────────────────────────────────────────────────────────────────────
PACK_MULTI_RE = re.compile(r'\b(\d{1,3})\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*(ml|cl|l|g|kg|oz|fl\s*oz)\b', re.IGNORECASE)
PACK_COUNT_RE = re.compile(r'\b(\d{1,3})\s*(pk|pack|ct|count|개입|입)(?![\w])', re.IGNORECASE)
WEIGHT_RANGE_RE = re.compile(r'\b(\d{1,4})\s*-\s*(\d{1,4})\s*(g|kg|ml|l|oz)\b', re.IGNORECASE)   # Aldi "Milka 81-100g"


def mask_pack_sizes(text: str) -> Tuple[str, List[str]]:
    """Glue pack tokens together so quantity regexes (which need whitespace
    boundaries) ignore them: '12 X 330ML' -> '12x330ml'. Returns masked text + tokens."""
    packs: List[str] = []

    def glue(m: re.Match) -> str:
        packs.append(m.group(0))
        return re.sub(r'\s+', '', m.group(0))
    for rx in (PACK_MULTI_RE, WEIGHT_RANGE_RE, PACK_COUNT_RE):
        text = rx.sub(glue, text)
    return text, packs


# ─────────────────────────────────────────────────────────────────────────────
# 9. MULTI-BUY / BOGO / PER-UNIT RATES
# ─────────────────────────────────────────────────────────────────────────────
MULTIBUY_RE = re.compile(r'\b(\d{1,2})\s*(?:for|/)\s*[\$€£]\s*(\d+(?:[.,]\d{2})?)\b', re.IGNORECASE)
BOGO_RE = re.compile(r'\b(?:bogo|buy\s*\d+\s*get\s*\d+(?:\s*(?:free|\d+\s*%\s*off))?)\b', re.IGNORECASE)
PER_UNIT_RATE_RE = re.compile(r'[\$€£₹]?\s*(\d+[.,]\d{1,3})\s*/\s*(lb|lbs|kg|g|100\s*g|oz|ea|each|l|ml)\b', re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# 10. INDIAN DIGIT GROUPING + COLUMN HEADERS
# ─────────────────────────────────────────────────────────────────────────────
INDIAN_GROUPED_NUM_RE = re.compile(r'(?<![\d,])\d{1,2}(?:,\d{2})+,\d{3}(?:\.\d{1,2})?(?![\d,])')   # 1,23,456.00


def parse_grouped_number(txt: str) -> Optional[float]:
    t = txt.replace(',', '')
    try:
        return float(t)
    except ValueError:
        return None


_HEADER_WORDS = re.compile(
    r'\b(?:qty|quantity|rate|mrp|amount|amt|price|hsn|disc(?:ount)?|total|unit|uom|description|desc|item|product|particulars|'
    r'cena|množství|mnozstvi|menge|preis|단가|수량|금액|상품명)\b', re.IGNORECASE)


def is_column_header_line(line: str) -> bool:
    """>=3 column words and no price-looking number (e.g. 'Item  Qty  Rate  Amount')."""
    s = line.strip()
    hits = len(_HEADER_WORDS.findall(s))
    return hits >= 3 and not _PRICE_ANY.search(s)


# ─────────────────────────────────────────────────────────────────────────────
# 11. OCR DIGIT REPAIR + UNICODE-SAFE CLEANING + LETTER TEST
# ─────────────────────────────────────────────────────────────────────────────
_PRICEISH_TOKEN = re.compile(r'(?<![A-Za-z])[\dOolI]{1,4}[.,][\dOolI]{2}(?![A-Za-z])')
_DIGIT_MAP = str.maketrans({'O': '0', 'o': '0', 'l': '1', 'I': '1'})


def repair_ocr_digits(text: str) -> str:
    """'3.9O' -> '3.90', 'l.99' -> '1.99'. Only touches tokens shaped like a price
    that contain at least one real digit."""
    def fix(m: re.Match) -> str:
        tok = m.group(0)
        return tok.translate(_DIGIT_MAP) if re.search(r'\d', tok) else tok
    return _PRICEISH_TOKEN.sub(fix, text)


def has_letters(s: str) -> bool:
    """Replacement for re.search(r'[A-Za-z가-힣]', s) - any script."""
    return any(ch.isalpha() for ch in s)


def clean_symbols_unicode(s: str) -> str:
    """Replacement for step 8 of clean_item_name_for_bert. Keeps letters/digits in
    any script (JALAPEÑO, ČOKOLÁDA, Hangul, Devanagari incl. combining marks)."""
    out = []
    for ch in s:
        if ch.isalnum() or ch in '. ' or unicodedata.category(ch).startswith('M'):
            out.append(ch)
        else:
            out.append(' ')
    return re.sub(r'\s{2,}', ' ', ''.join(out)).strip()
