#!/usr/bin/env python3
"""
Receipt Structurer (Spatial & Regex Pipeline)
=============================================
Converts raw OCR outputs (words + bounding boxes) or raw text lines into
structured receipt items:
  - item_name: Clean product description (preserves spec like 500g, 1L, 5입)
  - quantity: Purchase quantity (e.g., 2.0)
  - unit: Purchase unit ('pcs', 'kg', '개', '팩', '캔', etc.)
  - unit_price: Price per unit
  - total_price: Line total
  - bbox: Unified bounding box [x1, y1, x2, y2]
"""

import os
import re
from typing import List, Dict, Optional, Tuple, Any

from receipt_patterns_additions import (
    ADDED_NOISE_KEYWORDS,
    ADDED_NOISE_REGEX,
    ADDED_PRODUCT_PROTECTION,
    FIXED_NOISE_REPLACEMENTS,
    is_added_noise,
    parse_savings_modifier,
    classify_fee_line,
    parse_tax_line,
    parse_total_line,
    reconcile_items,
    split_trailing_flag,
    price_sign,
    mask_pack_sizes,
    is_column_header_line,
    repair_ocr_digits,
    has_letters,
    clean_symbols_unicode,
    _PRICE_ANY,
    _to_float,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. NOISE & METADATA LINE FILTER (Total, Subtotal, Tax, Payment headers)
# ─────────────────────────────────────────────────────────────────────────────
NOISE_KEYWORDS = [
    # Subtotal, Total, Balance (Context-Aware)
    r'sub\s*t[0o]tal', r'grand\s*t[0o]tal', r'net\s*t[0o]tal', r'amount\s*due',
    r'\b(?:balance\s*due|new\s*balance|acct\s*balance|ending\s*balance|bal\s*due|balance)\b',
    r'^\s*(?:[\*\-_#\s]*)(?:sub\s*|grand\s*|net\s*)?t[0o]tal(?:\s*due|\s*amount|\s*sales)?\b',
    r'\bt[0o]tal\s+(?:amount|due|tax|items?|savings?|price|bill|sale|charge|paid|purchase|tend|tendered|sales)\b',
    r'\b(?:purchase|sale)\s*t[0o]tal\b',
    r'\bt[0o]tal\s*purchase\b',
    r'\bt[0o]tal\s*due\b',
    r'^\s*(?:order\s*)?total\s*[:=]?\s*[\d\$\.\,]+\s*$',
    r'^\s*amount\s*[:=]?\s*[\d\$\.\,]+\s*$',
    r'^\s*sub\s*[/]?\s*tt[1l]\b',

    # Tips, Gratuity & Service charges (footers and calculation rows)
    r'\b(?:suggested\s*(?:tip|gratuity)|tip\s*suggestions?|tip\s*guide)\b',
    r'\b(?:gratuity|tip)\s*(?:amount|not\s*included|included|added)?\b',
    r'\b(?:\d{1,2}(?:\.\d+)?%)\s*(?:of\s*sale|gratuity|tip)\b',
    r'\bautomatic\s*gratuity\b',
    r'\b(?:service\s*charge|service\s*fees?|surcharge)\b',

    # Taxes & Tax Summaries
    r'\btax\d*\b', r'\bvat\b', r'\bgst\b', r'\bhst\b', r'\bqst\b', r'\bcgst\b', r'\bsgst\b',
    r'\bigst\b', r'sales\s*tax', r'service\s*tax',
    r'\b(?:ca|ny|tx|fl|wa|il|pa|oh|ga|nc|mi)?\s*tax\s*\d*(?:\.\d+)?%?\b',
    r'\bstate\s*tax\b', r'\blocal\s*tax\b', r'\bsalestx\b',
    r'\btax\s*\d*\s*[\d.]*%\b',
    r'\btaxable(?:\s*amount)?\b',
    r'\btax\s*rate\b',

    # Financial line discounts & savings (context-aware: avoids deleting "SAVINGS COFFEE")
    r'\b(?:total\s*discount|discount\s*applied|store\s*discount|coupon\s*discount)\b',
    r'^\s*discount\s*[:#\s]?\s*[\d\$\.\-]+\s*$',
    r'\b(?:total\s*savings?|your\s*savings?|customer\s*savings?|instant\s*savings?|member\s*savings?)\b',
    r'\b(?:you\s*saved(?:\s*today)?|loyalty\s*(?:card)?(?:\s*savings?)?|card\s*savings)\b',
    r'^\s*savings?\s*[\d\$\.\-]+\s*$',
    r'\b(?:promo\s*discount|promo\s*savings?|promo\s*code|promo\s*applied)\b',
    r'^\s*coupon\b', r'\bround\s*off\b',
    r'\bearn\s*[\d\w\s]*points\b',

    # Payments & Tenders (context-aware: avoids deleting "BIRTHDAY CARD", "CASH & CARRY")
    r'\b(?:cash\s*(?:tend|tender|tendered|given|paid|back|received)|cash\s*payment|cash\s*sale|cashless)\b',
    r'^\s*cash(?:\s*tendered)?\s*[:#\s]?\s*[\d\$\.\-]+\s*$',
    r'\b(?:change\s*due|change\s*given|change\s*returned)\b',
    r'^\s*change(?:\s*\(cash\))?\s*[:#\s]?\s*[\d\$\.\-]+\s*$',
    r'\b(?:credit\s*card|debit\s*card|cred[1i]t\s*card|master\s*card|visa\s*card|amex\s*card|card\s*ending|card\s*auth|card\s*appr)\b',
    r'\b(?:wells\s*fargo|chase|citi|bank\s*of\s*america|atm\s*charge|atm\b)\b',
    r'^\s*(?:visa|mastercard|amex|discover|debit|credit|payment)\s*[:#\s]?\s*[\d\$\.\,]+\s*$',
    r'\bdebit\s*tend\b', r'\bcash\s*tend\b', r'\beft\s*debit\b', r'\bpayment\s*declined\b',
    r'\b(?:visa|mastercard|amex|discover)\s+(?:credit|debit|ending|card|auth|balance|tend|payment|chip)\b',
    r'\b(?:upi|gpay|paytm)\s*(?:pay|payment|id|txn)\b',
    r'^\s*[*\-\s]*(?:amt\s*|cash\s*|debit\s*|card\s*)?tender(?:ed)?\s*[:=]?\s*[\$€£]?\s*\d',
    r'^\s*[*\-\s]*tend\s*[:=]?\s*[\$€£]?\s*\d',
    r'\bba[li1]ance\b', r'\bbalance\s*to\s*pay\b',
    r'\breceived\s*(?:money|cash|amount)\b',
    r'\breturn\s*value\b',
    r'^\s*(?:table|guest|server|station|order|chk|check)\s*[:#]\s*[\w/.\-]+',

    # Administrative & Column Headers
    r'\b(?:item\s*code|b\.code|qty\s*price|qty\s*uom|u\/price|u\.price|desc\s*qty)\b',
    r'\bthank\s*you\b', r'\bvisit\s*(?:us\s*)?again\b', r'\bcustomer\s*copy\b', r'\bmerchant\s*copy\b',
    r'^\s*receipt\b', r'\btax\s*invoice\b', r'\bcash\s*memo\b', r'\btoken\s*no\b', r'\border\s*no\b',
    r'\bclosed\s*bill\b', r'\bcheck\s*no\b', r'\bguest\s*check\b',
    r'\bgstin\b', r'\bfssai\b', r'\b(?:phone|tel)\s*[:.]\s*[\d(+]',
    r'\b(?:cashier|cshr|reg(?:ister)?)\s*[:#]\s*\w+',
    r'\bmana[g9]er\b', r'\bst#', r'\b[0o]p#', r'\bte#', r'\btr#', r'\btc[#\+]',
    r'\bref\s*#', r'\btrans\s*#', r'\bterminal\b', r'\bterm#', r'\btermid\b', r'\bcatid\b',
    r'\btid[:\s]', r'\bmerch\s*id\b',
    r'\bpos\s*title\b', r'\bkasir\s*rcpt\b',
    r'\bt[o0]ta[l1i]\s*(?:due|purchase|amount|tax)?\b',
    r'\baccount\b', r'\bacct\b', r'\bappr\s*c[0o]de\b', r'\bnetw[0o]rk\b',
    r'\bitems?\s*(?:sold|count)\b', r'\blayaway\b', r'\bcome\s*back\s*soon\b',
    r'\bsave\s*money\b', r'\blive\s*better\b', r'\bwe\s*sell\s*less\b',
    
    # Voids
    r'^\s*(?:\*+\s*)?void(?:ed)?\b', r'xvoidedentry', r'gst/hst', r'\bqst\d+',

    # Central European / Czech financial headers
    r'\bcelkem\b', r'\bmezisou[cč]et\b', r'\bplatebn[ií]\s*karta\b',
    r'\bvr[aá]ceno\b', r'\bdph\s*rekapitulace\b', r'\bsazba\s*dph\b',
    r'\bpro[cč]\s*utr[aá]c[ií]te\b', r'\bplatba\b', r'\bhotovost\b',
    
    # Korean metadata
    r'합계', r'소계', r'총액', r'받을\s*금액', r'받은\s*금액', r'거스름돈',
    r'과세', r'면세', r'부가세', r'세액', r'승인', r'가맹점',
    r'사업자', r'대표자', r'전화', r'매출', r'영수증'
]

# Protected legitimate products that might contain words like 'total', 'card', 'balance'
PRODUCT_PROTECTION_REGEX = re.compile(
    r'\b(?:'
    r'carb\s*balance|'
    r'colgate\s*total|fage\s*total|total\s*(?:cereal|raisin|bran|clusters|whole|tea|wellness)|'
    r'7\s*up|v8|a1\s*steak|'
    r'(?:birthday|greeting|index|playing)\s*cards?|'
    r'cash\s*(&|and)\s*carry|'
    r'2%\s*(?:reduced\s*fat\s*)?milk|1%\s*milk|100%\s*(?:orange\s*)?juice|'
    r'snap\s*peas?|sugar\s*snap|snap\s*pea|table\s*salt|tablets?|regular|reggiano|'
    r'chicken\s*tenders?|tender\s*loin|phone\s*charger|chk?n\s*(?:breast|thigh)|'
    r'granola\s*bar|protein\s*bar|soap\s*bar|nectar|agave|clubcard\s*price|'
    r'kirkland\s*signature|signature\s*(?:select|farms|kitchens)|choc(?:olate)?\s*chip|'
    r'chip\s*ahoy|potato\s*chips?|tortilla\s*chips?|cashew|cashier\s*pick'
    r')\b',
    re.IGNORECASE
)

NOISE_REGEX = re.compile(r'(?:' + '|'.join(NOISE_KEYWORDS) + r')', re.IGNORECASE)
SEPARATOR_REGEX = re.compile(r'^[\s\-_=*#~.:;/]{3,}$')

# Dates & Times: "2024-05-12", "31/03/2022", "14:32:01", "2026.04.14"
DATE_TIME_REGEX = re.compile(
    r'\b(?:\d{2,4}[-/.포]\d{1,2}[-/.포]\d{1,4}|\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?)\b'
)

# Phone numbers (e.g., 043-648-6522, 02-310-5038, +91-9876543210)
PHONE_REGEX = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b')

# Masked Card/Account numbers (e.g. 5387-33******-617*, 400**60)
MASKED_NUM_REGEX = re.compile(r'\b\d+[\*]+[\d\*]*\b|\b\d{4}[- ]\d{2,4}[\*]+[- \d\*]*\b')

# Street addresses: "8885 N FLORIDA AVE", "2457 Maplewood Lane", "123 Main St", "456 Oak Blvd"
STREET_ADDRESS_REGEX = re.compile(
    r'^\s*\d{1,6}\s+(?:[A-Za-z0-9#\-\.\s]+)\b(?:avenue|ave|street|st|road|rd|lane|ln|drive|dr|court|ct|boulevard|blvd|way|highway|hwy|parkway|pkwy|circle|cir|terrace|ter|place|pl)\b',
    re.IGNORECASE
)

# City, State, ZIP: "TAMPA FL 33604", "Brooksville OH 43101", "SAN JOSE CA 95112"
CITY_STATE_ZIP_REGEX = re.compile(
    r'\b[A-Za-z\s]+,?\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?\b'
)

# ─────────────────────────────────────────────────────────────────────────────
# RETAIL PRODUCT CODES (Walmart, Target, Costco, Supermarket inventory)
# ─────────────────────────────────────────────────────────────────────────────
# 1. Standalone Barcodes / UPC numbers with optional attached flag: e.g. "007225003712F", "007874235186"
UPC_BARCODE_REGEX = re.compile(r'(?:^|\s)\b\d{6,14}[A-Za-z]?\b(?=\s|$)', re.IGNORECASE)

# 2. Explicit SKU / PLU / Item / Dept / Barcode tags: e.g. "PLU 4011", "SKU# 98124", "DPT 92", "바코드 880104"
# Requires explicit colon/hash or 3+ digits to avoid stripping "ITEM -1.00" or item descriptions
RETAIL_CODE_TAGS_REGEX = re.compile(
    r'\b(?:plu|sku|prod|upc|code|art|pos|dpt|dept|tc|ref|바코드|품번|상품코드|코드)\s*[:#\-\+]?\s*\d+[0-9A-Za-z\-]*\b|\bitem\s*[:#\+]\s*\d+\b|\bitem\s+\d{3,}\b|\b(?:st|op|te|tr)\s*[:#\-\+]\s*\d+\b',
    re.IGNORECASE
)

# 3. Supermarket alphanumeric inventory noise: e.g. "*+8SP300g", "#004900" (must contain at least 1 digit to avoid deleting brand names like #Lucozade)
INVENTORY_TAG_REGEX = re.compile(r'(?:^|\s)[\*#+]{1,3}(?=[0-9A-Za-z]*\d)[0-9A-Z]{3,10}(?=\s|$)', re.IGNORECASE)

# 4. Trailing Tax / Food-Stamp flags: single letter (D, K, F, T, N, X, Y, A, B, O, R), multi-letter (TFA, TF, FA, FS, NT, TAX, GPMRJ, MRJ), void hyphen (-), or space-separated 0
TAX_FLAG_REGEX = re.compile(r'(?<=[\d\$원])\s*(?:GPMRJ|PMRJ|GMRJ|MRJ|GPMR|PMR|GMR|(?:\d+)?PRQ|RQ|TFA|TF|FA|FS|NT|TAX|GST|HST|PST|VAT|[FTNXDHKYABOR]|\-|\s+0)\s*$', re.IGNORECASE)

# 5. Internal POS Transaction Codes (safe to strip early without destroying tax/discount keywords):
POS_INTERNAL_CODE_REGEX = re.compile(r'\b(?:GPMRJ|PMRJ|GMRJ|MRJ|GPMR|PMR|GMR|(?:\d+)?PRQ|RQ|TFA|TF|FA|FS|NT)\b', re.IGNORECASE)

# 5B. POS Tax & Weight Descriptors in Item Names: e.g. TFA, FA, NET, GROSS, TARE, WT, GPMRJ, MRJ
POS_TAX_AND_WEIGHT_REGEX = re.compile(r'\b(?:GPMRJ|PMRJ|GMRJ|MRJ|GPMR|PMR|GMR|(?:\d+)?PRQ|RQ|TFA|TF|FA|FS|NT|TAX|GST|HST|PST|VAT|NET|GROSS|TARE|WT)\b', re.IGNORECASE)


# 6. Retail Department Section Headers: e.g. "21-GROCERY", "22-DAIRY", "27-PRODUCE", "31-MEATS"
DEPT_HEADER_REGEX = re.compile(
    r'^\s*\d{1,3}\s*[\-]\s*(?:grocery|dairy|frozen|produce|meats?|deli|bakery|seafood|home(?:\s*health)?|beauty(?:\s*products)?|apparel|general\s*merch(?:andise)?|pharmacy)\b',
    re.IGNORECASE
)

# 7. Promotional pricing tiers / limit rows: e.g. "$2.47 lmt 4, $3.77 ea", "$1.97 ea or 2/$3.00"
PROMO_TIER_REGEX = re.compile(
    r'(?:'
        r'[\$€£¥₩]?\d+(?:\.\d{2})?\s*(?:lmt|limit)\s*\d+[^0-9\$]*(?:[\$€£¥₩]?\d+(?:\.\d{2})?\s*(?:ea|each))?'
        r'|'
        r'[\$€£¥₩]?\d+(?:\.\d{2})?\s*(?:ea|each)\s+or\s+\d+/\s*[\$€£¥₩]?\d+(?:\.\d{2})?'
    r')',
    re.IGNORECASE
)

# 8. Multiplier with attached unit: "2 @ $2.47 ea", "3 @ $3.78", "Qty 2 @ $2.02 each"
MULTIPLIER_RATE_REGEX = re.compile(
    r'(?:\s|^)(?:qty\s*)?(\d+(?:\.\d+)?)\s*[@xX×*]\s*[\$€£¥₩]?\s*(\d+(?:[.,]\d{2})?)\s*(?:each|ea|pcs|units?|nos|/\s*[A-Za-z가-힣]+)?(?=\s|$)',
    re.IGNORECASE
)

# 9. Negative promotional discount lines: "PRIMO SLICED OFFER -1.00"
DISCOUNT_LINE_REGEX = re.compile(
    r'\b(?:offer|discount|coupon|saving|savings|rebate|loyalty|markdown)\b.*?\-[\$€£¥₩]?\s*(\d+[.,]\d{2})',
    re.IGNORECASE
)

# 10. Leading item count before barcode: "(3)05500055117" or "(2)4539"
PREFIX_COUNT_BARCODE = re.compile(r'^\s*\((\d+)\)\s*(?=\d{4,14}\b)')

# 11. Serial Number / Line Index Prefixes:
# Matches: "1. ", "01 ", "SL.NO 1", "SL# 1", "S.NO. 1", "SR 1", "ITEM 1", "#1", "(1)", "1- "
SERIAL_PREFIX_RE = re.compile(
    r'^\s*(?:'
        r'(?:(?:sl|sr|item|line|pos|s)\.?\s*(?:no|num|#)?\.?\s*\d{1,3}\b(?!\.\d))'
        r'|'
        r'(?:#\s*\d{1,3}\b(?!\.\d))'
        r'|'
        r'(?:\(?\d{1,3}\)?\s*[.)\-:]\s+)'
        r'|'
        r'(?:\(\d{1,3}\)\s+)'
        r'|'
        r'(?:0\d\s+)'
    r')\s*',
    re.IGNORECASE
)

# Child line starting with a serial index followed by an actual quantity expression or rate:
# Matches e.g. "1 2 @ 2.50", "1 2 x 2.50", "1 0.778 kg", "1 2 PCS", "1 2 PKT", "1 2 2.50 5.00", "1 2 5.00"
CHILD_SERIAL_THEN_QTY_RE = re.compile(
    r'^\s*(\d{1,3})\s+(?='
        r'(?:[1-9]\d?\s*(?:[@xX×*]|ea\b|each\b|pcs\b|pc\b|pkts?\b|packets?\b|units?\b|nos\b|[\$€£¥₩]|단가))'
        r'|'
        r'(?:\d+\.\d{2,3}\s*(?:kg|kgs|lb|lbs|g|gm))'
        r'|'
        r'(?:[1-9]\d?\s+[\$€£¥₩]?\d+[.,]\d{2})'
    r')',
    re.IGNORECASE
)


def strip_child_serial(line: str) -> str:
    """Strips leading item serial number from a child subrow so it is not confused as quantity."""
    s = line.strip()
    m1 = SERIAL_PREFIX_RE.match(s)
    if m1:
        return s[m1.end():].strip()
    m2 = CHILD_SERIAL_THEN_QTY_RE.match(s)
    if m2:
        return s[m2.end():].strip()
    return s


# ─────────────────────────────────────────────────────────────────────────────
# 2. PRICE & QUANTITY REGEX DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
# Precise price matching:
# - Prioritizes decimals so $2.18D always captures 2.18, never falling back to 2
# - Captures:
#   1. Explicit currency prefixed numbers: $2.18, $1.44, $5, ₩5000, ₹120
#   2. Thousands-separated currency: 4,850 or 1,234.50
#   3. Standard decimal prices: 2.88, 3.84, 10.48
#   4. Korean/Indian currency suffixed: 5000원, 150rs
#   5. Attached tax flags: 2.18D, 2.88N, 2.78X, 0.84X, 1.88 0, 3.98R
PRICE_TOKEN_REGEX = re.compile(
    r'(?:'
        r'([\-]?\s*[₹$€£¥₩]|(?<![A-Za-z])(?:usd|cad|eur|gbp|aud|nzd|inr|rs|k[cč]|ke|czk)\.?)\s*(\d+[.,，]\d{2}|\.\d{2}|\d{1,3}(?:[.,\s]\d{3})+(?:[.,，]\d{1,2})?|\d+)'
        r'|'
        r'(?<![A-Za-z0-9])([0-9]{1,3}(?:\.[0-9]{3})+,[0-9]{2})'
        r'|'
        r'(?<![A-Za-z0-9])(\d{1,3}(?:,\d{3})+(?:[.,，]\d{1,2})?)'
        r'|'
        r'(?<![A-Za-z0-9])([\-]?\s*(?:\d+[.,，]\d{2}|\.\d{2}))'
        r'|'
        r'(?<![A-Za-z0-9])(\d+)\s*(원|rs\.?|(?<![A-Za-z])(?:k[cč]|ke|czk))'
    r')'
    r'(?:\s*([FTNXDHKYABOR]|(?:\s+0(?![.,\d]))))?'
    r'(?:\s*[\-])?'
    r'\s*(원|rs\.?|(?<![A-Za-z])(?:k[cč]|ke|czk))?'
    r'(?![A-Za-z0-9])',
    re.IGNORECASE
)

RATE_TOKEN_REGEX = re.compile(
    r'(?:@|단가)\s*[₹$€£¥₩]?(?:k[cč]|ke|czk)?\s*(\d+(?:,\d{3})*(?:[.,，]\d{1,2})?)\s*(?:/\s*[A-Za-z가-힣]+)?',
    re.IGNORECASE
)

# Multipliers: "2 x", "x 2", "2*", "* 3", "3xOrganicBananas"
QTY_PREFIX_MULTIPLIER_RE = re.compile(r'^\s*(\d+(?:\.\d+)?)\s*[xX×*](?:\s+|(?![\d\s]))', re.IGNORECASE)
QTY_INFIX_MULTIPLIER_RE  = re.compile(r'(?:\s|^)[xX×*]\s*([1-9]\d?)(?!\.\d)(?=\s|$)', re.IGNORECASE)
QTY_SUFFIX_MULTIPLIER_RE = re.compile(r'(?:\s|^)(\d+(?:\.\d+)?)\s*[xX×*](?=\s|$)', re.IGNORECASE)

# Purchase Count Units (Korean + English): "2개", "2팩", "2캔", "3pcs", "2ea", "2 nos", "1 pkt"
PURCHASE_UNIT_REGEX = re.compile(
    r'(?:\s|^)(\d+(?:\.\d+)?)\s*(pcs|pc|ea|nos|no|pkts?|packets?|units?|개|병|봉|캔|팩|박스|통|줄)(?=\s|$|[.,;:])',
    re.IGNORECASE
)

# Weighed Produce: "1.250 kg", "0.500kg", "1.2 lb"
WEIGHED_UNIT_REGEX = re.compile(
    r'(?:\s|^)(\d+\.\d{2,3})\s*(kg|kgs|g|gm|lb|lbs)(?=\s|$)',
    re.IGNORECASE
)


def parse_price_match(m: re.Match) -> Optional[float]:
    """Extracts numeric float value from structured price regex match, supporting multi-locale decimal/thousands conventions."""
    raw = m.group(2) or m.group(3) or m.group(4) or m.group(5) or m.group(6)
    if not raw:
        return None
    txt = raw.replace('，', ',').replace(' ', '').strip()

    # 1. European dot thousands + comma decimal: e.g. "1.234,50" -> "1234.50"
    if re.search(r'^\d{1,3}(?:\.\d{3})+,\d{2}$', txt):
        txt = txt.replace('.', '').replace(',', '.')
    # 2. Comma decimal: e.g. "4,99" or "24,06" -> "4.99"
    elif re.search(r'^\d+,\d{2}$', txt):
        txt = txt.replace(',', '.')
    # 3. Standard US/UK/Korean comma thousands: e.g. "1,234.50" or "4,850" -> "1234.50"
    else:
        txt = txt.replace(',', '')

    try:
        val = float(txt)
        if 0.01 <= abs(val) <= 50_000_000:
            return val
    except ValueError:
        pass
    return None


def parse_float_price(text: str) -> Optional[float]:
    """Fallback converter for price token strings into clean float, supporting international comma/dot separators."""
    if not text:
        return None
    cleaned = text.replace('，', ',').replace(' ', '').replace('원', '')
    cleaned = re.sub(r'^[₹$€£¥₩]|rs\.?|inr|usd|cad|eur|gbp|aud|nzd|k[cč]|ke|czk', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'(?:[FTNXABOOHY]|\s+0)$', '', cleaned, flags=re.IGNORECASE)

    # 1. European dot thousands + comma decimal: e.g. "1.234,50" -> "1234.50"
    if re.search(r'^\d{1,3}(?:\.\d{3})+,\d{2}$', cleaned):
        cleaned = cleaned.replace('.', '').replace(',', '.')
    # 2. Comma decimal: e.g. "4,99" -> "4.99"
    elif re.search(r'^\d+,\d{2}$', cleaned):
        cleaned = cleaned.replace(',', '.')
    else:
        cleaned = cleaned.replace(',', '')

    try:
        val = float(cleaned)
        if 0.01 <= abs(val) <= 50_000_000:
            return val
    except ValueError:
        pass
    return None


def is_noise_or_header_line(line: str) -> bool:
    """Returns True if the line contains header, subtotal, tax, address, or separator info."""
    s = line.strip()
    if not s or len(s) < 2:
        return True
    if SEPARATOR_REGEX.match(s):
        return True
    if DEPT_HEADER_REGEX.match(s):
        return True
    if PRODUCT_PROTECTION_REGEX.search(s):
        return False
    if classify_fee_line(s):
        return False
    if is_column_header_line(s):
        return True
    if is_added_noise(s):
        return True

    # If line is a negative price item or discount (e.g. "ITEM -1.00" or "PRIMO SLICED OFFER -1.00"), do not reject
    if re.search(r'[\-]\s*[\$€£¥₩]?\s*\d+[.,]\d{2}\s*$', s):
        # Only reject if it explicitly matches pure refund/change/cash/discount keywords without item nouns
        if re.search(r'^\s*(?:change|cash|balance|subtotal|total|tendered)\b', s, re.IGNORECASE):
            return True
        return False

    # If the line contains an item followed by an inline tax descriptor (e.g. "MILK 2.50 TAX 0.20"), do not reject if item precedes it
    if re.search(r'^[A-Za-z가-힣]{2,}.*?\btax\b\s*[\$€£¥₩]?\s*\d+[.,]\d{2}', s, re.IGNORECASE):
        # Don't treat as noise header if it starts with a product name (not 'Sales Tax' or 'State Tax')
        if not re.search(r'^\s*(?:sales|state|local|city|vat|gst|hst|qst|ca|ny|tx)?\s*tax\b', s, re.IGNORECASE):
            return False

    if NOISE_REGEX.search(s):
        return True
    # If line is a valid promotional discount (e.g. "PRIMO SLICED OFFER -1.00"), do not reject
    if DISCOUNT_LINE_REGEX.search(s):
        return False
    # Filter out voided negative price lines (e.g. "$4.27-" or "$1.97-D")
    if re.search(r'\d+\.\d{2}\s*[\-]', s):
        return True
    if STREET_ADDRESS_REGEX.search(s):
        return True
    if CITY_STATE_ZIP_REGEX.search(s):
        return True
    # If line has no alphabetic or Hangul characters at all, reject
    if not re.search(r'[A-Za-z가-힣]', s):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 2B. GROCERY COMPOUND DICTIONARY & CONSERVATIVE OCR NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────
GROCERY_COMPOUNDS = [
    ("organicswholemilk", "organics whole milk"),
    ("organicwholemilk", "organic whole milk"),
    ("bronzecutrigatoni", "bronze cut rigatoni"),
    ("tillamooksharpcheddar", "Tillamook sharp cheddar"),
    ("sharpcheddar", "sharp cheddar"),
    ("organicbananas", "organic bananas"),
    ("wholemilk", "whole milk"),
    ("sourdoughboule", "sourdough boule"),
    ("sourdoughloaf", "sourdough loaf"),
    ("groundcoffee", "ground coffee"),
    ("browneggs", "brown eggs"),
    ("bronzecut", "bronze cut"),
    ("pulpfree", "pulp free"),
    ("familypk", "family pk"),
    ("wholegrain", "whole grain"),
    ("peanutbutter", "peanut butter"),
    ("icecream", "ice cream"),
    ("sourcream", "sour cream"),
    ("creamcheese", "cream cheese"),
    ("orangejuice", "orange juice"),
    ("applecider", "apple cider"),
]


def normalize_ocr_conservative(text: str) -> str:
    """
    Stage A: Conservative OCR normalization.
    Normalizes full-width punctuation, repairs safe character confusions (e.g. 1b -> lb in weight context,
    0rganic -> Organic), splits glued compound grocery words, and separates numbers from units.
    Does NOT destructively delete numbers or prices.
    """
    # 1. Punctuation normalization
    s = text.translate(str.maketrans({
        "，": ",",
        "．": ".",
        "：": ":",
        "。": ".",
    }))

    # 1B. OCR Digit Repair on price-shaped tokens (e.g. 3.9O -> 3.90, l.99 -> 1.99)
    s = repair_ocr_digits(s)

    # 2. Safe character repairs in weight / quantity contexts
    # Fix 1b -> lb after numbers or before rate/price
    s = re.sub(r'(\d+(?:\.\d+)?)\s*1b\b', r'\1 lb', s, flags=re.IGNORECASE)
    s = re.sub(r'\b1b\b(?=\s*[@\d\$]|$)', 'lb', s, flags=re.IGNORECASE)

    # 3. OCR 0/O repair on known prefixes (e.g. 0rganic -> Organic, 1.380rganic -> 1.38 Organic)
    s = re.sub(r'(\d+(?:\.\d+)?)\s*0rganic', r'\1 Organic', s, flags=re.IGNORECASE)
    s = re.sub(r'\b0rganic\b', 'Organic', s, flags=re.IGNORECASE)
    s = re.sub(r'(\d+(?:\.\d+)?)\s*0(?=[A-Z][a-z])', r'\1 O ', s)
    s = re.sub(r'\b0(?=[A-Z][a-z])', 'O ', s)

    # 4. Leading 1 for common produce/food words (1ime -> lime, 1emon -> lemon)
    s = re.sub(r'\b1(ime|emon|ettuce|amb|obster|entils|ager)\b', r'l\1', s, flags=re.IGNORECASE)

    # 5B. European Tax Flag Repair on Decimal Prices:
    # OCR commonly confuses trailing tax class 'B' with '8' (e.g. "19.908" -> "19.90 B", "26,908" -> "26.90 B")
    # or 'A' with '4' (e.g. "3,804" -> "3.80 A"). Only trigger when NOT followed by a unit of measure or rate symbol,
    # and strictly at the end of line, before currency, or before large column gaps.
    neg_unit = r'(?!\s*(?:kg|k9|ko|g|gm|lb|lbs|oz|ea|pcs|nos|x|@|/))'
    s = re.sub(r'(\d+[.,]\d{2})[8B]' + neg_unit + r'(?=\s*(?:[Kk][cč]|CZK)?\s*$|\s{2,})', r'\1 B', s)
    s = re.sub(r'(\d+[.,]\d{2})[4A]' + neg_unit + r'(?=\s*(?:[Kk][cč]|CZK)?\s*$|\s{2,})', r'\1 A', s)
    s = re.sub(r'(\d+[.,]\d{2})C' + neg_unit + r'(?=\s*(?:[Kk][cč]|CZK)?\s*$|\s{2,})', r'\1 C', s)

    # 5C. Czech Currency Normalization: "Ke1,90" -> "Kc 1,90", "Kc109,90" -> "Kc 109,90"
    s = re.sub(r'\b[Kk][ecEC]\s*(\d+[.,]\d{2})', r'Kc \1', s)
    s = re.sub(r'(\d+(?:\.\d+)?)\s*0\s+Kc\b', r'\1 @ Kc', s)

    # 6. Separate packaging units glued to words or numbers (e.g. 8oz -> 8 oz, 1gal -> 1 gal)
    s = re.sub(r'([A-Za-z]+)(\d+)\s*(gal|oz|lb|lbs|g|kg|ml|l|ct|pk)\b', r'\1 \2 \3', s, flags=re.IGNORECASE)
    s = re.sub(r'(\d+)\s*(gal|oz|lb|lbs|g|kg|ml|l|ct|pk)\b', r'\1 \2', s, flags=re.IGNORECASE)

    # 7. Normalize spaces
    return re.sub(r'\s{2,}', ' ', s).strip()


def parse_receipt_line_advanced(line: str) -> Optional[Dict[str, Any]]:
    """
    Parses a single receipt line into structured item details.
    Stage B: Structural extraction (prices, counts, weighed units, rates).
    Returns None if the line is noise, header, address, or contains no valid price.
    """
    original = line.strip()

    # 1. Check for fee or container deposit line (e.g. "CRV 0.05", "BAG FEE 0.10", "Leergutrückgabe -0,25", "CA CRV 6 x 0.05 = 0.30")
    fee_info = classify_fee_line(original)
    if fee_info:
        amts = list(_PRICE_ANY.finditer(original))
        if amts:
            tot_val = _to_float(amts[-1].group(0))
            if fee_info["kind"] == "deposit_refund" and tot_val and tot_val > 0:
                tot_val = -tot_val
            unit_p = fee_info.get("unit_price") or tot_val
            qty = fee_info.get("quantity") or 1.0
            fee_label = re.sub(r'[\d\.,\$=xX×@\-]+', ' ', original).strip()
            return {
                "item_name": fee_label if fee_label else fee_info["kind"].upper(),
                "quantity": qty,
                "quantity_explicit": qty > 1.0,
                "unit": "pcs",
                "unit_price": unit_p,
                "total_price": tot_val,
                "arithmetic_valid": True,
                "discount_amount": None,
                "tax_amount": None,
                "raw_line": original,
                "is_fee": True,
                "fee_kind": fee_info["kind"]
            }

    # 2. Check for savings modifier child row (e.g. "Card Savings 1.50-", "Prime Buy 1 Get 1 -4.25", "Digital Coupon -0.75")
    sav_info = parse_savings_modifier(original)
    if sav_info and sav_info.get("amount") is not None:
        disc_val = sav_info["amount"]
        return {
            "item_name": sav_info["label"].upper(),
            "quantity": 1.0,
            "quantity_explicit": False,
            "unit": "pcs",
            "unit_price": disc_val,
            "total_price": disc_val,
            "arithmetic_valid": True,
            "discount_amount": disc_val,
            "tax_amount": None,
            "raw_line": original,
            "is_discount": True
        }

    norm_for_noise = original.translate(str.maketrans({"，": ",", "．": ".", "：": ":", "。": "."}))
    if is_noise_or_header_line(norm_for_noise):
        return None

    # Check for promotional discount line: e.g. "PRIMO SLICED OFFER -1.00"
    m_disc = DISCOUNT_LINE_REGEX.search(original)
    if m_disc:
        disc_val = float(m_disc.group(1))
        disc_name = original[:m_disc.start()].strip(' -')
        return {
            "item_name": disc_name if disc_name else "DISCOUNT OFFER",
            "quantity": 1.0,
            "unit": "pcs",
            "unit_price": -disc_val,
            "total_price": -disc_val,
            "raw_line": original,
            "is_discount": True
        }

    # Normalize leading count barcode: e.g. "(3)05500055117" -> quantity=3
    leading_qty = None
    cleaned_orig = original
    m_lead_count = PREFIX_COUNT_BARCODE.match(cleaned_orig)
    if m_lead_count:
        leading_qty = float(m_lead_count.group(1))
        cleaned_orig = cleaned_orig[m_lead_count.end():].strip()

    # Strip promotional limit tiers: e.g. "$2.47 lmt 4, $3.77 ea"
    cleaned_orig = PROMO_TIER_REGEX.sub(" ", cleaned_orig)

    # Strip leading serial number / row index (e.g. "1.", "01", "SL# 1", "S.NO. 1", "#1", "(1)")
    cleaned_orig = SERIAL_PREFIX_RE.sub(" ", cleaned_orig).strip()

    # Clean leading symbol flags: e.g. "^#Schweppes" -> "Schweppes"
    cleaned_orig = re.sub(r'^[\\^#*~]+\s*', '', cleaned_orig)

    # Stage A: Conservative OCR Normalization
    cleaned = normalize_ocr_conservative(cleaned_orig)
    cleaned, _ = mask_pack_sizes(cleaned)

    # Strip alphanumeric barcodes, dates, times, phone numbers, masked cards, retail codes, and inventory tags
    cleaned = re.sub(r'\b[A-Za-z]{1,4}\d{5,}[A-Za-z]?\b|\b\d{5,}[A-Za-z]?\b', ' ', cleaned)
    cleaned = re.sub(r'([A-Za-z])(\d{6,})', r'\1 \2', cleaned)
    cleaned = re.sub(r'(\d{6,})([A-Za-z])', r'\1 \2', cleaned)
    cleaned = PHONE_REGEX.sub(" ", cleaned)
    cleaned = MASKED_NUM_REGEX.sub(" ", cleaned)
    cleaned = DATE_TIME_REGEX.sub(" ", cleaned)
    cleaned = RETAIL_CODE_TAGS_REGEX.sub(" ", cleaned)
    cleaned = UPC_BARCODE_REGEX.sub(" ", cleaned)
    cleaned = INVENTORY_TAG_REGEX.sub(" ", cleaned)
    # Strip internal POS machine codes early, but preserve TAX and WEIGHT keywords for price and rate extraction
    cleaned = POS_INTERNAL_CODE_REGEX.sub(" ", cleaned)

    unit_price = None
    total_price = None
    quantity = None
    unit = "pcs"

    # Step 2A: Extract Loose Weighed Unit and Rate (e.g. "0.352kgNET@$6.00/kg", "0.778kgNET$5.99/kg", "0.615ko04.39/Kg")
    m_w_loose = WEIGHT_RATE_LOOSE_RE.search(cleaned)
    if m_w_loose:
        w_val = parse_float_price(m_w_loose.group(1))
        r_val = parse_float_price(m_w_loose.group(2))
        matched_str = m_w_loose.group(0).lower()
        if 'lb' in matched_str:
            unit = 'lb'
        elif 'kg' in matched_str or 'k9' in matched_str or 'ko' in matched_str:
            unit = 'kg'
        elif 'g' in matched_str:
            unit = 'g'
        else:
            unit = 'kg'
        if w_val and w_val > 0:
            quantity = w_val
        if r_val and r_val > 0:
            unit_price = r_val
        cleaned = cleaned[:m_w_loose.start()] + " " + cleaned[m_w_loose.end():]

    # Step 2AA: Strip attached card savings line (e.g. "Card $2.84/kg Save -0.95")
    m_card_save = re.search(r'\b(?:card|member)\b.*?\b(?:save|savings?|off)\b\s*[\-]?[\$€£¥₩]?\s*\d+[.,]\d{2}', cleaned, re.IGNORECASE)
    if m_card_save:
        cleaned = cleaned[:m_card_save.start()] + " " + cleaned[m_card_save.end():]

    # Step 2AB: Columnar table math reconciliation where serial number and quantity appear together
    # e.g. "WHOLE MILK 2 2.50 5.00" or "WHOLE MILK 1 2 2.50 5.00" or "WHOLE MILK 2 5.00"
    m_tab_math = re.search(
        r'(?:^|\s)(?:(\d{1,3})\s+)?([1-9]\d?)\s+[\$€£¥₩]?\s*(\d+[.,]\d{2})\s+[\$€£¥₩]?\s*(\d+[.,]\d{2})\s*$',
        cleaned
    )
    if m_tab_math:
        q_cand = float(m_tab_math.group(2))
        u_cand = parse_float_price(m_tab_math.group(3))
        t_cand = parse_float_price(m_tab_math.group(4))
        if u_cand and t_cand and abs(round(q_cand * u_cand, 2) - t_cand) <= 0.05:
            prefix_txt = cleaned[:m_tab_math.start()].strip()
            if prefix_txt and re.search(r'[A-Za-z가-힣]', prefix_txt):
                quantity = q_cand
                unit_price = u_cand
                total_price = t_cand
                unit = "pcs"
                cleaned = prefix_txt

    if quantity is None:
        m_tab_single = re.search(
            r'(?:^|\s)(?:(\d{1,3})\s+)?([1-9]\d?)\s+[\$€£¥₩]?\s*(\d+[.,]\d{2})\s*$',
            cleaned
        )
        if m_tab_single:
            q_cand = float(m_tab_single.group(2))
            t_cand = parse_float_price(m_tab_single.group(3))
            prefix_txt = cleaned[:m_tab_single.start()].strip()
            # Must have letters and not end in a unit word like 100g, 500ml
            if prefix_txt and re.search(r'[A-Za-z가-힣]{2,}', prefix_txt) and not re.search(r'\b(?:g|gm|kg|ml|l|oz|lb|lbs|ct|pk)\s*$', prefix_txt, re.IGNORECASE):
                quantity = q_cand
                total_price = t_cand
                unit_price = round(t_cand / q_cand, 2)
                unit = "pcs"
                cleaned = prefix_txt

    # Step 2B: Extract Multipliers with attached rates BEFORE bare standalone rate
    # e.g. "2 @ 1.99", "3 x $2.47 ea", "Qty 2 @ $2.02 each", "Qty 20$2.02 each"
    m_qty_typo = re.search(r'\b(?:qty|uty|oty|count)\s*([1-9]\d?)[0Oo8]\s*[\$€£¥₩]?\s*(\d+[.,]\d{2})\s*(?:each|ea|pcs|units?|nos)?\b', cleaned, re.IGNORECASE)
    if m_qty_typo:
        quantity = float(m_qty_typo.group(1))
        unit_price = parse_float_price(m_qty_typo.group(2))
        unit = "pcs"
        cleaned = cleaned[:m_qty_typo.start()] + " " + cleaned[m_qty_typo.end():]

    if quantity is None:
        m_mult_rate = MULTIPLIER_RATE_REGEX.search(cleaned)
        if m_mult_rate:
            quantity = float(m_mult_rate.group(1))
            unit = "pcs"
            mult_p = parse_float_price(m_mult_rate.group(2))
            if mult_p is not None:
                unit_price = mult_p
            cleaned = cleaned[:m_mult_rate.start()] + " " + cleaned[m_mult_rate.end():]

    if quantity is None:
        m_qty_rate = re.search(r'(?:\s|^)(\d+(?:\.\d+)?)\s*[xX×*@]\s*([₹$€£¥₩]?\s*\d+(?:[.,]\d{2})?)(?=\s|$)', cleaned)
        if m_qty_rate:
            quantity = float(m_qty_rate.group(1))
            unit = "pcs"
            rate_val = parse_float_price(m_qty_rate.group(2))
            if rate_val is not None:
                unit_price = rate_val
            cleaned = cleaned[:m_qty_rate.start()] + " " + cleaned[m_qty_rate.end():]

    # Step 2C: Extract Standalone explicit Rate (@ 40.00 / @0.79 / 단가 4,500) if not already part of a multiplier
    if unit_price is None:
        rate_m = RATE_TOKEN_REGEX.search(cleaned)
        if rate_m:
            unit_price = parse_float_price(rate_m.group(1))
            cleaned = cleaned[:rate_m.start()] + " " + cleaned[rate_m.end():]

    # Step 2D: Extract Multi-count multiplier without @ (e.g. "4 0.59" meaning 4 @ 0.59)
    if quantity is None:
        m_count_rate = re.search(r'(?:\s|^)([1-9]\d?)\s+(?:[@o0\s])\s*[\$€£¥₩]?\s*(\d+[.,]\d{2})(?=\s|$)', cleaned)
        if m_count_rate:
            quantity = float(m_count_rate.group(1))
            unit = "pcs"
            c_rate = parse_float_price(m_count_rate.group(2))
            if c_rate is not None:
                unit_price = c_rate
            cleaned = cleaned[:m_count_rate.start()] + " " + cleaned[m_count_rate.end():]

    # 3B. Prefix Multiplier: "2 x Coca Cola", "3xOrganicBananas"
    if quantity is None:
        m_pref = QTY_PREFIX_MULTIPLIER_RE.match(cleaned)
        if m_pref:
            quantity = float(m_pref.group(1))
            cleaned = cleaned[m_pref.end():]

    # 3C. Multiplier like "2AT $0.67" or "2 AT"
    if quantity is None:
        qty_at = re.match(r'^\s*(\d+)\s*AT\b', cleaned, flags=re.IGNORECASE)
        if qty_at:
            quantity = float(qty_at.group(1))
            cleaned = cleaned[qty_at.end():]

    # 3D. Infix/Suffix: "x 2" or "2 *"
    if quantity is None:
        m_inf = QTY_INFIX_MULTIPLIER_RE.search(cleaned)
        if m_inf:
            quantity = float(m_inf.group(1))
            cleaned = cleaned[:m_inf.start()] + " " + cleaned[m_inf.end():]
        else:
            m_suf = QTY_SUFFIX_MULTIPLIER_RE.search(cleaned)
            if m_suf:
                quantity = float(m_suf.group(1))
                cleaned = cleaned[:m_suf.start()] + " " + cleaned[m_suf.end():]

    # 3E. Weighed unit with trailing unit: e.g. "1.38 Organic bananas lb"
    if quantity is None:
        m_weigh_split = re.match(r'^\s*(\d+\.\d{2,3})\s+(.+?)\s+(lb|lbs|kg|kgs|g|gm)\b', cleaned, flags=re.IGNORECASE)
        if m_weigh_split:
            quantity = float(m_weigh_split.group(1))
            unit = m_weigh_split.group(3).lower()
            cleaned = m_weigh_split.group(2) + " " + cleaned[m_weigh_split.end():]

    # 3F. Explicit count unit: "2개", "2팩", "3pcs", "2ea"
    if quantity is None:
        m_unit = PURCHASE_UNIT_REGEX.search(cleaned)
        if m_unit:
            quantity = float(m_unit.group(1))
            unit = m_unit.group(2).lower()
            cleaned = cleaned[:m_unit.start()] + " " + cleaned[m_unit.end():]

    # 3G. Weighed unit adjacent: "1.250 kg", "1.2 lb"
    if quantity is None:
        m_weigh = WEIGHED_UNIT_REGEX.search(cleaned)
        if m_weigh:
            quantity = float(m_weigh.group(1))
            unit = m_weigh.group(2).lower()
            cleaned = cleaned[:m_weigh.start()] + " " + cleaned[m_weigh.end():]

    # 3H. Leading column quantity: e.g. "1 O Organics whole milk 1 gal", "2 Barilla bronze cut..."
    # Protect brand numbers like "7 UP", "V8", "A1", or percentages "2% Milk"
    if quantity is None:
        m_lead = re.match(r'^\s*([1-9]\d?)\s+(?!%)(?!(?:up|v8|a1|way|star|grain|ply)\b)([A-Za-z가-힣])', cleaned, flags=re.IGNORECASE)
        if m_lead:
            quantity = float(m_lead.group(1))
            unit = "pcs"
            cleaned = cleaned[m_lead.end(1):].strip()

    # Step 4: Extract Price Tokens and assign contextual roles (total vs discount vs tax vs unit_rate)
    price_spans = []
    for m in PRICE_TOKEN_REGEX.finditer(cleaned):
        val = parse_price_match(m)
        if val is not None:
            # Check if preceded by minus or discount/savings keyword
            pre_ctx = cleaned[max(0, m.start() - 15):m.start()].lower()
            post_ctx = cleaned[m.end():min(len(cleaned), m.end() + 15)].lower()

            sign = price_sign(cleaned, m.start(), m.end())
            is_negative = (sign == -1) or ('-' in m.group(0))
            role = 'unknown'

            if re.search(r'\b(?:save|saving|savings|off|discount|rebate|minus)\b', pre_ctx) or is_negative:
                role = 'discount'
                if val > 0 and is_negative:
                    val = -val
            elif re.search(r'\b(?:tax|vat|gst|hst|pst|dph)\b', pre_ctx):
                role = 'tax'
            elif '/' in post_ctx or '@' in pre_ctx or re.search(r'\b(?:each|ea)\b', post_ctx):
                role = 'unit_rate'
            else:
                role = 'line_total'

            price_spans.append({
                "val": val,
                "role": role,
                "start": m.start(),
                "end": m.end(),
                "text": m.group(0)
            })

    # Step 5: Assign Total Price and Unit Price using Contextual Roles & Arithmetic Reconciliation
    product_candidates = [p for p in price_spans if p["role"] in ('line_total', 'unknown')]
    discount_candidates = [p for p in price_spans if p["role"] == 'discount']
    tax_candidates = [p for p in price_spans if p["role"] == 'tax']

    discount_amount = sum(d["val"] for d in discount_candidates) if discount_candidates else None
    tax_amount = sum(t["val"] for t in tax_candidates) if tax_candidates else None

    if not product_candidates:
        if discount_candidates:
            # Standalone discount line: e.g. "ITEM -1.00"
            total_price = discount_candidates[-1]["val"]
            for p in discount_candidates:
                cleaned = cleaned[:p["start"]] + " " + cleaned[p["end"]:]
        elif total_price is not None:
            # Total price was already extracted by tabular matcher
            pass
        elif unit_price is not None and quantity is not None and quantity > 0:
            total_price = round(unit_price * quantity, 2)
        else:
            return None
    elif len(product_candidates) == 1:
        total_price = product_candidates[0]["val"]
        # Clean out all price spans from item name
        for p in reversed(price_spans):
            cleaned = cleaned[:p["start"]] + " " + cleaned[p["end"]:]
    elif len(product_candidates) >= 2:
        # We have multiple numbers on the line. Reconcile with Quantity if available!
        p_last = product_candidates[-1]["val"]
        p_prev = product_candidates[-2]["val"]

        if quantity is not None and quantity > 1.0:
            # Check if p_last is total (p_prev * quantity ≈ p_last)
            if abs(round(p_prev * quantity, 2) - p_last) <= 0.05:
                total_price = p_last
                unit_price = p_prev
            # Or if p_prev is total and p_last was something else
            elif abs(round(p_last * quantity, 2) - p_prev) <= 0.05:
                total_price = p_prev
                unit_price = p_last
            else:
                total_price = p_last
                if unit_price is None:
                    unit_price = p_prev
        else:
            total_price = p_last
            if unit_price is None:
                unit_price = p_prev

        for p in reversed(price_spans):
            cleaned = cleaned[:p["start"]] + " " + cleaned[p["end"]:]

    # Step 6: Quantity Finalization & Explicit vs Assumed Flag
    quantity_explicit = True
    if quantity is None:
        qty_col = re.search(r'(?:\s{2,})([1-9]\d?)\s*$', cleaned)
        if qty_col:
            quantity = float(qty_col.group(1))
            unit = "pcs"
            cleaned = cleaned[:qty_col.start()].strip()

    if quantity is None or quantity <= 0:
        if leading_qty is not None:
            quantity = leading_qty
            quantity_explicit = True
        else:
            quantity = 1.0
            quantity_explicit = False

    # Arithmetic Consistency Validation
    arithmetic_valid = None
    if unit_price is None and total_price is not None and quantity > 0:
        unit_price = round(total_price / quantity, 2)
        arithmetic_valid = True
    elif unit_price is not None and total_price is not None and quantity > 0:
        expected_total = round(unit_price * quantity, 2)
        arithmetic_valid = abs(expected_total - total_price) <= 0.05

    # Step 7: Final Item Name Cleanup
    name = cleaned
    name = UPC_BARCODE_REGEX.sub(" ", name)
    name = RETAIL_CODE_TAGS_REGEX.sub(" ", name)
    name = POS_TAX_AND_WEIGHT_REGEX.sub(" ", name)
    name = TAX_FLAG_REGEX.sub(" ", name)
    name = re.sub(r'^[\\^#*~]+\s*', '', name)
    # Require whitespace/boundary before multiplier x/X/@ so brands like LaCroix are never mutilated!
    name = re.sub(r'(?:\s|^)[@*×xX]\s*\d+(?:\.\d+)?(?=\s|$)', ' ', name)
    name = re.sub(r'[₹$€£¥₩]|\b(?:usd|cad|eur|gbp|aud|nzd|inr|rs)\b', ' ', name, flags=re.IGNORECASE)
    # Strip lingering tax or save tokens from item name
    name = re.sub(r'\b(?:save|saving|savings|tax|vat|gst|hst|pst|dph)\b.*$', ' ', name, flags=re.IGNORECASE)
    name = re.sub(r'[\-_=*#~:;/|\\+()\[\]{}]+', ' ', name)
    name = re.sub(r'\s{2,}', ' ', name).strip(' .,;:/-')

    # Remove isolated 4-7 digit product codes in item name (e.g. "4930129 BANANAS")
    name = re.sub(r'(?:^|\s)\b\d{4,7}\b(?=\s|$)', ' ', name)
    # Remove lingering single flag letters like "F", "N", "X", "D", "K" at end (not O)
    name = re.sub(r'\s+[FTNXDHKY0]\s*$', '', name)
    # Remove intermediate invoice float sequences from table lines (e.g. "2 PKT 10.38 20.75 22.00 SR")
    name = re.sub(r'(?:\s+\d+\.\d{2}){2,}\s*$', '', name)
    name = re.sub(r'\s{2,}', ' ', name).strip(' .,;:/-')

    # Strip lingering leading serial index if quantity was extracted from modifier/child/table column
    if quantity is not None and (quantity > 1.0 or quantity_explicit):
        name = re.sub(r'^\s*[1-9]\d?\s+(?!(?:up|v8|a1|way|star|grain|ply)\b)(?=[A-Za-z가-힣])', '', name, flags=re.IGNORECASE)

    # Strip lingering trailing serial index from fused child row: e.g. "WHOLE MILK 1" -> "WHOLE MILK"
    if quantity is not None and (quantity > 1.0 or quantity_explicit):
        name = re.sub(r'\s+[1-9]\d?\s*$', '', name)
    name = name.strip(' .,;:/-')

    # If the item name was completely consumed by explicit Korean purchase count (e.g. "2개 3,000원"),
    # restore the item with its count unit so valid products are never dropped!
    if not name or len(name) < 2:
        if quantity is not None and unit != "pcs":
            name = f"ITEM ({int(quantity) if quantity.is_integer() else quantity}{unit})"
        elif total_price is not None:
            name = "ITEM"
        else:
            return None

    # Filter out standalone POS tax codes, tender words, or table labels that slipped through
    STANDALONE_REJECT = {
        "SR", "ZR", "GST", "HST", "PST", "QST", "VAT", "TAX", "CASH", "CHANGE",
        "TOTAL", "SUBTOTAL", "VISA", "MASTERCARD", "AMEX", "DISCOVER", "CREDIT",
        "DEBIT", "PAYMENT", "BALANCE", "AMOUNT", "GRATUITY", "TIP", "TABLE", "GUEST"
    }
    # Reject if name is purely digits/symbols or starts with tax flags
    if name.upper() in STANDALONE_REJECT or re.match(r'^(?:SR|ZR|GST|HST|VAT)\b', name, flags=re.IGNORECASE):
        return None
    # Reject if name has no actual letters or is just a numeric rate + tax code (e.g. '10.38 SR')
    if not re.search(r'[A-Za-z가-힣]{2,}', name) or re.match(r'^\s*[\d\.\,\s]+(?:SR|ZR|GST|HST|VAT)?\s*$', name, flags=re.IGNORECASE):
        return None

    return {
        "item_name": name,
        "quantity": quantity,
        "quantity_explicit": quantity_explicit,
        "unit": unit,
        "unit_price": unit_price,
        "total_price": total_price,
        "arithmetic_valid": arithmetic_valid,
        "discount_amount": discount_amount,
        "tax_amount": tax_amount,
        "raw_line": original
    }


def clean_item_name_for_bert(item_name: str) -> str:
    """
    Stage C: Semantic normalization for MobileBERT.
    Strips lingering numbers, unit specs, and symbols, fixes OCR character errors,
    and expands retail shorthand so MobileBERT receives pure semantic text for
    state-of-the-art category classification.
    """
    s = item_name

    # 0A. Strip leading serial number / line index (e.g. "1.", "01", "SL. 1", "SL# 1", "S.NO 1", "#1")
    s = SERIAL_PREFIX_RE.sub(' ', s)

    # 0. Strip leading multiplier math (e.g. "1 x", "3 *", "4X", "3xOrganicBananas")
    s = re.sub(r'^\s*\d+\s*[xX×*](?:\s+|(?![\d\s]))', ' ', s)

    # 0B. Separate glued leading count digits or OCR letter artifacts from English words (e.g. "1Coke" -> "Coke", "1Dietcoke" -> "Dietcoke", "ICoffee" -> "Coffee", "JCoffee" -> "Coffee")
    s = re.sub(r'^\s*[0-9IlJj]\s*(?=[A-Za-z]{3,}\b)', '', s)

    # 0C. Clean trailing unmatched parenthesis or multiplier remainder: e.g. "BudLight (2" -> "BudLight"
    s = re.sub(r'[\(\[\{]\s*\d*\s*$', '', s)

    # 1. OCR character repairs
    s = re.sub(r'(\d+(?:\.\d+)?)\s*0rganic', r'\1 Organic', s, flags=re.IGNORECASE)
    s = re.sub(r'\b0rganic', 'Organic', s, flags=re.IGNORECASE)
    s = re.sub(r'(\d+(?:\.\d+)?)\s*0([A-Za-z]+)', r'\1 O \2', s)
    s = re.sub(r'\b0([A-Za-z]+)', r'O \1', s)

    s = re.sub(r'([A-Za-z])0([A-Za-z])', r'\1O\2', s)   # S0UP -> SOUP
    s = re.sub(r'(\d+)\s*0[zZ]\b', r'\1 oz', s)         # 160Z -> 16 oz
    s = re.sub(r'(\d+(?:\.\d+)?)\s*1b\b', r'\1 lb', s)   # 1.2 1b -> 1.2 lb
    s = re.sub(r'([A-Za-z]+)(\d+)\s*1b\b', r'\1 \2 lb', s, flags=re.IGNORECASE)
    s = re.sub(r'\b1b\b', 'lb', s, flags=re.IGNORECASE)

    # Contextual leading 1 in words (1ime -> lime, 1emon -> lemon)
    s = re.sub(r'\b1(ime|emon|ettuce|amb|obster|entils|ager)\b', r'l\1', s, flags=re.IGNORECASE)
    # Fix 1-for-l inside words BEFORE unit separation (e.g. Mi1k -> Milk, Ga1lon -> Gallon, app1e -> apple)
    s = re.sub(r'([A-Za-z])1([A-Za-z])', r'\1l\2', s)
    # Separate % from glued text (e.g. 2%Mi1k -> 2% Milk)
    s = re.sub(r'%([A-Za-z])', r'% \1', s)
    # Split glued PascalCase / CamelCase words (e.g. OrganicBananas -> Organic Bananas)
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    # Repair OCR l-for-1 before units (e.g. lgal -> 1 gal, Coffeel2oz -> Coffee 12 oz)
    s = re.sub(r'([A-Za-z]+)[lI](\d+)\s*(oz|lb|g|kg|ml|l|gal|ct|pk)\b', r'\1 1\2 \3', s, flags=re.IGNORECASE)
    s = re.sub(r'\b[lI](\d+)\s*(oz|lb|g|kg|ml|l|gal|ct|pk)\b', r'1\1 \2', s, flags=re.IGNORECASE)
    s = re.sub(r'\b[lI](gal|oz|lb|kg|g|ml|ct|pk)\b', r'1 \1', s, flags=re.IGNORECASE)

    # Separate attached units from words (e.g. "wholemilk1gal" -> "wholemilk 1 gal", "SOUP500ML" -> "SOUP 500 ML")
    s = re.sub(r'([A-Za-z]+)(\d+)\s*(gal|oz|lb|lbs|g|kg|ml|l|ct|pk)\b', r'\1 \2 \3', s, flags=re.IGNORECASE)
    s = re.sub(r'([A-Za-z]+)(\d+)', r'\1 \2', s)
    s = re.sub(r'(\d+)([A-Za-z]+)', r'\1 \2', s)

    # Supermarket compound word recovery (restoring spaces lost in OCR)
    for pat, rep in GROCERY_COMPOUNDS:
        s = re.sub(re.escape(pat), f" {rep} ", s, flags=re.IGNORECASE)

    # 2. Expand common supermarket shorthand & abbreviations
    s = re.sub(r'\b[A-Za-z]{1,4}\d{5,}[A-Za-z]?\b|\b\d{5,}[A-Za-z]?\b', ' ', s)  # Alphanumeric barcodes (e.g. NPE194346097820, 041192100430F)
    s = re.sub(r'\b(FT|WT|NNP|NP|FA|NN|FS|CPN|DPT)\b', ' ', s, flags=re.IGNORECASE)  # POS status/tax flags
    s = re.sub(r'\bWHL\b', 'WHOLE', s, flags=re.IGNORECASE)
    s = re.sub(r'\bKRNL\s*CORN\b|\bKRNLCORN\b', 'KERNEL CORN', s, flags=re.IGNORECASE)
    s = re.sub(r'\bKRNL\b', 'KERNEL', s, flags=re.IGNORECASE)
    s = re.sub(r'\bTOMRTOES\b|\bTOMT\b|\bTOMTOES\b', 'TOMATOES', s, flags=re.IGNORECASE)
    s = re.sub(r'\b(?:ON\s+THE\s+)?VN\b', 'ON THE VINE', s, flags=re.IGNORECASE)
    s = re.sub(r'\bFRSTO\s+FLKS\b|\bFRST\s+FLKS\b', 'FROSTED FLAKES', s, flags=re.IGNORECASE)
    s = re.sub(r'\bFLKS\b', 'FLAKES', s, flags=re.IGNORECASE)
    s = re.sub(r'\bSTRWBRY\b', 'STRAWBERRY', s, flags=re.IGNORECASE)
    s = re.sub(r'\bBLUBRY\b', 'BLUEBERRY', s, flags=re.IGNORECASE)
    s = re.sub(r'\bPC\s*300\s*CT\b', 'PAPER CUPS 300 CT', s, flags=re.IGNORECASE)
    s = re.sub(r'\bGRND\b', 'GROUND', s, flags=re.IGNORECASE)
    s = re.sub(r'\bBTTLD\b', 'BOTTLED', s, flags=re.IGNORECASE)
    s = re.sub(r'\bBTL(?=[A-Za-z]|\b)', 'BOTTLE ', s, flags=re.IGNORECASE)
    s = re.sub(r'\bCOKE\b', 'Coca Cola', s, flags=re.IGNORECASE)
    s = re.sub(r'\bVEG\b', 'VEGETABLE', s, flags=re.IGNORECASE)
    s = re.sub(r'\b7\s*UP\b', 'SEVEN UP', s, flags=re.IGNORECASE)
    s = re.sub(r'\bV\s*8\b', 'V EIGHT VEGETABLE', s, flags=re.IGNORECASE)
    s = re.sub(r'\bA\s*1\b', 'A ONE STEAK', s, flags=re.IGNORECASE)
    s = re.sub(r'\b2\s*%(?:\s+|$)', 'TWO PERCENT ', s, flags=re.IGNORECASE)
    s = re.sub(r'\b1\s*%(?:\s+|$)', 'ONE PERCENT ', s, flags=re.IGNORECASE)
    s = re.sub(r'\b100\s*%(?:\s+|$)', 'PURE ', s, flags=re.IGNORECASE)
    s = re.sub(r'\bPRETZELBUN\b', 'PRETZEL BUN', s, flags=re.IGNORECASE)
    s = re.sub(r'\bCHSBAGEL\b', 'CHEESE BAGEL', s, flags=re.IGNORECASE)
    s = re.sub(r'\bCHS\s*BAGEL\b', 'CHEESE BAGEL', s, flags=re.IGNORECASE)
    s = re.sub(r'\bEG\s*SOUP\b', 'SOUP', s, flags=re.IGNORECASE)
    s = re.sub(r'\bEG\s*S[0O]UP\b', 'SOUP', s, flags=re.IGNORECASE)
    s = re.sub(r'\bEGS[0O]UP\b', 'SOUP', s, flags=re.IGNORECASE)
    s = re.sub(r'\bS[0O]UP\b', 'SOUP', s, flags=re.IGNORECASE)
    s = re.sub(r'\bMINIGO\b', 'YOGURT', s, flags=re.IGNORECASE)
    s = re.sub(r'\bJELL[\s\-_]*O\b', 'JELL O GELATIN', s, flags=re.IGNORECASE)
    s = re.sub(r'\bGVPNT\s*T?BUTTR\b', 'PEANUT BUTTER', s, flags=re.IGNORECASE)
    s = re.sub(r'\bPNT\s*T?BUTTR\b', 'PEANUT BUTTER', s, flags=re.IGNORECASE)
    s = re.sub(r'\bPNT\b', 'PEANUT', s, flags=re.IGNORECASE)
    s = re.sub(r'\bT?BUTTR\b', 'BUTTER', s, flags=re.IGNORECASE)
    s = re.sub(r'\bCHNK\s*CHKN\b', 'CHUNK CHICKEN', s, flags=re.IGNORECASE)
    s = re.sub(r'\bCHKN\b', 'CHICKEN', s, flags=re.IGNORECASE)
    s = re.sub(r'\bPARM\b', 'PARMESAN CHEESE', s, flags=re.IGNORECASE)
    s = re.sub(r'\bNITRIL\b', 'NITRILE GLOVES', s, flags=re.IGNORECASE)
    s = re.sub(r'\bFOLGERS\b', 'FOLGERS COFFEE', s, flags=re.IGNORECASE)
    s = re.sub(r'\b(?:SC\s+)?TWIST\s+UP\b', 'TWIST UP SODA', s, flags=re.IGNORECASE)

    # 3. Strip store brand prefixes that confuse semantic classification
    s = re.sub(r'\b(?:great\s+value|gv|kirkland(?:\s+signature)?|member\'?s\s+mark)\b', ' ', s, flags=re.IGNORECASE)

    # 4. Strip retail prefix tags: 바코드, 품번, 상품코드, SKU, PLU, ITEM
    s = re.sub(r'\b(?:바코드|품번|상품코드|코드|plu|sku|item|dpt|dept)\b', ' ', s, flags=re.IGNORECASE)

    # 5. Remove long barcodes and product codes (5+ digits)
    s = re.sub(r'\b\d{5,}\b', ' ', s)

    # 5B. Strip lingering POS tax flags and weight descriptors
    s = POS_TAX_AND_WEIGHT_REGEX.sub(" ", s)

    # 6. Normalize packaging units with optional decimals (e.g. 15.25OZ -> 15.25 OZ, 2LB -> 2 LB)
    s = re.sub(r'\b(\d+(?:\.\d+)?)\s*(oz|lb|lbs|g|kg|ml|l|ct|pk|pack|gal|ea)\b', r'\1 \2', s, flags=re.IGNORECASE)

    # 7. Strip isolated standalone numbers not attached to packaging units
    s = re.sub(r'(?<![\d\.])\b\d+(?:\.\d+)?\b(?![\d\.]|\s*(?:oz|lb|lbs|g|kg|ml|l|ct|pk|pack|gal|ea))\b', ' ', s, flags=re.IGNORECASE)

    # 8. Clean symbols: preserve decimal points within numbers, keep letters, digits, and combining marks in any script
    s = clean_symbols_unicode(s)

    # 9. Semantic refinements
    if s.upper() == 'BREAD':
        s = 'BAKERY BREAD'
    if re.search(r'\bbuns?\b', s, re.IGNORECASE) and not re.search(r'\bbakery\b', s, re.IGNORECASE):
        s = 'BAKERY ' + s
    if re.search(r'\bla\s*croix\b', s, re.IGNORECASE) and not re.search(r'\bwater\b', s, re.IGNORECASE):
        s = re.sub(r'\bla\s*croix\b', 'LaCroix sparkling water', s, flags=re.IGNORECASE)
    if re.search(r'\brigatoni\b', s, re.IGNORECASE) and not re.search(r'\bpasta\b', s, re.IGNORECASE):
        s = re.sub(r'\brigatoni\b', 'rigatoni pasta', s, flags=re.IGNORECASE)

    return s


# ─────────────────────────────────────────────────────────────────────────────
# 2C. MULTI-LINE PRODUCT-MODIFIER FUSER (Quantities & Prices Across Lines)
# ─────────────────────────────────────────────────────────────────────────────
# 1. Pure Price Line on Next Line: e.g. "14.99", "$5.99 F", "7.89", "9,900원", "3,000", "3.98 R"
PURE_PRICE_LINE_RE = re.compile(
    r'^\s*'
    r'(?:'
        r'([₹$€£¥₩]|(?:usd|cad|eur|gbp|aud|nzd|inr|rs\.?)\s*)?(\d+[.,，]\d{2}|\d{1,3}(?:,\d{3})+(?:[.,，]\d{1,2})?|\d{1,6})'
        r'(?:\s*([원]|rs\.?))?'
    r')'
    r'(?:\s*[FTNXDHKYABO0R\-])?'
    r'\s*$',
    re.IGNORECASE
)

# 2. Qty / Rate / Weight / Multiplier Modifier on Next Line:
# Examples:
#   "2 @ 9.99 19.98", "2 @ $9.99", "2 x 2.49 4.98", "2 x 2.49", "3 x 1.25", "2 EA 15.99"
#   "Qty 2 each @ 1.99 3.98", "Qty 2 @ 1.99", "Qty 2 3.98", "2 x", "3 *"
#   "1.38 lb @ 0.59 /lb 0.81", "1.38 lb 0.81", "2.5 lb 4.98", "0.650 kg 2.60"
#   "2개 @ 1,500 3,000", "2개 3,000", "2  1,100  2,200", "1  3,980"
SUBROW_MODIFIER_RE = re.compile(
    r'^\s*'
    r'(?:'
        # Explicit Qty prefix
        r'(?:qty|count)\s*[:#\s]?\s*\d+(?:\.\d+)?(?:\s*(?:each|ea|pcs|units?|nos))?'
        r'|'
        # Multiplier x, @, at, *
        r'\d+(?:\.\d+)?\s*(?:[xX×*]|@|\bat\b)'
        r'|'
        # Weighed produce unit
        r'\d+(?:\.\d+)?\s*(?:lb|lbs|kg|kgs|g|gm|oz|gal)\b'
        r'|'
        # Count packaging unit
        r'\d+(?:\.\d+)?\s*(?:ea|pcs|pkts?|units?|개|병|봉|캔|팩|박스|통|줄)\b'
        r'|'
        # Tabular space-separated numeric columns: "2  1,100  2,200" or "1  3,980"
        r'\d{1,2}\s+(?:[₹$€£¥₩]?\s*\d+(?:,\d{3})*(?:\.\d{2})?)\s+(?:[₹$€£¥₩]?\s*\d+(?:,\d{3})*(?:\.\d{2})?)'
        r'|'
        r'\d{1,2}\s+(?:[₹$€£¥₩]?\s*\d+(?:,\d{3})*(?:\.\d{2})?)'
    r')'
    r'.*$',
    re.IGNORECASE
)

# 3. Standalone Barcode / SKU on Next Line: e.g. "041192100430F", "880104301294"
BARCODE_ONLY_LINE_RE = re.compile(
    r'^\s*(?:(?:sku|upc|item|code|plu|dpt)\s*[:#\-\+]?\s*)?\d{7,14}[A-Za-z]?\s*$',
    re.IGNORECASE
)


def has_price_token(line: str) -> bool:
    """Checks if a string already contains a valid price token."""
    return bool(re.search(
        r'([₹$€£¥₩]|\b(?:usd|cad|eur|gbp|aud|nzd|inr|rs)\b)\s*\d+|\d+[.,]\d{2}|\d+\s*원',
        line,
        re.IGNORECASE
    ))


def is_boundary_or_header_line(line: str) -> bool:
    """Checks if a line is a divider, financial subtotal, tax, or receipt boundary."""
    s = line.strip()
    if not s or SEPARATOR_REGEX.match(s):
        return True
    if NOISE_REGEX.search(s):
        return True
    return False


# Modifier vocabulary for detecting child POS sub-rows
MODIFIER_ALLOWLIST_WORDS = {
    'ea', 'each', 'pcs', 'pc', 'net', 'tare', 'gross', 'wt', 'weight',
    'lb', 'lbs', 'kg', 'kgs', 'g', 'gm', 'oz', 'gal', 'at', 'qty', 'count', 'uty', 'oty', 'cty',
    'for', 'unit', 'units', 'item', 'items', 'nos', 'no', 'pk', 'pack', 'pkt', 'pkts', 'packet', 'packets',
    'btl', 'btls', 'bottle', 'bottles', 'can', 'cans', 'bx', 'box', 'boxes', 'carton', 'ctn',
    'sl', 'sr', 'sno',
    'tfa', 'tf', 'fa', 'fs', 'tax', 'f', 't', 'n', 'x', 'd', 'k', 'y', 'a', 'b', 'r',
    '단가', '개', '팩', '병', '봉', '캔', '박스', '통', '줄',
    'lmt', 'limit', 'max', 'min', 'or', 'save', 'saving', 'savings', 'off', 'reg', 'regular', 'promo',
    'card', 'member', 'club', 'rewards?',
    'zlevneno', 'sleva', 'akce', 'kc', 'kč', 'ks', 'dph',
    'gpmrj', 'pmrj', 'gmrj', 'mrj', 'gpmr', 'pmr', 'gmr', 'rq', '8prq', 'prq', 'bunch', 'bunched'
}

# Loose Weighed Unit and Rate: e.g. "0.778kgNET$5.99/kg", "0.615ko04.39/Kg", "0.950K951.74/K9", "0.300kg2.84/kg"
WEIGHT_RATE_LOOSE_RE = re.compile(
    r'(?:^|\s)(?:[Tt]\.|0\.|1\.)?(\d+(?:[.,]\d{1,3}))\s*(?:kg|k9|ko|k8|g|gm|lb|lbs|k|xys?|[0-9a-z]{1,3})?\s*(?:net|tare|gross|wt)?\s*(?:[@o08s]\s*)?[\$€£¥₩]?\s*(\d+[.,]\d{1,2})\s*/\s*(?:kg|k9|g|gm|lb|lbs|k|ea|unit)?',
    re.IGNORECASE
)

# Loose Multi-Buy Rate: e.g. "4 @ 0.59", "4 0.59", "Card 3/$9.00 Save -1.29", "Card $2.84/kg Save -0.95"
CARD_SAVINGS_ROW_RE = re.compile(
    r'^\s*(?:card|member|club|rewards?)\b.*?\b(?:save|savings?|off)\b.*$',
    re.IGNORECASE
)


def is_child_subrow_modifier(line: str) -> bool:
    """
    Returns True if a line represents a child POS transaction modifier/sub-row
    (quantity, packaging count, weighed unit tare/net, @ unit-rate, promo limit, or barcode)
    rather than an independent product line.
    """
    s = line.strip()
    if not s or len(s) < 2:
        return False
    if is_boundary_or_header_line(s) or DEPT_HEADER_REGEX.match(s):
        return False
    if BARCODE_ONLY_LINE_RE.match(s):
        return True
    if CARD_SAVINGS_ROW_RE.match(s):
        return True
    if WEIGHT_RATE_LOOSE_RE.search(s):
        return True

    # Standalone POS tax flag lines (e.g. "GPMRJ", "MRJ", "TFA")
    if re.match(r'^\s*(?:GPMRJ|PMRJ|GMRJ|MRJ|GPMR|PMR|GMR|8PRQ|PRQ|RQ|TFA|FA|TF|FS|TAX)\s*$', s, re.IGNORECASE):
        return True

    # Strip serial number prefix before evaluating words vs modifier tokens
    s_core = strip_child_serial(s)
    if WEIGHT_RATE_LOOSE_RE.search(s_core):
        return True

    # Strip numbers, prices, currency, units, punctuation
    cleaned = re.sub(r'[\$€£¥₩.,:;/*#@×xX\-\+()0-9]', ' ', s_core)
    words = [w.lower() for w in cleaned.split() if len(w) > 0]
    product_words = [w for w in words if w not in MODIFIER_ALLOWLIST_WORDS and len(w) > 1]

    # Must contain numeric values or rate symbol '@', and contain NO independent product nouns
    has_num_or_rate = bool(re.search(r'\d|@', s_core))
    if has_num_or_rate and len(product_words) == 0:
        return True
    return False


def pre_fuse_multiline_items(lines: List[str]) -> List[str]:
    """
    Generalized POS Parent-Child Fusing Algorithm:
    Fuses multi-line receipt items across 2, 3, or 4 lines:
      - Parent Item Line: Product Name (can be priced or unpriced)
      - Child Sub-Row Line: Modifiers (rates, weights, counts, promo tiers, barcodes)
      - Child Price Line: Pure Price line (if parent line was unpriced)
    """
    if not lines:
        return []

    fused = []
    i = 0
    n = len(lines)

    while i < n:
        curr = lines[i].strip()
        if not curr:
            i += 1
            continue

        # If curr is a boundary/noise line, department header, a standalone child subrow, or lacks letters/Hangul, keep as is
        if is_boundary_or_header_line(curr) or DEPT_HEADER_REGEX.match(curr) or is_child_subrow_modifier(curr) or not re.search(r'[A-Za-z가-힣]', curr):
            fused.append(curr)
            i += 1
            continue

        merged = curr
        j = i + 1
        has_price = has_price_token(curr)

        # Look ahead up to 3 lines for child modifier rows, barcodes, promo tiers, or pure prices
        while j < n and j <= i + 3:
            nxt = lines[j].strip()
            if not nxt:
                j += 1
                continue
            if is_boundary_or_header_line(nxt) or DEPT_HEADER_REGEX.match(nxt):
                break

            # Case 1: nxt is a child sub-row modifier (e.g. "4EA @ 2.59/EA", "0.778kg NET @ $5.99/kg", barcode, promo tier)
            if is_child_subrow_modifier(nxt):
                nxt_clean = strip_child_serial(nxt)
                merged += " " + nxt_clean
                has_price = has_price or has_price_token(nxt)
                j += 1
                continue

            # Case 2: Parent line had NO price, and nxt is a pure price line (e.g. "$14.99", "7.89", "9,900원")
            if not has_price and PURE_PRICE_LINE_RE.match(nxt):
                merged += " " + nxt
                has_price = True
                j += 1
                break

            # Not a child of this parent line
            break

        fused.append(merged)
        i = j

    return [l for l in fused if l]




# ─────────────────────────────────────────────────────────────────────────────
# 3. SPATIAL ROW CLUSTERING (ROBUST CENTROID-BASED)
# ─────────────────────────────────────────────────────────────────────────────
def cluster_ocr_boxes_into_lines(ocr_results: List[Any], max_y_diff_ratio: float = 0.5) -> List[Dict[str, Any]]:
    """
    Groups OCR word fragments into distinct rows by comparing each word's Y-center
    against active row centers, preventing transitive chain merging.
    """
    if not ocr_results:
        return []

    fragments = []
    for item in ocr_results:
        if len(item) == 3:
            box, text, score = item
        elif len(item) == 2:
            box, text = item
            score = 1.0
        else:
            continue

        text = text.strip()
        if not text or score < 0.2:
            continue

        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        height = max(y_max - y_min, 1.0)
        y_center = (y_min + y_max) / 2.0

        fragments.append({
            "text": text,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "y_center": y_center,
            "height": height
        })

    # Sort fragments top-to-bottom
    fragments.sort(key=lambda f: f["y_center"])

    # Cluster by matching Y-center with existing row centroid
    rows = []
    for f in fragments:
        best_row = None
        best_diff = float("inf")

        for r in rows:
            # Row centroid is average y_center of elements in row
            row_y_center = sum(el["y_center"] for el in r) / len(r)
            row_avg_h = sum(el["height"] for el in r) / len(r)
            diff = abs(f["y_center"] - row_y_center)
            
            # Allow max difference of 50% of the line height (or at most 18 pixels)
            max_allowed = max(min(row_avg_h * max_y_diff_ratio, 18.0), 8.0)
            if diff <= max_allowed and diff < best_diff:
                best_diff = diff
                best_row = r

        if best_row is not None:
            best_row.append(f)
        else:
            rows.append([f])

    # Sort elements in each row left-to-right
    merged_lines = []
    for r in rows:
        r.sort(key=lambda f: f["x_min"])
        line_text = " ".join(f["text"] for f in r)
        x_min = min(f["x_min"] for f in r)
        y_min = min(f["y_min"] for f in r)
        x_max = max(f["x_max"] for f in r)
        y_max = max(f["y_max"] for f in r)
        merged_lines.append({
            "line_text": line_text,
            "bbox": [x_min, y_min, x_max, y_max]
        })

    # Sort lines from top to bottom
    merged_lines.sort(key=lambda l: l["bbox"][1])
    return merged_lines
