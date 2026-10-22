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


# ─────────────────────────────────────────────────────────────────────────────
# 1. NOISE & METADATA LINE FILTER (Total, Subtotal, Tax, Payment headers)
# ─────────────────────────────────────────────────────────────────────────────
NOISE_KEYWORDS = [
    # Subtotal, Total, Balance (Context-Aware)
    r'sub\s*t[0o]tal', r'grand\s*t[0o]tal', r'net\s*t[0o]tal', r'amount\s*due',
    r'\b(?:balance\s*due|new\s*balance|acct\s*balance|ending\s*balance|bal\s*due)\b',
    r'^\s*(?:[\*\-_#\s]*)(?:sub\s*|grand\s*|net\s*)?t[0o]tal\b',
    r'\bt[0o]tal\s+(?:amount|due|tax|items?|savings?|price|bill|sale|charge|paid|purchase|tend|tendered)\b',
    r'\b(?:purchase|sale)\s*t[0o]tal\b',
    r'\bt[0o]tal\s*purchase\b',
    r'\bt[0o]tal\s*due\b',
    
    # Taxes
    r'\btax\d*\b', r'\bvat\b', r'\bgst\b', r'\bhst\b', r'\bqst\b', r'\bcgst\b', r'\bsgst\b',
    r'\bigst\b', r'sales\s*tax', r'service\s*charge', r'service\s*tax',
    r'\bstate\s*tax\b', r'\blocal\s*tax\b',
    r'\btax\s*\d*\s*[\d.]*%\b',
    
    # Financial line discounts & savings (context-aware: avoids deleting "SAVINGS COFFEE")
    r'\b(?:total\s*discount|discount\s*applied|store\s*discount|coupon\s*discount)\b',
    r'^\s*discount\s*[\d\$\.\-]+\s*$',
    r'\b(?:total\s*savings?|your\s*savings?|customer\s*savings?|instant\s*savings?|member\s*savings?)\b',
    r'\b(?:you\s*saved(?:\s*today)?|loyalty\s*(?:card)?(?:\s*savings?)?|card\s*savings)\b',
    r'^\s*savings?\s*[\d\$\.\-]+\s*$',
    r'\b(?:promo\s*discount|promo\s*savings?|promo\s*code|promo\s*applied)\b',
    r'^\s*coupon\b', r'\bround\s*off\b',
    r'\bearn\s*[\d\w\s]*points\b',
    
    # Payments (context-aware: avoids deleting "BIRTHDAY CARD", "CASH & CARRY")
    r'\b(?:cash\s*(?:tend|tender|tendered|given|paid|back)|cash\s*payment|cash\s*sale)\b',
    r'^\s*cash\s*[\d\$\.]+\s*$',
    r'\b(?:change\s*due|change\s*given|change\s*returned)\b',
    r'^\s*change\s*[\d\$\.]+\s*$',
    r'\b(?:credit\s*card|debit\s*card|cred[1i]t\s*card|master\s*card|visa\s*card|amex\s*card|card\s*ending|card\s*auth|card\s*appr)\b',
    r'\bdebit\s*tend\b', r'\bcash\s*tend\b', r'\beft\s*debit\b', r'\bpayment\s*declined\b',
    r'\b(?:visa|mastercard|amex|discover)\s+(?:credit|debit|ending|card|auth|balance|tend)\b',
    r'\b(?:upi|gpay|paytm)\s*(?:pay|payment|id|txn)\b',
    r'\btender(?:ed)?\b',
    r'\bba[li1]ance\b', r'\bbalance\s*to\s*pay\b',
    r'\breceived\s*(?:money|cash|amount)\b',
    r'\breturn\s*value\b',
    
    # Administrative & Greeting headers
    r'\bthank\s*you\b', r'\bvisit\s*(?:us\s*)?again\b', r'\bcustomer\s*copy\b', r'\bmerchant\s*copy\b',
    r'^\s*receipt\b', r'\btax\s*invoice\b', r'\bcash\s*memo\b', r'\btoken\s*no\b', r'\border\s*no\b',
    r'\bclosed\s*bill\b', r'\bcheck\s*no\b', r'\bguest\s*check\b',
    r'\bgstin\b', r'\bfssai\b', r'\btel[:.\s]', r'\bphone[:.\s]',
    r'\b(?:cashier|cshr|reg|register)\s*[:#\s]?\s*[A-Za-z0-9]+',
    r'\bmana[g9]er\b', r'\bst#', r'\b[0o]p#', r'\bte#', r'\btr#', r'\btc[#\+]',
    r'\bref\s*#', r'\btrans\s*#', r'\bterminal\b', r'\bterm#', r'\btermid\b', r'\bcatid\b',
    r'\btid[:\s]', r'\bmerch\s*id\b',
    r'\bpos\s*title\b', r'\bkasir\s*rcpt\b',
    r'\bt[o0]ta[l1i]\s*(?:due|purchase|amount|tax)?\b',
    r'\baccount\b', r'\bacct\b', r'\bappr\s*c[0o]de\b', r'\bnetw[0o]rk\b',
    r'\bitems?\s*sold\b', r'\blayaway\b', r'\bcome\s*back\s*soon\b',
    r'\bsave\s*money\b', r'\blive\s*better\b', r'\bwe\s*sell\s*less\b',
    
    # Voids
    r'\bvoid(?:ed)?\b', r'xvoidedentry', r'gst/hst', r'\bqst\d+',
    
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
    r'2%\s*(?:reduced\s*fat\s*)?milk|1%\s*milk|100%\s*(?:orange\s*)?juice'
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
RETAIL_CODE_TAGS_REGEX = re.compile(
    r'\b(?:plu|sku|item|prod|upc|code|art|pos|dpt|dept|tc|ref|st|op|te|tr|바코드|품번|상품코드|코드)\s*[:#\-\+]?\s*[0-9A-Za-z\-]+\b',
    re.IGNORECASE
)

# 3. Supermarket alphanumeric inventory noise: e.g. "*+8SP300g", "#004900" (must contain at least 1 digit to avoid deleting brand names like #Lucozade)
INVENTORY_TAG_REGEX = re.compile(r'(?:^|\s)[\*#+]{1,3}(?=[0-9A-Za-z]*\d)[0-9A-Z]{3,10}(?=\s|$)', re.IGNORECASE)

# 4. Trailing Tax / Food-Stamp flags: single letter (D, K, F, T, N, X, Y, A, B, O, R), void hyphen (-), or space-separated 0
TAX_FLAG_REGEX = re.compile(r'(?<=[\d\$원])\s*(?:[FTNXDHKYABOR]|\-|\s+0)\s*$', re.IGNORECASE)


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
        r'([₹$€£¥₩]|(?:usd|cad|eur|gbp|aud|nzd|inr|rs\.?))\s*(\d+[.,，]\d{2}|\d{1,3}(?:,\d{3})+(?:[.,，]\d{1,2})?|\d+)'
        r'|'
        r'(?<![A-Za-z0-9])(\d{1,3}(?:,\d{3})+(?:[.,，]\d{1,2})?)'
        r'|'
        r'(?<![A-Za-z0-9])(\d+[.,，]\d{2})'
        r'|'
        r'(?<![A-Za-z0-9])(\d+)\s*(원|rs\.?)'
    r')'
    r'(?:\s*([FTNXDHKYABOR]|\s+0))?'
    r'(?:\s*[\-])?'
    r'\s*(원|rs\.?)?'
    r'(?![A-Za-z0-9])',
    re.IGNORECASE
)

RATE_TOKEN_REGEX = re.compile(
    r'(?:@|단가)\s*[₹$€£¥₩]?\s*(\d+(?:,\d{3})*(?:[.,，]\d{1,2})?)\s*(?:/\s*[A-Za-z가-힣]+)?',
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
    """Extracts numeric float value from structured price regex match, supporting comma decimals."""
    raw = m.group(2) or m.group(3) or m.group(4) or m.group(5)
    if not raw:
        return None
    txt = raw.replace('，', ',')
    # Comma followed by exactly two digits is a decimal comma (e.g. 4,99 -> 4.99)
    if re.search(r'\d+,\d{2}$', txt):
        txt = re.sub(r',(\d{2})$', r'.\1', txt)
        txt = txt.replace(',', '')
    else:
        # Standard commas are thousands separators (e.g. 1,250 -> 1250)
        txt = txt.replace(',', '')
    try:
        val = float(txt)
        if 0.01 <= val <= 50_000_000:
            return val
    except ValueError:
        pass
    return None


def parse_float_price(text: str) -> Optional[float]:
    """Fallback converter for price token strings into clean float, supporting comma decimals."""
    if not text:
        return None
    cleaned = text.replace('，', ',').replace(' ', '').replace('원', '')
    cleaned = re.sub(r'^[₹$€£¥₩]|rs\.?|inr|usd|cad|eur|gbp|aud|nzd', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'[FTNXABOOHY0]$', '', cleaned, flags=re.IGNORECASE)
    if re.search(r'\d+,\d{2}$', cleaned):
        cleaned = re.sub(r',(\d{2})$', r'.\1', cleaned)
        cleaned = cleaned.replace(',', '')
    else:
        cleaned = cleaned.replace(',', '')
    try:
        val = float(cleaned)
        if 0.01 <= val <= 50_000_000:
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
    # If it's a known product name containing words like 'total', 'card', 'balance', protect it
    if PRODUCT_PROTECTION_REGEX.search(s):
        return False
    if NOISE_REGEX.search(s):
        return True
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

    # 5. Grocery compound word recovery (longest phrase first)
    for pat, rep in GROCERY_COMPOUNDS:
        s = re.sub(re.escape(pat), f" {rep} ", s, flags=re.IGNORECASE)

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
    norm_for_noise = original.translate(str.maketrans({"，": ",", "．": ".", "：": ":", "。": "."}))
    if is_noise_or_header_line(norm_for_noise):
        return None

    # Stage A: Conservative OCR Normalization
    cleaned = normalize_ocr_conservative(original)

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
    cleaned = TAX_FLAG_REGEX.sub(" ", cleaned)

    unit_price = None
    total_price = None
    quantity = None
    unit = "pcs"

    # Step 2: Extract explicit Rate (@ 40.00 / @0.79 / 단가 4,500)
    rate_m = RATE_TOKEN_REGEX.search(cleaned)
    if rate_m:
        unit_price = parse_float_price(rate_m.group(1))
        cleaned = cleaned[:rate_m.start()] + " " + cleaned[rate_m.end():]

    # Step 3: Extract Purchase Multiplier, Weighed Produce, or Leading Column Quantity
    # 3A. Explicit Qty modifier: "Qty 2 @ $3.15 each", "Qty 2 each", "Qty 4"
    if quantity is None:
        m_qty_each = re.search(
            r'\b(?:qty|count)\s*[:#\s]?\s*(\d+(?:\.\d+)?)\s*(?:@\s*[\$€£₩₹]?\s*\d+(?:[.,]\d+)?\s*)?(?:each|ea|pcs|units?|items?|nos)?\b',
            cleaned,
            re.IGNORECASE
        )
        if m_qty_each:
            quantity = float(m_qty_each.group(1))
            unit = "pcs"
            cleaned = cleaned[:m_qty_each.start()] + " " + cleaned[m_qty_each.end():]

    # 3AA. Explicit multiplier with attached unit price: "3 x 1.25", "2 @ 9.99", "2*4.50"
    if quantity is None:
        m_qty_rate = re.search(r'(?:\s|^)(\d+(?:\.\d+)?)\s*[xX×*@]\s*([₹$€£¥₩]?\s*\d+(?:[.,]\d{2})?)(?=\s|$)', cleaned)
        if m_qty_rate:
            quantity = float(m_qty_rate.group(1))
            unit = "pcs"
            rate_val = parse_float_price(m_qty_rate.group(2))
            if rate_val is not None:
                unit_price = rate_val
            cleaned = cleaned[:m_qty_rate.start()] + " " + cleaned[m_qty_rate.end():]

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

    # Step 4: Extract Price Tokens using rigorous pattern
    price_spans = []
    for m in PRICE_TOKEN_REGEX.finditer(cleaned):
        val = parse_price_match(m)
        if val is not None:
            price_spans.append({"val": val, "start": m.start(), "end": m.end(), "text": m.group(0)})

    # Step 5: Assign Total Price and Unit Price
    if not price_spans:
        if unit_price is not None and quantity is not None and quantity > 0:
            total_price = round(unit_price * quantity, 2)
        else:
            return None
    elif len(price_spans) == 1:
        total_price = price_spans[0]["val"]
        cleaned = cleaned[:price_spans[0]["start"]] + " " + cleaned[price_spans[0]["end"]:]
    elif len(price_spans) >= 2:
        total_price = price_spans[-1]["val"]
        prev_price = price_spans[-2]["val"]
        if unit_price is None:
            unit_price = prev_price

        for p in reversed(price_spans[-2:]):
            cleaned = cleaned[:p["start"]] + " " + cleaned[p["end"]:]

    # Step 6: Column Qty Fallback (must be 1-99, not 0)
    if quantity is None:
        qty_col = re.search(r'(?:\s|^)([1-9]\d?)\s*$', cleaned)
        if qty_col:
            quantity = float(qty_col.group(1))
            unit = "pcs"
            cleaned = cleaned[:qty_col.start()].strip()

    if quantity is None or quantity <= 0:
        quantity = 1.0

    if unit_price is None and total_price is not None and quantity > 0:
        unit_price = round(total_price / quantity, 2)

    # Step 7: Final Item Name Cleanup
    name = cleaned
    name = UPC_BARCODE_REGEX.sub(" ", name)
    name = RETAIL_CODE_TAGS_REGEX.sub(" ", name)
    name = TAX_FLAG_REGEX.sub(" ", name)
    # Require whitespace/boundary before multiplier x/X/@ so brands like LaCroix are never mutilated!
    name = re.sub(r'(?:\s|^)[@*×xX]\s*\d+(?:\.\d+)?(?=\s|$)', ' ', name)
    name = re.sub(r'[₹$€£¥₩]|\b(?:usd|cad|eur|gbp|aud|nzd|inr|rs)\b', ' ', name, flags=re.IGNORECASE)
    name = re.sub(r'[\-_=*#~:;/|\\+()\[\]{}]+', ' ', name)
    name = re.sub(r'\s{2,}', ' ', name).strip(' .,;:/-')

    # Remove isolated 4-7 digit product codes in item name (e.g. "4930129 BANANAS")
    name = re.sub(r'(?:^|\s)\b\d{4,7}\b(?=\s|$)', ' ', name)
    # Remove lingering single flag letters like "F", "N", "X", "D", "K" at end (not O)
    name = re.sub(r'\s+[FTNXDHKY0]\s*$', '', name)
    name = re.sub(r'\s{2,}', ' ', name).strip(' .,;:/-')

    if not name or len(name) < 2:
        return None

    return {
        "item_name": name,
        "quantity": quantity,
        "unit": unit,
        "unit_price": unit_price,
        "total_price": total_price,
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

    # 0. Strip leading multiplier math (e.g. "1 x", "3 *", "4X", "3xOrganicBananas")
    s = re.sub(r'^\s*\d+\s*[xX×*](?:\s+|(?![\d\s]))', ' ', s)

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

    # 6. Normalize packaging units with optional decimals (e.g. 15.25OZ -> 15.25 OZ, 2LB -> 2 LB)
    s = re.sub(r'\b(\d+(?:\.\d+)?)\s*(oz|lb|lbs|g|kg|ml|l|ct|pk|pack|gal|ea)\b', r'\1 \2', s, flags=re.IGNORECASE)

    # 7. Strip isolated standalone numbers not attached to packaging units
    s = re.sub(r'(?<![\d\.])\b\d+(?:\.\d+)?\b(?![\d\.]|\s*(?:oz|lb|lbs|g|kg|ml|l|ct|pk|pack|gal|ea))\b', ' ', s, flags=re.IGNORECASE)

    # 8. Clean symbols: preserve decimal points within numbers, keep letters, digits, and Hangul
    s = re.sub(r'(?<!\d)\.|\.(?!\d)', ' ', s)
    s = re.sub(r'[^A-Za-z0-9가-힣\.\s]', ' ', s)
    s = re.sub(r'\s{2,}', ' ', s).strip()

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


def is_multiline_item_candidate(line: str) -> bool:
    """True if line looks like an unpriced item description that needs following line prices/quantities."""
    s = line.strip()
    if not s or len(s) < 2:
        return False
    if is_boundary_or_header_line(s):
        return False
    if has_price_token(s):
        return False
    if not re.search(r'[A-Za-z가-힣]', s):
        return False
    return True


def pre_fuse_multiline_items(lines: List[str]) -> List[str]:
    """
    Fuses multi-line receipt items across 2 or 3 lines:
      - Line N:   Product Name (no price)
      - Line N+1: Pure Price, or Qty/Modifier/Weight, or Barcode
      - Line N+2: Price (if line N+1 had Barcode or Qty only)
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

        if is_multiline_item_candidate(curr):
            merged = curr
            j = i + 1
            # Look ahead up to 2 lines
            while j < n and j <= i + 2:
                nxt = lines[j].strip()
                if not nxt:
                    j += 1
                    continue
                if is_boundary_or_header_line(nxt):
                    break
                if BARCODE_ONLY_LINE_RE.match(nxt):
                    merged += " " + nxt
                    j += 1
                    continue
                if SUBROW_MODIFIER_RE.match(nxt):
                    merged += " " + nxt
                    j += 1
                    if has_price_token(nxt):
                        break
                    continue
                if PURE_PRICE_LINE_RE.match(nxt):
                    merged += " " + nxt
                    j += 1
                    break
                break

            if j > i + 1:
                fused.append(merged)
                i = j
                continue

        fused.append(curr)
        i += 1

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
