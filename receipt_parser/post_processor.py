import re
from typing import List, Dict, Any, Optional

"""
Receipt Post-Processing Module (Reserved for Future Use)
-------------------------------------------------------
Provides utilities for:
1. Stripping leading quantity numbers (e.g., "2 APPLE" -> "APPLE", Qty: 2)
2. Filtering store header/brand false positives in top receipt layout region
3. Standardizing extracted food item strings
"""

def clean_item_name(text: str) -> Dict[str, Any]:
    """
    Parses a receipt text line to extract quantity prefix and clean item name.
    
    Examples:
        "2 APPLE"       -> {"clean_text": "APPLE", "quantity": 2}
        "1x BANANA"     -> {"clean_text": "BANANA", "quantity": 1}
        "3 ORG SPINACH" -> {"clean_text": "ORG SPINACH", "quantity": 3}
    """
    text_str = text.strip()
    
    # Match leading quantity digit (e.g., "2 ", "2x ", "2X ", "2- ")
    match = re.match(r'^(\d+)\s*[xX\-]?\s+(.+)$', text_str)
    if match:
        qty = int(match.group(1))
        clean_name = match.group(2).strip()
        return {
            "raw_text": text_str,
            "clean_text": clean_name,
            "quantity": qty
        }
    
    return {
        "raw_text": text_str,
        "clean_text": text_str,
        "quantity": 1
    }


def is_likely_store_header(item: Dict[str, Any], img_height: Optional[int] = None) -> bool:
    """
    Identifies potential false positive store header words (e.g., "Green" in "Green Supermarket" logo).
    """
    text = item.get("text", "").strip()
    y1 = item.get("y1", 0)
    
    # Common store header brand words
    known_header_words = {"GREEN", "SUPERMARKET", "GROCERY", "MART", "STORE", "WHOLE", "FOODS", "MARKET"}
    
    # If the text is a single brand word positioned near top 15% of image
    if text.upper() in known_header_words:
        if img_height and y1 < (img_height * 0.15):
            return True
        elif not img_height:
            return True

    return False


def post_process_food_items(items: List[Dict[str, Any]], strip_quantities: bool = True) -> List[Dict[str, Any]]:
    """
    Main post-processor function to clean extracted food item list.
    
    Args:
        items: List of classified item dicts containing 'text', 'confidence', etc.
        strip_quantities: If True, separates quantity digit from item name.
        
    Returns:
        List of post-processed items with 'clean_text' and 'quantity' fields.
    """
    processed = []
    for item in items:
        cleaned_info = clean_item_name(item.get("text", ""))
        
        processed_record = dict(item)
        if strip_quantities:
            processed_record["text"] = cleaned_info["clean_text"]
            processed_record["quantity"] = cleaned_info["quantity"]
            processed_record["raw_text"] = cleaned_info["raw_text"]
            
        processed.append(processed_record)
        
    return processed
