import re
from dataclasses import dataclass, field
from typing import Optional

# ──────────────────────────────────────────────
# Data Model
# ──────────────────────────────────────────────

@dataclass
class ReceiptItem:
    name: str          # We will just put the raw text here
    raw_line: str
    quantity: Optional[str] = None
    unit: Optional[str] = None
    price: Optional[float] = None
    is_food: bool = True
    confidence: float = 1.0

    def to_dict(self):
        return {
            "name": self.name,
            "quantity": self.quantity,
            "unit": self.unit,
            "price": self.price,
            "is_food": self.is_food,
            "confidence": round(self.confidence, 3),
        }

@dataclass
class Receipt:
    store_name: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    items: list[ReceiptItem] = field(default_factory=list)
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None

    def to_dict(self):
        return {
            "store_name": self.store_name,
            "date": self.date,
            "time": self.time,
            "items": [item.to_dict() for item in self.items],
            "subtotal": self.subtotal,
            "tax": self.tax,
            "total": self.total
        }

# ──────────────────────────────────────────────
# Core Parser (NO REGEX FILTERING)
# ──────────────────────────────────────────────

class ReceiptParser:
    def parse_receipt(self, lines: list, confidences: list = None) -> Receipt:
        """
        Dumb parser: Keeps EVERYTHING. No regex filtering, no skipped lines.
        Outputs every line as an item so nothing is lost.
        """
        if confidences is None:
            confidences = [1.0] * len(lines)

        receipt = Receipt()
        
        if lines:
            receipt.store_name = lines[0].strip() # Assume first line is store

        for line, conf in zip(lines, confidences):
            raw = line.strip()

            if not raw:
                continue

            # We just add EVERYTHING to the items list so you don't lose any data
            receipt.items.append(ReceiptItem(
                name=raw,           # Keep the exact raw text
                raw_line=raw,
                confidence=conf,
            ))

        return receipt
