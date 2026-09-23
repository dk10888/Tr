import json

with open("combined_korean_receipts_dataset.json", "r", encoding="utf-8") as f:
    d = json.load(f)

print("--- Inspecting GT Labels in combined_korean_receipts_dataset.json ---")
for r in d["receipts"]:
    for item in r["items"]:
        txt = item["text"]
        lbl = item["label"]
        if any(term in txt for term in ["LPOINI", "EATZ", "사무니거", "완료되면", "대기런호가", "모장유무", "오직 맛"]):
            print(f"Receipt {r['receipt_id']} ({r['receipt_name']}) Line {item['line_no']}: '{txt}' -> GT Label: {lbl}")
