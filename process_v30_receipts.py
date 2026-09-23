import os
import sys
import json
import time
import fasttext

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "qq_json_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

REF_JSON_PATH = os.path.join(RESULTS_DIR, "fasttext_new_aug_v1_results.json")
V30_MODEL_PATH = os.path.join(BASE_DIR, "fastText_v30_korean", "fasttext_korean_food.ftz")
OUT_JSON_PATH = os.path.join(RESULTS_DIR, "fasttext_v30_korean_results.json")

def process_v30_receipts():
    print(f"📂 Loading FastText v3.0 model from: {V30_MODEL_PATH}")
    ft_v30 = fasttext.load_model(V30_MODEL_PATH)

    print(f"📂 Loading reference receipt items from: {REF_JSON_PATH}")
    with open(REF_JSON_PATH, "r", encoding="utf-8") as f:
        ref_data = json.load(f)

    total_receipts = len(ref_data["receipts"])
    total_lines_all = 0
    total_food_all = 0
    total_not_food_all = 0

    receipts_out = []

    for r_idx, r in enumerate(ref_data["receipts"]):
        r_id = r["receipt_id"]
        r_name = r["receipt_name"]

        items_out = []
        food_count = 0
        not_food_count = 0

        for item in r["items"]:
            line_no = item["line_no"]
            txt = item["text"].strip()
            total_lines_all += 1

            t0 = time.perf_counter()
            labels, probs = ft_v30.predict(txt)
            t1 = time.perf_counter()

            ms = round((t1 - t0) * 1000.0, 4)
            pred_lbl = labels[0].replace("__label__", "")
            prob = round(float(probs[0]), 4)

            if pred_lbl == "food":
                food_count += 1
                total_food_all += 1
            else:
                not_food_count += 1
                total_not_food_all += 1

            items_out.append({
                "line_no": line_no,
                "text": txt,
                "label": pred_lbl,
                "confidence": prob,
                "time_ms": ms
            })

        receipts_out.append({
            "receipt_id": r_id,
            "receipt_name": r_name,
            "line_count": len(items_out),
            "food_count": food_count,
            "not_food_count": not_food_count,
            "items": items_out
        })

    final_json = {
        "model": "FastText v3.0 (Korean Family-Based & Compositional Augmentation)",
        "postprocessed": True,
        "total_receipts": total_receipts,
        "total_lines": total_lines_all,
        "total_food_items": total_food_all,
        "total_not_food_items": total_not_food_all,
        "receipts": receipts_out
    }

    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Successfully processed {total_receipts} receipts ({total_lines_all} total lines)!")
    print(f"   Food items predicted     : {total_food_all}")
    print(f"   Not-food items predicted : {total_not_food_all}")
    print(f"💾 Saved JSON result to     : {OUT_JSON_PATH}")

if __name__ == "__main__":
    process_v30_receipts()
