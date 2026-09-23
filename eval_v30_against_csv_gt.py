import os
import csv
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_GT_PATH = os.path.join(BASE_DIR, "receipt_model_comparison_full.csv")
V30_JSON_PATH = os.path.join(BASE_DIR, "qq_json_results", "fasttext_v30_korean_results.json")
AUG1_JSON_PATH = os.path.join(BASE_DIR, "qq_json_results", "fasttext_new_aug_v1_results.json")
V23_JSON_PATH = os.path.join(BASE_DIR, "qq_json_results", "fasttext_v23_results.json")

def load_csv_gt():
    gt_rows = []
    with open(CSV_GT_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rec_id_str = r.get("receipt_id") or list(r.values())[0]
            gt_rows.append({
                "receipt_id": int(rec_id_str),
                "receipt_name": r["receipt_name"].strip(),
                "line_no": int(r["line_no"]),
                "text": r["text"].strip(),
                "ground_truth": r["ground_truth"].strip(),
                "is_unrecognizable_ocr": r["is_unrecognizable_ocr"].strip().lower() == "true"
            })
    return gt_rows

def load_json_predictions(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    pred_map = {}
    for r in data["receipts"]:
        r_name = r["receipt_name"]
        for item in r["items"]:
            key = (r_name, item["line_no"])
            pred_map[key] = {
                "label": item["label"],
                "text": item["text"].strip(),
                "confidence": item.get("confidence", 0.0),
                "time_ms": item.get("time_ms", 0.0)
            }
    return pred_map

def compute_detailed_metrics(gt_rows, pred_map, model_name, clean_only=False):
    tp = fp = tn = fn = 0
    skipped = 0
    latencies = []

    for row in gt_rows:
        if clean_only and row["is_unrecognizable_ocr"]:
            continue

        key = (row["receipt_name"], row["line_no"])
        if key not in pred_map:
            # Fallback search by text match in same receipt
            match = None
            for (rn, ln), val in pred_map.items():
                if rn == row["receipt_name"] and val["text"] == row["text"]:
                    match = val
                    break
            if not match:
                skipped += 1
                continue
            pred_val = match
        else:
            pred_val = pred_map[key]

        gt = row["ground_truth"]
        pred = pred_val["label"]
        latencies.append(pred_val.get("time_ms", 0.0))

        if pred == "food" and gt == "food":
            tp += 1
        elif pred == "food" and gt == "not_food":
            fp += 1
        elif pred == "not_food" and gt == "not_food":
            tn += 1
        elif pred == "not_food" and gt == "food":
            fn += 1

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0
    
    food_prec = tp / (tp + fp) if (tp + fp) else 0
    food_rec = tp / (tp + fn) if (tp + fn) else 0
    food_f1 = (2 * food_prec * food_rec) / (food_prec + food_rec) if (food_prec + food_rec) else 0

    notfood_prec = tn / (tn + fn) if (tn + fn) else 0
    notfood_rec = tn / (tn + fp) if (tn + fp) else 0
    notfood_f1 = (2 * notfood_prec * notfood_rec) / (notfood_prec + notfood_rec) if (notfood_prec + notfood_rec) else 0

    macro_f1 = (food_f1 + notfood_f1) / 2.0
    balanced_acc = (food_rec + notfood_rec) / 2.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "model": model_name,
        "subset": "Clean (Understandable OCR Only)" if clean_only else "All Lines (Full Set)",
        "total": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": acc,
        "food_prec": food_prec, "food_rec": food_rec, "food_f1": food_f1,
        "notfood_prec": notfood_prec, "notfood_rec": notfood_rec, "notfood_f1": notfood_f1,
        "macro_f1": macro_f1,
        "balanced_acc": balanced_acc,
        "avg_latency": avg_latency
    }

def main():
    print(f"📂 Reading Ground Truth CSV: {CSV_GT_PATH}")
    gt_rows = load_csv_gt()
    print(f"   Total Ground Truth Rows in CSV: {len(gt_rows)}")
    unrec_count = sum(1 for r in gt_rows if r["is_unrecognizable_ocr"])
    clean_count = len(gt_rows) - unrec_count
    print(f"   Understandable OCR Lines: {clean_count} | Unrecognizable OCR Lines: {unrec_count}")

    print("\n📂 Loading Model Predictions...")
    v30_preds = load_json_predictions(V30_JSON_PATH)
    aug1_preds = load_json_predictions(AUG1_JSON_PATH)
    v23_preds = load_json_predictions(V23_JSON_PATH)

    models = [
        ("FastText v3.0 (Family Aug) ★", v30_preds),
        ("FastText new_aug_v1 (54k)", aug1_preds),
        ("FastText v23 (Baseline)", v23_preds),
    ]

    all_results = []
    for name, p_map in models:
        m_full = compute_detailed_metrics(gt_rows, p_map, name, clean_only=False)
        m_clean = compute_detailed_metrics(gt_rows, p_map, name, clean_only=True)
        all_results.append(m_full)
        all_results.append(m_clean)

    print("\n" + "=" * 165)
    print(f" 📊 ACCURACY & METRICS DIRECTLY EVALUATED AGAINST receipt_model_comparison_full.csv GROUND TRUTH")
    print("=" * 165)
    print(f"{'Model Variant':<32} | {'Subset':<32} | {'Total':<6} | {'TP':<4} | {'FP':<4} | {'TN':<4} | {'FN':<4} | {'Accuracy':<9} | {'Food Prec':<10} | {'Food Rec':<9} | {'Food F1':<8} | {'Macro F1':<8}")
    print("-" * 165)

    for r in all_results:
        print(f"{r['model']:<32} | {r['subset']:<32} | {r['total']:<6} | {r['tp']:<4} | {r['fp']:<4} | {r['tn']:<4} | {r['fn']:<4} | {r['accuracy']*100:.2f}%   | {r['food_prec']*100:.2f}%    | {r['food_rec']*100:.2f}%   | {r['food_f1']*100:.2f}%   | {r['macro_f1']*100:.2f}%")

    print("=" * 165)

    # Detailed report breakdown for FastText v3.0
    v30_full = [r for r in all_results if r['model'] == "FastText v3.0 (Family Aug) ★" and r['subset'] == "All Lines (Full Set)"][0]
    v30_clean = [r for r in all_results if r['model'] == "FastText v3.0 (Family Aug) ★" and "Clean" in r['subset']][0]

    print("\n📌 DETAILED METRICS FOR FASTTEXT V3.0:")
    print("---------------------------------------------------------")
    print(f"1. ALL LINES (Full Set: {v30_full['total']} lines):")
    print(f"   • Accuracy               : {v30_full['accuracy']*100:.2f}%")
    print(f"   • Food Precision         : {v30_full['food_prec']*100:.2f}%")
    print(f"   • Food Recall            : {v30_full['food_rec']*100:.2f}%")
    print(f"   • Food F1 Score          : {v30_full['food_f1']*100:.2f}%")
    print(f"   • Not-Food Precision     : {v30_full['notfood_prec']*100:.2f}%")
    print(f"   • Not-Food Recall        : {v30_full['notfood_rec']*100:.2f}%")
    print(f"   • Not-Food F1 Score      : {v30_full['notfood_f1']*100:.2f}%")
    print(f"   • Macro-F1               : {v30_full['macro_f1']*100:.2f}%")
    print(f"   • Balanced Accuracy      : {v30_full['balanced_acc']*100:.2f}%")
    print(f"   • Confusion Matrix       : TP={v30_full['tp']}, FP={v30_full['fp']}, TN={v30_full['tn']}, FN={v30_full['fn']}")

    print("\n2. UNDERSTANDABLE OCR LINES ONLY (Clean Set: {v30_clean['total']} lines):")
    print(f"   • Accuracy               : {v30_clean['accuracy']*100:.2f}%")
    print(f"   • Food Precision         : {v30_clean['food_prec']*100:.2f}%")
    print(f"   • Food Recall            : {v30_clean['food_rec']*100:.2f}%")
    print(f"   • Food F1 Score          : {v30_clean['food_f1']*100:.2f}%")
    print(f"   • Not-Food Precision     : {v30_clean['notfood_prec']*100:.2f}%")
    print(f"   • Not-Food Recall        : {v30_clean['notfood_rec']*100:.2f}%")
    print(f"   • Not-Food F1 Score      : {v30_clean['notfood_f1']*100:.2f}%")
    print(f"   • Macro-F1               : {v30_clean['macro_f1']*100:.2f}%")
    print(f"   • Balanced Accuracy      : {v30_clean['balanced_acc']*100:.2f}%")
    print(f"   • Confusion Matrix       : TP={v30_clean['tp']}, FP={v30_clean['fp']}, TN={v30_clean['tn']}, FN={v30_clean['fn']}")

if __name__ == "__main__":
    main()
