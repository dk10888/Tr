import os
import sys
import json
import time
import fasttext

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "qq_json_results")

# Ground truth dataset
GT_JSON_PATH = os.path.join(BASE_DIR, "combined_korean_receipts_dataset.json")

# Model result paths
NEW_AUG_V1_JSON = os.path.join(RESULTS_DIR, "fasttext_new_aug_v1_results.json")
V23_JSON = os.path.join(RESULTS_DIR, "fasttext_v23_results.json")
V25_JSON = os.path.join(RESULTS_DIR, "fasttext_v25_results.json")
BERT_NEW_AUG_JSON = os.path.join(RESULTS_DIR, "korean_bert_new_aug_v1_results.json")

V30_MODEL_PATH = os.path.join(BASE_DIR, "fastText_v30_korean", "fasttext_korean_food.ftz")
V30_OUT_JSON = os.path.join(RESULTS_DIR, "fasttext_v30_korean_results.json")

UNREC_SNIPPETS = ["8 용", "L 무", "꽃", "히 올", "빠", "물 품", "움", "필들", "궁림", "삼접삼", "김퇴미개", "우사라보", "진계포자", "꿀", "PST)"]

def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_v30_evaluation():
    print(f"📂 Loading Official Ground Truth from: {GT_JSON_PATH}")
    gt_data = load_json_file(GT_JSON_PATH)

    gt_map = {}
    for r in gt_data["receipts"]:
        r_name = r["receipt_name"]
        for item in r["items"]:
            gt_map[(r_name, item["line_no"])] = item["label"].strip()

    print(f"  Total Ground Truth Annotated Lines: {len(gt_map)}")

    print(f"📂 Loading FastText v3.0 model from: {V30_MODEL_PATH}")
    ft_v30 = fasttext.load_model(V30_MODEL_PATH)

    print(f"📂 Loading reference predictions: {NEW_AUG_V1_JSON}")
    ref_data = load_json_file(NEW_AUG_V1_JSON)

    v30_results = json.loads(json.dumps(ref_data))

    total_lines = 0
    food_predicted = 0
    not_food_predicted = 0
    total_time_ms = 0.0

    for r in v30_results["receipts"]:
        for item in r["items"]:
            txt = item["text"].strip()
            total_lines += 1

            t0 = time.perf_counter()
            labels, probs = ft_v30.predict(txt)
            t1 = time.perf_counter()

            ms = (t1 - t0) * 1000.0
            total_time_ms += ms

            pred_lbl = labels[0].replace("__label__", "")
            prob = float(probs[0])

            if pred_lbl == "food":
                food_predicted += 1
            else:
                not_food_predicted += 1

            item["fasttext_v30"] = {
                "label": pred_lbl,
                "confidence": prob,
                "time_ms": ms
            }
            item["label"] = pred_lbl

    with open(V30_OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(v30_results, f, ensure_ascii=False, indent=2)

    avg_ms = total_time_ms / total_lines if total_lines else 0
    print(f"\n✅ Evaluated FastText v3.0 across {total_lines} lines!")
    print(f"   Food Items Predicted     : {food_predicted}")
    print(f"   Not-Food Items Predicted : {not_food_predicted}")
    print(f"   Avg Latency per Line    : {avg_ms:.4f} ms")
    print(f"💾 Saved predictions to     : {V30_OUT_JSON}")

    # Evaluate metric function matching generate_final_benchmark_table.py exactly
    def evaluate_model_json(path, model_key):
        data = load_json_file(path)
        
        tp_full = fp_full = tn_full = fn_full = 0
        tp_clean = fp_clean = tn_clean = fn_clean = 0

        for r in data["receipts"]:
            r_name = r["receipt_name"]
            for item in r["items"]:
                key = (r_name, item["line_no"])
                if key not in gt_map:
                    continue

                gt = gt_map[key]
                
                if model_key in item:
                    pred = item[model_key]["label"]
                elif "label" in item:
                    pred = item["label"]
                else:
                    pred = "not_food"

                txt = item["text"].strip()
                is_unrec = any(u in txt for u in UNREC_SNIPPETS) or (len(txt) <= 2 and not any(k in txt for k in ["밥", "죽", "국", "탕", "면", "차", "술", "떡"]))

                # Full
                if pred == "food" and gt == "food": tp_full += 1
                elif pred == "food" and gt == "not_food": fp_full += 1
                elif pred == "not_food" and gt == "not_food": tn_full += 1
                elif pred == "not_food" and gt == "food": fn_full += 1

                # Clean
                if not is_unrec:
                    if pred == "food" and gt == "food": tp_clean += 1
                    elif pred == "food" and gt == "not_food": fp_clean += 1
                    elif pred == "not_food" and gt == "not_food": tn_clean += 1
                    elif pred == "not_food" and gt == "food": fn_clean += 1

        def calc(tp, fp, tn, fn):
            tot = tp + fp + tn + fn
            acc = (tp + tn) / tot if tot else 0
            pf = tp / (tp + fp) if (tp + fp) else 0
            rf = tp / (tp + fn) if (tp + fn) else 0
            f1f = (2 * pf * rf) / (pf + rf) if (pf + rf) else 0

            pnf = tn / (tn + fn) if (tn + fn) else 0
            rnf = tn / (tn + fp) if (tn + fp) else 0
            f1nf = (2 * pnf * rnf) / (pnf + rnf) if (pnf + rnf) else 0

            macro = (f1f + f1nf) / 2.0
            return tot, tp, fp, tn, fn, acc, pf, rf, f1f, pnf, rnf, f1nf, macro

        return calc(tp_full, fp_full, tn_full, fn_full), calc(tp_clean, fp_clean, tn_clean, fn_clean)

    res_v23_full, res_v23_clean = evaluate_model_json(V23_JSON, "fasttext_v23")
    res_v25_full, res_v25_clean = evaluate_model_json(V25_JSON, "fasttext_v25")
    res_aug_full, res_aug_clean = evaluate_model_json(NEW_AUG_V1_JSON, "fasttext_new_aug_v1")
    res_v30_full, res_v30_clean = evaluate_model_json(V30_OUT_JSON, "fasttext_v30")

    print("\n" + "="*135)
    print(f" 🏆 AUTHORITATIVE BENCHMARK METRICS (GT Evaluated Across All 785 Receipt Items)")
    print("="*135)
    print(f"{'Model Variant':<32} | {'Subset':<10} | {'Acc':<8} | {'Food Prec':<9} | {'Food Rec':<8} | {'Food F1':<8} | {'NotFood F1':<10} | {'Macro F1':<8}")
    print("-" * 135)

    all_evals = [
        ("FastText v23 (Baseline)", "Full 785", res_v23_full),
        ("FastText v23 (Baseline)", "Clean 757", res_v23_clean),
        ("FastText v25 (Baseline)", "Full 785", res_v25_full),
        ("FastText v25 (Baseline)", "Clean 757", res_v25_clean),
        ("FastText new_aug_v1 (54k)", "Full 785", res_aug_full),
        ("FastText new_aug_v1 (54k)", "Clean 757", res_aug_clean),
        ("FastText v3.0 (Family Aug) ★", "Full 785", res_v30_full),
        ("FastText v3.0 (Family Aug) ★", "Clean 757", res_v30_clean),
    ]

    for model_name, sub, (tot, tp, fp, tn, fn, acc, pf, rf, f1f, pnf, rnf, f1nf, mac) in all_evals:
        print(f"{model_name:<32} | {sub:<10} | {acc*100:.1f}%   | {pf*100:.1f}%    | {rf*100:.1f}%   | {f1f*100:.1f}%   | {f1nf*100:.1f}%    | {mac*100:.1f}%")

    print("="*135)

    # Prediction Diff between new_aug_v1 and v3.0
    aug_data = load_json_file(NEW_AUG_V1_JSON)
    v30_data = load_json_file(V30_OUT_JSON)

    fixed = []
    regressed = []

    for r_aug, r_30 in zip(aug_data["receipts"], v30_data["receipts"]):
        r_name = r_aug["receipt_name"]
        for i_aug, i_30 in zip(r_aug["items"], r_30["items"]):
            key = (r_name, i_aug["line_no"])
            if key not in gt_map:
                continue
            gt = gt_map[key]

            p_aug = i_aug.get("label") or (i_aug.get("fasttext_new_aug_v1", {}).get("label"))
            p_30 = i_30.get("label") or (i_30.get("fasttext_v30", {}).get("label"))
            txt = i_aug["text"].strip()

            if p_aug != gt and p_30 == gt:
                fixed.append((txt, gt, p_aug, p_30))
            elif p_aug == gt and p_30 != gt:
                regressed.append((txt, gt, p_aug, p_30))

    print(f"\n🔍 PREDICTION DIFF ANALYSIS (FastText new_aug_v1 → FastText v3.0):")
    print(f"   Lines Fixed     (+) : {len(fixed)}")
    print(f"   Lines Regressed (-) : {len(regressed)}")
    print(f"   Net Change          : {len(fixed) - len(regressed):+d} lines")

    if fixed:
        print("\n✨ Fixed Lines in FastText v3.0:")
        for txt, gt, old_p, new_p in fixed[:15]:
            print(f"   • [{txt}] (GT: {gt}) | Old: {old_p} → New: {new_p}")

    if regressed:
        print("\n⚠️ Regressed Lines in FastText v3.0:")
        for txt, gt, old_p, new_p in regressed[:15]:
            print(f"   • [{txt}] (GT: {gt}) | Old: {old_p} → New: {new_p}")

if __name__ == "__main__":
    run_v30_evaluation()
