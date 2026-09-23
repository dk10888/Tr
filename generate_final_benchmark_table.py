import json
import os

# 1. Load Ground Truth Annotations
with open("combined_korean_receipts_dataset.json", "r", encoding="utf-8") as f:
    gt_data = json.load(f)

gt_map = {}
for r in gt_data["receipts"]:
    r_name = r["receipt_name"]
    for item in r["items"]:
        gt_map[(r_name, item["line_no"])] = item["label"].strip()

# 2. Function to evaluate a model JSON result
def evaluate_model_file(path, model_name):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Full set
    tp_full = fp_full = tn_full = fn_full = 0
    
    # Clean set (excluding unrecognizable noise / 1-char single noise)
    tp_clean = fp_clean = tn_clean = fn_clean = 0

    unrec_snippets = ["8 용", "L 무", "꽃", "히 올", "빠", "물 품", "움", "필들", "궁림", "삼접삼", "김퇴미개", "우사라보", "진계포자", "꿀", "PST)"]

    for r in data["receipts"]:
        r_name = r["receipt_name"]
        for item in r["items"]:
            key = (r_name, item["line_no"])
            if key not in gt_map:
                continue

            gt = gt_map[key]
            
            # Predict label: handle different JSON formats
            if "label" in item:
                pred = item["label"]
            elif "fasttext_v25" in item:
                pred = item["fasttext_v25"]["label"]
            elif "bert_korean_onnx" in item:
                pred = item["bert_korean_onnx"]["label"]
            else:
                pred = "not_food"

            txt = item["text"].strip()
            is_unrec = any(u in txt for u in unrec_snippets) or (len(txt) <= 2 and not any(k in txt for k in ["밥", "죽", "국", "탕", "면", "차", "술", "떡"]))

            # Full set stats
            if pred == "food" and gt == "food": tp_full += 1
            elif pred == "food" and gt == "not_food": fp_full += 1
            elif pred == "not_food" and gt == "not_food": tn_full += 1
            elif pred == "not_food" and gt == "food": fn_full += 1

            # Clean set stats
            if not is_unrec:
                if pred == "food" and gt == "food": tp_clean += 1
                elif pred == "food" and gt == "not_food": fp_clean += 1
                elif pred == "not_food" and gt == "not_food": tn_clean += 1
                elif pred == "not_food" and gt == "food": fn_clean += 1

    def calc(tp, fp, tn, fn):
        tot = tp + fp + tn + fn
        acc = (tp + tn) / tot if tot else 0
        p_f = tp / (tp + fp) if (tp + fp) else 0
        r_f = tp / (tp + fn) if (tp + fn) else 0
        f1_f = (2 * p_f * r_f) / (p_f + r_f) if (p_f + r_f) else 0

        p_nf = tn / (tn + fn) if (tn + fn) else 0
        r_nf = tn / (tn + fp) if (tn + fp) else 0
        f1_nf = (2 * p_nf * r_nf) / (p_nf + r_nf) if (p_nf + r_nf) else 0

        macro = (f1_f + f1_nf) / 2.0
        return tot, tp, fp, tn, fn, acc, p_f, r_f, f1_f, p_nf, r_nf, f1_nf, macro

    return {
        "model": model_name,
        "full": calc(tp_full, fp_full, tn_full, fn_full),
        "clean": calc(tp_clean, fp_clean, tn_clean, fn_clean)
    }

models_to_eval = [
    ("qq_json_results/fasttext_v23_results.json", "FastText v23"),
    ("qq_json_results/fasttext_v25_results.json", "FastText v25"),
    ("qq_json_results/korean_bert_onnx_results.json", "Korean BERT (KoELECTRA v25)"),
    ("qq_json_results/fasttext_clean_core_10k_results.json", "FastText Clean Core (~10.4k)")
]

results = [evaluate_model_file(p, m) for p, m in models_to_eval]

print("\n" + "="*140)
print(f"{'Model':<30} | {'Subset':<10} | {'Total':<6} | {'TP':<4} | {'FP':<4} | {'TN':<4} | {'FN':<4} | {'Accuracy':<9} | {'Food Prec':<10} | {'Food Rec':<9} | {'Food F1':<8} | {'NotFood Prec':<12} | {'NotFood Rec':<12} | {'Macro F1':<8}")
print("="*140)

for r in results:
    m_name = r["model"]
    tot_f, tp_f, fp_f, tn_f, fn_f, acc_f, pf_f, rf_f, f1f_f, pnf_f, rnf_f, f1nf_f, mac_f = r["full"]
    tot_c, tp_c, fp_c, tn_c, fn_c, acc_c, pf_c, rf_c, f1f_c, pnf_c, rnf_c, f1nf_c, mac_c = r["clean"]

    print(f"{m_name:<30} | {'full_lines':<10} | {tot_f:<6} | {tp_f:<4} | {fp_f:<4} | {tn_f:<4} | {fn_f:<4} | {acc_f*100:.2f}%   | {pf_f*100:.2f}%    | {rf_f*100:.2f}%   | {f1f_f*100:.2f}%   | {pnf_f*100:.2f}%      | {rnf_f*100:.2f}%      | {mac_f*100:.2f}%")
    print(f"{m_name:<30} | {'clean_lines':<10} | {tot_c:<6} | {tp_c:<4} | {fp_c:<4} | {tn_c:<4} | {fn_c:<4} | {acc_c*100:.2f}%   | {pf_c*100:.2f}%    | {rf_c*100:.2f}%   | {f1f_c*100:.2f}%   | {pnf_c*100:.2f}%      | {rnf_c*100:.2f}%      | {mac_c*100:.2f}%")
    print("-" * 140)
