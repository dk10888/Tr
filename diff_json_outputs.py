import json

with open("qq_json_results/fasttext_new_aug_v1_results.json", "r", encoding="utf-8") as f:
    aug_v1 = json.load(f)

with open("qq_json_results/fasttext_v30_korean_results.json", "r", encoding="utf-8") as f:
    v30 = json.load(f)

print("=== PROOF OF MODEL EXECUTION: LINE-BY-LINE DIFF (new_aug_v1 vs v3.0) ===")
print(f"{'Receipt Line Text':<35} | {'new_aug_v1 Label (Conf)':<30} | {'v3.0 Label (Conf)':<30}")
print("-" * 100)

diff_count = 0
for r_aug, r_v30 in zip(aug_v1["receipts"], v30["receipts"]):
    for i_aug, i_v30 in zip(r_aug["items"], r_v30["items"]):
        txt = i_aug["text"]
        lbl_aug = f"{i_aug['label']} ({i_aug['confidence']:.4f})"
        lbl_v30 = f"{i_v30['label']} ({i_v30['confidence']:.4f})"

        if i_aug['label'] != i_v30['label'] or abs(i_aug['confidence'] - i_v30['confidence']) > 0.05:
            diff_count += 1
            if diff_count <= 20:
                print(f"{txt[:35]:<35} | {lbl_aug:<30} | {lbl_v30:<30}")

print("-" * 100)
print(f"Total lines with different predictions or confidence scores: {diff_count} / {aug_v1['total_lines']}")
