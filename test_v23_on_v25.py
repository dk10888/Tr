import os
import csv
import time
import fasttext
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

def evaluate_v23_on_v25():
    v23_model_path = "fastText_v23/fasttext_korean_food.ftz"
    if not os.path.exists(v23_model_path):
        v23_model_path = "fastText_v23/fasttext_korean_food.bin"

    print(f"📂 Loading FastText v23 Model from: {v23_model_path}")
    model = fasttext.load_model(v23_model_path)

    v25_csv = "korean_receipt_dataset_v25.csv"
    v23_csv = "korean_receipt_dataset_v23.csv"

    # Load v23 texts to identify brand new samples in v25
    v23_texts = set()
    with open(v23_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            t = r.get("text", "").strip()
            if t: v23_texts.add(t)

    # Load full v25 dataset
    v25_texts = []
    v25_labels = []
    
    new_v25_texts = []
    new_v25_labels = []

    with open(v25_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            t = r.get("text", "").strip()
            l = r.get("label", "").strip()
            if t and l in ["food", "not_food"]:
                v25_texts.append(t)
                v25_labels.append(l)

                if t not in v23_texts:
                    new_v25_texts.append(t)
                    new_v25_labels.append(l)

    print(f"\n📊 Evaluation Breakdown:")
    print(f"  Total v25 Dataset Samples: {len(v25_texts)}")
    print(f"  Brand New Samples added in v25 (not in v23): {len(new_v25_texts)}")

    # 1. Evaluate on FULL v25 Dataset
    print("\n----------------------------------------------------------------------")
    print("1️⃣ EVALUATING v23 MODEL ON FULL v25 DATASET (74,623 samples)")
    print("----------------------------------------------------------------------")
    start_time = time.time()
    preds_full = []
    for t in v25_texts:
        lbls, _ = model.predict(t.replace("\n", " "))
        pred = lbls[0].replace("__label__", "") if lbls else "not_food"
        preds_full.append(pred)

    elapsed_full = time.time() - start_time
    acc_full = accuracy_score(v25_labels, preds_full)
    p_full, r_full, f1_full, _ = precision_recall_fscore_support(v25_labels, preds_full, average='binary', pos_label='food')

    print(f"  Full v25 Accuracy   : {acc_full * 100:.2f}%")
    print(f"  Food Precision      : {p_full * 100:.2f}%")
    print(f"  Food Recall         : {r_full * 100:.2f}%")
    print(f"  Food F1-Score       : {f1_full * 100:.2f}%")
    print(f"  Total Time          : {elapsed_full:.3f} seconds ({elapsed_full*1000000/len(v25_texts):.2f} μs/item)")
    print("\nFull v25 Classification Report:")
    print(classification_report(v25_labels, preds_full, digits=4))

    # 2. Evaluate on NEW v25 Samples Only (Out-Of-Distribution Generalization Test)
    if new_v25_texts:
        print("\n----------------------------------------------------------------------")
        print(f"2️⃣ EVALUATING v23 MODEL ON BRAND NEW v25 SAMPLES ONLY ({len(new_v25_texts)} unseen samples)")
        print("----------------------------------------------------------------------")
        start_time = time.time()
        preds_new = []
        for t in new_v25_texts:
            lbls, _ = model.predict(t.replace("\n", " "))
            pred = lbls[0].replace("__label__", "") if lbls else "not_food"
            preds_new.append(pred)

        elapsed_new = time.time() - start_time
        acc_new = accuracy_score(new_v25_labels, preds_new)
        p_new, r_new, f1_new, _ = precision_recall_fscore_support(new_v25_labels, preds_new, average='binary', pos_label='food')

        print(f"  New v25 Data Accuracy: {acc_new * 100:.2f}%")
        print(f"  Food Precision       : {p_new * 100:.2f}%")
        print(f"  Food Recall          : {r_new * 100:.2f}%")
        print(f"  Food F1-Score        : {f1_new * 100:.2f}%")
        print(f"  Total Time           : {elapsed_new:.3f} seconds")
        print("\nNew v25 Samples Classification Report:")
        print(classification_report(new_v25_labels, preds_new, digits=4))

if __name__ == "__main__":
    evaluate_v23_on_v25()
