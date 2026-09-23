import os
import sys
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YESTERDAY_DIR = os.path.join(BASE_DIR, "yesterday")

def build_structure():
    print(f"🚀 Building clean GitHub export directory: {YESTERDAY_DIR}")

    # Subdirectories
    dirs = [
        os.path.join(YESTERDAY_DIR, "models", "fasttext_v23"),
        os.path.join(YESTERDAY_DIR, "models", "fasttext_v25"),
        os.path.join(YESTERDAY_DIR, "models", "korean_bert_v25"),
        os.path.join(YESTERDAY_DIR, "models", "fasttext_new_aug_v1"),
        os.path.join(YESTERDAY_DIR, "models", "korean_bert_new_aug_v1"),
        os.path.join(YESTERDAY_DIR, "models", "fasttext_v30_korean"),
        os.path.join(YESTERDAY_DIR, "json_results"),
        os.path.join(YESTERDAY_DIR, "datasets"),
        os.path.join(YESTERDAY_DIR, "ocr_and_helpers"),
        os.path.join(YESTERDAY_DIR, "documentation")
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)

    def copy_file(src, dst):
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✓ Copied: {os.path.basename(src)} → {dst}")
        else:
            print(f"  ⚠️ Warning: Source file not found: {src}")

    # 1. Models
    print("\n1. Copying ONNX & FTZ Models...")
    copy_file(os.path.join(BASE_DIR, "fastText_v23", "fasttext_korean_food.ftz"),
              os.path.join(YESTERDAY_DIR, "models", "fasttext_v23", "fasttext_v23.ftz"))
    
    copy_file(os.path.join(BASE_DIR, "fastText_v25", "fasttext_korean_food.ftz"),
              os.path.join(YESTERDAY_DIR, "models", "fasttext_v25", "fasttext_v25.ftz"))

    copy_file(os.path.join(BASE_DIR, "korean_new", "korean_food_classifier_quant.onnx"),
              os.path.join(YESTERDAY_DIR, "models", "korean_bert_v25", "korean_food_classifier_v25_quant.onnx"))

    copy_file(os.path.join(BASE_DIR, "fastText_new_aug_v1", "fasttext_korean_food.ftz"),
              os.path.join(YESTERDAY_DIR, "models", "fasttext_new_aug_v1", "fasttext_new_aug_v1.ftz"))

    copy_file(os.path.join(BASE_DIR, "korean_new_augv1", "korean_food_classifier_quant.onnx"),
              os.path.join(YESTERDAY_DIR, "models", "korean_bert_new_aug_v1", "korean_food_classifier_new_aug_v1_quant.onnx"))

    copy_file(os.path.join(BASE_DIR, "fastText_v30_korean", "fasttext_korean_food.ftz"),
              os.path.join(YESTERDAY_DIR, "models", "fasttext_v30_korean", "fasttext_v30_korean.ftz"))

    # 2. JSON Results
    print("\n2. Copying Master JSON Prediction Files...")
    json_files = [
        "fasttext_v23_results.json",
        "fasttext_v25_results.json",
        "korean_bert_onnx_results.json",
        "fasttext_new_aug_v1_results.json",
        "korean_bert_new_aug_v1_results.json",
        "fasttext_v30_korean_results.json"
    ]
    for jf in json_files:
        copy_file(os.path.join(BASE_DIR, "qq_json_results", jf),
                  os.path.join(YESTERDAY_DIR, "json_results", jf))

    # 3. Datasets
    print("\n3. Copying Dataset Files...")
    datasets = [
        "korean_receipt_dataset_v23.csv",
        "korean_receipt_dataset_v25.csv",
        "korean_receipt_dataset_new_aug_v1.csv",
        "korean_receipt_dataset_v30_korean_focused.csv",
        "combined_korean_receipts_dataset.json",
        "receipt_model_comparison_full.csv"
    ]
    for ds in datasets:
        copy_file(os.path.join(BASE_DIR, ds),
                  os.path.join(YESTERDAY_DIR, "datasets", ds))

    # 4. OCR & Helper Scripts
    print("\n4. Copying OCR & Helper Scripts...")
    copy_file(os.path.join(BASE_DIR, "ocr_post_processor.py"),
              os.path.join(YESTERDAY_DIR, "ocr_and_helpers", "ocr_post_processor.py"))
    
    kt_path = os.path.join(BASE_DIR, "android_app", "app", "src", "main", "java", "com", "example", "receiptscanner", "ml", "MlKitOcrHelper.kt")
    copy_file(kt_path, os.path.join(YESTERDAY_DIR, "ocr_and_helpers", "MlKitOcrHelper.kt"))

    # 5. Documentation
    print("\n5. Copying Word Report & Documentation...")
    copy_file(os.path.join(BASE_DIR, "Receipt_Scanner_AI_Comprehensive_Benchmark_Report.docx"),
              os.path.join(YESTERDAY_DIR, "documentation", "Receipt_Scanner_AI_Comprehensive_Benchmark_Report.docx"))

    # Write README.md
    readme_path = os.path.join(YESTERDAY_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("""# Receipt Scanner AI Engine - GitHub Master Export (`yesterday/`)

This directory contains the complete artifact bundle for the **Korean Receipt Item Classification Engine**, including converted ONNX models for mobile Android deployment, evaluation JSONs, synthetic training datasets, OCR post-processing helpers, Kotlin MLKit integrations, and Word Document reports.

---

## Directory Sitemap

```
yesterday/
├── models/                         # All 6 Trained Models (ONNX + FTZ)
│   ├── fasttext_v23/
│   │   ├── fasttext_v23.onnx       # Converted ONNX Model (Android compatible)
│   │   └── fasttext_v23.ftz        # Compressed FastText Model
│   ├── fasttext_v25/
│   │   ├── fasttext_v25.onnx
│   │   └── fasttext_v25.ftz
│   ├── korean_bert_v25/
│   │   └── korean_food_classifier_v25_quant.onnx # Quantized ONNX BERT (14.6 MB)
│   ├── fasttext_new_aug_v1/
│   │   ├── fasttext_new_aug_v1.onnx
│   │   └── fasttext_new_aug_v1.ftz
│   ├── korean_bert_new_aug_v1/
│   │   └── korean_food_classifier_new_aug_v1_quant.onnx # Quantized ONNX BERT (14.6 MB)
│   └── fasttext_v30_korean/
│       ├── fasttext_v30_korean.onnx # Converted ONNX Model
│       └── fasttext_v30_korean.ftz
├── json_results/                   # Evaluation Output JSONs for all 6 models
│   ├── fasttext_v23_results.json
│   ├── fasttext_v25_results.json
│   ├── korean_bert_onnx_results.json
│   ├── fasttext_new_aug_v1_results.json
│   ├── korean_bert_new_aug_v1_results.json
│   └── fasttext_v30_korean_results.json
├── datasets/                       # Datasets & Ground Truth Files
│   ├── korean_receipt_dataset_v23.csv
│   ├── korean_receipt_dataset_v25.csv
│   ├── korean_receipt_dataset_new_aug_v1.csv # 54k Augmented Dataset
│   ├── korean_receipt_dataset_v30_korean_focused.csv # 230k Family Augmented Dataset
│   ├── combined_korean_receipts_dataset.json # Ground Truth Annotations
│   └── receipt_model_comparison_full.csv     # Full Line Benchmark Ground Truth
├── ocr_and_helpers/                # OCR Helpers & Post-Processors
│   ├── ocr_post_processor.py      # Python 3-Rule Post-Processing Pipeline
│   └── MlKitOcrHelper.kt           # Kotlin Android Word-Level BBox Merger
└── documentation/
    └── Receipt_Scanner_AI_Comprehensive_Benchmark_Report.docx # Formatted Word Report
```

---

## Deployment & Usage

### 1. Android ONNX Mobile Deployment
To load any model inside Android:
* Move the desired `.onnx` file (e.g. `yesterday/models/fasttext_v30_korean/fasttext_v30_korean.onnx` or `korean_food_classifier_new_aug_v1_quant.onnx`) to `android_app/app/src/main/assets/`.
* Use ONNX Runtime Android SDK (`com.microsoft.onnxruntime:onnxruntime-android`).

### 2. OCR Post-Processing Pipeline
Apply `ocr_post_processor.py` (or Kotlin `MlKitOcrHelper.kt`) prior to classification:
1. **Single-Character Noise Filtering**: Drops isolated 1-character OCR noise unless on the food whitelist.
2. **Price Digit Repair**: Fixes `0OO` -> `0,000` and `13,5UU` -> `13,500`.
3. **Bounding Box Merging**: Merges horizontally aligned text boxes on the same Y-axis row.
""")
    print(f"  ✓ Created: {readme_path}")
    print("\n🎉 Master export complete! All files organized inside `yesterday/` directory.")

if __name__ == "__main__":
    build_structure()
