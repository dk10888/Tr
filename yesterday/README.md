# Receipt Scanner AI Engine - GitHub Master Export (`yesterday/`)

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
