# Receipt Scanner AI Engine - GitHub Master Export (`yesterday/`)

This directory contains the complete artifact bundle for the **Korean Receipt Item Classification Engine**, including converted FastText ONNX models for mobile Android deployment, evaluation JSONs, synthetic training datasets, OCR post-processing helpers, Kotlin MLKit & ONNX integrations, and Word Document reports.

---

## Directory Sitemap

```
yesterday/
├── models/                         # All 8 Trained Mobile Models (ONNX Format)
│   ├── fasttext_v23/
│   │   ├── fasttext_v23_android.onnx # Converted FastText ONNX Model (8.7 MB)
│   │   └── fasttext_vocab.json     # FastText Vocab + FNV Hash Map
│   ├── fasttext_v25/
│   │   ├── fasttext_v25_android.onnx (11.0 MB)
│   │   └── fasttext_vocab.json
│   ├── korean_bert_v25/
│   │   └── korean_food_classifier_v25_quant.onnx # Quantized ONNX BERT (14.6 MB)
│   ├── fasttext_new_aug_v1/
│   │   ├── fasttext_new_aug_v1_android.onnx (7.7 MB)
│   │   └── fasttext_vocab.json
│   ├── korean_bert_new_aug_v1/
│   │   └── korean_food_classifier_new_aug_v1_quant.onnx # Quantized ONNX BERT (14.6 MB)
│   ├── fasttext_v30_korean/
│   │   ├── fasttext_v30_android.onnx (8.3 MB)
│   │   └── fasttext_vocab.json
│   ├── fasttext_v40_merged/
│   │   ├── fasttext_v40_android.onnx (8.1 MB)
│   │   └── fasttext_vocab.json
│   └── fasttext_v50_final/
│       ├── fasttext_v50_android.onnx (8.8 MB - Best Mobile Model)
│       └── fasttext_vocab.json
├── json_results/                   # Evaluation Output JSONs for all models
│   ├── fasttext_v23_results.json
│   ├── fasttext_v25_results.json
│   ├── korean_bert_onnx_results.json
│   ├── fasttext_new_aug_v1_results.json
│   ├── korean_bert_new_aug_v1_results.json
│   ├── fasttext_v30_korean_results.json
│   ├── fasttext_v40_merged_results.json
│   └── fasttext_v50_final_results.json
├── datasets/                       # Datasets & Ground Truth Files
│   ├── korean_receipt_dataset_v23.csv
│   ├── korean_receipt_dataset_v25.csv
│   ├── korean_receipt_dataset_new_aug_v1.csv # 54k Augmented Dataset
│   ├── korean_receipt_dataset_v30_korean_focused.csv # 230k Family Augmented Dataset
│   ├── korean_receipt_dataset_v40_merged.csv # 304k Merged Dataset
│   ├── korean_receipt_dataset_v50_final.csv # 4.5M Final Dataset
│   ├── combined_korean_receipts_dataset.json # Ground Truth Annotations
│   └── receipt_model_comparison_full.csv     # Full Line Benchmark Ground Truth
├── ocr_and_helpers/                # OCR Helpers & Post-Processors
│   ├── ocr_post_processor.py      # Python 3-Rule Post-Processing Pipeline
│   ├── MlKitOcrHelper.kt           # Kotlin Android Word-Level BBox Merger
│   └── FastTextOnnxClassifier.kt   # Kotlin Native FNV-Hash FastText ONNX Classifier
└── documentation/
    └── Receipt_Scanner_AI_Comprehensive_Benchmark_Report.docx # Formatted Word Report
```

---

## Android Mobile Deployment

### FastText ONNX Mobile Inference
1. Copy `fasttext_v50_android.onnx` and `fasttext_vocab.json` into `android_app/app/src/main/assets/`.
2. Use `FastTextOnnxClassifier.kt` in Kotlin:
   - Evaluates subword n-grams using signed FNV-1a hashing matching C++ FastText specs.
   - Embeds into ONNX Runtime `com.microsoft.onnxruntime:onnxruntime-android`.
   - Runs inference in ~2ms per receipt line with 0 dependencies on legacy JNI `.ftz` libraries.
