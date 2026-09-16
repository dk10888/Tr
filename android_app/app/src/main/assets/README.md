# Android App Assets

Place the following files in this directory before building the app:

## Required Files

| File | Size | Source |
|------|------|--------|
| `english_food_bert.onnx` | ~14 MB | Run `export_bert_to_onnx_android.py` locally |
| `english_vocab.txt` | ~200 KB | Extracted from `bert_tiny_v5_food_final/vocab.txt` |
| `korean_food_classifier_quant.onnx` | ~14 MB | Download from Colab after training |
| `korean_vocab.txt` | ~200 KB | Download from Colab (in the ZIP) |

## How to Get These Files

### English Model (from your existing trained model)
```bash
cd receipt_scanner
python3 export_bert_to_onnx_android.py
# Outputs to android_app/app/src/main/assets/
```

### Korean Model (from Google Colab)
1. Open `colab_korean_bert_train.py` in Google Colab
2. Runtime → GPU (T4)
3. Run all cells
4. Download `korean_food_android_assets.zip`
5. Extract here

## Notes
- ONNX files are automatically excluded from compression (set in build.gradle)
- GPU acceleration via NNAPI is auto-detected at runtime
- Falls back to CPU if NNAPI unavailable (e.g. emulator)
