# 🧾 Receipt Scanner & Parser

On-device receipt OCR, item extraction, and parser powered by PaddleOCR PP-OCRv6 ONNX models and Streamlit.

---

## 📁 Project Structure

```text
receipt_scanner/
├── models/
│   └── onnx/
│       ├── v6_medium_det.onnx       # PP-OCRv6 Medium Text Detection
│       ├── v6_medium_rec.onnx       # PP-OCRv6 Medium Text Recognition
│       ├── rec_keys.txt             # Character dictionary
│       └── inference.yml            # Model configuration
├── receipt_parser/
│   └── regex_parser.py              # Regex-based heuristic parser for receipt fields
├── Receipts/                        # COCO-annotated receipts dataset (train/valid/test)
├── data/                            # Extracted items, labeled data & checkpoints
├── ui/
│   └── app.py                       # Streamlit web interface with layout reconstruction
├── extract_items.py                 # Dataset item extractor with resume checkpointing
├── extract_training_data.py         # Full bounding box training data extractor
├── requirements.txt                 # Project dependencies (Windows & Unix compatible)
└── README.md
```

---

## 🚀 Setup Instructions (Windows)

### 1. Prerequisites
- **Python 3.9, 3.10, or 3.11** installed. (Check "Add Python to PATH" during installation).
- Git installed.

### 2. Clone & Setup Virtual Environment
Open **PowerShell** or **Command Prompt**:

```cmd
git clone https://github.com/dk10888/Tr.git
cd Tr
```

Create and activate a virtual environment:

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt `cmd`):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### 3. Install Dependencies
```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 💻 Running the Application

### Launch the Streamlit Web UI:
```cmd
streamlit run ui/app.py
```
Open your browser at `http://localhost:8501` to upload receipt images, view OCR extractions, download text, and export JSON results.

### Run Item Extraction on Dataset:
```cmd
python extract_items.py
```
This processes the receipt dataset, extracting item texts with auto-checkpointing (`data/extraction_checkpoint.jsonl`) and generates `data/all_items.txt`.
