"""
╔══════════════════════════════════════════════════════════════════════╗
║  Korean Receipt Food Classifier — Colab Training Script              ║
║  Model : monologg/koelectra-small-v3-discriminator (~14M params)     ║
║  Output: Korean BERT ONNX + quantized for Android ONNX Runtime       ║
╚══════════════════════════════════════════════════════════════════════╝

HOW TO USE IN COLAB:
  1. Runtime → Change runtime type → GPU (T4)
  2. Run all cells top-to-bottom
  3. Final cell downloads:
       - korean_food_classifier.onnx        (~55 MB FP32)
       - korean_food_classifier_quant.onnx  (~14 MB INT8 dynamic)
       - korean_vocab.txt                   (vocab for Android tokenizer)
       - korean_special_tokens.json         (special token IDs)
"""

# ─────────────────────────────────────────────────────────────────────
# CELL 1 — Install dependencies
# ─────────────────────────────────────────────────────────────────────
# %%
# !pip install -q transformers==4.40.0 datasets==2.19.0 torch==2.2.2 \
#              onnx==1.16.0 onnxruntime==1.18.0 onnxruntime-tools \
#              scikit-learn==1.4.2 evaluate==0.4.1 accelerate==0.29.3

# ─────────────────────────────────────────────────────────────────────
# CELL 2 — Mount Google Drive (to save model)
# ─────────────────────────────────────────────────────────────────────
# %%
# from google.colab import drive
# drive.mount('/content/drive')
# SAVE_DIR = "/content/drive/MyDrive/korean_food_model"

# ─────────────────────────────────────────────────────────────────────
# CELL 3 — Imports & Config
# ─────────────────────────────────────────────────────────────────────
# %%
import os, json, random, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    ElectraForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report, accuracy_score
import onnx
import onnxruntime as ort

# ── Config ────────────────────────────────────────────────────────────
MODEL_NAME    = "monologg/koelectra-small-v3-discriminator"
MAX_LEN       = 64       # receipt lines are short
BATCH_SIZE    = 32
EPOCHS        = 8
LR            = 2e-5
SEED          = 42
OUTPUT_DIR    = "./korean_food_model_output"
ONNX_PATH     = "./korean_food_classifier.onnx"
ONNX_QUANT    = "./korean_food_classifier_quant.onnx"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✅ Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────
# CELL 4 — Synthetic Korean Receipt Dataset
# ─────────────────────────────────────────────────────────────────────
# %%
# Label: 1 = food, 0 = not_food

FOOD_KO = [
    "김치찌개", "된장찌개", "부대찌개", "순두부찌개", "청국장",
    "비빔밥", "돌솥비빔밥", "볶음밥", "김밥", "주먹밥",
    "불고기", "삼겹살", "갈비", "갈비찜", "돼지갈비",
    "냉면", "물냉면", "비빔냉면", "막국수", "잡채",
    "삼계탕", "설렁탕", "곰탕", "육개장", "감자탕",
    "떡볶이", "순대", "어묵", "튀김", "핫도그",
    "라면", "짬뽕", "짜장면", "탕수육", "마파두부",
    "치킨", "후라이드치킨", "양념치킨", "반반치킨", "닭갈비",
    "피자", "파스타", "스파게티", "스테이크", "햄버거",
    "초밥", "회", "연어", "광어회", "참치",
    "된장국", "미역국", "콩나물국", "무국", "북어국",
    "잡채", "호박전", "김치전", "파전", "해물파전",
    "보쌈", "족발", "제육볶음", "오삼불고기", "낙지볶음",
    "갈치구이", "고등어구이", "삼치구이", "조기구이",
    "계란말이", "계란프라이", "스크램블에그",
    "떡국", "만두", "물만두", "군만두", "찐만두",
    "삼겹살 2인분", "소주 1병", "맥주 500cc",
    "아메리카노", "카페라떼", "카푸치노", "아이스티",
    "콜라", "사이다", "오렌지주스", "포도주스",
    "딸기케이크", "초코케이크", "치즈케이크", "티라미수",
    "아이스크림", "팥빙수", "딸기빙수", "망고빙수",
    "순대국밥", "뼈다귀해장국", "콩나물해장국",
    "쌀국수", "쌀떡", "찹쌀떡", "흑임자경단",
    "갈비 1인분", "불고기 1인분", "삼겹살 3인분",
    "라면 곱빼기", "짜장면 2그릇", "짬뽕 1그릇",
    "Americano", "Latte", "Cappuccino", "Green Tea",
    "Burger", "Sandwich", "Pizza", "Pasta", "Steak",
    "Chicken", "Fish & Chips", "Salad", "Soup",
    "Coca Cola", "Orange Juice", "Water",
    "Tiramisu", "Cheesecake", "Chocolate Cake",
    "김치찌개 1", "된장찌개 2", "비빔밥×2", "불고기*1",
    "삼겹살 (200g)", "갈비 (300g)", "냉면 보통",
    "치킨 1마리", "피자 L", "파스타 크림",
    "아메리카노 T", "카페라떼 ICE", "케이크 1조각",
]

NOT_FOOD_KO = [
    "합계", "소계", "총합계", "과세표준", "부가세", "합산금액",
    "결제금액", "청구금액", "총결제", "영수합계",
    "면세금액", "과세금액", "세액", "공급가액",
    "카드결제", "현금결제", "신용카드", "체크카드", "포인트",
    "할인", "쿠폰할인", "멤버십할인", "적립금사용",
    "거스름돈", "잔액", "선불카드", "상품권",
    "영수증번호", "주문번호", "테이블번호", "대기번호",
    "매장명", "사업자번호", "대표자명", "주소",
    "전화번호", "팩스번호", "이메일",
    "영업시간", "휴무일", "주차안내",
    "결제일시", "주문시간", "취소일시", "발행일",
    "2024-01-15", "2024/03/22", "15:30:00",
    "담당직원", "캐셔", "서버", "테이블", "좌석번호",
    "인원수", "방문인원",
    "배달료", "포장비", "서비스료", "봉사료",
    "예약금", "보증금", "추가요금",
    "반품", "환불", "취소", "재발행",
    "적립포인트", "사용포인트", "잔여포인트",
    "스탬프", "쿠폰번호", "이벤트코드",
    "Total", "Subtotal", "Tax", "VAT", "Service Charge",
    "Credit Card", "Cash", "Discount", "Coupon",
    "Receipt No.", "Table No.", "Order No.",
    "Date", "Time", "Staff", "Manager",
    "Tel", "Fax", "Address",
    "1004-0022", "REF: 29301", "**** **** 4521",
    "승인번호 123456", "단말기번호 00001",
    "가맹점번호 987654",
]

def augment_text(text):
    variants = [text]
    for qty in ["1", "2", "x2", "×1", "*2", "1개", "2개"]:
        variants.append(f"{text} {qty}")
    for price in ["12,000", "8,500", "25,000원", "₩15,000"]:
        variants.append(f"{text}  {price}")
    if any(c.isascii() and c.isalpha() for c in text):
        variants.append(text.upper())
        variants.append(text.title())
    return variants[:4]

def build_dataset():
    data = []
    for item in FOOD_KO:
        for v in augment_text(item):
            data.append({"text": v, "label": 1})
    for item in NOT_FOOD_KO:
        for v in augment_text(item):
            data.append({"text": v, "label": 0})
    random.shuffle(data)
    print(f"✅ Dataset: {len(data)} samples "
          f"({sum(d['label']==1 for d in data)} food, "
          f"{sum(d['label']==0 for d in data)} non-food)")
    return data

all_data = build_dataset()

# ─────────────────────────────────────────────────────────────────────
# CELL 5 — Train/Val Split & Dataset Class
# ─────────────────────────────────────────────────────────────────────
# %%
split      = int(0.85 * len(all_data))
train_data = all_data[:split]
val_data   = all_data[split:]
print(f"Train: {len(train_data)} | Val: {len(val_data)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class KoreanReceiptDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        enc = self.tokenizer(
            item["text"], max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get(
                "token_type_ids", torch.zeros(self.max_len, dtype=torch.long)
            ).squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }

train_ds = KoreanReceiptDataset(train_data, tokenizer, MAX_LEN)
val_ds   = KoreanReceiptDataset(val_data,   tokenizer, MAX_LEN)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print("✅ DataLoaders ready")

# ─────────────────────────────────────────────────────────────────────
# CELL 6 — Load Model
# ─────────────────────────────────────────────────────────────────────
# %%
model = ElectraForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "not_food", 1: "food"},
    label2id={"not_food": 0, "food": 1},
)
model = model.to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Loaded {MODEL_NAME} — {total_params/1e6:.1f}M parameters")

# ─────────────────────────────────────────────────────────────────────
# CELL 7 — Training Loop
# ─────────────────────────────────────────────────────────────────────
# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_dl) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)

def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                input_ids      = batch["input_ids"].to(DEVICE),
                attention_mask = batch["attention_mask"].to(DEVICE),
                token_type_ids = batch["token_type_ids"].to(DEVICE),
            )
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].numpy())
    return accuracy_score(all_labels, all_preds), all_preds, all_labels

best_val_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    t0 = time.time()
    for step, batch in enumerate(train_dl):
        optimizer.zero_grad()
        outputs = model(
            input_ids      = batch["input_ids"].to(DEVICE),
            attention_mask = batch["attention_mask"].to(DEVICE),
            token_type_ids = batch["token_type_ids"].to(DEVICE),
            labels         = batch["labels"].to(DEVICE),
        )
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += outputs.loss.item()
        if (step + 1) % 20 == 0:
            print(f"  Epoch {epoch+1} | Step {step+1}/{len(train_dl)} | "
                  f"Loss: {total_loss/(step+1):.4f}", flush=True)

    val_acc, _, _ = evaluate(model, val_dl)
    print(f"✅ Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_dl):.4f} | "
          f"Val Acc: {val_acc*100:.2f}% | Time: {time.time()-t0:.1f}s")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"  ⭐ Best model saved! ({val_acc*100:.2f}%)")

print(f"\n🏆 Best val accuracy: {best_val_acc*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────
# CELL 8 — Classification Report
# ─────────────────────────────────────────────────────────────────────
# %%
best_model = ElectraForSequenceClassification.from_pretrained(OUTPUT_DIR).to(DEVICE)
_, preds, labels = evaluate(best_model, val_dl)
print(classification_report(labels, preds, target_names=["not_food", "food"]))

# ─────────────────────────────────────────────────────────────────────
# CELL 9 — Export to ONNX (FP32)
# ─────────────────────────────────────────────────────────────────────
# %%
best_model.eval().cpu()
dummy = torch.ones(1, MAX_LEN, dtype=torch.long)
torch.onnx.export(
    best_model,
    (dummy, dummy, torch.zeros(1, MAX_LEN, dtype=torch.long)),
    ONNX_PATH,
    input_names  = ["input_ids", "attention_mask", "token_type_ids"],
    output_names = ["logits"],
    dynamic_axes = {
        "input_ids":      {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "token_type_ids": {0: "batch", 1: "seq"},
        "logits":         {0: "batch"},
    },
    opset_version = 14,
    do_constant_folding = True,
)
size_fp32 = os.path.getsize(ONNX_PATH) / 1024 / 1024
print(f"✅ ONNX FP32: {size_fp32:.1f} MB")
onnx.checker.check_model(onnx.load(ONNX_PATH))
print("✅ ONNX check passed")

# ─────────────────────────────────────────────────────────────────────
# CELL 10 — Dynamic INT8 Quantization (~14 MB for Android GPU)
# ─────────────────────────────────────────────────────────────────────
# %%
from onnxruntime.quantization import quantize_dynamic, QuantType

print("⚙️  Dynamic INT8 quantization...")
quantize_dynamic(
    model_input  = ONNX_PATH,
    model_output = ONNX_QUANT,
    weight_type  = QuantType.QInt8,
    extra_options = {"MatMulConstBOnly": True, "EnableSubgraph": True},
)
size_quant = os.path.getsize(ONNX_QUANT) / 1024 / 1024
print(f"✅ ONNX INT8: {size_quant:.1f} MB  (was {size_fp32:.1f} MB, {size_fp32/size_quant:.1f}x smaller)")

# ─────────────────────────────────────────────────────────────────────
# CELL 11 — Accuracy: FP32 vs Quantized
# ─────────────────────────────────────────────────────────────────────
# %%
def run_onnx(onnx_path, val_data):
    session = ort.InferenceSession(onnx_path)
    preds, labels = [], []
    for item in val_data:
        enc = tokenizer(item["text"], max_length=MAX_LEN, padding="max_length",
                        truncation=True, return_tensors="np")
        feeds = {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
            "token_type_ids": enc.get("token_type_ids",
                              np.zeros((1, MAX_LEN), dtype=np.int64)),
        }
        logits = session.run(["logits"], feeds)[0]
        preds.append(int(np.argmax(logits)))
        labels.append(item["label"])
    return accuracy_score(labels, preds)

acc_fp32  = run_onnx(ONNX_PATH,  val_data)
acc_quant = run_onnx(ONNX_QUANT, val_data)
print(f"FP32  ({size_fp32:.1f} MB): {acc_fp32*100:.2f}%")
print(f"INT8  ({size_quant:.1f} MB): {acc_quant*100:.2f}%")
print(f"Drop: {(acc_fp32-acc_quant)*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────
# CELL 12 — Export vocab + special tokens for Android
# ─────────────────────────────────────────────────────────────────────
# %%
import shutil
shutil.copy(os.path.join(OUTPUT_DIR, "vocab.txt"), "./korean_vocab.txt")

special_tokens = {
    "cls_token_id": tokenizer.cls_token_id,
    "sep_token_id": tokenizer.sep_token_id,
    "pad_token_id": tokenizer.pad_token_id,
    "unk_token_id": tokenizer.unk_token_id,
    "max_length":   MAX_LEN,
    "model_name":   MODEL_NAME,
    "labels":       {0: "not_food", 1: "food"},
}
with open("./korean_special_tokens.json", "w", encoding="utf-8") as f:
    json.dump(special_tokens, f, ensure_ascii=False, indent=2)

print("✅ korean_vocab.txt")
print("✅ korean_special_tokens.json")
print(json.dumps(special_tokens, indent=2, ensure_ascii=False))

# ─────────────────────────────────────────────────────────────────────
# CELL 13 — Zip & Download
# ─────────────────────────────────────────────────────────────────────
# %%
import zipfile
zip_path = "./korean_food_android_assets.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for src, arc in [
        (ONNX_QUANT,                   "korean_food_classifier_quant.onnx"),
        (ONNX_PATH,                    "korean_food_classifier.onnx"),
        ("./korean_vocab.txt",          "korean_vocab.txt"),
        ("./korean_special_tokens.json","korean_special_tokens.json"),
    ]:
        if os.path.exists(src):
            zf.write(src, arc)
            print(f"  + {arc} ({os.path.getsize(src)/1024/1024:.1f} MB)")

print(f"\n✅ ZIP: {zip_path} ({os.path.getsize(zip_path)/1024/1024:.1f} MB)")
# Uncomment to download:
# from google.colab import files; files.download(zip_path)
# Or save to Drive:
# shutil.copy(zip_path, "/content/drive/MyDrive/korean_food_model/")

# ─────────────────────────────────────────────────────────────────────
# CELL 14 — Quick inference test
# ─────────────────────────────────────────────────────────────────────
# %%
def predict(texts, onnx_path=ONNX_QUANT):
    sess = ort.InferenceSession(onnx_path)
    enc  = tokenizer(texts, max_length=MAX_LEN, padding="max_length",
                     truncation=True, return_tensors="np")
    feeds = {
        "input_ids":      enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
        "token_type_ids": enc.get("token_type_ids",
                          np.zeros((len(texts), MAX_LEN), dtype=np.int64)),
    }
    logits = sess.run(["logits"], feeds)[0]
    probs  = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    lbls   = {0: "not_food", 1: "food"}
    for t, p, prob in zip(texts, np.argmax(probs, -1), probs):
        icon = "🍜" if p == 1 else "🧾"
        print(f"  {icon} {t:22s} → {lbls[p]:8s} ({prob[p]*100:.1f}%)")

predict([
    "김치찌개", "된장찌개 2인분", "삼겹살 (200g)",
    "합계", "카드결제", "부가세",
    "아메리카노", "영수증번호", "총금액",
])
print("\n✅ Done! Place korean_food_classifier_quant.onnx + korean_vocab.txt")
print("   in android_app/app/src/main/assets/")
