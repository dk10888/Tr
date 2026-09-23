import os
import sys
import numpy as np
import fasttext
import torch
import torch.nn as nn
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class PyTorchFastTextModel(nn.Module):
    def __init__(self, input_matrix: np.ndarray, output_matrix: np.ndarray):
        super().__init__()
        vocab_size, embed_dim = input_matrix.shape
        num_classes, _ = output_matrix.shape

        self.embedding = nn.EmbeddingBag.from_pretrained(
            torch.from_numpy(input_matrix).float(),
            mode='mean',
            sparse=False
        )
        self.fc = nn.Linear(embed_dim, num_classes, bias=False)
        self.fc.weight.data = torch.from_numpy(output_matrix).float()

    def forward(self, input_ids: torch.Tensor, offsets: torch.Tensor):
        embed = self.embedding(input_ids, offsets)
        logits = self.fc(embed)
        probs = torch.softmax(logits, dim=-1)
        return probs

def convert_fasttext_model(bin_path: str, onnx_out_path: str):
    print(f"📂 Loading FastText model from: {bin_path}")
    ft = fasttext.load_model(bin_path)

    input_matrix = ft.get_input_matrix()   # Shape: [vocab_size, dim]
    output_matrix = ft.get_output_matrix() # Shape: [num_classes, dim]

    print(f"   Input Matrix  : {input_matrix.shape}")
    print(f"   Output Matrix : {output_matrix.shape}")

    model = PyTorchFastTextModel(input_matrix, output_matrix)
    model.eval()

    dummy_input_ids = torch.tensor([12, 45, 99], dtype=torch.long)
    dummy_offsets = torch.tensor([0], dtype=torch.long)

    os.makedirs(os.path.dirname(onnx_out_path), exist_ok=True)

    print(f"🚀 Exporting to ONNX: {onnx_out_path}")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_offsets),
        onnx_out_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids', 'offsets'],
        output_names=['probs'],
        dynamic_axes={
            'input_ids': {0: 'num_subwords'},
            'offsets': {0: 'batch_size'}
        }
    )

    print(f"✅ Created ONNX model: {onnx_out_path} ({os.path.getsize(onnx_out_path) / (1024*1024):.2f} MB)")

    # Verify ONNX Runtime
    session = ort.InferenceSession(onnx_out_path)
    res = session.run(None, {
        'input_ids': dummy_input_ids.numpy(),
        'offsets': dummy_offsets.numpy()
    })
    print(f"   ONNX Test Probabilities: {res[0]}")

def main():
    models_to_convert = [
        (os.path.join(BASE_DIR, "fastText_v23", "fasttext_korean_food.bin"),
         os.path.join(BASE_DIR, "yesterday", "models", "fasttext_v23", "fasttext_v23.onnx")),
        (os.path.join(BASE_DIR, "fastText_v25", "fasttext_korean_food.bin"),
         os.path.join(BASE_DIR, "yesterday", "models", "fasttext_v25", "fasttext_v25.onnx")),
        (os.path.join(BASE_DIR, "fastText_new_aug_v1", "fasttext_korean_food.bin"),
         os.path.join(BASE_DIR, "yesterday", "models", "fasttext_new_aug_v1", "fasttext_new_aug_v1.onnx")),
        (os.path.join(BASE_DIR, "fastText_v30_korean", "fasttext_korean_food.bin"),
         os.path.join(BASE_DIR, "yesterday", "models", "fasttext_v30_korean", "fasttext_v30_korean.onnx")),
    ]

    for bin_p, onnx_p in models_to_convert:
        if os.path.exists(bin_p):
            convert_fasttext_model(bin_p, onnx_p)
        else:
            print(f"⚠️ Model file not found: {bin_p}")

if __name__ == "__main__":
    main()
