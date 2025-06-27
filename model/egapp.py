import streamlit as st
import torch
import numpy as np
import os
import pickle
from PIL import Image
from types import SimpleNamespace

from image_grid_dataset import Combine
from patch_and_embed import image_to_patch_columns
from transformer import Transformer
from decoder import IDX2TOKEN, TOKEN2IDX

# Model config (from train_enc_dec.py)
MODEL_CONFIG = SimpleNamespace(**{
    "init_learning_rate": 1e-4,
    "min_learning_rate": 1e-6,
    "batch_size": 1024,
    "num_epochs": 100,
    "num_heads": 8,
    "num_encoders": 8,
    "num_patches": 16,
    "patch_size": 14,
    "stride": 14,
    "dim_patch": 196,
    "dim_proj_V": 25,
    "dim_proj_QK": 100,
    "dim_out": 49,
    "dim_in": 49,
    "mlp_hidden_dim": 25,
    "max_seq_len": 5,
    "dec_dim_in": 49,
    "dec_dim_out": 49,
    "num_decoders": 6,
    "dec_mask_num_heads": 8,
    "dec_cross_num_heads": 8,
})

MODEL_WEIGHTS = os.path.join(os.path.dirname(__file__), "model_weights", "enc_dec_final_epoch2025-06-27_14-14-01.pth")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "MNIST", "raw", "mnist_testset.pkl")

def load_mnist():
    with open(DATA_PATH, "rb") as f:
        fullset = pickle.load(f)
    return fullset

def get_random_grid(fullset):
    ds = Combine(fullset)
    combo, patch, labels = ds[0]  # combo: [56,56], patch: [16,14,14], labels: [4]
    return combo, patch, labels

def preprocess_for_model(img):
    return img.unsqueeze(0)

def run_model(model, device, img, max_seq_len=5):
    input_tokens = torch.full((1, max_seq_len), TOKEN2IDX["<pad>"], dtype=torch.long, device=device)
    input_tokens[0, 0] = TOKEN2IDX["<start>"]
    img_patches = image_to_patch_columns(img, patch_size=14, stride=14).to(device)
    model.eval()
    with torch.no_grad():
        for i in range(1, max_seq_len):
            logits = model(img_patches, input_tokens)
            next_token = logits[0, i-1].argmax().item()
            input_tokens[0, i] = next_token
            if next_token == TOKEN2IDX["<stop>"]:
                break
    pred_tokens = input_tokens[0, 1:].cpu().numpy()
    pred_digits = []
    for t in pred_tokens:
        if t == TOKEN2IDX["<stop>"]:
            break
        if t < 10:
            pred_digits.append(IDX2TOKEN[t])
    return pred_digits

def tensor_to_pil(img_tensor):
    arr = img_tensor.cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    arr = (arr * 0.3081 + 0.1307) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def main():
    st.title("MNIST 2x2 Grid Transformer")
    st.write("Click the button to generate a random 2x2 MNIST grid and predict the digits using the trained transformer model.")

    @st.cache_resource
    def load_all():
        fullset = load_mnist()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Transformer(MODEL_CONFIG).to(device)
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        return fullset, model, device

    fullset, model, device = load_all()

    if 'example' not in st.session_state:
        combo, patch, labels = get_random_grid(fullset)
        st.session_state['example'] = (combo, patch, labels)

    if st.button("Generate New Example"):
        combo, patch, labels = get_random_grid(fullset)
        st.session_state['example'] = (combo, patch, labels)

    combo, patch, labels = st.session_state['example']
    st.subheader("2x2 MNIST Grid")
    st.image(tensor_to_pil(combo), caption=f"True Digits: {labels.tolist()}", width=224)

    img = preprocess_for_model(combo).to(device)
    pred_digits = run_model(model, device, img)
    st.subheader("Model Prediction")

    # Compare and color each digit
    true_digits = [str(x) for x in labels.tolist()]
    pred_digits_str = [str(x) for x in pred_digits]
    colored = []
    for i, pred in enumerate(pred_digits_str):
        if i < len(true_digits) and pred == true_digits[i]:
            colored.append(f'<span style="color:green">{pred}</span>')
        else:
            colored.append(f'<span style="color:red">{pred}</span>')
    st.markdown("Predicted Digits: " + ' '.join(colored), unsafe_allow_html=True)

if __name__ == "__main__":
    main()