# evaluate.py

import torch
from model import OPiPredictor, LayerDecoder
from utils import load_trace_features, load_layer_sequences, load_opi_labels, edit_distance, mean_absolute_error
from evaluation_utils import evaluate_predictions

# Constants
NUM_OPS = 8
NUM_KERNELS = 10
INPUT_DIM = 20

# Load data
trace = load_trace_features("data/trace_features.csv").reshape(-1, NUM_KERNELS, INPUT_DIM)
opi = load_opi_labels("data/opi_labels.csv").reshape(-1, NUM_KERNELS, NUM_OPS)
sequences = load_layer_sequences("data/layer_sequences.txt")

# Load models
opi_model = OPiPredictor()
decoder_model = LayerDecoder(input_dim=NUM_OPS, vocab_size=9)
opi_model.load_state_dict(torch.load("opi_model.pth"))
decoder_model.load_state_dict(torch.load("decoder_model.pth"))
opi_model.eval()
decoder_model.eval()

# Predict full dataset
logits_all = []
pred_opi_all = []
true_opi_all = []
with torch.no_grad():
    for i in range(len(trace)):
        x = torch.tensor(trace[i:i+1], dtype=torch.float32)
        _, count_pred, fused = opi_model(x)
        logits = decoder_model(fused)
        logits = logits / 0.5
        logits[:, :, 0] -= 2.0

        logits_all.append(logits.squeeze())
        pred_opi_all.append(count_pred.squeeze().sum(dim=0).tolist())
        true_opi_all.append(opi[i].sum(axis=0).tolist())

# Evaluate prediction sequence accuracy
evaluate_predictions(logits_all, sequences)

# Extra: MAE for OPi count predictions
mae = mean_absolute_error(true_opi_all, pred_opi_all)
print(f"📏 Mean Absolute Error (MAE) for OPi counts: {mae:.2f}")
