# evaluate.py

import torch
from model import OPiPredictor, LayerDecoder
from utils import load_trace_features, load_layer_sequences
from evaluation_utils import evaluate_predictions

# Constants
NUM_OPS = 8
NUM_KERNELS = 10
INPUT_DIM = 20

# Load data
trace = load_trace_features("data/trace_features.csv").reshape(-1, NUM_KERNELS, INPUT_DIM)
sequences = load_layer_sequences("data/layer_sequences.txt")

# Load models
opi_model = OPiPredictor()
decoder_model = LayerDecoder(input_dim=NUM_OPS, vocab_size=9)
opi_model.load_state_dict(torch.load("opi_model.pth"))
decoder_model.load_state_dict(torch.load("decoder_model.pth"))
opi_model.eval()
decoder_model.eval()

# Get logits for full dataset
logits_all = []
with torch.no_grad():
    for i in range(len(trace)):
        x = torch.tensor(trace[i:i+1], dtype=torch.float32)
        _, _, fused = opi_model(x)
        logits = decoder_model(fused)
        logits = logits / 0.5
        logits[:, :, 0] -= 2.0
        logits_all.append(logits.squeeze())

# Evaluate
evaluate_predictions(logits_all, sequences)
