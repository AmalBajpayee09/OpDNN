# prediction_sample.py
import torch
from model import OPiPredictor, LayerDecoder
from utils import load_trace_features, load_layer_sequences

# Constants
NUM_OPS = 8
NUM_KERNELS = 10
INPUT_DIM = 20
VOCAB = ['Conv', 'BN', 'ReLU', 'Pool', 'Add', 'Mul', 'Dense', 'Dropout']
VOCAB_IDX = {i+1: v for i, v in enumerate(VOCAB)}  # Reverse map

# Load models
opi_model = OPiPredictor()
decoder_model = LayerDecoder(input_dim=NUM_OPS, vocab_size=len(VOCAB) + 1)
opi_model.load_state_dict(torch.load("opi_model.pth"))
decoder_model.load_state_dict(torch.load("decoder_model.pth"))
opi_model.eval()
decoder_model.eval()

# Load data (just 1 sample)
trace = load_trace_features("data/trace_features.csv").reshape(-1, NUM_KERNELS, INPUT_DIM)
sequences = load_layer_sequences("data/layer_sequences.txt")
def decode(indices):
    result = []
    prev = -1
    for idx in indices:
        if idx != 0 and idx != prev:
            result.append(VOCAB_IDX.get(idx, 'UNK'))
        prev = idx
    return result

#
# Predict and show first 10 samples
N = min(10, len(trace))
for i in range(N):
    x = torch.tensor(trace[i:i+1], dtype=torch.float32)
    true_seq = sequences[i]

    with torch.no_grad():
        _, _, fused = opi_model(x)
        logits = decoder_model(fused)
        logits = logits / 0.5
        logits[:, :, 0] -= 2.0
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    pred_indices = torch.argmax(log_probs, dim=2).squeeze().tolist()
    pred_seq = decode(pred_indices)

    print(f"\n🟢 True Layer Sequence [{i+1}]:")
    print(" →", " ".join(true_seq))
    print(f"\n🔵 Predicted Layer Sequence [{i+1}]:")
    print(" →", " ".join(pred_seq))

print("\n✅ Prediction completed for all samples.")