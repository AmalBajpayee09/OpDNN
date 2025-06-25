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
x = torch.tensor(trace[0:1], dtype=torch.float32)
true_seq = sequences[0]

# Predict
with torch.no_grad():
    _, _, fused = opi_model(x)
    logits = decoder_model(fused)
    logits = logits / 0.5
    logits[:, :, 0] -= 2.0
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

# Decode output (greedy)
pred_indices = torch.argmax(log_probs, dim=2).squeeze().tolist()

# Postprocess (remove blanks and repeats)
final_pred = []
prev = -1
for idx in pred_indices:
    if idx != 0 and idx != prev:
        final_pred.append(VOCAB_IDX.get(idx, 'UNK'))
    prev = idx

# Output
print("🟢 True Layer Sequence:")
print(" →", " ".join(true_seq))
print("\n🔵 Predicted Layer Sequence:")
print(" →", " ".join(final_pred))
print("\n✅ Prediction completed successfully!")
print("You can now use this prediction for further analysis or visualization.")