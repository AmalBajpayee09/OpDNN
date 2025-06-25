# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import OPiPredictor, LayerDecoder
from utils import load_trace_features, load_opi_labels, load_layer_sequences
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
NUM_OPS = 8
NUM_KERNELS = 10
INPUT_DIM = 20
VOCAB = ['Conv', 'BN', 'ReLU', 'Pool', 'Add', 'Mul', 'Dense', 'Dropout']
VOCAB_IDX = {v: i + 1 for i, v in enumerate(VOCAB)}  # CTC blank = 0
BATCH_SIZE = 16
EPOCHS = 100
loss_history = []

# Load and preprocess data
trace = load_trace_features("data/trace_features.csv").reshape(-1, NUM_KERNELS, INPUT_DIM)
opi = load_opi_labels("data/opi_labels.csv").reshape(-1, NUM_KERNELS, NUM_OPS)
sequences = load_layer_sequences("data/layer_sequences.txt")

def seq_to_idx(seq):
    return [VOCAB_IDX[w] for w in seq if w in VOCAB_IDX]

indexed_sequences = [seq_to_idx(seq)[:NUM_KERNELS] for seq in sequences]

min_len = min(len(trace), len(opi), len(indexed_sequences))
trace = trace[:min_len]
opi = opi[:min_len]
indexed_sequences = indexed_sequences[:min_len]

# Models
opi_model = OPiPredictor()
decoder_model = LayerDecoder(input_dim=NUM_OPS, vocab_size=len(VOCAB) + 1)

# Loss and optimizer
ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.AdamW(list(opi_model.parameters()) + list(decoder_model.parameters()), lr=0.0003)

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0.0
    opi_model.train()
    decoder_model.train()

    for i in range(0, min_len, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, min_len)
        batch_size = batch_end - i

        batch_x = torch.tensor(trace[i:batch_end], dtype=torch.float32)
        batch_x += 0.01 * torch.randn_like(batch_x)

        batch_y = torch.tensor(opi[i:batch_end], dtype=torch.float32)

        # Forward pass
        _, _, fused = opi_model(batch_x)
        fused += 0.01 * torch.randn_like(fused)
        logits = decoder_model(fused)

        # CTC postprocessing
        logits = logits / 0.5
        logits[:, :, 0] -= 2.0
        log_probs = nn.functional.log_softmax(logits, dim=2).transpose(0, 1)
        log_probs = torch.clamp(log_probs, min=-7, max=0)

        # Prepare target
        try:
            target_seq = [torch.tensor(indexed_sequences[i + j]) for j in range(batch_size)]
        except IndexError:
            continue
        if not target_seq:
            continue

        targets = torch.cat(target_seq)
        input_lengths = torch.full((batch_size,), NUM_KERNELS, dtype=torch.long)
        target_lengths = torch.tensor([len(seq) for seq in target_seq])

        if targets.numel() == 0 or torch.unique(targets).numel() <= 1:
            continue
        if torch.any(target_lengths > NUM_KERNELS) or log_probs.isnan().any() or log_probs.isinf().any():
            continue

        # Loss
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        if loss.isnan() or loss.isinf() or loss.item() < 1e-4:
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.4f}")
    loss_history.append(total_loss)

# Save models
torch.save(opi_model.state_dict(), "opi_model.pth")
torch.save(decoder_model.state_dict(), "decoder_model.pth")
print("✅ Models saved: opi_model.pth, decoder_model.pth")

# Plot loss
# Save accuracy graph (optional if you track epoch-wise accuracy)
plt.figure(figsize=(8, 5))
plt.plot(loss_history, color='blue', marker='o')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("training_loss_curve.png")
print("📉 Loss graph saved as training_loss_curve.png")

