# evaluation_utils.py
import torch
import numpy as np
from difflib import SequenceMatcher
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

VOCAB = ['Conv', 'BN', 'ReLU', 'Pool', 'Add', 'Mul', 'Dense', 'Dropout']
IDX_TO_VOCAB = {i + 1: v for i, v in enumerate(VOCAB)}  # For decoding

def decode_ctc(indices):
    result = []
    prev = -1
    for idx in indices:
        if idx != 0 and idx != prev:
            result.append(IDX_TO_VOCAB.get(idx, 'UNK'))
        prev = idx
    return result

def levenshtein_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()

def evaluate_predictions(logits_all, true_sequences):
    predictions = []
    ground_truths = []
    ratios = []

    with open("predictions_log.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["True Sequence", "Predicted Sequence", "Levenshtein Ratio"])

        for logits, true_seq in zip(logits_all, true_sequences):
            probs = torch.nn.functional.log_softmax(logits, dim=-1)
            pred_indices = torch.argmax(probs, dim=-1).tolist()
            pred_seq = decode_ctc(pred_indices)

            predictions.append(pred_seq)
            ground_truths.append(true_seq)
            ratio = levenshtein_ratio(pred_seq, true_seq)
            ratios.append(ratio)

            writer.writerow([" ".join(true_seq), " ".join(pred_seq), f"{ratio:.2f}"])

    acc = sum([levenshtein_ratio(p, t) > 0.8 for p, t in zip(predictions, ground_truths)]) / len(predictions)
    avg_lev = sum(ratios) / len(ratios)

    print(f"🔍 Accuracy: {acc:.2f}")
    print(f"✏️  Levenshtein Similarity: {avg_lev:.2f}")
    print("✅ predictions_log.csv saved")

    # Confusion Matrix
    true_flat = [layer for seq in ground_truths for layer in seq]
    pred_flat = [layer for seq in predictions for layer in seq]

    # Match lengths
    min_len = min(len(true_flat), len(pred_flat))
    true_flat = true_flat[:min_len]
    pred_flat = pred_flat[:min_len]

    labels = VOCAB
    cm = confusion_matrix(true_flat, pred_flat, labels=labels)
    print("📊 Confusion Matrix:")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation=45, cmap='Blues', colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("📊 confusion_matrix.png saved")

    # ✅ Print sample mismatches
    print("\n🔍 Sample mismatch:")
    for i in range(min(3, len(predictions))):
        print(f"\nGT: {' '.join(ground_truths[i])}")
        print(f"PR: {' '.join(predictions[i])}")

    plt.show()
