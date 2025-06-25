# utils.py
import numpy as np
from difflib import SequenceMatcher

def load_trace_features(path):
    return np.loadtxt(path, delimiter=',')

def load_opi_labels(path):
    return np.loadtxt(path, delimiter=',')

def load_layer_sequences(path):
    with open(path, 'r') as f:
        sequences = [line.strip().split() for line in f]
    return sequences

def edit_distance(a, b):
    return 1 - SequenceMatcher(None, a, b).ratio()

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
