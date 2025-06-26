# utils.py

import numpy as np
from difflib import SequenceMatcher

def load_trace_features(path):
    """Load trace feature vectors from CSV file"""
    return np.loadtxt(path, delimiter=',')

def load_opi_labels(path):
    """Load OPi labels (existence + count matrix)"""
    return np.loadtxt(path, delimiter=',')

def load_layer_sequences(path):
    """Load true layer sequences from text file"""
    with open(path, 'r') as f:
        sequences = [line.strip().split() for line in f]
    return sequences

def edit_distance(a, b):
    """ Levenshtein-based error rate (1 - similarity) """
    return 1 - SequenceMatcher(None, a, b).ratio()

def mean_absolute_error(y_true, y_pred):
    """Mean Absolute Error for OPi count regression"""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
