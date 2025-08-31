import numpy as np


def relu(x):
    """
    Computes the Rectified Linear Unit activation function.
    """
    return np.maximum(0, x)


def layer_norm(x, eps=1e-5):
    """
    Applies Layer Normalization to the input tensor.
    """
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def softmax(x, axis=-1):
    """
    Computes softmax along a specified axis, with a numerically stable trick.
    """
    # Subtract max for numerical stability before exponentiating
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def get_positional_encoding(seq_len, d_model):
    """
    Generates positional encoding matrix using sine and cosine functions.
    """
    position = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    # Term for the denominator in the sine/cosine functions
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)  # Even indices
    pe[:, 1::2] = np.cos(position * div_term)  # Odd indices
    return pe


def create_causal_mask(seq_len):
    """
    Creates a causal mask for attention of shape (seq_len, seq_len).
    """
    # np.triu with k=1 gives the upper triangle, excluding the diagonal.
    # We set these positions to a large negative number to zero them out in softmax.
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
    return mask
