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


def main():
    """
    Demonstrates the core components of a transformer's self-attention
    mechanism using NumPy. This is a reimplementation of the PyTorch version.
    """
    # Let's define seq_len for clarity in comments. Our sentence has 6 words.
    seq_len = 6

    # --- 1. Tokenization and Vocabulary Creation ---
    sentence = "Life is short, eat dessert first"

    tokens = sentence.replace(",", "").split()
    print(f"Tokens: {tokens}")

    vocab = sorted(set(tokens))
    print(f"Vocab: {vocab}")

    # Create a word-to-index mapping
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    print(f"Word-to-index mapping: {word_to_idx}")

    # --- 2. Integer Encoding ---
    # Convert tokens to their integer representations using the vocabulary.
    # Shape: (seq_len,) -> (6,)
    sentence_int = np.array([word_to_idx[token] for token in tokens])
    print(f"\nInteger-encoded sentence: {sentence_int}")

    # --- 3. Word Embeddings ---
    # In PyTorch, nn.Embedding is a lookup table. We can simulate this with a NumPy array.
    # Set a seed for reproducibility to match the PyTorch script's intent
    np.random.seed(123)
    vocab_size = len(vocab)
    # d_model is the dimension of the embedding vectors.
    embedding_dim = 2  # d_model

    # Create the embedding matrix with random weights.
    # Shape: (vocab_size, d_model) -> (6, 2)
    embedding_matrix = np.random.rand(vocab_size, embedding_dim)
    print(f"\nEmbedding matrix (weights) shape: {embedding_matrix.shape}")

    # Get the embeddings for our sentence by indexing into the matrix.
    # This is our input matrix X.
    # Shape: (seq_len, d_model) -> (6, 2)
    embedded_sentence = embedding_matrix[sentence_int]
    print(f"Word embeddings shape: {embedded_sentence.shape}")

    # --- 3.5 Positional Encoding ---
    # The self-attention mechanism doesn't inherently know the order of tokens.
    # We add positional encodings to the word embeddings to give the model
    # information about the position of each token in the sequence.
    d_model = embedded_sentence.shape[1]
    pos_encoding = get_positional_encoding(seq_len, d_model)
    print(f"\nPositional encoding shape: {pos_encoding.shape}")

    # Add positional encodings to the word embeddings.
    # This is the final input to the transformer block.
    # Shape: (seq_len, d_model) -> (6, 2)
    input_embeddings = embedded_sentence + pos_encoding
    print(
        f"Input embeddings (with positional encoding) shape: {input_embeddings.shape}"
    )

    # --- 4. Defining Q, K, V Weight Matrices ---
    # d_model represents the size of each word vector (embedding dimension)

    # Define the dimensions for query, key, and value vectors.
    # In a real transformer, d_q = d_k = d_v = d_model / num_heads.
    # For simplicity, we define them directly. The dot-product attention
    # requires d_q and d_k to be equal.
    d_q, d_k, d_v = 4, 4, 8

    # Initialize weight matrices with random values. These matrices project the
    # input embeddings into the Q, K, and V spaces.
    # W_query shape: (d_q, d_model) -> (4, 2)
    W_query = np.random.rand(d_q, d_model)
    # W_key shape: (d_k, d_model) -> (4, 2)
    W_key = np.random.rand(d_k, d_model)
    # W_value shape: (d_v, d_model) -> (8, 2)
    W_value = np.random.rand(d_v, d_model)

    # --- 5. Calculating Q, K, V vectors for the whole sentence ---
    # The operation is W @ x.T, then transpose the result. This is equivalent to X @ W.T
    # queries shape: (seq_len, d_q) -> (6, 4)
    queries = (W_query @ input_embeddings.T).T
    # keys shape: (seq_len, d_k) -> (6, 4)
    keys = (W_key @ input_embeddings.T).T
    # values shape: (seq_len, d_v) -> (6, 8)
    values = (W_value @ input_embeddings.T).T

    print(f"\nqueries.shape: {queries.shape}")
    print(f"keys.shape: {keys.shape}")
    print(f"values.shape: {values.shape}")

    # --- 6. Calculating Full Attention Scores ---
    # omega(i,j) = q(i) . k(j)
    # We compute the dot product of all query vectors with all key vectors.
    # queries shape: (seq_len, d_q) -> (6, 4)
    # keys.T shape: (d_k, seq_len) -> (4, 6)
    # omega shape: (seq_len, seq_len) -> (6, 6)
    omega = queries @ keys.T
    print(f"\nAttention scores (omega) matrix shape: {omega.shape}")

    # --- 7. Applying Softmax for Full Attention Weights Matrix ---
    # Apply scaled dot-product attention: softmax( (Q @ K.T) / sqrt(d_k) )
    # attention_weights shape: (seq_len, seq_len) -> (6, 6)
    attention_weights = softmax(omega / np.sqrt(d_k), axis=1)
    print(f"Attention weights matrix shape: {attention_weights.shape}")

    # --- 8. Computing the Context Vectors for all tokens ---
    # The context vector is the weighted sum of the value vectors.
    # attention_weights shape: (seq_len, seq_len) -> (6, 6)
    # values shape: (seq_len, d_v) -> (6, 8)
    # context_vectors shape: (seq_len, d_v) -> (6, 8)
    context_vectors = attention_weights @ values
    print(f"Context vectors shape: {context_vectors.shape}")

    # --- 9. Add & Norm (Post-Attention) ---
    # In a real transformer, the output of multi-head attention is projected back
    # to d_model. We'll simulate this with a projection matrix W_O.
    # W_O shape: (d_model, d_v) -> (2, 8)
    W_O = np.random.rand(d_model, d_v)
    # projected_context shape: (seq_len, d_model) -> (6, 2)
    projected_context = context_vectors @ W_O.T

    # Add the residual (skip) connection
    residual_1 = input_embeddings + projected_context
    # Apply Layer Normalization
    norm_1 = layer_norm(residual_1)
    print(f"\nOutput of first Add & Norm layer shape: {norm_1.shape}")

    # --- 10. Feed-Forward Network (FFN) ---
    # FFN consists of two linear layers with a ReLU activation in between.
    d_ff = 16  # Dimension of the inner FFN layer
    # W_ff1 shape: (d_ff, d_model) -> (16, 2)
    W_ff1 = np.random.rand(d_ff, d_model)
    # W_ff2 shape: (d_model, d_ff) -> (2, 16)
    W_ff2 = np.random.rand(d_model, d_ff)

    # ffn_output shape: (seq_len, d_model) -> (6, 2)
    ffn_output = relu(norm_1 @ W_ff1.T) @ W_ff2.T
    print(f"Output of FFN shape: {ffn_output.shape}")

    # --- 11. Add & Norm (Post-FFN) ---
    # Add the second residual connection
    residual_2 = norm_1 + ffn_output
    # Apply Layer Normalization
    norm_2 = layer_norm(residual_2)
    print(f"Output of second Add & Norm layer shape: {norm_2.shape}")

    print(f"\nFinal output of the transformer block:\n{norm_2}")


if __name__ == "__main__":
    main()
