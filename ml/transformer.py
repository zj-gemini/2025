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


class TransformerEncoderBlock:
    """
    A single block of a Transformer encoder, implemented from scratch with NumPy.
    This class encapsulates multi-head attention and a feed-forward network.
    """

    def __init__(self, d_model, num_heads, d_ff, d_k, d_v):
        """
        Initializes the weights for the encoder block.
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.d_k = d_k
        self.d_v = d_v
        self.d_q = d_k  # Query and Key dimensions must be equal

        # --- Initialize weights for Multi-Head Attention ---
        # W_queries is a list of `num_heads` matrices. Each matrix has shape (d_q, d_model).
        self.W_queries = [
            np.random.rand(self.d_q, self.d_model) for _ in range(self.num_heads)
        ]
        # W_keys is a list of `num_heads` matrices. Each matrix has shape (d_k, d_model).
        self.W_keys = [
            np.random.rand(self.d_k, self.d_model) for _ in range(self.num_heads)
        ]
        # W_values is a list of `num_heads` matrices. Each matrix has shape (d_v, d_model).
        self.W_values = [
            np.random.rand(self.d_v, self.d_model) for _ in range(self.num_heads)
        ]
        # Final projection matrix for attention output
        self.W_O = np.random.rand(self.d_model, self.num_heads * self.d_v)

        # --- Initialize weights for Feed-Forward Network ---
        self.W_ff1 = np.random.rand(self.d_ff, self.d_model)
        self.W_ff2 = np.random.rand(self.d_model, self.d_ff)

    def forward(self, x):
        """
        Performs the forward pass for the encoder block.
        x: Input tensor of shape (seq_len, d_model)
        """
        # 1. Multi-Head Attention
        all_context_vectors = []
        for i in range(self.num_heads):
            W_query, W_key, W_value = (
                self.W_queries[i],
                self.W_keys[i],
                self.W_values[i],
            )
            # The operation x @ W.T is equivalent to (W @ x.T).T but is more direct.
            queries = x @ W_query.T
            keys = x @ W_key.T
            values = x @ W_value.T

            omega = queries @ keys.T
            attention_weights = softmax(omega / np.sqrt(self.d_k), axis=1)
            context_vectors_h = attention_weights @ values
            all_context_vectors.append(context_vectors_h)

        # 2. Concatenate and Project
        concatenated_context = np.concatenate(all_context_vectors, axis=-1)
        projected_context = concatenated_context @ self.W_O.T

        # 3. Add & Norm (Post-Attention)
        residual_1 = x + projected_context
        norm_1 = layer_norm(residual_1)

        # 4. Feed-Forward Network
        ffn_output = relu(norm_1 @ self.W_ff1.T) @ self.W_ff2.T

        # 5. Add & Norm (Post-FFN)
        residual_2 = norm_1 + ffn_output
        norm_2 = layer_norm(residual_2)

        return norm_2


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
    # Shape: (seq_len,)
    sentence_int = np.array([word_to_idx[token] for token in tokens])
    print(f"\nInteger-encoded sentence: {sentence_int}")

    # --- 3. Word Embeddings ---
    # In PyTorch, nn.Embedding is a lookup table. We can simulate this with a NumPy array.
    # Set a seed for reproducibility to match the PyTorch script's intent
    np.random.seed(123)
    vocab_size = len(vocab)
    # d_model is the dimension of the embedding vectors.
    d_model = 8

    # Create the embedding matrix with random weights.
    # Shape: (vocab_size, d_model)
    embedding_matrix = np.random.rand(vocab_size, d_model)
    print(f"\nEmbedding matrix (weights) shape: {embedding_matrix.shape}")

    # Get the embeddings for our sentence by indexing into the matrix.
    # This is our input matrix X.
    # Shape: (seq_len, d_model)
    embedded_sentence = embedding_matrix[sentence_int]
    print(f"Word embeddings shape: {embedded_sentence.shape}")

    # --- 3.5 Positional Encoding ---
    # The self-attention mechanism doesn't inherently know the order of tokens.
    # We add positional encodings to the word embeddings to give the model
    # information about the position of each token in the sequence.
    # pos_encoding shape: (seq_len, d_model)
    pos_encoding = get_positional_encoding(seq_len, d_model)
    print(f"\nPositional encoding shape: {pos_encoding.shape}")

    # Add positional encodings to the word embeddings.
    # This is the final input to the transformer block.
    # Shape: (seq_len, d_model)
    input_embeddings = embedded_sentence + pos_encoding
    print(
        f"Input embeddings (with positional encoding) shape: {input_embeddings.shape}"
    )

    # --- 4. Define Model Hyperparameters ---
    d_ff = 16
    num_heads = 2

    # The model's dimension must be divisible by the number of heads.
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    # In the standard transformer architecture, the dimension of the value vector
    # is set to d_model / num_heads. This ensures the total computation remains
    # consistent when splitting the model into multiple heads.
    d_v = d_model // num_heads

    # The dimension of query and key can be set independently, but d_q must equal d_k.
    d_k = 4

    # --- 5. Instantiate and Run the Transformer Block ---
    print("\n--- Initializing Transformer Encoder Block ---")
    print(
        f"Hyperparameters: d_model={d_model}, num_heads={num_heads}, d_k={d_k}, d_v={d_v}"
    )
    encoder_block = TransformerEncoderBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        d_k=d_k,
        d_v=d_v,
    )

    print("\n--- Running Forward Pass ---")
    final_output = encoder_block.forward(input_embeddings)

    print(f"\nFinal output of the transformer block shape: {final_output.shape}")
    print(f"Final output:\n{final_output.round(3)}")


if __name__ == "__main__":
    main()
