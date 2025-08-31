import numpy as np

from utils import relu, layer_norm, softmax


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

    def forward(self, x, mask=None):
        """
        Performs the forward pass for the encoder block.
        x: Input tensor of shape (seq_len, d_model)
        mask: Optional attention mask of shape (seq_len, seq_len)
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
            # Shape: (seq_len, d_q)
            queries = x @ W_query.T
            # Shape: (seq_len, d_k)
            keys = x @ W_key.T
            # Shape: (seq_len, d_v)
            values = x @ W_value.T

            # Shape: (seq_len, seq_len)
            omega = queries @ keys.T
            # Apply the mask (if provided) to the attention scores
            if mask is not None:
                omega += mask

            # Calculate attention weights: scale dot products by sqrt(d_k) and apply softmax.
            # The scaling, as described in the original Transformer paper, prevents the
            # dot products from growing too large, which helps stabilize training by
            # preventing the softmax function from saturating.
            # Shape: (seq_len, seq_len)
            attention_weights = softmax(omega / np.sqrt(self.d_k), axis=1)
            # Shape: (seq_len, d_v)
            context_vectors_h = attention_weights @ values
            all_context_vectors.append(context_vectors_h)

        # 2. Concatenate and Project
        # Shape: (seq_len, num_heads * d_v)
        concatenated_context = np.concatenate(all_context_vectors, axis=-1)
        # Shape: (seq_len, d_model)
        projected_context = concatenated_context @ self.W_O.T

        # 3. Add & Norm (Post-Attention)
        # Shape: (seq_len, d_model)
        residual_1 = x + projected_context
        norm_1 = layer_norm(residual_1)

        # 4. Feed-Forward Network
        ffn_output = relu(norm_1 @ self.W_ff1.T) @ self.W_ff2.T

        # 5. Add & Norm (Post-FFN)
        residual_2 = norm_1 + ffn_output
        norm_2 = layer_norm(residual_2)

        return norm_2
