import numpy as np

from utils import relu, layer_norm, softmax


class TransformerDecoderBlock:
    """
    A single block of a Transformer decoder, implemented from scratch with NumPy.
    This block consists of masked multi-head self-attention, multi-head cross-attention,
    and a feed-forward network.
    """

    def __init__(self, d_model, num_heads, d_ff, d_k, d_v):
        """
        Initializes the weights for the decoder block.
        """
        np.random.seed(42)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.d_k = d_k
        self.d_v = d_v
        self.d_q = d_k  # Query and Key dimensions must be equal

        # --- Initialize weights for Masked Multi-Head Self-Attention ---
        # Each shape: (d_q, d_model)
        self.W_queries_self = [
            np.random.rand(self.d_q, self.d_model) for _ in range(self.num_heads)
        ]
        # Each shape: (d_k, d_model)
        self.W_keys_self = [
            np.random.rand(self.d_k, self.d_model) for _ in range(self.num_heads)
        ]
        # Each shape: (d_v, d_model)
        self.W_values_self = [
            np.random.rand(self.d_v, self.d_model) for _ in range(self.num_heads)
        ]
        # Shape: (d_model, num_heads * d_v)
        self.W_O_self = np.random.rand(self.d_model, self.num_heads * self.d_v)

        # --- Weights for Multi-Head Cross-Attention ---
        # Queries come from the decoder, Keys and Values from the encoder.
        # Each shape: (d_q, d_model)
        self.W_queries_cross = [
            np.random.rand(self.d_q, self.d_model) for _ in range(self.num_heads)
        ]
        # Each shape: (d_k, d_model)
        self.W_keys_cross = [
            np.random.rand(self.d_k, self.d_model) for _ in range(self.num_heads)
        ]
        # Each shape: (d_v, d_model)
        self.W_values_cross = [
            np.random.rand(self.d_v, self.d_model) for _ in range(self.num_heads)
        ]
        # Shape: (d_model, num_heads * d_v)
        self.W_O_cross = np.random.rand(self.d_model, self.num_heads * self.d_v)

        # --- Initialize weights for Feed-Forward Network ---
        # Shape: (d_ff, d_model)
        self.W_ff1 = np.random.rand(self.d_ff, self.d_model)
        # Shape: (d_model, d_ff)
        self.W_ff2 = np.random.rand(self.d_model, self.d_ff)

    def _multi_head_attention(
        self,
        queries_input,
        keys_input,
        values_input,
        W_queries,
        W_keys,
        W_values,
        W_O,
        mask=None,
    ):
        """
        Helper function for multi-head attention.
        queries_input: Shape (query_seq_len, d_model)
        keys_input: Shape (key_seq_len, d_model)
        values_input: Shape (value_seq_len, d_model) where key_seq_len == value_seq_len
        """
        all_context_vectors = []
        for i in range(self.num_heads):
            W_query, W_key, W_value = W_queries[i], W_keys[i], W_values[i]

            # Project inputs to Q, K, V spaces
            # Shape: (query_seq_len, d_q)
            queries = queries_input @ W_query.T
            # Shape: (key_seq_len, d_k)
            keys = keys_input @ W_key.T
            # Shape: (value_seq_len, d_v)
            values = values_input @ W_value.T

            # Calculate attention scores
            # Shape: (query_seq_len, key_seq_len)
            omega = queries @ keys.T
            if mask is not None:
                # The mask shape should be compatible with omega's shape
                omega += mask

            # Scale and apply softmax
            # Shape: (query_seq_len, key_seq_len)
            attention_weights = softmax(omega / np.sqrt(self.d_k), axis=-1)

            # Get context vectors
            # Shape: (query_seq_len, d_v)
            context_vectors_h = attention_weights @ values
            all_context_vectors.append(context_vectors_h)

        # Concatenate and project
        # Shape: (query_seq_len, num_heads * d_v)
        concatenated_context = np.concatenate(all_context_vectors, axis=-1)
        # Shape: (query_seq_len, d_model)
        projected_context = concatenated_context @ W_O.T
        return projected_context

    def forward(self, x, encoder_output, causal_mask=None, padding_mask=None):
        """
        Performs the forward pass for the decoder block.
        x: Input tensor from the previous decoder layer.
           Shape: (target_seq_len, d_model)
        encoder_output: Output from the encoder stack.
                        Shape: (source_seq_len, d_model)
        causal_mask: Causal attention mask for self-attention.
                     Shape: (target_seq_len, target_seq_len)
        padding_mask: Padding mask for cross-attention (optional).
        """
        # 1. Masked Multi-Head Self-Attention (decoder attends to itself)
        # Shape: (target_seq_len, d_model)
        self_attn_output = self._multi_head_attention(
            queries_input=x,
            keys_input=x,
            values_input=x,
            W_queries=self.W_queries_self,
            W_keys=self.W_keys_self,
            W_values=self.W_values_self,
            W_O=self.W_O_self,
            mask=causal_mask,
        )

        # 2. Add & Norm (Post-Self-Attention)
        # Shape: (target_seq_len, d_model)
        residual_1 = x + self_attn_output
        # Shape: (target_seq_len, d_model)
        norm_1 = layer_norm(residual_1)

        # 3. Multi-Head Cross-Attention (decoder attends to encoder output)
        # Shape: (target_seq_len, d_model)
        cross_attn_output = self._multi_head_attention(
            queries_input=norm_1,
            keys_input=encoder_output,
            values_input=encoder_output,
            W_queries=self.W_queries_cross,
            W_keys=self.W_keys_cross,
            W_values=self.W_values_cross,
            W_O=self.W_O_cross,
            mask=padding_mask,
        )

        # 4. Add & Norm (Post-Cross-Attention)
        # Shape: (target_seq_len, d_model)
        residual_2 = norm_1 + cross_attn_output
        # Shape: (target_seq_len, d_model)
        norm_2 = layer_norm(residual_2)

        # 5. Feed-Forward Network
        # Shape: (target_seq_len, d_model)
        ffn_output = relu(norm_2 @ self.W_ff1.T) @ self.W_ff2.T

        # 6. Add & Norm (Post-FFN)
        # Shape: (target_seq_len, d_model)
        residual_3 = norm_2 + ffn_output
        # Shape: (target_seq_len, d_model)
        norm_3 = layer_norm(residual_3)

        return norm_3
