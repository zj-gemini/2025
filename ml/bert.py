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
            queries = x @ W_query.T
            keys = x @ W_key.T
            values = x @ W_value.T

            omega = queries @ keys.T
            # Apply the mask (if provided) to the attention scores
            if mask is not None:
                omega += mask

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


class MiniBERT:
    """
    A simplified BERT-like model composed of multiple Transformer Encoder Blocks.
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff, d_k, d_v):
        """
        Initializes the BERT model with a stack of encoder blocks.
        """
        self.num_layers = num_layers
        self.encoder_blocks = [
            TransformerEncoderBlock(d_model, num_heads, d_ff, d_k, d_v)
            for _ in range(num_layers)
        ]

    def forward(self, x, mask=None):
        """
        Performs the forward pass for the entire BERT model.
        x: Input tensor of shape (seq_len, d_model)
        mask: Optional mask to apply to attention scores.
        """
        # Pass the input through each encoder block in sequence
        for block in self.encoder_blocks:
            x = block.forward(x, mask=mask)
        return x


class BERTForCausalLM:
    """
    Wraps MiniBERT with a language model head for auto-regressive generation.
    """

    def __init__(self, vocab_size, num_layers, d_model, num_heads, d_ff, d_k, d_v):
        """
        Initializes the BERT model and the language model head.
        """
        self.bert = MiniBERT(num_layers, d_model, num_heads, d_ff, d_k, d_v)
        self.d_model = d_model
        # The LM head is a linear layer that maps from d_model to vocab_size
        self.lm_head_weights = np.random.rand(vocab_size, d_model)

    def generate(
        self,
        input_ids,
        embedding_matrix,
        context_length,
        pad_token_id,
        idx_to_word,
        max_new_tokens=10,
        stop_token_id=None,
    ):
        """
        Generates a sequence of tokens auto-regressively.
        """
        generated_ids = list(input_ids)

        # Create the causal mask and positional encodings once outside the loop,
        # as they are constant for a fixed context_length.
        # The mask must be the full context_length because the model input
        # is padded to context_length. The attention scores matrix inside
        # the model will have dimensions (context_length, context_length),
        # so the mask must match.
        # Shape: (context_length, context_length)
        causal_mask = create_causal_mask(context_length)
        # Shape: (context_length, d_model)
        pos_encoding = get_positional_encoding(context_length, self.d_model)

        for i in range(max_new_tokens):
            print(f"\n--- Generation Step {i+1} ---")
            current_seq_len = len(generated_ids)
            if current_seq_len >= context_length:
                print("Max sequence length reached. Halting generation.")
                break

            current_words = [idx_to_word[id] for id in generated_ids]
            print(f"Current generated sequence: {' '.join(current_words)}")

            # 1. Prepare fixed-length input with padding
            # Shape: (context_length,)
            context_ids = np.full((context_length,), pad_token_id, dtype=int)
            context_ids[:current_seq_len] = generated_ids
            context_words = [idx_to_word[id] for id in context_ids]
            print(
                f"Model input (padded to {context_length}): {' '.join(context_words)}"
            )

            # 2. Get embeddings for the padded input
            current_embeddings = embedding_matrix[context_ids]

            # 3. Add the pre-calculated fixed-size positional encoding
            # Shape: (context_length, d_model)
            model_input = current_embeddings + pos_encoding

            # 4. Pass the fixed-length input through the BERT encoder
            # Shape: (context_length, d_model)
            bert_output = self.bert.forward(model_input, mask=causal_mask)

            # 5. Use the output of the *last actual token* (before padding) for prediction
            # The index is current_seq_len - 1
            # Shape: (d_model,)
            last_token_output = bert_output[current_seq_len - 1, :]

            # 6. Project to vocabulary space to get logits
            # Shape: (vocab_size,)
            logits = last_token_output @ self.lm_head_weights.T
            print(f"Logits for next token (shape {logits.shape}): {logits.round(2)}")

            # 7. Get the most likely next token (greedy decoding)
            # Shape: scalar
            next_token_id = np.argmax(logits)
            next_word = idx_to_word[next_token_id]
            print(f"Predicted next token: '{next_word}' (ID: {next_token_id})")
            generated_ids.append(next_token_id)

            if next_token_id == stop_token_id:
                print("Stop token generated. Halting generation.")
                break

        return generated_ids


def main():
    """
    Demonstrates the core components of a transformer's self-attention
    mechanism using NumPy. This is a reimplementation of the PyTorch version.
    """
    # --- 1. Tokenization and Vocabulary Creation ---
    # Add special tokens for generation
    base_sentence = "Life is short eat dessert first"
    tokens = base_sentence.split()
    print(f"Tokens: {tokens}")

    # Add special tokens for padding, starting, and stopping generation
    special_tokens = ["[PAD]", "[START]", "[STOP]"]
    vocab = sorted(list(set(tokens)) + special_tokens)
    print(f"Vocab: {vocab}")

    # Create a word-to-index mapping
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for word, i in word_to_idx.items()}
    print(f"Word-to-index mapping: {word_to_idx}")

    # --- 2. Define Model Hyperparameters & Embeddings ---
    # Set a seed for reproducibility
    np.random.seed(123)
    vocab_size = len(vocab)
    context_length = 15  # Define the fixed context window size for the model
    d_model = 8
    d_ff = 16
    num_heads = 2
    num_layers = 3  # Number of encoder blocks to stack

    # Create the embedding matrix with random weights. This is passed to the model.
    embedding_matrix = np.random.rand(vocab_size, d_model)
    print(f"\nEmbedding matrix (weights) shape: {embedding_matrix.shape}")

    # The model's dimension must be divisible by the number of heads.
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    # In the standard transformer architecture, the dimension of the value vector
    # is set to d_model / num_heads. This ensures the total computation remains
    # consistent when splitting the model into multiple heads.
    d_v = d_model // num_heads

    # The dimension of query and key can be set independently, but d_q must equal d_k.
    d_k = 4

    # --- 3. Instantiate and Run the Generative BERT Model ---
    print("\n--- Initializing Generative BERT Model ---")
    print(
        f"Hyperparameters: vocab_size={vocab_size}, num_layers={num_layers}, d_model={d_model}, num_heads={num_heads}, d_k={d_k}, d_v={d_v}"
    )
    generative_bert = BERTForCausalLM(
        vocab_size=vocab_size,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        d_k=d_k,
        d_v=d_v,
    )

    print("\n--- Running Auto-Regressive Generation ---")
    # Start generation with a prompt, e.g., the '[START]' token
    input_ids = [word_to_idx["[START]"]]
    pad_token_id = word_to_idx["[PAD]"]
    stop_token_id = word_to_idx["[STOP]"]

    generated_sequence_ids = generative_bert.generate(
        input_ids,
        embedding_matrix,
        context_length,
        pad_token_id,
        idx_to_word,
        max_new_tokens=5,
        stop_token_id=stop_token_id,
    )

    generated_words = [idx_to_word[id] for id in generated_sequence_ids]
    print(f"\nGenerated sequence IDs: {generated_sequence_ids}")
    print(f"Generated sequence: {' '.join(generated_words)}")


if __name__ == "__main__":
    main()
