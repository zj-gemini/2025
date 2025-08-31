import numpy as np

from encoder import TransformerEncoderBlock


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


class DecoderOnlyTransformer:
    """
    A simplified GPT-like, decoder-only transformer composed of multiple blocks.
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff, d_k, d_v):
        """
        Initializes the model with a stack of transformer blocks.
        """
        self.num_layers = num_layers
        self.encoder_blocks = [
            TransformerEncoderBlock(d_model, num_heads, d_ff, d_k, d_v)
            for _ in range(num_layers)
        ]

    def forward(self, x, mask=None):
        """
        Performs the forward pass for the entire BERT model.
        x: Input tensor of shape (context_length, d_model)
        mask: Optional mask to apply to attention scores.
        """
        # Pass the input through each encoder block in sequence
        for block in self.encoder_blocks:
            x = block.forward(x, mask=mask)
        return x


class GPTForCausalLM:
    """
    Wraps the DecoderOnlyTransformer with a language model head for
    auto-regressive generation (Causal Language Modeling).
    """

    def __init__(
        self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, d_k, d_v
    ):
        """
        Initializes the GPT model and the language model head.
        """
        np.random.seed(123)
        self.transformer = DecoderOnlyTransformer(
            num_layers, d_model, num_heads, d_ff, d_k, d_v
        )
        self.d_model = d_model
        self.context_length = context_length

        # --- Initialize Model Weights ---
        # Word embedding lookup table
        self.embedding_matrix = np.random.rand(vocab_size, d_model)
        # The LM head is a linear layer that maps from d_model to vocab_size
        self.lm_head_weights = np.random.rand(vocab_size, d_model)

    def generate(
        self,
        input_ids,
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
        causal_mask = create_causal_mask(self.context_length)
        # Shape: (context_length, d_model)
        pos_encoding = get_positional_encoding(self.context_length, self.d_model)

        for i in range(max_new_tokens):
            print(f"\n--- Generation Step {i+1} ---")
            current_seq_len = len(generated_ids)
            if current_seq_len >= self.context_length:
                print("Max sequence length reached. Halting generation.")
                break

            current_words = [idx_to_word[id] for id in generated_ids]
            print(f"Current generated sequence: {' '.join(current_words)}")

            # 1. Prepare fixed-length input with padding
            # Shape: (context_length,)
            context_ids = np.full((self.context_length,), pad_token_id, dtype=int)
            context_ids[:current_seq_len] = generated_ids
            context_words = [idx_to_word[id] for id in context_ids]
            print(
                f"Model input (padded to {self.context_length}): {' '.join(context_words)}"
            )

            # 2. Get embeddings for the padded input
            current_embeddings = self.embedding_matrix[context_ids]

            # 3. Add the pre-calculated fixed-size positional encoding
            # Shape: (context_length, d_model)
            model_input = current_embeddings + pos_encoding

            # 4. Pass the fixed-length input through the BERT encoder
            # Shape: (context_length, d_model)
            transformer_output = self.transformer.forward(model_input, mask=causal_mask)

            # 5. Use the output of the *last actual token* (before padding) for prediction
            # The index is current_seq_len - 1
            # Shape: (d_model,)
            last_token_output = transformer_output[current_seq_len - 1, :]

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

    # --- 2. Define Model Hyperparameters ---
    # Set a seed for reproducibility
    vocab_size = len(vocab)
    context_length = 10  # Define the fixed context window size for the model
    d_model = 8
    d_ff = 16
    num_heads = 2
    num_layers = 3  # Number of encoder blocks to stack

    # The model's dimension must be divisible by the number of heads.
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    # In the standard transformer architecture, the dimension of the value vector
    # is set to d_model / num_heads. This ensures the total computation remains
    # consistent when splitting the model into multiple heads.
    d_v = d_model // num_heads

    # The dimension of query and key can be set independently, but d_q must equal d_k.
    d_k = 4

    # --- 3. Instantiate and Run the Generative BERT Model ---
    print("\n--- Initializing Generative GPT-style Model ---")
    print(
        f"Hyperparameters: vocab_size={vocab_size}, context_length={context_length}, num_layers={num_layers}, d_model={d_model}, num_heads={num_heads}, d_k={d_k}, d_v={d_v}"
    )
    generative_model = GPTForCausalLM(
        vocab_size=vocab_size,
        context_length=context_length,
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

    generated_sequence_ids = generative_model.generate(
        input_ids,
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
