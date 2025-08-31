import numpy as np

from utils import softmax, get_positional_encoding, create_causal_mask
from encoder import TransformerEncoderBlock
from decoder import TransformerDecoderBlock


class Transformer:
    """An Encoder-Decoder Transformer model."""

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        d_k,
        d_v,
        source_vocab_size,
        target_vocab_size,
        max_seq_len,
    ):
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        np.random.seed(42)

        # Embeddings and Positional Encoding
        # Shape: (source_vocab_size, d_model)
        self.source_embedding = np.random.rand(source_vocab_size, d_model)
        # Shape: (target_vocab_size, d_model)
        self.target_embedding = np.random.rand(target_vocab_size, d_model)
        # Shape: (max_seq_len, d_model)
        self.positional_encoding = get_positional_encoding(max_seq_len, d_model)

        # Encoder and Decoder Stacks
        self.encoder_blocks = [
            TransformerEncoderBlock(d_model, num_heads, d_ff, d_k, d_v)
            for _ in range(num_layers)
        ]
        self.decoder_blocks = [
            TransformerDecoderBlock(d_model, num_heads, d_ff, d_k, d_v)
            for _ in range(num_layers)
        ]

        # Final linear layer
        # Shape: (target_vocab_size, d_model)
        self.final_linear = np.random.rand(target_vocab_size, d_model)

    def encode(self, source_seq, source_mask=None):
        """Processes the source sequence through the encoder stack."""
        # 1. Embed source sequence and add positional encoding.
        # Shape: (source_seq_len, d_model)
        source_embedded = (
            self.source_embedding[source_seq]
            + self.positional_encoding[: source_seq.shape[0], :]
        )
        # 2. Pass through encoder blocks.
        # We use 'x' as a generic variable to represent the data flowing through the layers.
        # It starts as the embedded source and is updated by each block.
        x = source_embedded
        for block in self.encoder_blocks:
            x = block.forward(x, mask=source_mask)
        return x

    def decode(self, target_seq, encoder_output, target_mask=None):
        """
        Processes the target sequence and encoder output through the decoder stack.
        target_seq: Shape (target_seq_len,)
        encoder_output: Shape (source_seq_len, d_model)
        target_mask: Shape (target_seq_len, target_seq_len)
        """
        # 1. Embed target sequence and add positional encoding.
        # Shape: (target_seq_len, d_model)
        target_embedded = (
            self.target_embedding[target_seq]
            + self.positional_encoding[: target_seq.shape[0], :]
        )
        # 2. Pass through decoder blocks.
        # We use 'x' as a generic variable to represent the data flowing through the layers.
        x = target_embedded
        for block in self.decoder_blocks:
            x = block.forward(x, encoder_output, causal_mask=target_mask)
        # 3. Final projection to get logits.
        # Shape: (target_seq_len, target_vocab_size)
        return x @ self.final_linear.T

    def forward(self, source_seq, target_seq, source_mask=None, target_mask=None):
        """
        Performs a forward pass through the entire Encoder-Decoder Transformer.

        Args:
            source_seq (np.array): An array of token IDs for the source sequence.
                                   Shape: (source_seq_len,)
            target_seq (np.array): An array of token IDs for the target sequence, used as input to the decoder.
                                   During training (with teacher forcing), this is the ground-truth
                                   target sequence shifted to the right (e.g., starting with a [START] token).
                                   Shape: (target_seq_len,)
            source_mask (np.array, optional): Mask to apply to the encoder's self-attention. Defaults to None.
            target_mask (np.array, optional): Causal mask to apply to the decoder's self-attention to prevent
                                              it from looking ahead. Defaults to None.

        Returns:
            np.array: The raw, unnormalized scores (logits) for each token in the vocabulary.
                      Shape: (target_seq_len, target_vocab_size)
        """
        # 1. Encoder Pass: Process the source sequence.
        # The encoder creates a rich contextual representation of the input sequence.
        # Shape: (source_seq_len, d_model)
        encoder_output = self.encode(source_seq, source_mask)

        # 2. Decoder Pass: Generate the target sequence.
        # This pass uses the ground-truth target sequence for "teacher forcing".
        # Shape: (target_seq_len, target_vocab_size)
        logits = self.decode(target_seq, encoder_output, target_mask)
        return logits

    def generate(
        self, source_seq, start_token_id, stop_token_id, idx_to_word, max_len=15
    ):
        """
        Performs auto-regressive generation (inference).
        It generates the target sequence one token at a time.
        """
        # 1. Encode the source sequence once. This output is reused for each decoding step.
        print("--- Encoding Source Sequence ---")
        # Shape: (source_seq_len, d_model)
        encoder_output = self.encode(source_seq)
        print(f"Encoder output shape: {encoder_output.shape}")

        # 2. Initialize the generated sequence with the start token.
        # This is a list of token IDs.
        generated_ids = [start_token_id]

        for i in range(max_len - 1):
            print(f"\n--- Generation Step {i+1} ---")
            current_words = [idx_to_word.get(idx, "[UNK]") for idx in generated_ids]
            print(f"Current sequence: {' '.join(current_words)}")

            # a. Prepare the current sequence as input to the decoder.
            # Shape: (current_seq_len,)
            current_target_seq = np.array(generated_ids)

            # b. Create a causal mask for the current sequence length.
            # Shape: (current_seq_len, current_seq_len)
            causal_mask = create_causal_mask(len(current_target_seq))

            # c. Get the logits from the decoder.
            # We pass the sequence generated so far.
            # Shape: (current_seq_len, target_vocab_size)
            logits = self.decode(current_target_seq, encoder_output, causal_mask)
            print(f"Decoder output (logits) shape: {logits.shape}")

            # d. We only care about the prediction for the *next* token,
            # so we take the logits for the last token in the sequence.
            # Shape: (target_vocab_size,)
            last_token_logits = logits[-1, :]
            print(
                f"Logits for next token (shape {last_token_logits.shape}): {last_token_logits.round(2)}"
            )

            # e. Apply softmax and select the most likely next token (greedy decoding).
            probabilities = softmax(last_token_logits)
            # Shape: scalar
            next_token_id = np.argmax(probabilities)
            # Shape: string
            next_word = idx_to_word.get(next_token_id, "[UNK]")
            print(
                f"Probabilities for next token (top 5): {np.sort(probabilities)[-5:][::-1].round(3)}"
            )
            print(f"Predicted next token: '{next_word}' (ID: {next_token_id})")

            # f. Append the new token to our sequence.
            generated_ids.append(next_token_id)

            # g. If the model predicts the stop token, we're done.
            if next_token_id == stop_token_id:
                print("\nStop token generated. Halting generation.")
                break
        else:
            print("\nMax length reached. Halting generation.")

        return generated_ids


def main():
    """
    Demonstrates a full Encoder-Decoder Transformer model for a sequence-to-sequence task.
    """
    # --- 1. Setup: Vocabulary and Data ---
    source_sentence = "Life is short eat dessert first"
    target_sentence = "[START] first dessert eat short is Life [STOP]"
    source_tokens, target_tokens = source_sentence.split(), target_sentence.split()

    vocab = sorted(list(set(source_tokens + target_tokens)))
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for word, i in word_to_idx.items()}
    vocab_size = len(vocab)

    # Shape: (source_seq_len,)
    source_ids = np.array([word_to_idx[token] for token in source_tokens])
    # Shape: (target_seq_len,)
    target_ids = np.array([word_to_idx[token] for token in target_tokens])

    # --- 2. Define Model Hyperparameters ---
    num_layers, d_model, num_heads, d_ff, max_seq_len = 2, 32, 4, 64, 20
    d_v = d_model // num_heads
    d_k = 6

    # --- 3. Instantiate the Transformer Model ---
    model = Transformer(
        num_layers,
        d_model,
        num_heads,
        d_ff,
        d_k,
        d_v,
        vocab_size,
        vocab_size,
        max_seq_len,
    )

    # --- 4. Run Auto-Regressive Inference (Generation) ---
    print("\n--- Running Auto-Regressive Inference ---")
    start_token_id = word_to_idx["[START]"]
    stop_token_id = word_to_idx["[STOP]"]

    # generated_ids is a list of token IDs.
    generated_ids = model.generate(
        source_ids,
        start_token_id=start_token_id,
        stop_token_id=stop_token_id,
        idx_to_word=idx_to_word,
        max_len=10,  # Set a max generation length
    )

    # --- 5. Display Results ---
    generated_words = [idx_to_word.get(idx, "[UNK]") for idx in generated_ids]
    print(f"\nEncoder Input:   {' '.join(source_tokens)}")
    print(f"Generated Output: {' '.join(generated_words)}")
    print(f"Ground Truth:     {' '.join(target_tokens)}")


if __name__ == "__main__":
    main()
