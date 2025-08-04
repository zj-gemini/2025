import math
import dataclasses


@dataclasses.dataclass
class Node:
    prob: float
    tokens: list[str]


class UnigramTokenizer:
    def __init__(self, vocab):
        """
        vocab: dict mapping token -> probability (0 < p <= 1)
        Convert probabilities to log-probabilities for numerical stability.
        """
        self.log_vocab = {token: math.log(prob) for token, prob in vocab.items()}
        self.max_token_len = max(len(token) for token in vocab)

    def tokenize(self, text):
        """
        Uses Viterbi (DP) to find the best tokenization of text.
        dp[i]: (max_log_prob, best_tokenization) for text[:i]
        For each position i, try all possible tokens ending at i.
        """
        n = len(text)
        # dp[i] holds the best tokenization for text[:i]
        dp = [Node(-float("inf"), []) for _ in range(n + 1)]
        dp[0] = Node(0.0, [])  # Base case: empty string

        for i in range(n):
            # Try all possible tokens starting at i
            for token_len in range(1, self.max_token_len + 1):
                if i + token_len > n:
                    break
                token = text[i : i + token_len]
                if token in self.log_vocab:
                    log_prob = dp[i].prob + self.log_vocab[token]
                    # Update if this split gives higher probability
                    if log_prob > dp[i + token_len].prob:
                        dp[i + token_len] = Node(log_prob, dp[i].tokens + [token])
        return dp[n].tokens


def test():
    # Example vocab: tokens and their probabilities
    vocab = {
        "i": 0.1,
        "like": 0.05,
        "sam": 0.02,
        "sung": 0.02,
        "samsung": 0.03,
        "mobile": 0.04,
        "ice": 0.02,
        "cream": 0.03,
        "icecream": 0.04,
        "man": 0.03,
        "go": 0.02,
        "mango": 0.04,
    }
    tokenizer = UnigramTokenizer(vocab)
    print(
        tokenizer.tokenize("ilikesamsungmobile")
    )  # ['i', 'like', 'samsung', 'mobile']
    print(tokenizer.tokenize("ilikeicecream"))  # ['i', 'like', 'icecream']
    print(tokenizer.tokenize("ilikeicecreamman"))  # ['i', 'like', 'icecream', 'man']
    print(tokenizer.tokenize("mango"))  # ['mango']
    print(tokenizer.tokenize("samsungmobile"))  # ['samsung', 'mobile']


test()
