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
        self.vocab = vocab
        self.log_vocab = {token: math.log(prob) for token, prob in vocab.items()}

    def tokenize(self, text):
        """
        Uses Viterbi (DP) to find the best tokenization of text.
        dp[i]: (max_log_prob, best_tokenization) for text[:i]
        For each position i, try all possible tokens ending at i.
        """
        n = len(text)
        dp = [Node(-float("inf"), []) for _ in range(n + 1)]
        dp[0] = Node(0.0, [])  # Base case: empty string

        for i in range(1, n + 1):
            # Try all possible tokens ending at position i
            for j in range(max(0, i - 20), i):  # Limit token length for efficiency
                token = text[j:i]
                if token in self.log_vocab:
                    log_prob = dp[j].prob + self.log_vocab[token]
                    # Update if this split gives higher probability
                    if log_prob > dp[i].prob:
                        dp[i] = Node(log_prob, dp[j].tokens + [token])
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
