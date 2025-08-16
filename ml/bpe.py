# Toy BPE tokenizer training and tokenization

from collections import Counter, defaultdict


class BPETokenizer:
    def __init__(self, vocab_size=9, end_token="</eow>"):
        self.vocab_size = vocab_size
        self.end_token = end_token
        self.vocab = set()
        self.merges = []

    def train(self, corpus):
        # Step 1: Prepare corpus as list of symbol sequences
        # Each word is split into chars + </eow>
        corpus_tokens = []
        for word, count in corpus:
            tokens = list(word) + [self.end_token]
            for _ in range(count):
                corpus_tokens.append(tokens[:])

        # Step 2: Build initial vocab (all chars + </eow>)
        self.vocab = set()
        for tokens in corpus_tokens:
            self.vocab.update(tokens)

        # Step 3: Iteratively merge most frequent pairs
        while len(self.vocab) < self.vocab_size:
            # Count all symbol pairs
            pairs = Counter()
            for tokens in corpus_tokens:
                for i in range(len(tokens) - 1):
                    pairs[(tokens[i], tokens[i + 1])] += 1
            if not pairs:
                break
            # Find most frequent pair
            best_pair, _ = pairs.most_common(1)[0]
            self.merges.append(best_pair)
            # Merge the best pair in all tokens
            new_corpus_tokens = []
            for tokens in corpus_tokens:
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                        new_tokens.append(tokens[i] + tokens[i + 1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                new_corpus_tokens.append(new_tokens)
            corpus_tokens = new_corpus_tokens
            # Update vocab
            self.vocab = set()
            for tokens in corpus_tokens:
                self.vocab.update(tokens)

    def tokenize(self, word):
        # Tokenize a word using learned merges
        tokens = list(word) + [self.end_token]
        for merge in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == merge:
                    tokens[i : i + 2] = [tokens[i] + tokens[i + 1]]
                else:
                    i += 1
        return tokens


if __name__ == "__main__":
    # Training corpus: [("banana", 2), ("bandana", 1)]
    corpus = [("banana", 2), ("bandana", 1)]
    bpe = BPETokenizer(vocab_size=9, end_token="</eow>")
    bpe.train(corpus)

    print("Merges:", bpe.merges)
    print("Vocab:", bpe.vocab)

    # Tokenize test words
    for word in ["banana", "band", "bandana"]:
        print(f"{word} -> {bpe.tokenize(word)}")

# Output:
# Merges: [('a', 'n'), ('b', 'an'), ('an', 'a'), ('ana', '</eow>'), ('ban', 'ana</eow>'), ('ban', 'd'), ('band', 'ana</eow>')]
# Vocab: {'banana</eow>', 'bandana</eow>'}
# banana -> ['banana</eow>']
# band -> ['band', '</eow>']
# bandana -> ['bandana</eow>']
