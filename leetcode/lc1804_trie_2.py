from collections import defaultdict


class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        # Number of words ending here
        self.count = 0
        # Number of words passing through here (including ending here)
        self.prefix_count = 0


class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            node = node.children[ch]
            node.prefix_count += 1
        node.count += 1

    def countWordsEqualTo(self, word: str) -> int:
        node = self.root
        for ch in word:
            if ch not in node.children:
                return 0
            node = node.children[ch]
        return node.count

    def countWordsStartingWith(self, prefix: str) -> int:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return 0
            node = node.children[ch]
        return node.prefix_count

    def erase(self, word: str) -> None:
        node = self.root
        stack = []
        for ch in word:
            stack.append((node, ch))
            node = node.children[ch]
            node.prefix_count -= 1
        node.count -= 1
        # Optional: Clean up nodes with no children and no words
        for parent, ch in reversed(stack):
            child = parent.children[ch]
            if child.prefix_count == 0 and child.count == 0:
                del parent.children[ch]
            else:
                break


# Unit tests
def test_trie():
    trie = Trie()
    trie.insert("apple")
    trie.insert("apple")
    assert trie.countWordsEqualTo("apple") == 2
    assert trie.countWordsStartingWith("app") == 2
    trie.erase("apple")
    assert trie.countWordsEqualTo("apple") == 1
    assert trie.countWordsStartingWith("app") == 1
    trie.erase("apple")
    assert trie.countWordsStartingWith("app") == 0
    trie.insert("banana")
    trie.insert("band")
    trie.insert("bandana")
    assert trie.countWordsStartingWith("ban") == 3
    assert trie.countWordsEqualTo("banana") == 1
    trie.erase("banana")
    assert trie.countWordsStartingWith("ban") == 2
    assert trie.countWordsEqualTo("banana") == 0
    print("All tests passed.")


if __name__ == "__main__":
    test_trie()
