from collections import defaultdict


class Node:
    def __init__(self):
        self.children = defaultdict(Node)
        self.is_end = False


class Trie:
    def __init__(self):
        # Root node does not hold any character
        self.root = Node()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            node = node.children[ch]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True


# Regular test code
def test():
    trie = Trie()
    trie.insert("apple")
    print(trie.search("apple"))  # True
    print(trie.search("app"))  # False
    print(trie.startsWith("app"))  # True
    trie.insert("app")
    print(trie.search("app"))  # True
    trie.insert("banana")
    print(trie.search("banana"))  # True
    print(trie.startsWith("ban"))  # True
    print(trie.startsWith("bat"))  # False


test()
