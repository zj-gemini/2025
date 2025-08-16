from typing import List
from collections import defaultdict


class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.word = None  # Store word at end node for fast lookup


class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        # Build Trie
        root = TrieNode()
        for word in words:
            node = root
            for ch in word:
                if ch not in node.children:
                    # Create a new TrieNode if the character is not present
                    node.children[ch] = TrieNode()
                node = node.children[ch]
            node.word = word

        m, n = len(board), len(board[0])
        result = set()

        def dfs(x, y, node):
            ch = board[x][y]
            if ch not in node.children:
                return
            nxt = node.children[ch]
            if nxt.word:
                result.add(nxt.word)
                nxt.word = None  # Avoid duplicate results, not neccessary.

            board[x][y] = None  # Mark visited
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and board[nx][ny] is not None:
                    dfs(nx, ny, nxt)
            board[x][y] = ch  # Restore

        for i in range(m):
            for j in range(n):
                dfs(i, j, root)

        return list(result)


# Regular test code
def test():
    sol = Solution()
    board1 = [
        ["o", "a", "a", "n"],
        ["e", "t", "a", "e"],
        ["i", "h", "k", "r"],
        ["i", "f", "l", "v"],
    ]
    words1 = ["oath", "pea", "eat", "rain"]
    print(sorted(sol.findWords(board1, words1)))  # Output: ['eat', 'oath']

    board2 = [["a", "b"], ["c", "d"]]
    words2 = ["abcb"]
    print(sol.findWords(board2, words2))  # Output: []

    board3 = [["a"]]
    words3 = ["a"]
    print(sol.findWords(board3, words3))  # Output: ['a']

    board4 = [["a", "b"], ["c", "d"]]
    words4 = ["abcd", "acdb", "bacd"]
    print(sorted(sol.findWords(board4, words4)))  # Output:


test()
