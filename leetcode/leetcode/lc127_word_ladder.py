from typing import List
from collections import deque, defaultdict


class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if endWord not in wordList:
            return 0

        L = len(beginWord)
        # Preprocess: build all generic states mapping to words
        # For example, "hot" -> "*ot", "h*t", "ho*"
        all_combo_dict = defaultdict(list)
        for word in wordList:
            for i in range(L):
                pattern = word[:i] + "*" + word[i + 1 :]
                all_combo_dict[pattern].append(word)

        # BFS queue: (current_word, current_level)
        queue = deque([(beginWord, 1)])
        visited = set([beginWord])
        while queue:
            word, level = queue.popleft()
            # Try changing each letter in the current word
            for i in range(L):
                pattern = word[:i] + "*" + word[i + 1 :]
                # For all words matching this pattern
                for next_word in all_combo_dict[pattern]:
                    if next_word == endWord:
                        return level + 1  # Found the end word
                    if next_word not in visited:
                        visited.add(next_word)
                        queue.append((next_word, level + 1))
                # Clear the pattern list to avoid revisiting
                all_combo_dict[pattern] = []
        return 0  # No transformation found


def test():
    sol = Solution()
    print(
        sol.ladderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"])
    )  # 5
    print(sol.ladderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log"]))  # 0
    print(sol.ladderLength("a", "c", ["a", "b", "c"]))  # 2


# Uncomment to run tests
test()
