from typing import List
from collections import deque
import dataclasses


@dataclasses.dataclass
class Node:
    word: str
    step: int  # Number of steps from beginWord to this word


class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        word_set = set(wordList)  # Use a set for O(1) lookups and removals
        if endWord not in word_set or beginWord == endWord:
            return 0  # No possible transformation

        # BFS queue initialization
        q = deque([Node(beginWord, 1)])
        word_set.discard(beginWord)  # Remove beginWord from set to avoid revisiting

        while q:
            cur = q.pop()  # Get current node (word and step count)
            visited = set()  # Track words visited in this BFS layer

            # Try all possible next words in the word_set
            for next in word_set:
                n_diff = 0  # Count character differences
                for c1, c2 in zip(cur.word, next):
                    if c1 != c2:
                        n_diff += 1
                        if n_diff > 1:
                            break  # More than one letter difference, skip

                if n_diff != 1:
                    continue  # Only consider words differing by one letter

                if next == endWord:
                    return cur.step + 1  # Found the endWord, return steps

                q.appendleft(Node(next, cur.step + 1))  # Add next word to BFS queue
                visited.add(next)  # Mark as visited for this layer

            # Remove all visited words from word_set to prevent revisiting
            for to_remove in visited:
                word_set.discard(to_remove)

        return 0  # No transformation sequence found


def test():
    sol = Solution()
    # Test case 1: Large word list, expect 5
    print(
        sol.ladderLength(
            "qa",
            "sq",
            [
                "si",
                "go",
                "se",
                "cm",
                "so",
                "ph",
                "mt",
                "db",
                "mb",
                "sb",
                "kr",
                "ln",
                "tm",
                "le",
                "av",
                "sm",
                "ar",
                "ci",
                "ca",
                "br",
                "ti",
                "ba",
                "to",
                "ra",
                "fa",
                "yo",
                "ow",
                "sn",
                "ya",
                "cr",
                "po",
                "fe",
                "ho",
                "ma",
                "re",
                "or",
                "rn",
                "au",
                "ur",
                "rh",
                "sr",
                "tc",
                "lt",
                "lo",
                "as",
                "fr",
                "nb",
                "yb",
                "if",
                "pb",
                "ge",
                "th",
                "pm",
                "rb",
                "sh",
                "co",
                "ga",
                "li",
                "ha",
                "hz",
                "no",
                "bi",
                "di",
                "hi",
                "qa",
                "pi",
                "os",
                "uh",
                "wm",
                "an",
                "me",
                "mo",
                "na",
                "la",
                "st",
                "er",
                "sc",
                "ne",
                "mn",
                "mi",
                "am",
                "ex",
                "pt",
                "io",
                "be",
                "fm",
                "ta",
                "tb",
                "ni",
                "mr",
                "pa",
                "he",
                "lr",
                "sq",
                "ye",
            ],
        )
    )  # Output: 5

    return

    # Test case 2: Standard example, expect 5
    print(
        sol.ladderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"])
    )  # Output: 5
    # Test case 3: No possible transformation, expect 0
    print(
        sol.ladderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log"])
    )  # Output: 0
    # Test case 4: Short transformation, expect 2
    print(sol.ladderLength("a", "c", ["a", "b", "c"]))  # Output: 2


# Uncomment to run tests
test()
