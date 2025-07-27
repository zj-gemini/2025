from typing import List
from collections import deque, defaultdict
import dataclasses


@dataclasses.dataclass
class Node:
    word: str
    step: int


class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        word_set = set(wordList)
        if endWord not in word_set or beginWord == endWord:
            return 0

        q = deque([Node(beginWord, 1)])
        word_set.discard(beginWord)
        while q:
            cur = q.pop()
            visited = set()
            for next in word_set:
                n_diff = 0
                for c1, c2 in zip(cur.word, next):
                    if c1 != c2:
                        n_diff += 1
                        if n_diff > 1:
                            break
                if n_diff != 1:
                    continue
                if next == endWord:
                    return cur.step + 1
                q.appendleft(Node(next, cur.step + 1))
                visited.add(next)
            for to_remove in visited:
                word_set.discard(to_remove)
        return 0


def test():
    sol = Solution()
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
    )  # 5
    return
    print(
        sol.ladderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"])
    )  # 5
    print(sol.ladderLength("hit", "cog", ["hot", "dot", "dog", "lot", "log"]))  # 0
    print(sol.ladderLength("a", "c", ["a", "b", "c"]))  # 2


# Uncomment to run tests
test()
