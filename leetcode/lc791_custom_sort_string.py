from collections import Counter


class Solution:
    def customSortString(self, order: str, s: str) -> str:
        """
        Sorts the string s according to the custom order given in order.
        Characters not in order can appear in any position after those in order.
        """
        # Create a mapping from character to its position in 'order'
        score = {ch: i for i, ch in enumerate(order)}
        # Convert s to a list for sorting
        s_list = list(s)
        # Sort s_list using the score mapping, defaulting to a large value for chars not in order
        s_list.sort(key=lambda ch: score.get(ch, 1000))
        # Return the sorted list as a string
        return "".join(s_list)


class SolutionCounter:
    def customSortString(self, order: str, s: str) -> str:
        """
        Sorts the string s according to the custom order given in order.
        Characters not in order can appear in any position after those in order.
        """
        count = Counter(s)
        result = []
        # Add characters in the order specified by 'order'
        for ch in order:
            if ch in count:
                result.append(ch * count[ch])
                del count[ch]
        # Add remaining characters (not in 'order') in any order
        for ch, freq in count.items():
            result.append(ch * freq)
        return "".join(result)


# Unit tests
def test():
    sol = Solution()
    # Example 1
    assert sol.customSortString("cba", "abcd") in ["cbad", "cbda"]
    # Example 2
    assert sol.customSortString("bcafg", "abcd") in ["bcad", "bcda"]
    # Example 3: All characters in order
    assert sol.customSortString("abc", "abc") == "abc"
    # Example 4: Characters in s not in order
    assert sol.customSortString("kqep", "pekeq") in ["kqeep", "kqpee"]
    # Example 5: order is empty
    assert sol.customSortString("", "abc") in ["abc", "acb", "bac", "bca", "cab", "cba"]
    # Example 6: s is empty
    assert sol.customSortString("abc", "") == ""
    print("All tests passed.")


if __name__ == "__main__":
    test()
