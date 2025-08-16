class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # To optimize space, always use the shorter string for columns
        if len(text1) < len(text2):
            text1, text2 = text2, text1
        m, n = len(text1), len(text2)
        # prev and curr are 1D DP arrays representing previous and current rows
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                # If current characters match, extend the LCS by 1
                if text1[i - 1] == text2[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    # Otherwise, take the max LCS by skipping a character from either string
                    curr[j] = max(prev[j], curr[j - 1])
            # Move to the next row
            prev = curr
        # The last cell contains the length of the longest common subsequence
        return prev[n]


# Regular test code
def test():
    sol = Solution()
    print(sol.longestCommonSubsequence("abcde", "ace"))  # Output: 3
    print(sol.longestCommonSubsequence("abc", "abc"))  # Output: 3
    print(sol.longestCommonSubsequence("abc", "def"))  # Output: 0
    print(sol.longestCommonSubsequence("", ""))  # Output: 0
    print(sol.longestCommonSubsequence("a", ""))  # Output: 0
    print(sol.longestCommonSubsequence("", "a"))  # Output: 0
    print(sol.longestCommonSubsequence("a", "a"))  # Output: 1
    print(sol.longestCommonSubsequence("a", "b"))  # Output: 0
    print(sol.longestCommonSubsequence("abcde", "abfce"))  # Output: 3


test()
