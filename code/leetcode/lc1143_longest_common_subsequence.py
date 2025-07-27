class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        # The longgest common subsquece with first i characters of text1 and first j characters of text2
        dp = [[""] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + text1[i - 1]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], key=len)
        print(dp[m][n])  # Debugging output to see the DP table
        return len(dp[m][n])


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


test()
