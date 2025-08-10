from collections import Counter


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        """
        Returns the minimum window substring of s such that every character in t is included.
        If no such window exists, returns an empty string.
        """
        if not t or not s:
            return ""
        need = Counter(t)  # Count of each character needed from t
        min_len = float("inf")  # Minimum window length found
        start = 0  # Start index of the minimum window
        lo = 0  # Left pointer of the window
        n_valid_chs = 0  # Number of valid characters included in the window

        # Expand the window by moving 'hi' (right pointer)
        for hi, ch in enumerate(s):
            need[ch] -= 1  # Use one occurrence of ch
            # Only when ch is still needed (not redundant), update n_valid_chs
            if need[ch] >= 0:
                n_valid_chs += 1
            # Try to shrink the window from the left as long as all chars are covered
            while n_valid_chs == len(t):
                # Current [lo, hi] is a valid substring
                window_len = hi - lo + 1
                if window_len < min_len:
                    min_len = window_len
                    start = lo  # Update the start of the minimum window
                ch_remove = s[lo]
                need[ch_remove] += 1  # Put back the character at 'lo'
                # If we now need more of ch_remove, decrease n_valid_chs
                if need[ch_remove] > 0:
                    n_valid_chs -= 1
                lo += 1  # Move left pointer to shrink the window
        # Return the minimum window substring, or "" if not found
        return s[start : start + min_len] if min_len != float("inf") else ""


class SolutionCopilot:
    def minWindow(self, s: str, t: str) -> str:
        """
        Returns the minimum window substring of s such that every character in t is included.
        If no such window exists, returns an empty string.
        """
        if not t or not s:
            return ""
        need = Counter(t)  # Count of each character needed from t
        missing = len(t)  # Total number of characters still needed
        left = start = end = 0  # Window pointers and result indices
        min_len = float("inf")  # Minimum window length found

        # Expand the window by moving 'right'
        for right, char in enumerate(s):
            if need[char] > 0:
                missing -= 1  # Found a needed character
            need[
                char
            ] -= 1  # Decrement count for this character (can go negative if extra)

            # When all characters are matched, try to shrink the window from the left
            while missing == 0:
                # Update result if this window is smaller than previous ones
                if right - left + 1 < min_len:
                    min_len = right - left + 1
                    start, end = left, right + 1
                # Move left pointer to try and minimize window
                need[s[left]] += 1  # Put back the character at 'left'
                if need[s[left]] > 0:
                    missing += 1  # Now missing this character again
                left += 1  # Shrink window from the left

        # Return the minimum window substring, or "" if not found
        return s[start:end] if min_len != float("inf") else ""


# Unit tests
def test():
    sol = Solution()
    # Example 1
    assert sol.minWindow("ADOBECODEBANC", "ABC") == "BANC"
    # Example 2
    assert sol.minWindow("a", "a") == "a"
    # Example 3
    assert sol.minWindow("a", "aa") == ""
    # Example 4: t not in s
    assert sol.minWindow("abc", "xyz") == ""
    # Example 5: s and t are empty
    assert sol.minWindow("", "") == ""
    # Example 6: t is empty
    assert sol.minWindow("abc", "") == ""
    # Example 7: s is empty
    assert sol.minWindow("", "a") == ""
    print("All tests passed.")


if __name__ == "__main__":
    test()
