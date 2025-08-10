# Solution for LeetCode 1004: Max Consecutive Ones III

from typing import List
import unittest


class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        """
        Finds the longest subarray containing only 1s after flipping at most k 0s to 1s.
        Uses a sliding window approach.
        """
        left = 0  # Left pointer of the window
        zeros = 0  # Number of zeros in the current window
        max_len = 0  # Maximum length found
        for right in range(len(nums)):
            if nums[right] == 0:
                zeros += 1  # Increment zero count if current is 0
            # If there are more than k zeros, shrink window from the left
            while zeros > k:
                if nums[left] == 0:
                    zeros -= 1  # Decrement zero count as we move left pointer
                left += 1
            # Update max_len if current window is larger
            max_len = max(max_len, right - left + 1)
        return max_len


# Unit tests
class TestLongestOnes(unittest.TestCase):
    def setUp(self):
        self.sol = Solution()

    def test_example1(self):
        # Flip at most 2 zeros to get max 6 consecutive 1s
        self.assertEqual(self.sol.longestOnes([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], 2), 6)

    def test_example2(self):
        # No flips allowed, max consecutive 1s is 3
        self.assertEqual(self.sol.longestOnes([0, 0, 1, 1, 1, 0, 0], 0), 3)

    def test_all_ones(self):
        # All ones, flipping any zeros doesn't change result
        self.assertEqual(self.sol.longestOnes([1, 1, 1, 1], 2), 4)

    def test_all_zeros(self):
        # All zeros, can flip at most 2 to get 2 consecutive 1s
        self.assertEqual(self.sol.longestOnes([0, 0, 0, 0], 2), 2)

    def test_empty(self):
        # Empty array, result is 0
        self.assertEqual(self.sol.longestOnes([], 2), 0)

    def test_k_greater_than_zeros(self):
        # k is larger than number of zeros, can flip all zeros
        self.assertEqual(self.sol.longestOnes([1, 0, 1, 0, 1], 10), 5)


if __name__ == "__main__":
    unittest.main()
