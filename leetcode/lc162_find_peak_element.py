from typing import List


class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        """
        Finds a peak element and returns its index.
        A peak element is one that is strictly greater than its neighbors.
        Uses binary search for O(log n) time.
        """
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            # If mid element is less than its right neighbor, peak must be to the right
            if nums[mid] < nums[mid + 1]:
                left = mid + 1
            else:
                # Peak is at mid or to the left
                right = mid
        return left  # or right, since left == right


# Unit tests
def test():
    sol = Solution()
    # Example 1: Peak at index 2 (2)
    assert sol.findPeakElement([1, 2, 3, 1]) == 2
    # Example 2: Peak at index 5 (6)
    assert sol.findPeakElement([1, 2, 1, 3, 5, 6, 4]) in [1, 5]
    # Single element
    assert sol.findPeakElement([1]) == 0
    # Two elements, peak at index 1
    assert sol.findPeakElement([1, 2]) == 1
    # Two elements, peak at index 0
    assert sol.findPeakElement([2, 1]) == 0
    # All elements equal, any index is a peak
    assert sol.findPeakElement([3, 3, 3, 3]) in [0, 1, 2, 3]
    print("All tests passed.")


if __name__ == "__main__":
    test()
