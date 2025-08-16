from typing import List
import bisect


class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        """
        Returns the starting and ending position of a given target value in a sorted array.
        If target is not found, returns [-1, -1].
        Uses binary search for O(log n) time.
        """
        if not nums:
            return [-1, -1]
        # Find the leftmost (first) index where target could be inserted
        left = bisect.bisect_left(nums, target)
        # If left is out of bounds or not equal to target, target is not present
        if left >= len(nums) or nums[left] != target:
            return [-1, -1]
        # Find the rightmost (last+1) index where target could be inserted
        right = bisect.bisect_right(nums, target)
        # right - 1 is the last occurrence of target
        return [left, right - 1]


class SolutionOld:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        """
        Returns the starting and ending position of a given target value in a sorted array.
        If target is not found, returns [-1, -1].
        Uses binary search for O(log n) time.
        """

        def find_left():
            left, right = 0, len(nums) - 1
            res = -1
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] < target:
                    left = mid + 1
                else:
                    if nums[mid] == target:
                        res = mid  # Update result, but keep searching left
                    right = mid - 1
            return res

        def find_right():
            left, right = 0, len(nums) - 1
            res = -1
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] > target:
                    right = mid - 1
                else:
                    if nums[mid] == target:
                        res = mid  # Update result, but keep searching right
                    left = mid + 1
            return res

        left = find_left()
        right = find_right()
        return [left, right] if left != -1 else [-1, -1]


# Unit tests
def test():
    sol = Solution()
    # Example 1
    assert sol.searchRange([5, 7, 7, 8, 8, 10], 8) == [3, 4]
    # Example 2
    assert sol.searchRange([5, 7, 7, 8, 8, 10], 6) == [-1, -1]
    # Example 3: Single element, found
    assert sol.searchRange([1], 1) == [0, 0]
    # Example 4: Single element, not found
    assert sol.searchRange([1], 0) == [-1, -1]
    # Example 5: All elements are the target
    assert sol.searchRange([2, 2, 2, 2, 2], 2) == [0, 4]
    # Example 6: Target at the beginning
    assert sol.searchRange([1, 2, 3, 4, 5], 1) == [0, 0]
    # Example 7: Target at the end
    assert sol.searchRange([1, 2, 3, 4, 5], 5) == [4, 4]
    print("All tests passed.")


if __name__ == "__main__":
    test()
