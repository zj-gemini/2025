from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
        Returns indices of the two numbers such that they add up to target.
        Uses a hash map to store previously seen numbers.
        """
        num_to_index = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in num_to_index:
                return [num_to_index[complement], i]
            num_to_index[num] = i
        return []


# Unit tests
def test():
    sol = Solution()
    # Example 1
    assert sol.twoSum([2, 7, 11, 15], 9) == [0, 1]
    # Example 2
    assert sol.twoSum([3, 2, 4], 6) == [1, 2]
    # Example 3
    assert sol.twoSum([3, 3], 6) == [0, 1]
    # Example 4: Negative numbers
    assert sol.twoSum([-1, -2, -3, -4, -5], -8) == [2, 4]
    # Example 5: No solution
    assert sol.twoSum([1, 2, 3], 7) == []
    print("All tests passed.")


if __name__ == "__main__":
    test()
