from typing import List
from collections import defaultdict


class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        """
        Returns the total number of continuous subarrays whose sum equals to k.
        Uses a hashmap to store the cumulative sum frequencies.
        """
        count = 0
        cum_sum = 0
        sum_freq = defaultdict(int)
        sum_freq[0] = 1  # There is one way to have a sum of 0 (empty prefix)
        for num in nums:
            cum_sum += num
            # If (cum_sum - k) has been seen, there is a subarray ending here with sum k
            count += sum_freq[cum_sum - k]
            sum_freq[cum_sum] += 1
        return count


# Unit tests
def test():
    sol = Solution()
    # Example 1
    assert sol.subarraySum([1, 1, 1], 2) == 2
    # Example 2
    assert sol.subarraySum([1, 2, 3], 3) == 2
    # Example 3: Negative numbers
    assert sol.subarraySum([1, -1, 0], 0) == 3
    # Example 4: All zeros
    assert sol.subarraySum([0, 0, 0, 0], 0) == 10
    # Example 5: No subarray found
    assert sol.subarraySum([1, 2, 3], 7) == 0
    print("All tests passed.")


if __name__ == "__main__":
    test()
