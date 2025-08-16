from typing import List
import bisect


class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        """
        Returns the length of the longest strictly increasing subsequence.
        Uses patience sorting (O(n log n) time).
        """
        if not nums:
            return 0

        # tails[i] will be the smallest possible tail of an increasing subsequence of length i+1
        # For example, tails[0] is the smallest tail of all length-1 increasing subsequences,
        # tails[1] is the smallest tail of all length-2 increasing subsequences, etc.
        tails = []

        for num in nums:
            # Use binary search to find the leftmost index in tails where num can be placed.
            # If num is greater than all tails, bisect_left returns len(tails) (i.e., append).
            # Otherwise, it returns the first index where tails[idx] >= num.
            idx = bisect.bisect_left(tails, num)

            if idx == len(tails):
                # num is greater than any element in tails, so it extends the largest subsequence.
                tails.append(num)
            else:
                # num could be a new, smaller tail for a subsequence of length idx+1.
                # This does NOT mean we've found a longer subsequence, but we keep tails as small as possible.
                tails[idx] = num

            # Example trace:
            # nums = [10, 9, 2, 5, 3, 7, 101, 18]
            # tails evolves as:
            # [10]
            # [9]
            # [2]
            # [2, 5]
            # [2, 3]
            # [2, 3, 7]
            # [2, 3, 7, 101]
            # [2, 3, 7, 18]
            # The length of tails is the answer (4).

        # The length of tails is the length of the LIS.
        return len(tails)


# Unit tests
def test():
    sol = Solution()
    # Example 1: LIS is [2,3,7,101], length 4
    assert sol.lengthOfLIS([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    # Example 2: LIS is [0,1,2,3], length 4
    assert sol.lengthOfLIS([0, 1, 0, 3, 2, 3]) == 4
    # Example 3: All elements are the same, LIS is [7], length 1
    assert sol.lengthOfLIS([7, 7, 7, 7, 7, 7, 7]) == 1
    # Example 4: Single element, LIS is [1], length 1
    assert sol.lengthOfLIS([1]) == 1
    # Example 5: LIS is [4,8,9], length 3
    assert sol.lengthOfLIS([4, 10, 4, 3, 8, 9]) == 3
    print("All tests passed.")


if __name__ == "__main__":
    test()
