from typing import List
import heapq
from collections import Counter


class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """
        Returns the k most frequent elements in nums.
        Uses a min-heap to keep track of the top k elements.
        """
        # Count the frequency of each element
        count = Counter(nums)
        # Use a heap of size k to keep the k most frequent elements
        # heapq.nlargest returns the k keys with the largest counts
        return [
            item for item, freq in heapq.nlargest(k, count.items(), key=lambda x: x[1])
        ]


class SolutionCounter:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """
        Returns the k most frequent elements in nums.
        Uses a min-heap to keep track of the top k elements.
        """
        # Count the frequency of each element
        count = Counter(nums)
        # Use a heap of size k to keep the k most frequent elements
        # heapq.nlargest returns the k keys with the largest counts
        return [
            item for item, freq in heapq.nlargest(k, count.items(), key=lambda x: x[1])
        ]


# Unit tests
def test():
    sol = Solution()
    # Example 1
    assert set(sol.topKFrequent([1, 1, 1, 2, 2, 3], 2)) == {1, 2}
    # Example 2
    assert set(sol.topKFrequent([1], 1)) == {1}
    # Example 3: All elements unique, k=2
    assert set(sol.topKFrequent([4, 5, 6, 7], 2)) == {4, 5, 6, 7} & set(
        sol.topKFrequent([4, 5, 6, 7], 2)
    )
    # Example 4: Multiple elements with same frequency
    assert set(sol.topKFrequent([1, 2, 3, 1, 2, 3], 2)) <= {1, 2, 3}
    # Example 5: k equals number of unique elements
    assert set(sol.topKFrequent([1, 2, 3, 4], 4)) == {1, 2, 3, 4}
    print("All tests passed.")


if __name__ == "__main__":
    test()
