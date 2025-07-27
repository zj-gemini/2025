import heapq
from typing import List


class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # Use a min-heap of size k
        heap = nums[:k]
        heapq.heapify(heap)
        for num in nums[k:]:
            if num > heap[0]:
                heapq.heappushpop(heap, num)
        return heap[0]


def test():
    sol = Solution()
    print(sol.findKthLargest([3, 2, 1, 5, 6, 4], 2))  # Output: 5
    print(sol.findKthLargest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4))  # Output: 4
    print(sol.findKthLargest([1], 1))  # Output: 1
    print(sol.findKthLargest([2, 1], 2))  # Output: 1


test()
