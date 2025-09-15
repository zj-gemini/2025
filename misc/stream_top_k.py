import heapq
from typing import List


class KthLargest:
    """
    Design a class to find the kth largest element in a stream.

    This class maintains a min-heap of size k. The root of this min-heap
    is always the kth largest element seen so far.

    - __init__: Initializes the heap with the initial numbers. It keeps only
      the k largest elements, so the heap size is at most k.
    - add: Adds a new element to the stream. If the heap is smaller than k,
      the element is added. If the heap is full (size k), the new element
      is added only if it's larger than the smallest element in the heap
      (the root). This keeps the heap updated with the k largest elements.
    """

    def __init__(self, k: int, nums: List[int]):
        self.heap = nums
        self.k = k
        # Convert the list into a min-heap
        heapq.heapify(self.heap)
        # Pop elements until the heap size is k
        while len(self.heap) > self.k:
            heapq.heappop(self.heap)

    def add(self, val: int) -> int:
        """
        Adds a value to the stream and returns the current kth largest element.
        """
        if len(self.heap) < self.k:
            # If the heap isn't full, just add the new element
            heapq.heappush(self.heap, val)
        elif val > self.heap[0]:
            # If the new value is larger than the smallest element in the heap (the root),
            # replace the root with the new value. This is more efficient than a separate push and pop.
            heapq.heappushpop(self.heap, val)

        # The root of the min-heap is the kth largest element
        return self.heap[0]


# --- Unit Tests ---
def test_kth_largest():
    """
    Tests the KthLargest class with a sequence of operations.
    """
    # LeetCode Example
    k = 3
    nums = [4, 5, 8, 2]
    kth_largest = KthLargest(k, nums)
    # Initial state: heap contains [4, 5, 8]. The root (kth largest) is 4.
    assert kth_largest.add(3) == 4  # heap: [4, 5, 8]. 3 is smaller than 4, no change.
    assert kth_largest.add(5) == 5  # heap: [5, 5, 8]. 5 > 4, so 4 is replaced by 5.
    assert kth_largest.add(10) == 5  # heap: [5, 8, 10]. 10 > 5, so 5 is replaced by 10.
    assert kth_largest.add(9) == 8  # heap: [8, 9, 10]. 9 > 5, so 5 is replaced by 9.
    assert kth_largest.add(4) == 8  # heap: [8, 9, 10]. 4 is smaller than 8, no change.

    # Test with empty initial list
    kth_largest_empty = KthLargest(1, [])
    assert kth_largest_empty.add(-3) == -3
    assert kth_largest_empty.add(-2) == -2
    assert kth_largest_empty.add(-4) == -2
    assert kth_largest_empty.add(0) == 0
    assert kth_largest_empty.add(4) == 4

    print("All test cases passed!")


if __name__ == "__main__":
    test_kth_largest()
