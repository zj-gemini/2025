from typing import List


class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        Merges all overlapping intervals and returns a list of non-overlapping intervals.
        """
        if not intervals:
            return []
        # Sort intervals by start time
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            last_end = merged[-1][1]
            if start <= last_end:
                # Overlap: merge with the last interval
                merged[-1][1] = max(last_end, end)
            else:
                # No overlap: add as a new interval
                merged.append([start, end])
        return merged


# Unit tests
def test():
    sol = Solution()
    # Example 1
    assert sol.merge([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
    # Example 2
    assert sol.merge([[1, 4], [4, 5]]) == [[1, 5]]
    # No overlap
    assert sol.merge([[1, 2], [3, 4], [5, 6]]) == [[1, 2], [3, 4], [5, 6]]
    # All overlap
    assert sol.merge([[1, 10], [2, 3], [4, 5]]) == [[1, 10]]
    # Single interval
    assert sol.merge([[1, 2]]) == [[1, 2]]
    # Nested intervals
    assert sol.merge([[1, 4], [2, 3]]) == [[1, 4]]
    print("All tests passed.")


if __name__ == "__main__":
    test()
