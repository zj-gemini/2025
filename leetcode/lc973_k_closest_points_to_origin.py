from typing import List
import heapq

from collections import Counter


class SolutionCounter:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        distances = Counter()
        for i, (x, y) in enumerate(points):
            distances[i] = -(x * x + y * y)
        return [points[item[0]] for item in distances.most_common(k)]


class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        """
        Returns the k closest points to the origin (0, 0).
        Uses a heap for efficiency.
        """
        # Compute squared distance for each point and use nsmallest to get k closest
        return heapq.nsmallest(k, points, key=lambda p: p[0] ** 2 + p[1] ** 2)


class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        """
        Returns the k closest points to the origin (0, 0).
        Uses a heap for efficiency.
        """
        # Compute squared distance for each point and use nsmallest to get k closest
        return heapq.nsmallest(k, points, key=lambda p: p[0] ** 2 + p[1] ** 2)


# Unit tests
def test():
    sol = Solution()
    # Example 1
    assert sorted(sol.kClosest([[1, 3], [-2, 2]], 1)) == [[-2, 2]]
    # Example 2
    result = sol.kClosest([[3, 3], [5, -1], [-2, 4]], 2)
    assert sorted(result) == sorted([[3, 3], [-2, 4]])
    # Example 3: k equals number of points
    assert sorted(sol.kClosest([[1, 2], [2, 1], [0, 0]], 3)) == sorted(
        [[1, 2], [2, 1], [0, 0]]
    )
    # Example 4: All points at origin
    assert sorted(sol.kClosest([[0, 0], [0, 0], [0, 0]], 2)) == [[0, 0], [0, 0]]
    print("All tests passed.")


if __name__ == "__main__":
    test()
