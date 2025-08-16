from typing import List


class Solution:
    def findBuildings(self, heights: List[int]) -> List[int]:
        """
        Returns a list of indices of buildings that have an ocean view.
        A building has an ocean view if all the buildings to its right are shorter.
        """
        n = len(heights)
        result = []
        max_height = float("-inf")
        # Traverse from right to left
        for i in range(n - 1, -1, -1):
            if heights[i] > max_height:
                result.append(i)
                max_height = heights[i]
        # Reverse to get indices in increasing order
        return result[::-1]


# Unit tests
def test():
    sol = Solution()
    # Example 1
    assert sol.findBuildings([4, 2, 3, 1]) == [0, 2, 3]
    # Example 2
    assert sol.findBuildings([4, 3, 2, 1]) == [0, 1, 2, 3]
    # Example 3
    assert sol.findBuildings([1, 3, 2, 4]) == [3]
    # Example 4: All same height
    assert sol.findBuildings([2, 2, 2, 2]) == [3]
    # Example 5: Single building
    assert sol.findBuildings([10]) == [0]
    # Example 6: Increasing heights
    assert sol.findBuildings([1, 2, 3, 4]) == [3]
    print("All tests passed.")


if __name__ == "__main__":
    test()
