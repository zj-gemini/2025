from typing import List
from collections import deque


class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        """
        Returns the length of the shortest clear path in a binary matrix from top-left to bottom-right.
        A clear path consists of only 0s, and moves can be made in 8 directions.
        Returns -1 if no such path exists.
        """
        n = len(grid)
        if grid[0][0] != 0 or grid[n - 1][n - 1] != 0:
            return -1  # Start or end blocked

        # Directions: 8 possible moves (including diagonals)
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        queue = deque([(0, 0, 1)])  # (row, col, path_length)
        visited = [[False] * n for _ in range(n)]
        visited[0][0] = True

        while queue:
            r, c, length = queue.popleft()
            if r == n - 1 and c == n - 1:
                return length
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < n
                    and 0 <= nc < n
                    and not visited[nr][nc]
                    and grid[nr][nc] == 0
                ):
                    visited[nr][nc] = True
                    queue.append((nr, nc, length + 1))
        return -1


# Unit tests
def test():
    sol = Solution()
    # Example 1: Path exists
    grid1 = [[0, 1], [1, 0]]
    assert sol.shortestPathBinaryMatrix(grid1) == 2
    # Example 2: No path
    grid2 = [[0, 1, 1], [1, 1, 0], [1, 1, 0]]
    assert sol.shortestPathBinaryMatrix(grid2) == -1
    # Example 3: 1x1 grid, open
    grid3 = [[0]]
    assert sol.shortestPathBinaryMatrix(grid3) == 1
    # Example 4: 1x1 grid, blocked
    grid4 = [[1]]
    assert sol.shortestPathBinaryMatrix(grid4) == -1
    # Example 5: Larger grid, path exists
    grid5 = [[0, 0, 0], [1, 1, 0], [1, 1, 0]]
    assert sol.shortestPathBinaryMatrix(grid5) == 4
    print("All tests passed.")


if __name__ == "__main__":
    test()
