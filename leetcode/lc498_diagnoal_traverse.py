from typing import List


class Solution:
    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        """
        Returns all elements of the matrix in diagonal order.
        Traverses the matrix in a zigzag (up-right and down-left) diagonal pattern.
        """
        m = len(mat)
        if not m:
            return []
        n = len(mat[0])
        if not n:
            return []

        dirs = [(-1, 1), (1, -1)]  # Directions: up-right, down-left
        i_dir = 0  # Current direction index: 0 for up-right, 1 for down-left

        rst = []  # Result list to store the traversal order

        i = j = 0  # Start from the top-left corner
        while True:
            rst.append(mat[i][j])
            # If we've reached the bottom-right corner, we're done
            if i == m - 1 and j == n - 1:
                break
            # Calculate the next cell in the current direction
            next_i, next_j = i + dirs[i_dir][0], j + dirs[i_dir][1]
            # Check if the next cell is out of bounds
            i_out = not (0 <= next_i < m)
            j_out = not (0 <= next_j < n)
            if i_out and j_out:
                # If both row and column are out of bounds,
                # move down if possible, else move right
                if i + 1 < m:
                    i += 1
                else:
                    j += 1
                i_dir = 1 - i_dir  # Switch direction
            elif i_out:
                # If row is out of bounds, move right and switch direction
                j += 1
                i_dir = 1 - i_dir
            elif j_out:
                # If column is out of bounds, move down and switch direction
                i += 1
                i_dir = 1 - i_dir
            else:
                # If not out of bounds, continue in the current direction
                i, j = next_i, next_j
        return rst


# Unit tests
def test():
    sol = Solution()
    # Example 1
    mat1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert sol.findDiagonalOrder(mat1) == [1, 2, 4, 7, 5, 3, 6, 8, 9]
    # Example 2: 1x1 matrix
    mat2 = [[42]]
    assert sol.findDiagonalOrder(mat2) == [42]
    # Example 3: 2x3 matrix
    mat3 = [[1, 2, 3], [4, 5, 6]]
    assert sol.findDiagonalOrder(mat3) == [1, 2, 4, 5, 3, 6]
    # Example 4: 3x2 matrix
    mat4 = [[1, 2], [3, 4], [5, 6]]
    assert sol.findDiagonalOrder(mat4) == [1, 2, 3, 5, 4, 6]
    # Example 5: Empty matrix
    mat5 = []
    assert sol.findDiagonalOrder(mat5) == []
    print("All tests passed.")


if __name__ == "__main__":
    test()
