from typing import Optional, List
from collections import defaultdict, deque


# Definition for a binary tree node.
class TreeNode:
    def __init__(
        self,
        val: int = 0,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None,
    ):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        Returns the vertical order traversal of a binary tree.
        For nodes in the same row and column, sorts by value.
        """
        # List to hold tuples of (col, row, value)
        nodes = []

        # BFS traversal to record row, col, and value for each node
        queue = deque([(root, 0, 0)])  # (node, row, col)
        while queue:
            node, row, col = queue.popleft()
            if node:
                nodes.append((col, row, node.val))
                queue.append((node.left, row + 1, col - 1))
                queue.append((node.right, row + 1, col + 1))

        # Sort by col, then row, then value
        nodes.sort()
        res = []
        prev_col = float("-inf")
        for col, row, value in nodes:
            if col != prev_col:
                res.append([])
                prev_col = col
            res[-1].append(value)
        return res


class SolutionMine:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        # col, row -> list
        nums_per_pos = defaultdict(dict)
        # node, row, col
        q = deque([(root, 0, 0)])
        while q:
            node, row, col = q.popleft()
            if row not in nums_per_pos[col]:
                nums_per_pos[col][row] = []
            nums_per_pos[col][row].append(node.val)
            if node.left:
                q.append((node.left, row + 1, col - 1))
            if node.right:
                q.append((node.right, row + 1, col + 1))
        res = []
        for col in sorted(nums_per_pos):
            sub_list = []
            for row in nums_per_pos[col]:
                sub_list.extend(sorted(nums_per_pos[col][row]))
            res.append(sub_list)
        return res


# Helper to build tree from list (LeetCode style)
def build_tree(nodes: List[Optional[int]]) -> Optional[TreeNode]:
    """
    Builds a binary tree from a list of values (level-order).
    None values represent missing children.
    """
    if not nodes:
        return None
    root = TreeNode(nodes[0])
    queue = [root]
    i = 1
    while queue and i < len(nodes):
        curr = queue.pop(0)
        # Assign left child if available
        if i < len(nodes) and nodes[i] is not None:
            curr.left = TreeNode(nodes[i])
            queue.append(curr.left)
        i += 1
        # Assign right child if available
        if i < len(nodes) and nodes[i] is not None:
            curr.right = TreeNode(nodes[i])
            queue.append(curr.right)
        i += 1
    return root


# Unit tests
def test():
    sol = Solution()
    # Example 1
    root1 = build_tree([3, 9, 20, None, None, 15, 7])
    assert sol.verticalTraversal(root1) == [[9], [3, 15], [20], [7]]
    # Example 2
    root2 = build_tree([1, 2, 3, 4, 5, 6, 7])
    assert sol.verticalTraversal(root2) == [[4], [2], [1, 5, 6], [3], [7]]
    # Example 3
    root3 = build_tree([1, 2, 3, 4, 6, 5, 7])
    assert sol.verticalTraversal(root3) == [[4], [2], [1, 5, 6], [3], [7]]
    # Single node
    root4 = build_tree([42])
    assert sol.verticalTraversal(root4) == [[42]]
    print("All tests passed.")


if __name__ == "__main__":
    test()
