from typing import Optional, List
from collections import deque, defaultdict


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
    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        Returns the vertical order traversal of a binary tree's nodes' values.
        Uses BFS to ensure top-to-bottom and left-to-right order within each column.
        """
        if not root:
            return []
        nums_per_pos = defaultdict(list)  # pos -> List of node values at that column
        layer = [(root, 0)]  # Each element is (node, column index)
        while layer:
            next_layer = []
            for node, pos in layer:
                nums_per_pos[pos].append(node.val)
                # Left child goes to column pos-1, right child to pos+1
                if node.left:
                    next_layer.append((node.left, pos - 1))
                if node.right:
                    next_layer.append((node.right, pos + 1))
            layer = next_layer

        # Collect results from leftmost to rightmost column
        return [nums_per_pos[pos] for pos in sorted(nums_per_pos.keys())]


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
    assert sol.verticalOrder(root1) == [[9], [3, 15], [20], [7]]
    # Example 2
    root2 = build_tree([3, 9, 8, 4, 0, 1, 7])
    assert sol.verticalOrder(root2) == [[4], [9], [3, 0, 1], [8], [7]]
    # Example 3
    root3 = build_tree(
        [1, 2, 3, 4, 10, 9, 11, None, 5, None, None, None, None, None, None, None, 6]
    )
    assert sol.verticalOrder(root3) == [[4], [2, 5], [1, 10, 9, 6], [3], [11]]
    # Empty tree
    root4 = build_tree([])
    assert sol.verticalOrder(root4) == []
    # Single node
    root5 = build_tree([42])
    assert sol.verticalOrder(root5) == [[42]]
    print("All tests passed.")


if __name__ == "__main__":
    test()
