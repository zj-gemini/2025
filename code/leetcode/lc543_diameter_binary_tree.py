from typing import Optional, List


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
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        """
        Returns the length of the diameter of the binary tree.
        The diameter is the length of the longest path between any two nodes in the tree.
        """
        self.max_diameter = 0

        def depth(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            left_depth = depth(node.left)
            right_depth = depth(node.right)
            # Update the diameter at this node
            self.max_diameter = max(self.max_diameter, left_depth + right_depth)
            # Return the height of the tree rooted at this node
            return 1 + max(left_depth, right_depth)

        depth(root)
        return self.max_diameter


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
    # Example 1: [1,2,3,4,5] => diameter is 3 (path: 4-2-1-3 or 5-2-1-3)
    root1 = build_tree([1, 2, 3, 4, 5])
    assert sol.diameterOfBinaryTree(root1) == 3
    # Example 2: [1,2] => diameter is 1
    root2 = build_tree([1, 2])
    assert sol.diameterOfBinaryTree(root2) == 1
    # Example 3: [1] => diameter is 0
    root3 = build_tree([1])
    assert sol.diameterOfBinaryTree(root3) == 0
    # Example 4: [] => diameter is 0
    root4 = build_tree([])
    assert sol.diameterOfBinaryTree(root4) == 0
    print("All tests passed.")


if __name__ == "__main__":
    test()
