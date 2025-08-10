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
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        """
        Returns the values of the nodes you can see from the right side of the binary tree.
        Uses level-order traversal (BFS) to collect the last node of each level.
        """
        if not root:
            return []
        result = []
        layer = [root]
        while layer:
            result.append(layer[-1].val)  # Rightmost node at this level
            # Build the next layer
            next_layer = []
            for child in layer:
                if child.left:
                    next_layer.append(child.left)
                if child.right:
                    next_layer.append(child.right)
            layer = next_layer
        return result


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
        if i < len(nodes) and nodes[i] is not None:
            curr.left = TreeNode(nodes[i])
            queue.append(curr.left)
        i += 1
        if i < len(nodes) and nodes[i] is not None:
            curr.right = TreeNode(nodes[i])
            queue.append(curr.right)
        i += 1
    return root


# Unit tests
def test():
    sol = Solution()
    # Example 1: [1,2,3,None,5,None,4] => [1,3,4]
    root1 = build_tree([1, 2, 3, None, 5, None, 4])
    assert sol.rightSideView(root1) == [1, 3, 4]
    # Example 2: [1,None,3] => [1,3]
    root2 = build_tree([1, None, 3])
    assert sol.rightSideView(root2) == [1, 3]
    # Example 3: [] => []
    root3 = build_tree([])
    assert sol.rightSideView(root3) == []
    # Example 4: [1,2] => [1,2]
    root4 = build_tree([1, 2])
    assert sol.rightSideView(root4) == [1, 2]
    # Example 5: [1,2,3,4] => [1,3,4]
    root5 = build_tree([1, 2, 3, 4])
    assert sol.rightSideView(root5) == [1, 3, 4]
    print("All tests passed.")


if __name__ == "__main__":
    test()
