from typing import List, Optional


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


class Solution_dfs:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """
        Returns the preorder traversal of a binary tree's nodes' values.
        Preorder: root -> left -> right
        """
        result = []

        def dfs(node: Optional[TreeNode]):
            if not node:
                return
            result.append(node.val)
            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return result


class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """
        Iterative preorder traversal using a stack.
        Preorder: root -> left -> right
        """
        if root is None:
            return []

        stack = [root]  # Start with the root node
        output = []

        while stack:
            node = stack.pop()  # Pop the top node
            output.append(node.val)  # Visit the node

            # Push right child first so left child is processed first
            if node.right is not None:
                stack.append(node.right)
            if node.left is not None:
                stack.append(node.left)

        return output


# Helper function to build tree from list (LeetCode style input)
def build_tree(nodes: List[Optional[int]]) -> Optional[TreeNode]:
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
    # Example 1
    root1 = build_tree([1, None, 2, 3])
    assert sol.preorderTraversal(root1) == [1, 2, 3]
    # Example 2
    root2 = build_tree([1, 2, 3, 4, 5, None, 8, None, None, 6, 7, 9])
    assert sol.preorderTraversal(root2) == [1, 2, 4, 5, 6, 7, 3, 8, 9]
    # Example 3
    root3 = build_tree([])
    assert sol.preorderTraversal(root3) == []
    # Example 4
    root4 = build_tree([1])
    assert sol.preorderTraversal(root4) == [1]
    print("All tests passed.")


if __name__ == "__main__":
    test()
