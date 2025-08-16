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


class SolutionTrace:
    def lowestCommonAncestor(
        self, root: TreeNode, p: TreeNode, q: TreeNode
    ) -> TreeNode:
        """
        Finds the lowest common ancestor (LCA) of two nodes in a binary tree.
        This version uses a clear path-finding approach.
        """

        def find_path(node, target, path):
            """
            Helper function to find the path from root to the target node.
            Returns True if found, and fills 'path' with the nodes along the way.
            """
            if not node:
                return False
            path.append(node)
            if node == target:
                return True
            # Search left or right subtree
            if find_path(node.left, target, path) or find_path(
                node.right, target, path
            ):
                return True
            path.pop()  # Backtrack if not found in this path
            return False

        # Find paths from root to p and root to q
        path_p, path_q = [], []
        find_path(root, p, path_p)
        find_path(root, q, path_q)

        # Compare the paths to find the last common node
        lca = None
        for u, v in zip(path_p, path_q):
            if u == v:
                lca = u
            else:
                break
        return lca


class SolutionHardToUnderstand:
    def lowestCommonAncestor(
        self, root: TreeNode, p: TreeNode, q: TreeNode
    ) -> TreeNode:
        """
        Finds the lowest common ancestor (LCA) of two nodes in a binary tree.
        """
        # Base case: if root is None, or root is p or q, return root.
        # If root is p or q, then root is the LCA (if the other node is in its subtree).
        if not root or root == p or root == q:
            return root
        # Recursively search for p and q in the left and right subtrees
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        # If both left and right are not None, it means p and q are found in different subtrees,
        # so root is their lowest common ancestor.
        if left and right:
            return root
        # If only one side is not None, return that side (it could be p, q, or their ancestor)
        return left if left else right


# Helper to build tree from list (LeetCode style)
def build_tree(nodes: List[Optional[int]]) -> Optional[TreeNode]:
    """
    Builds a binary tree from a list of values (level-order).
    None values represent missing children.
    Returns the root and a dict mapping value to TreeNode.
    """
    if not nodes:
        return None, {}
    root = TreeNode(nodes[0])
    queue = [root]
    val_to_node = {nodes[0]: root}
    i = 1
    while queue and i < len(nodes):
        curr = queue.pop(0)
        # Assign left child if available
        if i < len(nodes) and nodes[i] is not None:
            curr.left = TreeNode(nodes[i])
            queue.append(curr.left)
            val_to_node[nodes[i]] = curr.left
        i += 1
        # Assign right child if available
        if i < len(nodes) and nodes[i] is not None:
            curr.right = TreeNode(nodes[i])
            queue.append(curr.right)
            val_to_node[nodes[i]] = curr.right
        i += 1
    return root, val_to_node


# Unit tests
def test():
    sol = Solution()
    # Example 1: root = [3,5,1,6,2,0,8,None,None,7,4], p = 5, q = 1, LCA = 3
    root1, nodes1 = build_tree([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4])
    p1, q1 = nodes1[5], nodes1[1]
    assert sol.lowestCommonAncestor(root1, p1, q1).val == 3
    # Example 2: p = 5, q = 4, LCA = 5
    p2, q2 = nodes1[5], nodes1[4]
    assert sol.lowestCommonAncestor(root1, p2, q2).val == 5
    # Example 3: root = [1,2], p = 1, q = 2, LCA = 1
    root2, nodes2 = build_tree([1, 2])
    p3, q3 = nodes2[1], nodes2[2]
    assert sol.lowestCommonAncestor(root2, p3, q3).val == 1
    print("All tests passed.")


if __name__ == "__main__":
    test()
