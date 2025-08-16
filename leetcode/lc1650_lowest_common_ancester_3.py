# Definition for a Node.
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None


class Solution:
    def lowestCommonAncestor(self, p: "Node", q: "Node") -> "Node":
        """
        Finds the lowest common ancestor of nodes p and q in a binary tree with parent pointers.
        """
        # Use a set to record all ancestors of p
        ancestors = set()
        while p:
            ancestors.add(p)
            p = p.parent
        # Traverse ancestors of q, return the first one found in p's ancestors
        while q:
            if q in ancestors:
                return q
            q = q.parent
        return None  # Should not happen if p and q are in the tree


# Helper to build tree from list (LeetCode style) and set parent pointers
def build_tree_with_parent(nodes):
    """
    Builds a binary tree from a list of values (level-order), returns root and a dict of val->Node.
    None values represent missing children.
    """
    if not nodes:
        return None, {}
    root = Node(nodes[0])
    queue = [root]
    val_to_node = {nodes[0]: root}
    i = 1
    while queue and i < len(nodes):
        curr = queue.pop(0)
        # Assign left child if available
        if i < len(nodes) and nodes[i] is not None:
            curr.left = Node(nodes[i])
            curr.left.parent = curr
            queue.append(curr.left)
            val_to_node[nodes[i]] = curr.left
        i += 1
        # Assign right child if available
        if i < len(nodes) and nodes[i] is not None:
            curr.right = Node(nodes[i])
            curr.right.parent = curr
            queue.append(curr.right)
            val_to_node[nodes[i]] = curr.right
        i += 1
    return root, val_to_node


# Unit tests
def test():
    sol = Solution()
    # Example 1
    root1, nodes1 = build_tree_with_parent([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4])
    p1, q1 = nodes1[5], nodes1[1]
    assert sol.lowestCommonAncestor(p1, q1).val == 3
    # Example 2
    p2, q2 = nodes1[5], nodes1[4]
    assert sol.lowestCommonAncestor(p2, q2).val == 5
    # Example 3
    root2, nodes2 = build_tree_with_parent([1, 2])
    p3, q3 = nodes2[1], nodes2[2]
    assert sol.lowestCommonAncestor(p3, q3).val == 1
    print("All tests passed.")


if __name__ == "__main__":
    test()
