from typing import Optional, List
from collections import deque


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x: int):
        self.val = x
        self.left: Optional["TreeNode"] = None
        self.right: Optional["TreeNode"] = None


class Codec:
    def serialize(self, root):
        """
        Encodes a tree to a single string using preorder traversal (root, left, right).
        Uses "null" as a placeholder for missing children.
        """

        def dfs(node):
            if not node:
                vals.append("null")  # Mark missing node
                return
            vals.append(str(node.val))  # Record node value
            dfs(node.left)  # Serialize left subtree
            dfs(node.right)  # Serialize right subtree

        vals = []
        dfs(root)
        return ",".join(vals)

    def deserialize(self, data):
        """
        Decodes your encoded data to tree using preorder traversal.
        Recursively reconstructs the tree by reading values in order.
        """

        def dfs():
            val = next(vals)
            if val == "null":
                return None  # No node here
            node = TreeNode(int(val))
            node.left = dfs()  # Rebuild left subtree
            node.right = dfs()  # Rebuild right subtree
            return node

        vals = iter(data.split(","))
        return dfs()


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


# Helper to convert tree to list (LeetCode style)
def tree_to_list(root: Optional[TreeNode]) -> List[Optional[int]]:
    """
    Converts a binary tree to a list (level-order), using None for missing children.
    Trims trailing None values for compactness.
    """
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        if node:
            result.append(node.val)
            queue.append(node.left)
            queue.append(node.right)
        else:
            result.append(None)
    # Remove trailing None values for compact output
    while result and result[-1] is None:
        result.pop()
    return result


# Unit tests
def test():
    codec = Codec()
    # Example 1: Tree with left and right children
    root1 = build_tree([1, 2, 3, None, None, 4, 5])
    data1 = codec.serialize(root1)
    assert tree_to_list(codec.deserialize(data1)) == [1, 2, 3, None, None, 4, 5]
    # Example 2: Empty tree
    root2 = build_tree([])
    data2 = codec.serialize(root2)
    assert tree_to_list(codec.deserialize(data2)) == []
    # Example 3: Single node
    root3 = build_tree([42])
    data3 = codec.serialize(root3)
    assert tree_to_list(codec.deserialize(data3)) == [42]
    print("All tests passed.")


if __name__ == "__main__":
    test()
