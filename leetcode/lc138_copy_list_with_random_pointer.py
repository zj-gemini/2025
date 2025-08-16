from typing import Optional, List, Union


# Definition for a Node.
class Node:
    def __init__(self, x: int, next: "Node" = None, random: "Node" = None):
        self.val = int(x)
        self.next = next
        self.random = random


class Solution:
    def copyRandomList(self, head: "Optional[Node]") -> "Optional[Node]":
        """
        Creates a deep copy of a linked list with random pointers.
        Uses a hash map to map original nodes to their copies.
        """
        if not head:
            return None
        old_to_new = {}
        # First pass: copy all nodes and store mapping
        curr = head
        while curr:
            old_to_new[curr] = Node(curr.val)
            curr = curr.next
        # Second pass: assign next and random pointers
        curr = head
        while curr:
            if curr.next:
                old_to_new[curr].next = old_to_new.get(curr.next)
            if curr.random:
                old_to_new[curr].random = old_to_new.get(curr.random)
            curr = curr.next
        return old_to_new[head]


# Helper to build a linked list from LeetCode-style input
def build_linked_list(data: List[List[Union[int, None]]]) -> Optional[Node]:
    if not data:
        return None
    nodes = [Node(val) for val, _ in data]
    for i, (_, rand_idx) in enumerate(data):
        if i < len(nodes) - 1:
            nodes[i].next = nodes[i + 1]
        if rand_idx is not None:
            nodes[i].random = nodes[rand_idx]
    return nodes[0]


# Helper to convert linked list to LeetCode-style output
def linked_list_to_list(head: Optional[Node]) -> List[List[Union[int, None]]]:
    if not head:
        return []
    nodes = []
    node_to_index = {}
    curr = head
    idx = 0
    while curr:
        node_to_index[curr] = idx
        nodes.append(curr)
        curr = curr.next
        idx += 1
    result = []
    for i, node in enumerate(nodes):
        rand_idx = node_to_index[node.random] if node.random else None
        result.append([node.val, rand_idx])
    return result


# Unit tests
def test():
    sol = Solution()
    # Example 1
    head1 = build_linked_list([[7, None], [13, 0], [11, 4], [10, 2], [1, 0]])
    copied1 = sol.copyRandomList(head1)
    assert linked_list_to_list(copied1) == [
        [7, None],
        [13, 0],
        [11, 4],
        [10, 2],
        [1, 0],
    ]
    # Example 2
    head2 = build_linked_list([[1, 1], [2, 1]])
    copied2 = sol.copyRandomList(head2)
    assert linked_list_to_list(copied2) == [[1, 1], [2, 1]]
    # Example 3
    head3 = build_linked_list([[3, None], [3, 0], [3, None]])
    copied3 = sol.copyRandomList(head3)
    assert linked_list_to_list(copied3) == [[3, None], [3, 0], [3, None]]
    # Example 4: Empty list
    head4 = build_linked_list([])
    copied4 = sol.copyRandomList(head4)
    assert linked_list_to_list(copied4) == []
    print("All tests passed.")


if __name__ == "__main__":
    test()
