from typing import List, Optional
import heapq


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"{self.val}->{self.next}"


class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        heap = []
        # Initialize the heap with the head node of each list
        for i, node in enumerate(lists):
            if node:
                heapq.heappush(heap, (node.val, i, node))
        dummy = ListNode(0)
        curr = dummy
        while heap:
            val, i, node = heapq.heappop(heap)
            curr.next = node
            curr = curr.next
            if node.next:
                heapq.heappush(heap, (node.next.val, i, node.next))
        return dummy.next


# Helper functions for testing
def build_linked_list(arr):
    dummy = ListNode(0)
    curr = dummy
    for x in arr:
        curr.next = ListNode(x)
        curr = curr.next
    return dummy.next


def linked_list_to_list(node):
    res = []
    while node:
        res.append(node.val)
        node = node.next
    return res


# Unit tests
def test_merge_k_lists():
    s = Solution()
    lists = [
        build_linked_list([1, 4, 5]),
        build_linked_list([1, 3, 4]),
        build_linked_list([2, 6]),
    ]
    merged = s.mergeKLists(lists)
    assert linked_list_to_list(merged) == [1, 1, 2, 3, 4, 4, 5, 6]

    lists = []
    merged = s.mergeKLists(lists)
    assert linked_list_to_list(merged) == []

    lists = [build_linked_list([])]
    merged = s.mergeKLists(lists)
    assert linked_list_to_list(merged) == []

    lists = [build_linked_list([1])]
    merged = s.mergeKLists(lists)
    assert linked_list_to_list(merged) == [1]

    print("All tests passed.")


if __name__ == "__main__":
    test_merge_k_lists()
