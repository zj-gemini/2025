from typing import Dict

RESERVED_HEAD_KEY = "HEAD"
RESERVED_TAIL_KEY = "TAIL"


class Node:
    def __init__(self, key: int, value: int):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: Dict[int, Node] = {}  # Explicit typing
        self.head = Node(RESERVED_HEAD_KEY, 0)
        self.tail = Node(RESERVED_TAIL_KEY, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        """Remove a node from the linked list."""
        pre, next = node.prev, node.next
        pre.next = next
        next.prev = pre

    def _add(self, node: Node) -> None:
        """Add a node right after the head."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Update the value and move to the head
            node = self.cache[key]
            node.value = value
            self.get(key)
        else:
            node = Node(key, value)
            self._add(node)
            self.cache[key] = node
            if len(self.cache) > self.capacity:
                last_node = self.tail.prev
                self._remove(last_node)
                del self.cache[last_node.key]


def test():
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    print(cache.get(1))  # 1
    cache.put(3, 3)
    print(cache.get(2))  # -1
    cache.put(4, 4)
    print(cache.get(1))  # -1
    print(cache.get(3))  # 3
    print(cache.get(4))  # 4


test()
