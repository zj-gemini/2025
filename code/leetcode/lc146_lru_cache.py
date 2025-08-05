# LRU
import dataclasses
from typing import Dict
from collections import OrderedDict


# Define two way linked node
@dataclasses.dataclass
class Node:
    key: int
    val: int
    next = None
    pre = None


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: Dict[int, Node] = {}
        # need dummy head and tail nodes to help look up
        self.head = Node(-1, 0)
        self.tail = Node(-1, 0)
        self.head.next = self.tail
        self.tail.pre = self.head

    # Remove node from the list
    def _remove(self, node: Node) -> None:
        pre, next = node.pre, node.next
        pre.next = next
        next.pre = pre

    # Add a node to front of the list
    def _add(self, node: Node) -> None:
        pre, next = self.head, self.head.next
        node.pre = pre
        node.next = next
        pre.next = node
        next.pre = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._remove(node)
        self._add(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache[key].val = value
            _ = self.get(key)
        else:
            self.cache[key] = Node(key, value)
            self._add(self.cache[key])
            if len(self.cache) > self.capacity:
                # Remove the last node in the list
                node = self.tail.pre
                del self.cache[node.key]
                self._remove(node)


class LRUCacheSimple:

    def __init__(self, capacity: int):
        self.cache: OrderedDict[int, int] = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        # Simply move it to the end
        val = self.cache[key]
        del self.cache[key]
        self.cache[key] = val
        return val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache[key] = value
            _ = self.get(key)
            return
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            _ = self.cache.popitem(last=False)


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
