import dataclasses
from typing import Dict
from collections import defaultdict, OrderedDict


@dataclasses.dataclass
class Node:
    """Node to store value and frequency for each key."""

    val: int
    freq: int = 0


class LFUCache:
    """
    LFU (Least Frequently Used) Cache implementation.
    Supports get and put in O(1) average time.
    When capacity is reached, evicts the least frequently used key.
    If multiple keys have the same frequency, evicts the least recently used among them.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: Dict[int, Node] = {}  # key -> Node(val, freq)
        self.freq_to_keys = defaultdict(
            OrderedDict
        )  # freq -> OrderedDict of keys (for LRU)
        self.min_freq = 0  # Tracks the minimum frequency in the cache

    def get(self, key: int) -> int:
        """
        Returns the value of the key if present, else -1.
        Also updates the frequency and recency of the key.
        """
        if key not in self.cache:
            return -1
        self._update_freq(key)
        return self.cache[key].val

    def put(self, key: int, value: int) -> None:
        """
        Inserts or updates the value of the key.
        If the cache reaches capacity, evicts the LFU key (LRU among ties).
        """
        if self.capacity == 0:
            return

        if key in self.cache:
            # Update value and frequency
            self.cache[key].val = value
            self._update_freq(key)
            return

        # Evict if at capacity
        if len(self.cache) >= self.capacity:
            lru_keys = self.freq_to_keys[self.min_freq]
            key_to_remove, _ = lru_keys.popitem(last=False)  # Remove LRU among min_freq
            del self.cache[key_to_remove]
            # Clean up empty frequency bucket
            if not lru_keys:
                del self.freq_to_keys[self.min_freq]

        # Insert new key with freq 1
        self.cache[key] = Node(value, 1)
        self.freq_to_keys[1][key] = None
        self.min_freq = 1  # Reset min_freq for new key

    def _update_freq(self, key: int):
        """
        Helper to update frequency and recency of a key.
        Moves the key to the next frequency bucket.
        Updates min_freq if needed.
        """
        freq = self.cache[key].freq
        # Remove key from current frequency's OrderedDict
        del self.freq_to_keys[freq][key]
        # If no keys left at this frequency, clean up and update min_freq
        if not self.freq_to_keys[freq]:
            del self.freq_to_keys[freq]
            if self.min_freq == freq:
                self.min_freq = freq + 1
        # Add key to next higher frequency's OrderedDict
        self.freq_to_keys[freq + 1][key] = None
        self.cache[key].freq = freq + 1


# Unit tests
import unittest


class TestLFUCache(unittest.TestCase):
    def test_leetcode_input(self):
        # LeetCode example input/output
        lfu = LFUCache(2)
        lfu.put(1, 1)  # cache=[1,_], cnt(1)=1
        lfu.put(2, 2)  # cache=[2,1], cnt(2)=1, cnt(1)=1
        self.assertEqual(lfu.get(1), 1)  # cache=[1,2], cnt(1)=2
        lfu.put(3, 3)  # evicts key 2, cache=[3,1], cnt(3)=1, cnt(1)=2
        self.assertEqual(lfu.get(2), -1)
        self.assertEqual(lfu.get(3), 3)  # cache=[3,1], cnt(3)=2
        lfu.put(4, 4)  # evicts key 1, cache=[4,3], cnt(4)=1, cnt(3)=2
        self.assertEqual(lfu.get(1), -1)
        self.assertEqual(lfu.get(3), 3)  # cache=[3,4], cnt(3)=3
        self.assertEqual(lfu.get(4), 4)  # cache=[4,3], cnt(4)=2

    def test_zero_capacity(self):
        # Cache with zero capacity should never store anything
        lfu = LFUCache(0)
        lfu.put(1, 1)
        self.assertEqual(lfu.get(1), -1)

    def test_update(self):
        # Test updating value and eviction order
        lfu = LFUCache(2)
        lfu.put(1, 1)
        lfu.put(2, 2)
        lfu.put(1, 10)  # update value for key 1
        self.assertEqual(lfu.get(1), 10)
        lfu.put(3, 3)  # evicts key 2
        self.assertEqual(lfu.get(2), -1)


if __name__ == "__main__":
    unittest.main()
